"""Calendar windows and raw-return exports must not use their prediction labels."""
from pathlib import Path
import inspect

import nbformat
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from takehome import fitters

ROOT = Path(__file__).resolve().parents[1]


def calendar_folds(index, **kwargs):
    assert hasattr(fitters, 'calendar_walk_forward_folds'), 'Missing calendar-year fold generator'
    return fitters.calendar_walk_forward_folds(index, **kwargs)


def test_calendar_windows_are_two_years_not_bar_counts():
    index = pd.date_range('2016-01-02 06:05', '2025-07-01', freq='17h', tz='America/New_York')
    index = index[np.arange(len(index)) % 7 != 1]
    folds = calendar_folds(index)
    assert len(folds) == 4
    for i, fold in enumerate(folds):
        boundary = index[0] + pd.DateOffset(years=2*(i+1))
        end = index[0] + pd.DateOffset(years=2*(i+2))
        start = boundary - pd.DateOffset(years=2)
        assert fold['train_start'] == index.searchsorted(start)
        assert fold['train_stop'] == fold['predict_start'] == index.searchsorted(boundary)
        assert fold['predict_stop'] == index.searchsorted(end)
        assert fold['train_start_time'] == start
        assert fold['predict_stop_time'] == end
    assert folds[-1]['predict_stop'] == len(index)
    assert len({f['train_stop']-f['train_start'] for f in folds}) > 1


def test_calendar_leap_day_anchor_and_partial_last_window():
    index = pd.date_range('2016-02-29 06:05', '2021-03-01', freq='D', tz='America/New_York')
    folds = calendar_folds(index)
    assert [f['predict_start_time'] for f in folds] == [
        pd.Timestamp('2018-02-28 06:05', tz=index.tz),
        pd.Timestamp('2020-02-29 06:05', tz=index.tz),
    ]
    assert folds[-1]['predict_stop'] == len(index)
    assert calendar_folds(index[:10]) == []


@pytest.mark.parametrize('change', ['unsorted', 'duplicate', 'nat', 'range'])
def test_calendar_requires_sorted_unique_datetime_index(change):
    index = pd.date_range('2015', periods=6, freq='YS')
    if change == 'unsorted': index = index[::-1]
    if change == 'duplicate': index = index.insert(0, index[0])
    if change == 'nat': index = index.insert(len(index), pd.NaT)
    if change == 'range': index = pd.RangeIndex(6)
    assert hasattr(fitters, 'calendar_walk_forward_folds')
    with pytest.raises(ValueError):
        fitters.calendar_walk_forward_folds(index)


def design():
    index = pd.date_range('2014-01-02', '2024-07-01', freq='14D', tz='America/New_York')
    rng = np.random.default_rng(32)
    X = rng.normal(size=(len(index), 3))
    y = .2 + X @ [.3, -.1, .07] + rng.normal(0, .01, len(index))
    return index, X, y


def test_calendar_batch_paths_equal_independent_two_year_fits():
    index, X, y = design()
    folds = calendar_folds(index)
    assert 'folds' in inspect.signature(fitters.BatchedFitters).parameters
    config = dict(alpha=.03, decay=.97, fit_intercept=True, tol=1e-12)
    path = fitters.BatchedFitters(fitters.BatchLasso(3, **config), folds=folds).fit(X, y)
    for f in folds:
        a,b,c,d = (f[k] for k in ('train_start','train_stop','predict_start','predict_stop'))
        ref = fitters.BatchLasso(3, **config).fit(X[a:b], y[a:b])
        assert_allclose(path.prediction_[c:d], ref.predict(X[c:d]), atol=1e-8)
    reconstructed = np.einsum('ij,ij->i', X, path.get_coefs(lag=1)) + path.get_intercepts(lag=1)
    assert_allclose(path.prediction_, reconstructed, equal_nan=True)
    boundary = folds[1]['predict_start']
    altered = y.copy(); altered[boundary:] += 100
    alternate = fitters.BatchedFitters(fitters.BatchLasso(3, **config), folds=folds).fit(X, altered)
    assert_allclose(path.prediction_[:folds[1]['predict_stop']],
                    alternate.prediction_[:folds[1]['predict_stop']], equal_nan=True)
    altered = y.copy(); altered[:folds[2]['train_start']] += 100
    alternate = fitters.BatchedFitters(fitters.BatchLasso(3, **config), folds=folds).fit(X, altered)
    assert_allclose(path.prediction_[folds[2]['predict_start']:folds[2]['predict_stop']],
                    alternate.prediction_[folds[2]['predict_start']:folds[2]['predict_stop']], atol=1e-8)


def test_rolling_frozen_stream_matches_batch_while_live_history_stays_continuous():
    index, X, y = design()
    folds = calendar_folds(index)
    m = fitters.StreamingWeightedLasso(3, .97, .03, fit_intercept=True,
        store_history=True, tol=1e-11, max_iter=20000)
    out = fitters.stream_at_folds(m, X, y, folds, audit_folds=range(len(folds)))
    batch = fitters.walk_forward_sweep({'lasso': fitters.BatchLasso(3,.97,.03,fit_intercept=True)},
                                      X, y, folds=folds)['lasso']
    assert_allclose(out['frozen'], batch.prediction_, equal_nan=True, atol=1e-7)
    ref = fitters.StreamingWeightedLasso(3, .97,.03,fit_intercept=True,tol=1e-11,max_iter=20000)
    assert_allclose(out['live'], ref.fit_predict(X,y), equal_nan=True, atol=1e-10)
    assert m.n_seen_ == len(y) and len(m.get_coefs()) == len(y)
    for i, f in enumerate(folds):
        moments = fitters.batch_moments(X[f['train_start']:f['train_stop']],
            y[f['train_start']:f['train_stop']], np.ones(f['train_stop']-f['train_start']), .97)
        assert_allclose(out['states'][i][0], moments[0], atol=1e-10)
        assert_allclose(out['states'][i][2], moments[2], atol=1e-10)


def test_explicit_folds_reject_future_training_and_overlap():
    index,X,y = design()
    folds = calendar_folds(index)
    bad = [dict(f) for f in folds]; bad[0]['train_stop'] += 1
    with pytest.raises(ValueError):
        fitters.walk_forward_sweep({'ridge':fitters.BatchRidge(3)}, X, y, folds=bad)
    bad = [dict(f) for f in folds]; bad[1]['predict_start'] -= 1
    with pytest.raises(ValueError):
        fitters.walk_forward_sweep({'ridge':fitters.BatchRidge(3)}, X, y, folds=bad)


def test_forecast_export_is_raw_return_predictions_and_blackout_only(tmp_path):
    assert (ROOT/'src/takehome/forecast.py').exists(), 'Missing forecast export'
    from takehome.forecast import write_lasso_forecasts
    index, X, y = design()
    raw = pd.DataFrame({'x1': X[:,0], 'x99':X[:,1], 'cashflow':X[:,2],
                        'volume':1., 'ret_5m':y}, index=index.rename('msgStamp'))
    cutoff = index[-1] - pd.DateOffset(years=2)
    raw.loc[raw.index >= cutoff, 'ret_5m'] = np.nan
    raw.iloc[5, raw.columns.get_loc('ret_5m')] = np.nan
    data = tmp_path/'input.parquet'; raw.to_parquet(data)
    result = write_lasso_forecasts(data, tmp_path/'output', hl=10, alpha=.03)
    submitted = pd.read_parquet(result['oos_parquet'])
    assert list(submitted.columns) == ['forecast']
    assert submitted.index.equals(raw.loc[raw.index >= cutoff].index)
    assert np.isfinite(submitted.forecast).all()
    train = raw.loc[(raw.index>=cutoff-pd.DateOffset(years=2)) & (raw.index<cutoff)]
    xx = train[['x1','x99','cashflow']].to_numpy()
    ref=fitters.BatchLasso(3,2**(-1/10),.03,fit_intercept=True).fit(xx,train.ret_5m.to_numpy(),
                            W=np.isfinite(train.ret_5m).astype(float).to_numpy())
    assert_allclose(submitted.forecast,ref.predict(raw.loc[submitted.index,['x1','x99','cashflow']].to_numpy()),atol=1e-8)
    validation=pd.read_parquet(result['validation_parquet'])
    assert validation.index.max() < pd.Timestamp(result['metadata']['research_cutoff'])
    assert validation.forecast.notna().all()
    csv=pd.read_csv(result['oos_csv'])
    assert list(csv.columns)==['msgStamp','forecast']
    assert_allclose(csv.forecast,submitted.forecast,rtol=1e-12,atol=1e-14)


def test_export_rejects_non_withheld_final_two_years(tmp_path):
    assert (ROOT/'src/takehome/forecast.py').exists(), 'Missing forecast export'
    from takehome.forecast import write_lasso_forecasts
    index,X,y=design()
    raw=pd.DataFrame({'x1':X[:,0],'cashflow':X[:,1],'volume':1.,'ret_5m':y},index=index.rename('msgStamp'))
    data=tmp_path/'input.parquet';raw.to_parquet(data)
    with pytest.raises(ValueError,match='withheld|blackout|missing'):
        write_lasso_forecasts(data,tmp_path/'output')


def test_all_notebooks_use_the_same_two_year_fold_generator():
    for path in (ROOT/'notebooks').glob('*.ipynb'):
        nb=nbformat.read(path,4)
        code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
        assert 'calendar_walk_forward_folds' in code, path.name
        assert 'train_size=None' not in code.replace(' ',''), path.name
        assert 'min_train_size=252*288' not in code.replace(' ',''), path.name
        if path.stem.startswith('takehome'):
            assert any('batch_forecast_export' in c.metadata.get('tags',[]) for c in nb.cells)


def test_explicit_partial_fold_history_reconstructs_only_predicted_rows():
    _, X, y = design()
    folds = [dict(train_start=0, train_stop=10, predict_start=10, predict_stop=20)]
    path = fitters.BatchedFitters(fitters.BatchRidge(3), folds=folds).fit(X, y)
    reconstructed = (np.einsum('ij,ij->i', X, path.get_coefs(lag=1))
                     + path.get_intercepts(lag=1))
    assert_allclose(reconstructed, path.prediction_, equal_nan=True)


def test_explicit_fold_generator_can_be_refitted_without_losing_windows():
    index, X, y = design()
    folds = calendar_folds(index)
    path = fitters.walk_forward_sweep({'ridge': fitters.BatchRidge(3)}, X, y,
                                     folds=(f.copy() for f in folds))['ridge']
    expected = path.prediction_.copy()
    assert len(path.explicit_folds) == len(folds)
    path.fit(X, y)
    assert_allclose(path.prediction_, expected, equal_nan=True)

"""The diagnostic must change sizing only, not fitting, timing or the main default."""
from pathlib import Path
import inspect

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_allclose

from takehome import features

ROOT = Path(__file__).resolve().parents[1]


def sample(n=80):
    rng = np.random.default_rng(418)
    idx = pd.date_range('2020-01-01', periods=n, freq='12h', tz='America/New_York')
    X = pd.DataFrame(rng.normal(size=(n, 2)), index=idx, columns=['x1', 'x2'])
    r = pd.Series(rng.normal(0, .01, n), index=idx, name='ret_5m')
    return X, r


def asset_pnl(X, r, hl=5):
    assert 'asset_vol' in inspect.signature(features.standalone_pnl).parameters, 'Missing opt-in asset volatility sizing'
    return features.standalone_pnl(X, r, hl, asset_vol=True)


@pytest.mark.parametrize('series', [False, True])
def test_asset_vol_formula_has_row_alignment_and_lagged_returns(series):
    X, r = sample()
    X.iloc[10, 0] = 0.; X.iloc[20, 1] = np.nan
    r.iloc[14] = 0.; r.iloc[25] = np.nan
    if series:
        X = X.x1
    denominator = features.ts_std(X, 5).mul(features.ts_std(r.shift(), 5), axis=0).replace(0, np.nan)
    expected = X.div(denominator).mul(r, axis=0)
    actual = asset_pnl(X, r)
    (assert_series_equal if series else assert_frame_equal)(actual, expected)
    assert actual.shape == X.shape
    assert actual.iloc[:5].isna().all().all()


def test_asset_vol_denominator_does_not_see_current_or_future_return():
    X, r = sample()
    old = asset_pnl(X, r)
    altered = r.copy(); altered.iloc[35:] *= 1000
    new = asset_pnl(X, altered)
    assert_frame_equal(old.iloc[:35], new.iloc[:35])
    assert_allclose(new.iloc[35] / altered.iloc[35], old.iloc[35] / r.iloc[35])
    assert_frame_equal(old.iloc[:35], asset_pnl(X.iloc[:35], r.iloc[:35]))


def test_asset_vol_is_invariant_to_positive_measurement_unit_changes():
    X, r = sample()
    base = asset_pnl(X, r)
    assert_allclose(asset_pnl(X * 1e-8, r), base, rtol=1e-10, equal_nan=True)
    assert_allclose(asset_pnl(X, r * 100), base, rtol=1e-10, equal_nan=True)


def test_asset_vol_does_not_fill_blackout_labels_or_zero_volatility():
    X, r = sample()
    r.iloc[35:] = np.nan
    assert asset_pnl(X, r).iloc[35:].isna().all().all()
    assert asset_pnl(X * 0, r).isna().all().all()
    assert asset_pnl(X, r * 0).isna().all().all()


def test_asset_vol_rejects_misaligned_index():
    X, r = sample()
    assert 'asset_vol' in inspect.signature(features.standalone_pnl).parameters
    with pytest.raises(ValueError, match='index'):
        asset_pnl(X, r.iloc[:-1])


def test_default_backtest_is_unchanged_and_asset_flag_is_forwarded():
    X, r = sample()
    expected = X.div(features.ts_std(X, 5).pow(2).replace(0, np.nan)).mul(r, axis=0)
    assert_frame_equal(features.standalone_pnl(X, r, 5), expected)
    pnl = asset_pnl(X, r)
    assert_frame_equal(features.backtest(X, r, 5, asset_vol=True), pnl)
    assert_frame_equal(features.backtest(X, r, 5), expected)


def test_batched_asset_evaluation_matches_materialized_pnl_and_meta():
    X, r = sample()
    pnl = asset_pnl(X, r)
    daily, summary, meta = features.evaluate_features(
        [X[['x1']], X[['x2']]], r, 5, meta=True, meta_hl=7, positive_only=True, asset_vol=True)
    assert_frame_equal(daily, pnl.resample('D').sum())
    assert_allclose(summary.loc[X.columns, 'pnl_observations'], pnl.count())
    assert_series_equal(meta, features.combine_pnls(pnl, 7, positive_only=True))


def base_notebook():
    path = ROOT / 'notebooks/takehome.ipynb'
    return nbformat.read(path if path.exists() else ROOT / 'notebooks/01_eda.ipynb', as_version=4)


def test_main_notebook_is_renamed_and_comparison_is_a_separate_copy():
    assert (ROOT / 'notebooks/takehome.ipynb').exists(), 'Main notebook has not been renamed'
    assert not (ROOT / 'notebooks/01_eda.ipynb').exists()
    assert not (ROOT / 'notebooks/takehome_asset_vol.ipynb').exists()
    assert (ROOT / 'notebooks/asset_vol.ipynb').exists()


def test_only_h1_is_kept_as_a_lookahead_with_h0_control():
    nb = base_notebook()
    tags = {t for c in nb.cells for t in c.metadata.get('tags', []) if t.startswith('oracle_lead_')}
    assert tags == {'oracle_lead_0', 'oracle_lead_1'}
    X, r = sample()
    df = X.assign(ret_5m=r)
    ns = dict(df=df, x_cols=X.columns, rows=np.arange(len(X)), HL=5, META_HL=7,
              evaluate_features=features.evaluate_features, sharpe=features.sharpe,
              display_results=lambda *args, **kwargs: None, display=lambda *args: None)
    for tag in ('oracle_lead_0', 'oracle_lead_1'):
        cell = next(c for c in nb.cells if tag in c.metadata.get('tags', []))
        exec(cell.source, ns)
        assert_series_equal(ns['oracle_returns'], r.where(np.arange(len(r)) < len(r)-1))
    assert [row['h'] for row in ns['oracle_records']] == [0, 1]


def test_coefficient_plot_omits_first_hl_rows_without_trimming_beta(monkeypatch):
    nb = base_notebook()
    cell = next(c for c in nb.cells if c.cell_type == 'code' and 'ax.plot(clock,' in c.source)
    X, r = sample(30)
    original = X.copy(deep=True)
    shown = []
    monkeypatch.setattr(plt, 'show', lambda: shown.append(plt.gcf()))
    ns = dict(np=np, pd=pd, plt=plt, beta=X, HL=7, yhat=r, online_yhat=r)
    exec(cell.source, ns)
    line = shown[0].axes[0].lines[0]
    assert len(line.get_ydata()) == len(X)-7
    assert_allclose(line.get_ydata(), X.iloc[7:, 0])
    assert_allclose(line.get_xdata(), X.index[7:].as_unit('ns').asi8 / 86_400_000_000_000)
    assert_frame_equal(ns['beta'], original)
    plt.close('all')

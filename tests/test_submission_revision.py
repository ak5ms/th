"""Submission contracts: causal features, safe sizing and notebook-owned Ridge export."""
from pathlib import Path
import inspect
import json
import nbformat
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal
from takehome import features, data
from takehome.normalization import signal_weights

ROOT = Path(__file__).resolve().parents[1]
RULE = dict(method='std', floor_fraction=.25, cap=3., lag=1)


def sample(n=200):
    rng = np.random.default_rng(64)
    ix = pd.date_range('2020-01-01', periods=n, freq='6h', tz='America/New_York', name='msgStamp')
    X = pd.DataFrame(rng.normal(size=(n, 3)), index=ix, columns=['x1','x2','x100'])
    r = pd.Series(rng.normal(0,.01,n), index=ix, name='ret_5m')
    return X, r


def test_submission_sizing_is_explicit_and_legacy_default_is_unchanged():
    assert 'normalization' in inspect.signature(features.backtest).parameters
    X, r = sample()
    expected = signal_weights(X, 5, **RULE).mul(r, axis=0)
    assert_frame_equal(features.backtest(X,r,5,normalization=RULE), expected)
    old = X.div(features.ts_std(X,5)**2).mul(r,axis=0)
    assert_frame_equal(features.backtest(X,r,5), old)
    with pytest.raises(ValueError, match='asset_vol|normalization'):
        features.backtest(X,r,5,normalization=RULE,asset_vol=True)


def test_normalized_feature_blocks_and_meta_equal_dense_reference():
    assert 'normalization' in inspect.signature(features.evaluate_features).parameters
    X, r = sample()
    X.iloc[15:20,0] = 0
    X.iloc[25:28,1] = np.nan
    w = signal_weights(X, 5, **RULE)
    pnl = w.mul(r,axis=0)
    got, summary, meta = features.evaluate_features(
        [X[['x1']],X[['x2','x100']]],r,5,normalization=RULE,meta=True,meta_hl=7)
    assert_frame_equal(got,pnl.resample('D').sum())
    assert_series_equal(meta,features.combine_pnls(pnl,7))
    assert_allclose(summary.loc[X.columns,'max_abs_weight'],w.abs().max())
    assert summary.max_abs_weight.max() <= 3


def test_sizing_cap_and_prefix_invariance_even_on_collapse_and_outlier():
    assert 'normalization' in inspect.signature(features.backtest).parameters
    X,r=sample(600)
    X.iloc[30:400]=.0001; X.iloc[410:430]=0; X.iloc[450:460]=np.nan
    X.iloc[530]=1e12
    full=features.backtest(X,r,5,normalization=RULE)
    prefix=features.backtest(X.iloc[:300],r.iloc[:300],5,normalization=RULE)
    assert_frame_equal(prefix,full.iloc[:300])
    assert (full.abs().le(3*r.abs(),axis=0)|full.isna()).all().all()
    changed=r*100
    assert_allclose(features.backtest(X,changed,5,normalization=RULE),full*100,equal_nan=True)


def test_dszl_design_matches_grouped_reference_and_is_prefix_causal():
    assert hasattr(features,'dszl_design')
    X,_=sample()
    X.iloc[10:15,0]=np.nan;X.iloc[50,1]=np.inf;X.iloc[40:45,2]=0
    reference=features.dszl(X,hl=3).replace([np.inf,-np.inf],np.nan).fillna(0)
    actual=features.dszl_design(X,hl=3,batch_size=2)
    assert_frame_equal(actual,reference)
    assert actual.to_numpy().flags.c_contiguous
    assert_frame_equal(features.dszl_design(X.iloc[:100],hl=3,batch_size=1),actual.iloc[:100])
    changed=X.copy();changed.iloc[100:]*=1000
    assert_frame_equal(features.dszl_design(changed,hl=3).iloc[:100],actual.iloc[:100])
    assert_frame_equal(features.dszl_design(X,hl=3,batch_size=1),actual)


@pytest.mark.parametrize('as_column',[False,True])
def test_batched_parquet_reader_preserves_exact_window_and_column_order(tmp_path,as_column):
    assert hasattr(data,'read_parquet_window')
    X,r=sample()
    raw=X.assign(ret_5m=r)
    path=tmp_path/'sample.parquet'
    (raw.reset_index() if as_column else raw).to_parquet(path)
    got=data.read_parquet_window(path,start=X.index[10],stop=X.index[120],columns=['x100','x1'],batch_size=13)
    assert_frame_equal(got,raw.loc[(raw.index>=X.index[10])&(raw.index<X.index[120]),['x100','x1']],check_freq=False)


def cells(name='takehome.ipynb'):
    return nbformat.read(ROOT/'notebooks'/name,4).cells


def tagged(tag,name='takehome.ipynb'):
    found=[c for c in cells(name) if tag in c.metadata.get('tags',[])]
    assert len(found)==1,f'{name}: missing/duplicated {tag}'
    return found[0].source


def test_submission_notebook_structure_and_visible_todos():
    assert not (ROOT/'notebooks/02_normalization.ipynb').exists()
    assert not (ROOT/'notebooks/takehome_asset_vol.ipynb').exists()
    assert (ROOT/'notebooks/asset_vol.ipynb').exists()
    assert not (ROOT/'run_forecast.py').exists()
    assert not (ROOT/'src/takehome/forecast.py').exists()
    nb=cells();md='\n'.join(c.source for c in nb if c.cell_type=='markdown')
    assert 'Asset-volatility experiment: same forecasts' not in md
    assert '<span style="color:red">TODO' in md
    assert 't-stat' in md and 'weighting by volume' in md
    oracle=next(i for i,c in enumerate(nb) if 'oracle_lead_1' in c.metadata.get('tags',[]))
    model=next(i for i,c in enumerate(nb) if 'regression_setup' in c.metadata.get('tags',[]))
    ar=next(i for i,c in enumerate(nb) if 'alpha_forecast_controls' in c.metadata.get('tags',[]))
    assert oracle < ar < model
    assert 'dszl_design' in tagged('regression_setup')
    assert 'RIDGE_CONFIG' in tagged('batch_oos_sweep')
    assert 'RIDGE_CONFIG' in tagged('submission_ridge_fit')
    assert 'write_lasso_forecasts' not in '\n'.join(c.source for c in nb)
    assert 'normalization=SIGNAL_RULE' in tagged('regression_setup')


def export_namespace(tmp_path, *, missing_blackout=True):
    from takehome.fitters import BatchRidge, calendar_walk_forward_folds, walk_forward_sweep
    ix=pd.date_range('2014-01-01','2025-06-01',freq='17h',tz='America/New_York',name='msgStamp')
    rng=np.random.default_rng(72)
    raw=pd.DataFrame(dict(x1=rng.normal(size=len(ix)),x99=rng.normal(size=len(ix)),
        cashflow=rng.normal(size=len(ix)),volume=rng.uniform(.5,2,len(ix))),index=ix)
    transformed=features.dszl_design(features.with_cashflow_feature(raw)[['x1','x99','x100']],hl=3)
    raw['ret_5m']=transformed.to_numpy()@np.array([.03,-.02,.015])+.002+rng.normal(0,.001,len(ix))
    boundary=ix[-1]-pd.DateOffset(years=2)
    if missing_blackout: raw.loc[raw.index>=boundary,'ret_5m']=np.nan
    raw.iloc[17,raw.columns.get_loc('ret_5m')]=np.nan
    source=tmp_path/'data.parquet';raw.to_parquet(source)
    _,split=data.training_data(raw[['ret_5m']])
    config=dict(alpha=.07,decay=.97,fit_intercept=True)
    research=raw.loc[(raw.index>=split['first_label'])&(raw.index<split['cutoff'])]
    folds=calendar_walk_forward_folds(research.index,train_years=2,test_years=2)
    validation=walk_forward_sweep({'ridge':BatchRidge(3,**config)},transformed.loc[research.index].to_numpy(),
        research.ret_5m.to_numpy(),W=np.isfinite(research.ret_5m).astype(float).to_numpy(),folds=folds)['ridge']
    pred=pd.Series(validation.prediction_,index=research.index,name='forecast').dropna().to_frame()
    ns=dict(Path=Path,pd=pd,np=np,json=json,os=__import__('os'),ROOT=tmp_path,DATA_PATH=source,
        read_parquet_window=data.read_parquet_window, dszl_design=features.dszl_design,
        BatchRidge=BatchRidge,WINDOW_YEARS=2,DSZL_HL=3,BATCH_SIZE=2,HL=5,SIGNAL_RULE=RULE,
        fit_columns=['x1','x99','x100'],file_last=ix[-1],split=split,RIDGE_CONFIG=config,
        display=lambda *a:None,research_forecast=pred,folds=folds,selected_stats={'n':len(pred)})
    return ns,raw,transformed


def test_actual_notebook_export_matches_manual_dszl_ridge_and_round_trips(tmp_path,monkeypatch):
    monkeypatch.delenv('FORECAST_OUTPUT_DIR',raising=False)
    ns,raw,X=export_namespace(tmp_path)
    for tag in ['submission_design','submission_ridge_fit','submission_export']:
        exec(tagged(tag),ns)
    ix=raw.index; boundary=ns['final_boundary']; lo=ns['final_train_start']
    mask=(ix>=lo)&(ix<boundary);out=ix>=boundary
    y=raw.loc[mask,'ret_5m'].to_numpy()
    ref=ns['BatchRidge'](3,**ns['RIDGE_CONFIG']).fit(X.loc[mask].to_numpy(),y,W=np.isfinite(y).astype(float))
    expected=pd.DataFrame({'forecast':ref.predict(X.loc[out].to_numpy())},index=ix[out])
    assert_frame_equal(ns['submission_forecast'],expected,atol=1e-12,check_freq=False)
    assert_allclose(ns['final_design'],X.loc[ns['final_index']],atol=0,rtol=0)
    folder=tmp_path/'forecasts'
    parquet=folder/'ridge_oos.parquet'; assert parquet.read_bytes()[:4]==b'PAR1'
    assert_frame_equal(pd.read_parquet(parquet),expected,check_freq=False)
    csv=pd.read_csv(folder/'ridge_oos.csv');assert list(csv)==['msgStamp','forecast']
    assert_allclose(csv.forecast,expected.forecast,atol=1e-14)
    m=json.loads((folder/'ridge_metadata.json').read_text())
    assert m['model']=='BatchRidge' and m['model_config']==ns['RIDGE_CONFIG']
    assert m['feature_transform']=='dszl' and not m['blackout_targets_used'] and not m['heldout_scoring']
    assert m['oos_rows']==int(out.sum())
    assert ns['research_forecast'].index.max()<ns['split']['cutoff']


def test_actual_export_cell_rejects_observed_blackout_labels(tmp_path):
    ns,_,_=export_namespace(tmp_path,missing_blackout=False)
    with pytest.raises(AssertionError,match='Blackout'):
        exec(tagged('submission_design'),ns)


def test_main_and_diagnostic_use_identical_fixed_configuration():
    import ast
    results=[]
    for source in [tagged('submission_config'),tagged('diagnostic_setup','asset_vol.ipynb')]:
        statements=[]
        for stmt in ast.parse(source).body:
            if not isinstance(stmt,ast.Assign):continue
            names={n.id for target in stmt.targets for n in ast.walk(target) if isinstance(n,ast.Name)}
            if names & {'HL','SIGNAL_RULE','RIDGE_CONFIG'}:statements.append(stmt)
        ns={};exec(compile(ast.Module(body=statements,type_ignores=[]),'<config>','exec'),ns)
        results.append({k:ns[k] for k in ['HL','DSZL_HL','WINDOW_YEARS','SIGNAL_RULE','RIDGE_CONFIG']})
    assert results[0]==results[1]
    assert results[0]['WINDOW_YEARS']==2 and results[0]['DSZL_HL']==10


def test_new_markdown_has_no_accidental_latex_control_characters():
    for name in ['takehome.ipynb','asset_vol.ipynb']:
        for cell in cells(name):
            if cell.cell_type=='markdown':
                assert not (set(cell.source) & {chr(i) for i in range(32) if i not in (9,10)}),name



def test_rolling_stream_audit_distinguishes_scored_updates_and_frozen_terminal_fits():
    from takehome.fitters import StreamingWeightedLasso, stream_at_folds
    rng=np.random.default_rng(909)
    X=rng.normal(size=(90,3));X[:,1]=X[:,0]+.1*X[:,1]
    y=X@np.array([.3,-.1,.2])+rng.normal(0,.02,len(X))
    folds=[dict(train_start=0,train_stop=30,predict_start=30,predict_stop=60),
           dict(train_start=30,train_stop=60,predict_start=60,predict_stop=90)]
    config=dict(n_features=3,decay=.97,alpha=.001,max_iter=1,tol=1e-12,fit_intercept=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        model=StreamingWeightedLasso(**config,store_history=True)
        result=stream_at_folds(model,X,y,folds)
        assert 'live_failed_oos_updates' in result
        manual=StreamingWeightedLasso(**config,store_history=True)
        pre=manual.fit_predict(X[:30],y[:30]);before=manual.n_failed_
        warmup_converged=manual.converged_
        rest=manual.fit_predict(X[30:],y[30:])
        assert result['live_failed_warmup_updates']==before
        assert result['live_failed_oos_updates']==manual.n_failed_-before
        assert result['live_warmup_converged']==warmup_converged
        assert result['live_failed_oos_updates']>0
        assert_allclose(result['live'],np.r_[pre,rest],equal_nan=True)
        assert_allclose(model.get_coefs(),manual.get_coefs())
        for i,f in enumerate(folds):
            ref=StreamingWeightedLasso(**config)
            ref.fit(X[f['train_start']:f['train_stop']],y[f['train_start']:f['train_stop']])
            assert result['frozen_converged'][i]==ref.converged_
            assert result['frozen_failed_updates'][i]==ref.n_failed_
            assert result['frozen_kkt_tolerances'][i]==ref.kkt_tolerance_

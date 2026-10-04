"""The previous test block chooses the next block, never itself or the holdout."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from takehome import features
from takehome.fitters import BatchRidge, BatchLasso, calendar_walk_forward_folds


def selection_module():
    assert importlib.util.find_spec('takehome.selection') is not None, 'Missing lagged OOS selection'
    from takehome import selection
    return selection


def windows():
    return [dict(train_start=10*i,train_stop=10*(i+1),predict_start=10*(i+1),predict_stop=10*(i+2)) for i in range(3)]


def test_combined_design_is_exact_raw_then_dszl_and_causal():
    assert hasattr(features, 'raw_dszl_design'), 'Missing combined design'
    rng=np.random.default_rng(37)
    ix=pd.date_range('2019-01-01',periods=140,freq='12h',tz='America/New_York')
    X=pd.DataFrame(rng.normal(size=(len(ix),3)),index=ix,columns=['x1','x2','x100'])
    X.iloc[9:12,0]=np.nan;X.iloc[22,1]=np.inf;X.iloc[27:30,2]=0
    expected=pd.concat([X.add_prefix('raw:'),features.dszl(X,3).add_prefix('dszl:')],axis=1).replace([np.inf,-np.inf],np.nan).fillna(0)
    out=features.raw_dszl_design(X,hl=3,batch_size=2)
    assert_frame_equal(out,expected)
    assert out.to_numpy().flags.c_contiguous
    assert_frame_equal(features.raw_dszl_design(X.iloc[:77],3,1),out.iloc[:77])
    changed=X.copy();changed.iloc[77:]*=10000
    assert_frame_equal(features.raw_dszl_design(changed,3,1).iloc[:77],out.iloc[:77])


def test_selection_uses_previous_test_mse_and_has_no_static_first_choice():
    select=selection_module().select_previous_test
    y=np.zeros(40);y[20:30]=2
    p={.1:np.zeros(40),1.:np.full(40,2.)}
    result=select(p,y,windows(),embargo=1)
    assert result.choices_==[None,.1,1.]
    assert np.isnan(result.prediction_[:20]).all()
    assert_allclose(result.prediction_[20:30],0)
    assert_allclose(result.prediction_[30:],2)
    assert result.selection_.validation_fold.tolist()==[0,1]
    assert result.selection_.validation_rows.tolist()==[9,9]
    assert (result.selection_.validation_stop<=result.selection_.predict_start).all()


def test_current_test_and_embargoed_boundary_labels_cannot_change_current_choice():
    select=selection_module().select_previous_test
    y=np.zeros(40);p={.1:np.zeros(40),1.:np.ones(40)}
    before=select(p,y,windows())
    y[20:]=5000;y[19]=5000
    after=select(p,y,windows())
    assert after.choices_[1]==before.choices_[1]
    assert_allclose(after.prediction_[:30],before.prediction_[:30],equal_nan=True)
    assert after.choices_[2]!=before.choices_[2]


def test_selection_ignores_holdout_targets_and_uses_matched_rows_for_candidates():
    select=selection_module().select_previous_test
    y=np.zeros(40);y[30:]=np.nan
    p={.1:np.zeros(40),1.:np.ones(40)}
    p[.1][10]=np.nan  # Disqualify this candidate, do not grant it favorable coverage.
    out=select(p,y,windows())
    assert out.choices_[1]==1.
    assert out.choices_[2]==.1
    assert np.isfinite(out.prediction_[30:]).all()
    assert np.isnan(out.validation_scores_.query('fold == 2').mse).all()


def test_ties_are_deterministic_and_missing_validation_does_not_fall_back_to_static():
    select=selection_module().select_previous_test
    p={1.:np.ones(40),.1:np.ones(40)}
    out=select(p,np.ones(40),windows())
    assert out.choices_[1:]==[1.,1.]  # insertion order documented tie breaker
    y=np.ones(40);y[10:20]=np.nan
    with pytest.raises(ValueError,match='validation'):
        select(p,y,windows())


@pytest.mark.parametrize('family',['ridge','lasso'])
def test_tuned_path_matches_independent_fit_and_future_perturbation(family):
    mod=selection_module();rng=np.random.default_rng(81)
    X=rng.normal(size=(40,3));y=X@np.array([.3,-.2,.1])+rng.normal(0,.15,40)
    model=lambda a: BatchRidge(3,alpha=a,decay=.97) if family=='ridge' else BatchLasso(3,.97,a,fit_intercept=True,tol=1e-11)
    models={.001:model(.001),.1:model(.1),10.:model(10.)}
    result=mod.walk_forward_select(models,X,y,folds=windows())
    for i,f in enumerate(windows()[1:],1):
        ref=model(result.choices_[i]).fit(X[f['train_start']:f['train_stop']],y[f['train_start']:f['train_stop']])
        assert_allclose(result.prediction_[f['predict_start']:f['predict_stop']],ref.predict(X[f['predict_start']:f['predict_stop']]),atol=1e-7)
    path=result.path_
    recovered=np.einsum('ij,ij->i',X,path.get_coefs(lag=1))+path.get_intercepts(lag=1)
    assert_allclose(recovered,result.prediction_,equal_nan=True,atol=1e-12)
    new_y=y.copy();new_y[20:]+=100
    changed=mod.walk_forward_select(models,X,new_y,folds=windows())
    assert_allclose(changed.prediction_[:30],result.prediction_[:30],equal_nan=True,atol=1e-12)


def test_streaming_frozen_fits_follow_selected_lasso_schedule_not_static_alpha():
    mod=selection_module();rng=np.random.default_rng(13)
    X=rng.normal(size=(40,2));y=X@np.array([.4,-.2])+rng.normal(0,.2,40)
    grid={.01:BatchLasso(2,.97,.01,fit_intercept=True),.3:BatchLasso(2,.97,.3,fit_intercept=True)}
    tuned=mod.walk_forward_select(grid,X,y,folds=windows())
    out=mod.stream_selected_folds(tuned,X,y,tol=1e-10,max_iter=30000)
    assert_allclose(out['frozen'],tuned.prediction_,equal_nan=True,atol=1e-7)
    assert out['audit'].alpha.tolist()==tuned.choices_[1:]
    assert out['audit'].terminal_converged.all()
    assert np.isnan(out['live'][:20]).all()


def test_ar_candidates_choose_previous_feature_test_block_and_mature_next_value():
    mod=selection_module();rng=np.random.default_rng(69)
    ix=pd.date_range('2014-01-01',periods=50,freq='D')
    x=pd.Series(rng.normal(size=50),index=ix,name='x1')
    folds=windows()
    out,info=mod.forecast_alpha_selected(x,folds=folds,alphas=[.001,.1],hl=5,min_train=4)
    assert np.isnan(out.iloc[:20]).all()
    changed=x.copy();changed.iloc[20:]+=1000
    altered,other=mod.forecast_alpha_selected(changed,folds=folds,alphas=[.001,.1],hl=5,min_train=4)
    assert info['selection'].iloc[0].candidate==other['selection'].iloc[0].candidate
    # t=19 forecasts x20; row 19 must not choose the model used at t=20.
    assert info['selection'].iloc[0].validation_stop==19


def test_live_stream_ages_embargo_rows_without_learning_their_labels():
    from takehome.fitters import StreamingWeightedLasso
    mod=selection_module();rng=np.random.default_rng(24)
    X=rng.normal(size=(40,2));y=X@np.array([.4,-.2])+rng.normal(0,.2,40)
    folds=[dict(f,train_stop=f['train_stop']-2) for f in windows()]
    grid={.01:BatchLasso(2,.8,.01,fit_intercept=True),.3:BatchLasso(2,.8,.3,fit_intercept=True)}
    tuned=mod.walk_forward_select(grid,X,y,folds=folds)
    out=mod.stream_selected_folds(tuned,X,y,tol=1e-10,max_iter=30000)
    for i in (1,2):
        f=folds[i];a,b,c,d=(f[k] for k in ('train_start','train_stop','predict_start','predict_stop'))
        ref=StreamingWeightedLasso(2,.8,tuned.choices_[i],fit_intercept=True,tol=1e-10,max_iter=30000)
        ref.fit(X[a:b],y[a:b])
        ref.fit(X[b:c],np.full(c-b,np.nan),W=np.zeros(c-b))
        expected=ref.fit_predict(X[c:d],y[c:d])
        assert_allclose(out['live'][c:d],expected,atol=1e-12)



def test_notebook_alpha_control_masks_rows_before_first_selected_fold():
    import ast,nbformat
    nb=nbformat.read(Path(__file__).resolve().parents[1]/'notebooks/takehome.ipynb',4)
    src=next(c.source for c in nb.cells if 'alpha_forecast_controls' in c.metadata.get('tags',[]))
    definition=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef))
    ix=pd.date_range('2020',periods=20,freq='D')
    frame=pd.DataFrame({'x1':np.arange(20,dtype=float)+1, 'x2':np.arange(20,dtype=float)+3},index=ix)
    ns=dict(np=np,pd=pd,df=frame,x_cols=frame.columns,rows=np.arange(20),alpha_first=10,AR_MIN_TRAIN=3,
            alpha_folds=[dict(predict_start=10,predict_stop=15),dict(predict_start=15,predict_stop=20)],
            ar_fold_choices={'x1':[None,.1],'x2':[.1,.1]})
    exec(compile(ast.Module(body=[definition],type_ignores=[]),'<controls>','exec'),ns)
    out=pd.concat(list(ns['matched_alpha_controls']()),axis=1)
    assert out.iloc[:10].isna().all().all()
    expected=frame.iloc[10:].copy();expected.iloc[:5,0]=np.nan
    assert_frame_equal(out.iloc[10:],expected)



def test_feature_selection_waits_for_completed_validation_without_static_fallback():
    mod=selection_module()
    y=np.zeros(40);y[10:20]=np.nan;y[20:30]=1
    preds={10.:np.zeros(40),.1:np.ones(40)}
    assert 'allow_missing_history' in __import__('inspect').signature(mod.select_previous_test).parameters
    result=mod.select_previous_test(preds,y,windows(),allow_missing_history=True)
    assert result.choices_==[None,None,.1]
    assert np.isnan(result.prediction_[:30]).all()
    assert_allclose(result.prediction_[30:],1)
    assert result.selection_.status.tolist()==['waiting_for_validation','selected']
    preds={a:np.full(40,np.nan) for a in (10.,.1)}
    with pytest.raises(ValueError,match='validation'):
        mod.select_previous_test(preds,np.ones(40),windows(),allow_missing_history=True)


def test_sparse_ar_feature_is_unavailable_until_previous_test_has_mature_examples():
    mod=selection_module();rng=np.random.default_rng(508)
    x=pd.Series(rng.normal(size=40),index=pd.date_range('2014',periods=40,freq='D'),name='late')
    x.iloc[:21]=np.nan
    pred,info=mod.forecast_alpha_selected(x,folds=windows(),alphas=[1.,.01],hl=5,min_train=3)
    assert pred.iloc[:30].isna().all()
    assert pred.iloc[30:].notna().all()
    assert info['choices'][:2]==[None,None]
    assert info['selection'].iloc[0].status=='waiting_for_validation'
    changed=x.copy();changed.iloc[30:]*=1000
    _,other=mod.forecast_alpha_selected(changed,folds=windows(),alphas=[1.,.01],hl=5,min_train=3)
    assert info['choices'][-1]==other['choices'][-1]

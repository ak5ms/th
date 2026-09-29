"""Pre-update forecasts, normalized ridge loss and both backtest conventions."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from takehome.fitters import StreamingWeightedLasso


def data(n=120, p=6):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(n, p))
    return X, .4 + X @ np.arange(p)/p + rng.normal(0, .2, n)


def model(record=False):
    return StreamingWeightedLasso(6, .96, .03, fit_intercept=True, store_history=record, max_iter=50000)


def test_fit_predict_previous_coefficients_across_chunks():
    X, y = data(); w = np.ones(len(y)); w[:3]=0; w[17:21]=0
    ref = model(True).fit(X,y,W=w)
    expected=np.full(len(y),np.nan)
    expected[1:] = (ref.get_coefs()[:-1]*X[1:]).sum(1)+ref.get_intercepts()[:-1]
    expected[:4]=np.nan
    for size in [1,17,200]:
        m=model()
        assert hasattr(m,'fit_predict')
        assert_allclose(m.fit_predict(X,y,W=w,chunk_size=size), expected, atol=1e-9,equal_nan=True)
        assert not m.store_history and not m.history and m.n_seen_==len(y)


def test_fit_predict_no_current_target_leak_and_preserves_history():
    X,y=data(); a=model(True); pa=a.fit_predict(X,y,chunk_size=11)
    other=y.copy(); other[51:]+=17
    pb=model().fit_predict(X,other)
    assert_allclose(pa[:52],pb[:52],equal_nan=True)
    assert_allclose(a.get_coefs(),model(True).fit(X,y).get_coefs(),atol=1e-9)


def test_fit_predict_continuation_and_input_validation():
    X,y=data(60); m=model()
    a=m.fit_predict(X[:25],y[:25],chunk_size=7)
    b=m.fit_predict(X[25:],y[25:],chunk_size=9)
    assert_allclose(np.r_[a,b],model().fit_predict(X,y),equal_nan=True)
    with pytest.raises(ValueError): m.fit_predict(X,y,chunk_size=0)


def test_ridge_matches_direct_normal_equations_and_cvxpy():
    import cvxpy as cp
    from takehome.fitters import BatchRidge
    X,y=data(90); w=np.linspace(.1,3,len(y)); w[:3]=0; a=w/w.sum()
    mx=a@X; my=a@y; scale=np.sqrt(a@((X-mx)**2)); Z=(X-mx)/scale
    beta=np.linalg.solve(Z.T@(a[:,None]*Z)+.02*np.eye(6),Z.T@(a*(y-my)))/scale
    m=BatchRidge(6,alpha=.02).fit(X,y,W=w)
    assert_allclose(m.coef,beta,atol=1e-11)
    assert_allclose(m.intercept_,my-mx@beta,atol=1e-11)
    coef=cp.Variable(6); bias=cp.Variable()
    problem=cp.Problem(cp.Minimize(cp.sum_squares(cp.multiply(np.sqrt(a),y-X@coef-bias))/2
                                  + .02*cp.sum_squares(cp.multiply(scale,coef))/2))
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-11,tol_feas=1e-11,tol_gap_rel=1e-11)
    assert_allclose(m.coef,coef.value,atol=2e-7)
    m2=BatchRidge(6,alpha=.02).fit(X*7,y,W=w*1e8)
    assert_allclose(m2.predict(X*7),m.predict(X),atol=1e-11)


def test_backtest_matches_standalone_and_literal_formula():
    from takehome import features
    idx=pd.date_range('2020-01-01',periods=120,freq='1h')
    p=pd.Series(np.sin(np.arange(120)*.3)+.2,index=idx)
    r=pd.Series(np.cos(np.arange(120)*.3),index=idx)
    for norm in ['variance','zscore']:
        denom=features.ts_std(p,10)**2 if norm=='variance' else features.ts_zscore(p,10)
        expected=p.div(denom.replace(0,np.nan)).mul(r).replace([np.inf,-np.inf],np.nan)
        got=features.backtest(p,r,hl=10,normalization=norm)
        assert_allclose(got,expected,equal_nan=True)
        assert_allclose(got[:70],features.backtest(p[:70],r[:70],10,normalization=norm),equal_nan=True)
    expected=features.standalone_pnl(p.to_frame('p'),r,hl=10).p
    assert_allclose(features.backtest(p,r,10),expected,equal_nan=True)
    with pytest.raises(ValueError): features.backtest(p,r,10,normalization='unknown')
    with pytest.raises(ValueError): features.backtest(p,r.iloc[:-1],10)


def test_ridge_constant_missing_and_invalid_weights():
    from takehome.fitters import BatchRidge
    X,y=data(40); X[:,1]=0; X[:,2]=4
    w=np.ones(len(y)); w[:3]=0; X[:3]=np.nan; y[:3]=np.nan
    fitted=BatchRidge(6).fit(X,y,W=w)
    assert_allclose(fitted.coef[1:3],0,atol=1e-12)
    for a in [0,-1,np.nan]:
        with pytest.raises(ValueError): BatchRidge(6,alpha=a)
    with pytest.raises(ValueError): BatchRidge(6).fit(X,y,W=0)


def test_cached_cd_matches_direct_cyclic_updates():
    from takehome.fitters import _coordinate_descent
    X,y=data(80); G=X.T@X/len(y); h=X.T@y/len(y); ref=np.zeros(6)
    for _ in range(2000):
        for j in range(6):
            rho=h[j]-G[j]@ref+G[j,j]*ref[j]
            ref[j]=np.sign(rho)*max(abs(rho)-.03,0)/G[j,j]
    actual=np.zeros(6); _coordinate_descent(G,h,actual,.03,2000,1e-11)
    assert_allclose(actual,ref,atol=1e-9)

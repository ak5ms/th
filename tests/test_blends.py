import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from takehome.features import combine_pnls, _blend_totals, evaluate_features, standalone_pnl


def frame():
    rng = np.random.default_rng(22)
    return pd.DataFrame(rng.normal(size=(80,5)),index=pd.date_range('2020-01-01',periods=80,freq='5min',tz='UTC'))


def test_matches_explicit_weight_normalize_shift_and_sum():
    pnl=frame()
    pnl.iloc[10:15,2]=np.nan
    pnl.iloc[:6,1]=np.nan
    for positive in [False,True]:
        score=pnl.ewm(halflife=10).mean()/pnl.ewm(halflife=10).std()
        score=score.replace([np.inf,-np.inf],np.nan).fillna(0.)
        if positive: score=score.clip(lower=0)
        weights=score.div(score.abs().sum(axis=1).replace(0,np.nan),axis=0)
        expected=weights.shift().mul(pnl).sum(axis=1)
        assert_allclose(combine_pnls(pnl,10,positive),expected,atol=1e-14)
        gross=weights.abs().sum(axis=1)
        assert ((abs(gross-1)<1e-12)|(gross==0)).all()


def test_batched_combination_is_global_not_per_batch():
    pnl=frame()
    pnl.iloc[:5]=np.nan
    numerator,gross=np.zeros(len(pnl)),np.zeros(len(pnl))
    for block in [pnl.iloc[:,:2],pnl.iloc[:,2:4],pnl.iloc[:,4:]]:
        n,g=_blend_totals(block,10)
        numerator+=n.to_numpy()
        gross+=g.to_numpy()
    actual=np.divide(numerator,gross,out=np.zeros_like(gross),where=gross>0)
    assert_allclose(actual,combine_pnls(pnl,10),atol=1e-14)


def test_all_zero_and_missing_stay_finite():
    pnl=frame()*0
    pnl.iloc[:10,:]=np.nan
    assert_allclose(combine_pnls(pnl),0.)


def test_future_data_do_not_affect_earlier_blend():
    pnl=frame()
    old=combine_pnls(pnl,10)
    pnl.iloc[40:]*=-1e3
    assert_allclose(old.iloc[:40],combine_pnls(pnl,10).iloc[:40])


def test_evaluate_features_meta_matches_direct_pnl():
    X=frame()
    returns=pd.Series(np.linspace(-.1,.1,len(X)),index=X.index)
    daily,summary,meta=evaluate_features([X.iloc[:,:2],X.iloc[:,2:]],returns,hl=3,meta=True,meta_hl=10)
    expected=combine_pnls(standalone_pnl(X,returns,hl=3),hl=10)
    assert_allclose(meta,expected,atol=1e-12)
    assert daily.shape[1]==len(X.columns) and len(summary)==len(X.columns)

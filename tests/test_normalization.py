"""Formula, prefix-invariance and adversarial tests for signal-only sizing."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_series_equal, assert_frame_equal
from takehome import normalization


def weights(x, **kwargs):
    assert hasattr(normalization, 'signal_weights'), 'Missing signal-only normalization'
    return normalization.signal_weights(x, **kwargs)


def sample(n=300):
    return pd.Series(np.random.default_rng(31).normal(size=n),
                     index=pd.date_range('2020-01-01', periods=n, freq='5min'), name='alpha')


@pytest.mark.parametrize('method', ['std', 'variance', 'rms'])
def test_formula_uses_previous_scale_and_only_initial_reference(method):
    x = sample(); x.iloc[25:28] = 0; x.iloc[31] = np.nan
    clean = x.replace(0, np.nan)
    window = clean.ewm(halflife=10, min_periods=10, ignore_na=True)
    s = ((clean**2).ewm(halflife=10, min_periods=10, ignore_na=True).mean()**.5
         if method == 'rms' else window.std()).shift(1).replace(0, np.nan)
    ref = s.where(s.notna().cumsum().eq(1)).ffill()
    denom = s.clip(lower=.25*ref)
    expected = x/denom * (ref/denom if method == 'variance' else 1)
    expected = expected.clip(-3, 3)
    assert_series_equal(weights(x, hl=10, method=method), expected)
    assert expected.iloc[:10].isna().all()


@pytest.mark.parametrize('method', ['std', 'variance', 'rms'])
def test_future_and_end_of_sample_do_not_change_past(method):
    x=sample()
    first=weights(x, hl=10, method=method)
    altered=x.copy(); altered.iloc[100:]=1e12
    assert_series_equal(first.iloc[:100],weights(altered,hl=10,method=method).iloc[:100])
    assert_series_equal(first.iloc[:100],weights(x.iloc[:100],hl=10,method=method))


@pytest.mark.parametrize('method', ['std', 'variance', 'rms'])
def test_measurement_units_and_column_partition_are_invariant(method):
    x=sample(); opts=dict(hl=10,method=method)
    original=weights(x,**opts)
    assert_allclose(weights(x*1e-8,**opts),original,rtol=1e-9,atol=1e-10,equal_nan=True)
    assert_allclose(weights(x*-100,**opts),-original,rtol=1e-9,atol=1e-10,equal_nan=True)
    X=pd.DataFrame({'a':x,'b':x*100})
    got=weights(X,**opts)
    expected=pd.concat({c:weights(X[c],**opts) for c in X},axis=1)
    assert_frame_equal(got,expected)


@pytest.mark.parametrize('method', ['std', 'variance', 'rms'])
def test_cap_survives_collapse_missingness_and_unseen_spike(method):
    x=sample(1000); x.iloc[60:900]=.001; x.iloc[910:930]=0
    x.iloc[940:950]=np.nan; x.iloc[975]=1e12; x.iloc[990]=-1e12
    got=weights(x,hl=5,method=method)
    assert not np.isinf(got).any()
    assert got.abs().max()<=3
    assert got.iloc[975]==3 and -3 <= got.iloc[990] < 0
    # The first positive spike raises the later denominator; the second need not hit its cap.
    negative = x.copy(); negative.iloc[975] = -1e12
    assert weights(negative,hl=5,method=method).iloc[975] == -3
    finite=got.notna()
    assert (np.sign(got[finite])==np.sign(x[finite])).all()
    r=pd.Series(np.random.default_rng(9).normal(0,.01,len(x)),index=x.index)
    assert ((got*r).abs()[finite]<=3*r.abs()[finite]).all()


def test_uncapped_variance_retains_old_path_up_to_fixed_past_scale():
    x=sample(); s=x.ewm(halflife=10,min_periods=10,ignore_na=True).std().shift(1)
    initial=s.dropna().iloc[0]
    expected=initial*x/s**2
    assert_allclose(weights(x,hl=10,method='variance',floor_fraction=0,cap=None),expected,equal_nan=True)


def test_vol_floor_alone_does_not_guarantee_bound_on_new_outlier():
    x=sample(); x.iloc[150]=1e10
    assert weights(x,hl=10,method='variance',cap=None).abs().max()>1e9
    assert weights(x,hl=10,method='std',cap=None).abs().max()>1e9
    assert weights(x,hl=10,method='rms',cap=None).abs().max()>1e9


def test_missing_and_zero_inputs_have_explicit_policy():
    x=sample(); x.iloc[30:80]=0; x.iloc[80:90]=np.nan; x.iloc[95]=np.inf
    w=weights(x,hl=10)
    assert w.iloc[30:80].eq(0).all() and w.iloc[80:90].isna().all() and np.isnan(w.iloc[95])
    assert weights(x*0,hl=10).isna().all()
    assert weights(x*0+2,hl=10,method='std').isna().all()
    assert_allclose(weights(x*0+2,hl=10,method='rms').dropna(),1)


def test_explicit_reference_is_used_not_refitted_on_test_data():
    x=sample()
    std=x.ewm(halflife=10,min_periods=10,ignore_na=True).std().shift(1)
    expected=(2*x/std.clip(lower=.5)**2).clip(-3,3)
    assert_series_equal(weights(x,hl=10,method='variance',reference=2.),expected)
    X=pd.DataFrame({'a':x,'b':3*x}); refs=pd.Series({'a':2.,'b':6.})
    assert_frame_equal(weights(X,hl=10,method='variance',reference=refs),
        pd.concat({c:weights(X[c],hl=10,method='variance',reference=refs[c]) for c in X},axis=1))


@pytest.mark.parametrize('kwargs',[{'hl':0},{'method':'bad'},{'floor_fraction':-1},
    {'cap':0},{'cap':np.inf},{'lag':-1},{'min_periods':1},{'reference':0}])
def test_bad_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError): weights(sample(),**kwargs)


def test_unsorted_and_duplicate_time_inputs_fail():
    x=sample()
    with pytest.raises(ValueError): weights(x.iloc[::-1])
    with pytest.raises(ValueError): weights(pd.concat([x,x]))

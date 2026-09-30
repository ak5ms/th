"""Timing and independent-loss checks for alpha forecasts and zero-aware volatility."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_series_equal, assert_frame_equal
from takehome import features, fitters


def sample(n=90):
    rng = np.random.default_rng(73)
    x = rng.normal(size=n)
    for t in range(2, n):
        x[t] += .55*x[t-1] - .2*x[t-2] + .1
    return pd.Series(x, index=pd.date_range('2020-01-01', periods=n, freq='5min', tz='UTC'), name='x1')


def forecast(x, **kwargs):
    assert hasattr(fitters, 'forecast_alpha'), 'Missing AR(2) alpha-forecast helper'
    return fitters.forecast_alpha(x, **kwargs)


def test_std_zeros_and_missing_are_ignored_before_estimation():
    x = pd.Series([1., 2., 0., np.nan, 0., 4., np.inf, -3.])
    clean = x.replace([0, np.inf, -np.inf], np.nan)
    expected = clean.ewm(halflife=2, min_periods=2, ignore_na=True).std().replace(0, np.nan)
    assert_series_equal(features.ts_std(x, 2), expected)
    assert features.ts_std(pd.Series([0.]*10), 2).isna().all()
    assert features.ts_std(pd.Series([2.]*10), 2).isna().all()
    actual = features.ts_std(pd.DataFrame({'a': x, 'b': x*3}), 2)
    assert_frame_equal(actual, pd.concat({'a': expected, 'b': expected*3}, axis=1))


def test_volatility_does_not_decay_across_zero_missing_run():
    x = pd.Series([1., -2., 3.] + [0.]*500 + [np.nan]*50 + [5.])
    got = features.ts_std(x, 2)
    assert_allclose(got.iloc[3:-1], got.iloc[2])
    ref = pd.Series([1., -2., 3., 5.]).ewm(halflife=2, min_periods=2, ignore_na=True).std()
    assert got.iloc[-1] == pytest.approx(ref.iloc[-1])


def test_forecast_is_future_invariant_and_uses_current_observed_value():
    x = sample()
    kwargs = dict(hl=10, alpha=.03, min_train=8, tol=1e-11)
    pred, info = forecast(x, **kwargs)
    assert pred.index.equals(x.index) and pred.name == x.name
    assert pred.iloc[:9].isna().all() and np.isfinite(pred.iloc[9:]).all()
    changed = x.copy(); changed.iloc[50:] += 100
    other, _ = forecast(changed, **kwargs)
    assert_allclose(pred.iloc[:50], other.iloc[:50], equal_nan=True)
    prefix, _ = forecast(x.iloc[:50], **kwargs)
    assert_series_equal(pred.iloc[:50], prefix)
    assert info['failed_updates'] == 0


def test_forecast_matches_direct_cvxpy_loss_at_checkpoints():
    x = sample(50)
    x.iloc[15] = np.nan
    pred, info = forecast(x, hl=10, alpha=.03, min_train=8, tol=1e-11)
    z = x/info['scale']
    train = pd.concat([z.shift(1), z.shift(2)], axis=1).to_numpy()
    weights = (np.isfinite(train).all(axis=1) & np.isfinite(z)).astype(float)
    for t in [12, 23, 49]:
        model = fitters.CvxpyWeightedLasso(2, 2**(-1/10), .03,
                    fit_intercept=True, tol=1e-12).fit(train[:t+1], z.iloc[:t+1].to_numpy(), W=weights[:t+1])
        # At decision t we know x_t and x_{t-1}, never x_{t+1}.
        expected = model.predict(np.array([z.iloc[t], z.iloc[t-1]]))*info['scale']
        assert pred.iloc[t] == pytest.approx(expected, abs=1e-7)
    assert pred.iloc[15:17].isna().all()


def test_forecast_units_and_batch_column_partition_do_not_change_answers():
    x = sample()
    kwargs = dict(hl=7, alpha=.1, min_train=8)
    pred, _ = forecast(x, **kwargs)
    scaled, _ = forecast(x*-100, **kwargs)
    assert_allclose(scaled, pred*-100, rtol=2e-6, atol=1e-7, equal_nan=True)
    assert hasattr(fitters, 'forecast_alpha_blocks'), 'Missing column-batched alpha forecasts'
    X = pd.concat([x, (2*x).rename('x2')], axis=1)
    stats=[]
    got = pd.concat(list(fitters.forecast_alpha_blocks(X, batch_size=1, diagnostics=stats, **kwargs)),axis=1)
    other = pd.concat(list(fitters.forecast_alpha_blocks(X, batch_size=2, **kwargs)),axis=1)
    assert_frame_equal(got, other)
    assert_allclose(got.x1, pred, equal_nan=True)
    assert len(stats)==2 and set(d['feature'] for d in stats)=={'x1','x2'}


def test_unavailable_alpha_stays_unforecastable():
    pred, info = forecast(sample(10)*np.nan, min_train=8)
    assert pred.isna().all() and info['n_updates']==0
    with pytest.raises(ValueError): forecast(sample().iloc[::-1])
    with pytest.raises(ValueError): forecast(sample(), min_train=1)

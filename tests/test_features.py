import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from takehome.features import sharpe, standalone_pnl, ts_standardize, ts_std, ts_zscore


def frame():
    idx = pd.date_range('2020-01-01', periods=40, freq='5min', tz='UTC')
    rng = np.random.default_rng(17)
    return pd.DataFrame(rng.normal(size=(40, 3)), index=idx, columns=['x1', 'x2', 'x3'])


def test_ewm_helpers_match_requested_pandas_formula():
    x = frame()
    s = x.ewm(halflife=5, min_periods=5).std()
    assert_frame_equal(ts_std(x, 5), s)
    assert_frame_equal(ts_standardize(x, 5), x / s)
    assert_frame_equal(ts_zscore(x, 5), (x - x.ewm(halflife=5, min_periods=5).mean()) / s)


def test_pnl_aligns_returns_to_rows_not_feature_names():
    x = frame()
    r = pd.Series(np.arange(40)/1000, index=x.index)
    pnl = standalone_pnl(x, r, hl=5)
    expected = x.apply(lambda s: s / s.ewm(halflife=5, min_periods=5).std()**2)
    expected = expected.mul(r, axis=0)
    assert_frame_equal(pnl, expected)
    assert pnl.shape == x.shape
    assert pnl.iloc[:4].isna().all().all()
    assert np.isfinite(pnl.iloc[4:]).all().all()
    daily = pnl.resample('D').sum()
    assert_frame_equal(daily, expected.resample('D').sum())


def test_feature_scaling_and_pnl_prefix_do_not_use_future_data():
    x = frame()
    r = pd.Series(.01, index=x.index)
    assert_frame_equal(standalone_pnl(x, r, 5).iloc[:20], standalone_pnl(x.iloc[:20], r.iloc[:20], 5))


def test_zero_volatility_is_missing_not_infinite():
    x = frame()
    x['constant'] = 1
    r = pd.Series(.01, index=x.index)
    assert standalone_pnl(x, r, 5)['constant'].isna().all()
    assert np.isnan(sharpe(pd.DataFrame({'zero': [0., 0.]}))['zero'])


def test_sharpe_is_unannualized_sample_mean_over_sample_std():
    x = frame()
    np.testing.assert_allclose(sharpe(x), x.mean()/x.std())


def test_sharpe_accepts_a_single_series():
    x = frame()['x1']
    assert sharpe(x) == x.mean() / x.std()

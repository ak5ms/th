"""Small direct references for transformations, chronological fitting and batching."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal

from takehome import features
from takehome.fitters import StreamingWeightedLasso


def helper(name):
    assert hasattr(features, name), f'Missing transformation helper: {name}'
    return getattr(features, name)


def sample(n=80):
    rng = np.random.default_rng(42)
    index = pd.date_range('2020-01-01', periods=n, freq='12h', tz='America/New_York', name='msgStamp')
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=index, columns=['x1', 'x2', 'x3', 'x4'])
    y = pd.Series(rng.normal(size=n), index=index, name='ret_5m')
    return X, y


def test_dszl_uses_each_clock_group_and_preserves_index():
    X, _ = sample()
    X.iloc[7, 1] = np.nan
    result = helper('dszl')(X, hl=10)
    expected = X.copy() * np.nan
    for clock in np.unique(X.index.time):
        group = X.loc[X.index.time == clock]
        expected.loc[group.index] = features.ts_standardize(group, 10)
    assert_frame_equal(result, expected)
    assert result.iloc[:18].isna().all().all()
    changed = X.copy()
    changed.loc[changed.index.hour == 12] *= 17
    assert_frame_equal(helper('dszl')(changed, 10).loc[X.index.hour == 0], result.loc[X.index.hour == 0])


def test_dszl_is_prefix_invariant_and_rejects_unsorted_index():
    X, _ = sample()
    assert_frame_equal(helper('dszl')(X, 3).iloc[:30], helper('dszl')(X.iloc[:30], 3))
    with pytest.raises(ValueError):
        helper('dszl')(X.iloc[::-1], 3)


def test_interactions_are_raw_products_no_squares_and_all_rows():
    X, _ = sample()
    X.iloc[3, 0] = np.nan
    blocks = list(helper('pairwise_features')(X, kind='product', batch_size=2))
    assert all(len(block) == len(X) and len(block.columns) <= 2 for block in blocks)
    result = pd.concat(blocks, axis=1)
    assert result.shape == (len(X), 6)
    assert_series_equal(result['x1*x2'], (X.x1 * X.x2).rename('x1*x2'))
    assert 'x1*x1' not in result and 'x2*x1' not in result


def test_residuals_equal_direct_weighted_prefix_regressions():
    X, _ = sample(40)
    x = X.x1
    y = (3 + 2 * x + X.x2 / 5).rename('y')
    x = x.copy(); y = y.copy()
    x.iloc[5:9] = np.nan
    y.iloc[15] = np.nan
    hl = 7
    actual = helper('pair_residual')(y, x, hl=hl)
    valid = np.isfinite(x) & np.isfinite(y)
    expected = np.full(len(y), np.nan)
    for t in range(len(y)):
        prior = np.flatnonzero(valid.iloc[:t].to_numpy())
        if valid.iloc[t] and len(prior) >= 2:
            a = 2 ** (-(t - 1 - prior) / hl)
            design = x.iloc[prior].to_numpy()[:, None]
            beta = np.linalg.lstsq(design * np.sqrt(a[:, None]), y.iloc[prior] * np.sqrt(a), rcond=None)[0]
            expected[t] = y.iloc[t] - beta[0] * x.iloc[t]
    assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)


def test_residual_current_response_does_not_refit_its_own_prediction():
    X, _ = sample(30)
    base = helper('pair_residual')(X.x1, X.x2, hl=5)
    changed = X.x1.copy()
    changed.iloc[20] += 100
    other = helper('pair_residual')(changed, X.x2, hl=5)
    assert_allclose(other.iloc[:20], base.iloc[:20], equal_nan=True)
    assert other.iloc[20] - base.iloc[20] == pytest.approx(100)
    assert_allclose(helper('pair_residual')(X.x1.iloc[:20], X.x2.iloc[:20], 5), base.iloc[:20], equal_nan=True)


def test_residual_batch_covers_both_directions_without_self_pairs():
    X, _ = sample(30)
    result = pd.concat(list(helper('pairwise_features')(X, kind='residual', hl=5, batch_size=3)), axis=1)
    assert result.shape == (30, 12)
    for name, target, predictor in [('x1~x2', 'x1', 'x2'), ('x2~x1', 'x2', 'x1')]:
        expected = helper('pair_residual')(X[target], X[predictor], 5)
        assert_allclose(result[name], expected, equal_nan=True)
    assert 'x1~x1' not in result


def test_pair_arguments_fail_loudly():
    X, _ = sample()
    with pytest.raises(ValueError):
        list(helper('pairwise_features')(X, kind='typo'))
    with pytest.raises(ValueError):
        list(helper('pairwise_features')(X, batch_size=0))
    with pytest.raises(ValueError):
        helper('pair_residual')(X.x1, X.x2.iloc[:-1], 5)


def test_batched_evaluation_equals_full_materialized_daily_pnl():
    X, y = sample(50)
    X.iloc[8:13, 0] = np.nan
    evaluate = helper('evaluate_features')
    daily, summary = evaluate([X.iloc[:, :2], X.iloc[:, 2:]], y, hl=5)
    pnl = features.standalone_pnl(X, y, hl=5)
    assert_frame_equal(daily, pnl.resample('D').sum())
    assert_allclose(summary.loc[X.columns, 'daily_mean_over_std'], features.sharpe(daily))
    assert_allclose(summary.loc[X.columns, 'pnl_observations'], pnl.count())
    assert_series_equal(summary.loc[X.columns, 'first_pnl_msgStamp'],
                        pnl.apply(lambda x: x.first_valid_index()).rename('first_pnl_msgStamp'), check_names=False)
    for kind in ['product', 'residual']:
        a, _ = evaluate(helper('pairwise_features')(X, kind=kind, hl=5, batch_size=1), y, hl=5)
        b, _ = evaluate(helper('pairwise_features')(X, kind=kind, hl=5, batch_size=5), y, hl=5)
        assert_frame_equal(a, b)


def test_undefined_signals_not_reported_as_valid_sharpes():
    X, y = sample(30)
    X.loc[:, :] = np.nan
    daily, summary = helper('evaluate_features')([X], y, hl=5)
    assert daily.eq(0).all().all()
    assert summary['daily_mean_over_std'].isna().all()
    assert summary.pnl_observations.eq(0).all()


@pytest.mark.parametrize('fit_intercept', [False, True])
def test_one_column_zero_penalty_matches_batch_and_zero_weight_decay(fit_intercept):
    X, _ = sample(60)
    x = X.x1.to_numpy()[:, None] + 8
    y = 3 + 2 * x[:, 0] + X.x2.to_numpy() / 10
    weights = np.linspace(.2, 3, len(y))
    weights[6:10] = 0
    fitter = StreamingWeightedLasso(1, .91, 0, fit_intercept=fit_intercept, store_history=True).fit(x, y, W=weights)
    for n in [4, 9, 20, 60]:
        a = weights[:n] * .91 ** np.arange(n-1, -1, -1)
        design = np.column_stack([np.ones(n), x[:n, 0]]) if fit_intercept else x[:n]
        beta = np.linalg.lstsq(design * np.sqrt(a[:, None]), y[:n] * np.sqrt(a), rcond=None)[0]
        expected_coef = beta[-1]
        expected_intercept = beta[0] if fit_intercept else 0
        assert fitter.get_coefs()[n-1, 0] == pytest.approx(expected_coef, abs=1e-10)
        assert fitter.get_intercepts()[n-1] == pytest.approx(expected_intercept, abs=1e-10)
    assert fitter.n_failed_ == 0


def test_display_results_has_two_separate_figures_and_all_series(monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from takehome import plots

    assert hasattr(plots, 'display_results'), 'Missing shared display_results helper'
    daily = pd.DataFrame({'a': [1., 2., -1., .5], 'b': [-1., 3., 2., .5], 'zero': [0., 0., 0., 0.]},
                         index=pd.date_range('2020-01-01', periods=4))
    shown = []
    monkeypatch.setattr(plt, 'show', lambda: shown.append(plt.gcf()))
    scores = plots.display_results(daily, 'Example')
    assert len(shown) == 2 and shown[0] is not shown[1]
    assert len(shown[0].axes[0].lines) == len(daily.columns)
    for j, line in enumerate(shown[0].axes[0].lines):
        assert_allclose(line.get_ydata(), daily.iloc[:, j].cumsum())
    assert_allclose(scores, features.sharpe(daily), equal_nan=True)
    assert sum(p.get_height() for p in shown[1].axes[0].patches) == scores.notna().sum()
    assert not plt.get_fignums()


@pytest.mark.parametrize('fit_intercept', [False, True])
def test_scalar_fast_path_matches_generic_coordinate_descent_history(fit_intercept):
    X, y = sample(90)
    x = X[['x1']].to_numpy()
    weights = np.where(np.arange(len(y)) % 7 == 0, 0., 1.)
    scalar = StreamingWeightedLasso(1, .93, 0, fit_intercept=fit_intercept, store_history=True).fit(x, y, W=weights)
    # A zero second column forces the original multidimensional CD path without changing the model.
    generic = StreamingWeightedLasso(2, .93, 0, fit_intercept=fit_intercept, store_history=True).fit(
        np.column_stack([x[:, 0], np.zeros(len(x))]), y, W=weights)
    assert_allclose(scalar.get_coefs()[:, 0], generic.get_coefs()[:, 0], atol=1e-12, rtol=1e-10)
    assert_allclose(scalar.get_intercepts(), generic.get_intercepts(), atol=1e-12, rtol=1e-10)
    assert_allclose(scalar.C_, generic.C_[:1, :1], atol=1e-12)
    assert_allclose(scalar.c_, generic.c_[:1], atol=1e-12)

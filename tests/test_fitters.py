"""Independent batch objectives, not another copy of the streaming algorithm."""
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.linear_model import Lasso

from takehome.fitters import StreamingWeightedLasso


def sample(seed=7, n=80, p=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * np.arange(1, p + 1) + .4
    y = .7 + X @ np.linspace(-.8, .5, p) + rng.normal(size=n) * .15
    return X, y, rng.lognormal(0, 1, n)


def reference(X, y, weights, decay, alpha, intercept):
    a = weights * decay ** np.arange(len(y) - 1, -1, -1)
    keep = a > 0
    X, y, a = X[keep], y[keep], a[keep]
    mean = np.average(X, axis=0, weights=a)
    scale = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=a))
    # Constant nonzero columns need an RMS fallback without an intercept.
    rms = np.sqrt(np.average(X ** 2, axis=0, weights=a))
    scale = np.where(scale > 0, scale, 1 if intercept else rms)
    scale = np.where(scale > 0, scale, 1)
    ref = Lasso(alpha=alpha, fit_intercept=intercept, tol=1e-12, max_iter=100000)
    ref.fit(X / scale, y, sample_weight=a)
    return ref.coef_ / scale, ref.intercept_


@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('decay', [1., .94])
def test_every_prefix_matches_weighted_batch_lasso(intercept, decay):
    X, y, weights = sample(n=50)
    m = StreamingWeightedLasso(4, decay, .07, fit_intercept=intercept, max_iter=50000, tol=1e-10)
    for t in range(len(y)):
        m.partial_fit(X[t], y[t], weight=weights[t])
        if t < 8:
            continue
        coef, b = reference(X[:t+1], y[:t+1], weights[:t+1], decay, .07, intercept)
        assert_allclose(m.coef, coef, atol=2e-7)
        assert_allclose(m.intercept_, b, atol=2e-7)
        assert m.converged_
        assert m.kkt_violation_ <= m.kkt_tolerance_


@pytest.mark.parametrize('intercept', [False, True])
def test_alpha_zero_matches_direct_weighted_least_squares(intercept):
    X, y, weights = sample()
    m = StreamingWeightedLasso(4, .98, 0, fit_intercept=intercept, max_iter=50000, tol=1e-11).fit(X, y, W=weights)
    a = weights * .98 ** np.arange(len(y)-1, -1, -1)
    A = np.column_stack([np.ones(len(y)), X]) if intercept else X
    beta = np.linalg.lstsq(A * np.sqrt(a[:, None]), y * np.sqrt(a), rcond=None)[0]
    assert_allclose(m.predict(X), A @ beta, atol=1e-8)


def test_one_feature_soft_threshold_formula():
    X, y, weights = sample(p=1)
    m = StreamingWeightedLasso(1, .96, .12, tol=1e-12).fit(X, y, W=weights)
    a = weights * .96 ** np.arange(len(y)-1, -1, -1)
    mu = np.average(X[:, 0], weights=a)
    sigma = np.sqrt(np.average((X[:, 0] - mu)**2, weights=a))
    rho = np.sum(a * X[:, 0] * y)
    threshold = .12 * sigma * a.sum()
    expected = np.sign(rho) * max(abs(rho) - threshold, 0) / np.sum(a * X[:, 0]**2)
    assert_allclose(m.coef[0], expected, atol=1e-11)


def test_diagonal_W_and_common_weight_rescaling():
    X, y, w = sample()
    models = [StreamingWeightedLasso(4, .97, .08).fit(X, y, W=W)
              for W in (w, np.diag(w), w * 1e20)]
    for m in models[1:]:
        assert_allclose(m.coef, models[0].coef, atol=1e-9)


def test_feature_unit_rescaling_preserves_predictions():
    X, y, w = sample()
    scale = np.array([.01, -12., 1e4, .3])
    a = StreamingWeightedLasso(4, .97, .04, tol=1e-11).fit(X, y, W=w)
    b = StreamingWeightedLasso(4, .97, .04, tol=1e-11).fit(X * scale, y, W=w)
    assert_allclose(a.predict(X), b.predict(X * scale), atol=1e-8)
    assert_allclose(a.coef, b.coef * scale, atol=1e-8)


def test_weighted_moments_match_explicit_matrix():
    X, y, w = sample()
    m = StreamingWeightedLasso(4, .96, .1).fit(X, y, W=w)
    a = w * .96 ** np.arange(len(y)-1, -1, -1)
    mean = np.average(X, axis=0, weights=a)
    ym = np.average(y, weights=a)
    assert_allclose(m.Wsum, a.sum())
    assert_allclose(m.mean_x_, mean)
    assert_allclose(m.mean_y_, ym)
    assert_allclose(m.C_, (X-mean).T @ (a[:, None] * (X-mean)))
    assert_allclose(m.c_, (X-mean).T @ (a * (y-ym)))


def test_chunks_partial_fit_and_history_match():
    X, y, w = sample()
    a = StreamingWeightedLasso(4, .95, .08, store_history=True).fit(X, y, W=w)
    b = StreamingWeightedLasso(4, .95, .08, store_history=True)
    b.fit(X[:30], y[:30], W=w[:30]).fit(X[30:], y[30:], W=w[30:])
    c = StreamingWeightedLasso(4, .95, .08, store_history=True)
    for x, yy, ww in zip(X, y, w):
        c.partial_fit(x, yy, ww)
    assert_allclose(a.get_coefs(), b.get_coefs(), atol=0, rtol=0)
    assert_allclose(a.get_coefs(), c.get_coefs(), atol=0, rtol=0)
    assert_allclose(a.get_intercepts(), b.get_intercepts(), atol=0, rtol=0)
    assert a.get_coefs().shape == X.shape
    assert a.get_intercepts().shape == y.shape
    copy = a.get_coefs()
    copy[:] = 999
    assert not np.any(a.coef == 999)


def test_skipped_row_advances_decay_without_fake_observation():
    X, y, w = sample()
    w[10:19] = 0
    X[10:19], y[10:19] = np.nan, np.nan
    m = StreamingWeightedLasso(4, .92, .08, tol=1e-11).fit(X, y, W=w)
    coef, b = reference(X, y, w, .92, .08, False)
    assert_allclose(m.coef, coef, atol=1e-8)
    old = m.coef.copy()
    total = m.Wsum
    m.partial_fit(np.full(4, np.nan), np.nan, weight=0)
    assert_allclose(m.coef, old, atol=0, rtol=0)
    assert_allclose(m.Wsum, .92 * total)


@pytest.mark.parametrize('intercept', [False, True])
def test_zero_and_constant_columns_stay_finite(intercept):
    rng = np.random.default_rng(5)
    x = rng.normal(size=100)
    X = np.column_stack([x, np.zeros(100), np.full(100, 3.)])
    y = 4 + .8 * x
    m = StreamingWeightedLasso(3, 1, .1, fit_intercept=intercept, tol=1e-11).fit(X, y)
    coef, b = reference(X, y, np.ones(100), 1, .1, intercept)
    assert np.isfinite(m.coef).all()
    assert m.coef[1] == 0
    assert_allclose(m.predict(X), X @ coef + b, atol=1e-8)


def test_large_offset_intercept_is_stable():
    X, y, w = sample()
    shifted = X + 1e7
    a = StreamingWeightedLasso(4, .95, .1, fit_intercept=True, tol=1e-11).fit(X, y, W=w)
    b = StreamingWeightedLasso(4, .95, .1, fit_intercept=True, tol=1e-11).fit(shifted, y, W=w)
    assert_allclose(a.predict(X), b.predict(shifted), atol=5e-7)


def test_future_values_do_not_change_stored_past_coefficients():
    X, y, w = sample()
    a = StreamingWeightedLasso(4, .95, .1, store_history=True).fit(X, y, W=w)
    y[40:] += 999
    b = StreamingWeightedLasso(4, .95, .1, store_history=True).fit(X, y, W=w)
    assert_allclose(a.get_coefs()[:40], b.get_coefs()[:40], atol=0, rtol=0)


def test_pickle_resume_and_no_history_default():
    X, y, w = sample()
    a = StreamingWeightedLasso(4, .95, .1).fit(X[:40], y[:40], W=w[:40])
    b = pickle.loads(pickle.dumps(a))
    a.fit(X[40:], y[40:], W=w[40:])
    b.fit(X[40:], y[40:], W=w[40:])
    assert_allclose(a.coef, b.coef, atol=0, rtol=0)
    assert a.history == []
    with pytest.raises(ValueError, match='store_history'):
        a.get_coefs()


def test_nonconvergence_is_reported():
    X, y, w = sample()
    with pytest.warns(RuntimeWarning, match='converge'):
        m = StreamingWeightedLasso(4, 1, .01, max_iter=1, tol=1e-14).fit(X, y, W=w)
    assert m.n_failed_ > 0
    assert not m.converged_


def test_rejects_correlated_W_negative_weights_and_positive_weight_missing_rows():
    X, y, w = sample(n=10)
    m = StreamingWeightedLasso(4, .95, .1)
    for W in [np.ones((10, 10)), -np.ones(10), np.full(10, np.nan), np.ones(9)]:
        with pytest.raises(ValueError):
            m.fit(X, y, W=W)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        m.fit(X, y)
    assert m.Wsum == 0


@pytest.mark.parametrize('kwargs', [dict(decay=0), dict(decay=1.1), dict(alpha=-1),
                                    dict(n_features=0), dict(tol=0), dict(max_iter=0)])
def test_invalid_configuration(kwargs):
    args = dict(n_features=2, decay=.99, alpha=.1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        StreamingWeightedLasso(**args)


def test_duplicate_features_compare_predictions_and_objective_not_nonunique_coefs():
    X, y, w = sample(p=2)
    X = np.column_stack([X, X[:, 0]])
    m = StreamingWeightedLasso(3, .97, .08, fit_intercept=True, tol=1e-11).fit(X, y, W=w)
    coef, b = reference(X, y, w, .97, .08, True)
    assert_allclose(m.predict(X), X @ coef + b, atol=1e-8)


def test_external_KKT_conditions_for_weighted_scaled_objective():
    X, y, w = sample()
    m = StreamingWeightedLasso(4, .95, .2, tol=1e-11).fit(X, y, W=w)
    a = w * .95 ** np.arange(len(y)-1, -1, -1)
    mean = np.average(X, axis=0, weights=a)
    std = np.sqrt(np.average((X-mean)**2, axis=0, weights=a))
    gradient = X.T @ (a * (X @ m.coef - y)) / a.sum() / std
    active = m.coef != 0
    assert np.max(np.abs(gradient[active] + .2 * np.sign(m.coef[active]))) < 1e-8
    assert np.all(np.abs(gradient[~active]) <= .2 + 1e-8)

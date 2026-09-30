"""Independent objectives, exact fold chronology, and a compiled estimator core."""
import pickle
import numpy as np
import pandas as pd
import pytest
from numba import njit
from numpy.testing import assert_allclose
from takehome import fitters, features


def sample(n=60):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, 3)) * [1, 3, .2] + .4
    y = .1 + X @ [.3, -.2, .1] + rng.normal(size=n) * .03
    w = rng.lognormal(size=n)
    w[8:12] = 0
    return X, y, w


def test_core_is_a_real_jitclass_and_runs_inside_njit():
    assert hasattr(fitters, 'StreamingWeightedLasso_'), 'Missing compiled estimator class'
    cls = fitters.StreamingWeightedLasso_
    assert hasattr(cls, 'class_type')
    @njit
    def compiled(X, y, w):
        m = cls(3, .97, .04, 20000, 1e-11, True, True)
        predictions = m.fit(X, y, w)
        return predictions, m.get_coefs(), m.get_intercepts(), m.coef
    X, y, w = sample()
    p, b, a, final = compiled(X, y, w)
    m = fitters.StreamingWeightedLasso(3, .97, .04, max_iter=20000, tol=1e-11,
                                     fit_intercept=True, store_history=True)
    assert_allclose(m.fit_predict(X, y, W=w), p, equal_nan=True, atol=0, rtol=0)
    assert_allclose(m.get_coefs(), b, atol=0, rtol=0)
    assert_allclose(m.get_intercepts(), a, atol=0, rtol=0)
    assert_allclose(m.coef, final, atol=0, rtol=0)


def test_jitclass_wrapper_pickle_chunks_and_preupdate_predictions():
    X, y, w = sample()
    m = fitters.StreamingWeightedLasso(3, .97, .04, fit_intercept=True, store_history=True)
    p = m.fit_predict(X[:25], y[:25], W=w[:25])
    restored = pickle.loads(pickle.dumps(m))
    p = np.r_[p, m.fit_predict(X[25:], y[25:], W=w[25:])]
    restored.fit(X[25:], y[25:], W=w[25:])
    assert_allclose(m.get_coefs(), restored.get_coefs(), atol=0, rtol=0)
    expected = np.r_[np.nan, np.einsum('ij,ij->i', X[1:], m.get_coefs()[:-1]) + m.get_intercepts()[:-1]]
    assert_allclose(p, expected, equal_nan=True)


@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('alpha', [0., .04])
def test_batch_lasso_matches_independent_row_loss(intercept, alpha):
    assert hasattr(fitters, 'BatchLasso')
    X, y, w = sample()
    a = fitters.BatchLasso(3, .97, alpha, fit_intercept=intercept, tol=1e-12).fit(X, y, W=w)
    b = fitters.CvxpyWeightedLasso(3, .97, alpha, fit_intercept=intercept, tol=1e-12).fit(X, y, W=w)
    assert_allclose(a.predict(X), b.predict(X), atol=5e-7)
    assert abs(a.objective_ - b.objective_) < 1e-9


def test_decayed_batch_ridge_matches_explicit_cvxpy_loss():
    import cvxpy as cp
    X, y, w = sample()
    model = fitters.BatchRidge(3, alpha=.03, decay=.97).fit(X, y, W=w)
    a = w * .97 ** np.arange(len(y)-1, -1, -1); a /= a.sum()
    mean = a @ X
    scale = np.sqrt(a @ (X-mean)**2)
    beta, bias = cp.Variable(3), cp.Variable()
    loss = .5 * cp.sum_squares(cp.multiply(np.sqrt(a), y-X@beta-bias))
    problem = cp.Problem(cp.Minimize(loss + .015 * cp.sum_squares(cp.multiply(scale, beta))))
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-12, tol_feas=1e-12, tol_gap_rel=1e-12)
    assert_allclose(model.predict(X), X@beta.value+bias.value, atol=1e-8)


def test_walkforward_sweep_uses_previous_fold_and_is_prefix_invariant():
    assert hasattr(fitters, 'walk_forward_sweep')
    X, y, w = sample(n=47)
    config = dict(min_train_size=12, step=7, train_size=24, gap=2)
    models = {'ridge': fitters.BatchRidge(3, .03, decay=.97),
              'lasso': fitters.BatchLasso(3, .97, .04, fit_intercept=True)}
    paths = fitters.walk_forward_sweep(models, X, y, W=w, **config)
    for name, path in paths.items():
        assert np.isnan(path.prediction_[:14]).all()
        for fold in path.folds_:
            a, b, c, d = [fold[k] for k in ('train_start','train_stop','predict_start','predict_stop')]
            assert b == c-2 and b-a <= 24
            ref = (fitters.BatchRidge(3, .03, decay=.97) if name=='ridge' else
                   fitters.BatchLasso(3, .97, .04, fit_intercept=True)).fit(X[a:b], y[a:b], W=w[a:b])
            assert_allclose(path.prediction_[c:d], ref.predict(X[c:d]), atol=1e-8)
        expected = np.einsum('ij,ij->i', X, path.get_coefs(lag=1)) + path.get_intercepts(lag=1)
        assert_allclose(path.prediction_, expected, equal_nan=True)
        assert_allclose(models[name].coef, 0)
    changed = y.copy(); changed[28:] += 100
    altered = fitters.walk_forward_sweep(models, X, changed, W=w, **config)
    for name in paths:
        assert_allclose(paths[name].prediction_[:29], altered[name].prediction_[:29], equal_nan=True)


def test_batch_frozen_stream_matches_independent_lasso_at_same_cutoffs():
    assert hasattr(fitters, 'walk_forward_sweep')
    X, y, w = sample()
    paths = fitters.walk_forward_sweep({'lasso': fitters.BatchLasso(3, .97, .04, fit_intercept=True)},
                                     X, y, W=w, min_train_size=15, step=9)
    stream = fitters.StreamingWeightedLasso(3, .97, .04, fit_intercept=True, tol=1e-11, max_iter=20000)
    last = 0
    for fold in paths['lasso'].folds_:
        stop, first, end = [fold[k] for k in ('train_stop','predict_start','predict_stop')]
        stream.fit(X[last:stop], y[last:stop], W=w[last:stop]); last=stop
        assert_allclose(stream.predict(X[first:end]), paths['lasso'].prediction_[first:end], atol=5e-7)


def test_cashflow_feature_is_last_raw_ratio_not_clipped_and_preserves_input():
    assert hasattr(features, 'with_cashflow_feature')
    df = pd.DataFrame({'x1':[1,2,3,4], 'x99':[2,3,4,5],
                       'cashflow':[2.,-6.,1.,np.nan], 'volume':[1.,2.,0.,1.]})
    result = features.with_cashflow_feature(df)
    assert result.columns[-1] == 'x100' and 'x100' not in df
    assert_allclose(result.x100, [2.,-3.,np.nan,np.nan], equal_nan=True)


def test_forward_feature_shifts_do_not_cross_the_training_boundary():
    X = pd.DataFrame({'x1':np.arange(40.)}, index=pd.date_range('2020-01-01',periods=40,freq='5min'))
    training = X.iloc[:30]
    for h in [1,2,6,12,24]:
        oracle = training.shift(-h)
        assert oracle.iloc[-h:].isna().all().all()
        assert_allclose(oracle.iloc[:-h,0], training.iloc[h:,0])

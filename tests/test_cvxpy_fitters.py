"""Compare the declared convex loss and the causal walk-forward contract."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from takehome.fitters import StreamingWeightedLasso, CvxpyWeightedLasso, BatchedFitters


def sample(n=72, p=3):
    rng = np.random.default_rng(31)
    X = rng.normal(size=(n, p)) * np.arange(1, p+1) + .4
    y = .2 + X @ np.linspace(-.7, .4, p) + rng.normal(0, .1, n)
    w = (rng.uniform(1800, 2200, n) * rng.lognormal(4, 1, n))**2
    return X, y, w


def objective(X, y, w, coef, bias, decay, alpha):
    a = w * decay**np.arange(len(y)-1, -1, -1)
    keep = a > 0
    X, y, a = X[keep], y[keep], a[keep]
    a /= a.sum()
    s = np.sqrt(a @ (X - np.average(X, weights=a, axis=0))**2)
    return (a @ (y - X @ coef - bias)**2) / 2 + alpha * (s @ abs(coef))


@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('alpha', [0., .08])
def test_cvxpy_matches_online_loss_and_predictions(intercept, alpha):
    X, y, w = sample()
    w[10:15] = 0
    X[10:15] = np.nan
    y[10:15] = np.nan
    online = StreamingWeightedLasso(3, .96, alpha, fit_intercept=intercept,
                                   max_iter=50000, tol=1e-11, store_history=True).fit(X, y, W=w)
    for n in [20, 40, 72]:
        ref = CvxpyWeightedLasso(3, .96, alpha, fit_intercept=intercept, tol=1e-12).fit(X[:n], y[:n], W=w[:n])
        b, intercept_online = online.get_coefs()[n-1], online.get_intercepts()[n-1]
        assert_allclose(b, ref.coef, atol=3e-7, rtol=3e-7)
        assert_allclose(intercept_online, ref.intercept_, atol=3e-7)
        lo = objective(X[:n], y[:n], w[:n], b, intercept_online, .96, alpha)
        lc = objective(X[:n], y[:n], w[:n], ref.coef, ref.intercept_, .96, alpha)
        assert abs(lo-lc) < 1e-9
        assert_allclose(ref.objective_, lc, atol=1e-12)
        assert ref.status_ == 'optimal'


def test_cvxpy_weight_vector_matrix_and_units():
    X, y, w = sample()
    models = [CvxpyWeightedLasso(3, .98, .04, tol=1e-12).fit(X, y, W=weights)
              for weights in [w, np.diag(w), w*1e20]]
    for m in models[1:]:
        assert_allclose(m.coef, models[0].coef, atol=1e-8)
    multiplier = np.array([.01, -30, 1e4])
    changed = CvxpyWeightedLasso(3, .98, .04, tol=1e-12).fit(X*multiplier, y, W=w)
    assert_allclose(changed.predict(X*multiplier), models[0].predict(X), atol=3e-7)


def test_cvxpy_return_sized_response_and_large_offset():
    X, y, w = sample()
    X += 1e7
    y *= 1e-4
    cvx = CvxpyWeightedLasso(3, .97, 1e-5, fit_intercept=True, tol=1e-12).fit(X, y, W=w)
    online = StreamingWeightedLasso(3, .97, 1e-5, fit_intercept=True, tol=1e-13).fit(X, y, W=w)
    assert_allclose(cvx.predict(X), online.predict(X), atol=3e-9)


def test_cvxpy_rejects_invalid_weights_and_resets():
    X, y, w = sample(n=8)
    model = CvxpyWeightedLasso(3)
    for W in [-w, np.ones((8,8)), np.ones(7), np.full(8, np.nan)]:
        with pytest.raises(ValueError): model.fit(X, y, W=W)
    model.fit(X, y)
    model.fit(np.full_like(X,np.nan), np.full_like(y,np.nan), W=0)
    assert_allclose(model.coef, 0)
    assert model.status_ == 'no_observations'


def test_batched_expanding_step_one_matches_online_post_and_lagged():
    X, y, w = sample(n=26)
    proto = CvxpyWeightedLasso(3, .98, .07, fit_intercept=True, tol=1e-12)
    batch = BatchedFitters(proto, min_train_size=10, step=1).fit(X, y, W=w)
    online = StreamingWeightedLasso(3, .98, .07, fit_intercept=True, tol=1e-11,
                                   store_history=True).fit(X, y, W=w)
    assert_allclose(batch.get_coefs()[9:], online.get_coefs()[9:], atol=4e-7)
    assert_allclose(batch.get_coefs(lag=1)[10:], online.get_coefs(lag=1)[10:], atol=4e-7)
    assert_allclose(batch.get_intercepts(lag=1)[10:], online.get_intercepts(lag=1)[10:], atol=4e-7)
    assert np.isnan(batch.get_coefs()[:9]).all()
    assert_allclose(proto.coef, 0)


def test_batched_rolling_gap_and_no_future_leak():
    X, y, w = sample(n=43)
    config = dict(min_train_size=8, train_size=12, step=6, gap=2)
    proto = CvxpyWeightedLasso(3, .95, .06, fit_intercept=True, tol=1e-12)
    batch = BatchedFitters(proto, **config).fit(X, y, W=w)
    for fold in batch.folds_:
        start,end,first,last = (fold[k] for k in ['train_start','train_stop','predict_start','predict_stop'])
        assert end <= first-2 and end-start <= 12
        ref = CvxpyWeightedLasso(3,.95,.06,fit_intercept=True,tol=1e-12).fit(X[start:end],y[start:end],W=w[start:end])
        assert_allclose(batch.get_coefs(lag=1)[first:last],np.tile(ref.coef,(last-first,1)),atol=1e-10)
    altered_y, altered_w = y.copy(), w.copy()
    altered_y[25:] += 1000
    altered_w[25:] *= 100
    altered = BatchedFitters(proto, **config).fit(X,altered_y,W=altered_w)
    assert_allclose(batch.get_coefs(lag=1)[:26],altered.get_coefs(lag=1)[:26],equal_nan=True)
    history = batch.get_coefs()
    batch.fit(X,y,W=w)
    assert_allclose(history,batch.get_coefs(),equal_nan=True)
    history[:] = 99
    assert not np.any(batch.get_coefs() == 99)


@pytest.mark.parametrize('kwargs', [dict(step=0),dict(min_train_size=0),dict(gap=-1),
                                    dict(train_size=2,min_train_size=3)])
def test_bad_fold_parameters(kwargs):
    with pytest.raises(ValueError): BatchedFitters(CvxpyWeightedLasso(2),**kwargs)

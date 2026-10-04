"""Variance-scaled inputs and zero-intercept constrained/unconstrained estimators."""
import inspect
import pickle
from pathlib import Path

import cvxpy as cp
import nbformat
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from takehome import features, fitters, selection

ROOT = Path(__file__).resolve().parents[1]


def scaled_design(X, hl=4, variance_hl=7, batch_size=2):
    assert hasattr(features, 'variance_scaled_design'), 'Missing pre-fit EWM-variance division'
    return features.variance_scaled_design(X, hl=hl, variance_hl=variance_hl, batch_size=batch_size)


def raw_frame():
    rng = np.random.default_rng(765)
    index = pd.date_range('2018-01-01', periods=400, freq='12h', tz='America/New_York')
    X = pd.DataFrame(rng.normal(size=(400, 3)), index=index, columns=['x1', 'x2', 'x3'])
    X.iloc[1:10, 0] = 0
    X.iloc[20:26, 1] = np.nan
    X.iloc[30, 2] = np.inf
    return X


def test_prefit_transform_is_literal_variance_not_std_and_imputes_only_after_division():
    X = raw_frame()
    raw = X.rename(columns=lambda c: 'raw:'+c)
    transformed = features.dszl(X, 4).rename(columns=lambda c: 'dszl:'+c)
    both = pd.concat([raw, transformed], axis=1)
    variance = both.replace([0, np.inf, -np.inf], np.nan).ewm(
        halflife=7, min_periods=7, ignore_na=True).var().replace(0, np.nan)
    expected = both.div(variance).replace([np.inf, -np.inf], np.nan).fillna(0)
    actual = scaled_design(X)
    assert_frame_equal(actual, expected)
    assert actual.to_numpy().flags.c_contiguous
    assert actual.shape == (400, 6)
    assert actual.iloc[:6].eq(0).all().all()
    std_version = both.div(np.sqrt(variance)).replace([np.inf, -np.inf], np.nan).fillna(0)
    assert not np.allclose(actual, std_version)


def test_transform_is_prefix_causal_batch_invariant_and_state_not_reset_at_fold():
    X = raw_frame()
    full = scaled_design(X)
    assert_frame_equal(scaled_design(X.iloc[:180]), full.iloc[:180])
    changed = X.copy(); changed.iloc[180:] *= 1000
    assert_frame_equal(scaled_design(changed).iloc[:180], full.iloc[:180])
    assert_frame_equal(scaled_design(X, batch_size=1), full)
    chunks = []
    for name in X:
        chunks.append(scaled_design(X[[name]]))
    assert_frame_equal(pd.concat(chunks, axis=1).reindex(columns=full.columns), full)
    assert not scaled_design(X.iloc[180:]).equals(full.iloc[180:])
    # Current features are available, but future features cannot enter the variance.
    changed = X.copy(); changed.iloc[80] *= 4
    assert_frame_equal(scaled_design(changed).iloc[:80], full.iloc[:80])
    assert not np.allclose(scaled_design(changed).iloc[80], full.iloc[80])


def test_zero_constant_and_missing_features_never_produce_infinite_design():
    X = raw_frame()
    X['zero'] = 0.; X['constant'] = 7.; X['missing'] = np.nan
    actual = scaled_design(X)
    assert np.isfinite(actual.to_numpy()).all()
    for name in ('zero', 'constant', 'missing'):
        assert actual[['raw:'+name, 'dszl:'+name]].eq(0).all().all()


@pytest.mark.parametrize('kwargs', [{'variance_hl':0}, {'variance_hl':True}, {'variance_hl':1.5}, {'variance_hl':1}])
def test_invalid_variance_half_life_is_rejected(kwargs):
    assert hasattr(features, 'variance_scaled_design')
    with pytest.raises(ValueError):
        scaled_design(raw_frame(), **kwargs)


def design():
    rng = np.random.default_rng(144)
    X = rng.normal(size=(150, 4))*[1., 2., .4, 3.] + [2., -3., 1., .3]
    y = 1.4 + X @ np.array([.8, -.3, 0., .2]) + rng.normal(size=len(X))*.2
    w = rng.uniform(.2, 2, len(y)); w[17:25] = 0; y[17:25] = np.nan
    return X, y, w


@pytest.mark.parametrize('name', ['BatchRidge', 'BatchLasso', 'CvxpyWeightedLasso', 'StreamingWeightedLasso'])
def test_every_public_fitter_defaults_to_zero_intercept(name):
    cls = getattr(fitters, name)
    assert inspect.signature(cls).parameters['fit_intercept'].default is False
    X, y, w = design()
    kwargs = dict(n_features=4, decay=.97, alpha=.03)
    if name == 'StreamingWeightedLasso': kwargs.update(max_iter=20000, tol=1e-11, store_history=True)
    model = cls(**kwargs).fit(X, y, W=w)
    assert model.fit_intercept is False and model.intercept_ == 0.
    assert_allclose(model.predict(X), X @ model.coef, atol=0, rtol=0)
    if name == 'StreamingWeightedLasso':
        assert np.array_equal(model.get_intercepts(), np.zeros(len(X)))
    # A zero-intercept fit must not secretly add back training means.
    if name == 'BatchRidge':
        mx, my, C, c, _ = fitters.batch_moments(X, y, w, .97)
        raw_G = C + np.outer(mx, mx); raw_h = c + mx*my
        expected = np.linalg.solve(raw_G + .03*np.diag(model.scale_**2), raw_h)
        assert_allclose(model.coef, expected, atol=1e-10)


@pytest.mark.parametrize('name', ['BatchRidge', 'BatchLasso', 'CvxpyWeightedLasso', 'StreamingWeightedLasso'])
def test_nonnegative_without_intercept_matches_independent_row_loss(name):
    cls = getattr(fitters, name)
    assert 'nonneg' in inspect.signature(cls).parameters, 'Missing nonnegative variant'
    X, y, w = design()
    config = dict(n_features=4, decay=.97, alpha=.03, fit_intercept=False, nonneg=True)
    if name == 'StreamingWeightedLasso': config.update(max_iter=30000, tol=1e-11, store_history=True)
    model = cls(**config).fit(X, y, W=w)
    keep = w > 0
    ages = np.arange(len(y)-1, -1, -1)
    a = w[keep]*.97**ages[keep]; a /= a.sum()
    x, target = X[keep], y[keep]
    mx = a @ x; std = np.sqrt(a @ (x-mx)**2)
    beta = cp.Variable(4, nonneg=True)
    loss = cp.sum_squares(cp.multiply(np.sqrt(a), x@beta-target))/2
    penalty = .03*cp.sum_squares(cp.multiply(std, beta))/2 if name == 'BatchRidge' else .03*cp.norm1(cp.multiply(std, beta))
    problem = cp.Problem(cp.Minimize(loss+penalty))
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
    assert problem.status == cp.OPTIMAL
    assert model.intercept_ == 0. and (model.coef >= 0).all()
    assert_allclose(model.predict(X), X@beta.value, atol=3e-6, rtol=3e-6)
    if name == 'StreamingWeightedLasso':
        assert (model.get_coefs() >= 0).all() and (model.get_intercepts() == 0).all()
        copied = pickle.loads(pickle.dumps(model))
        assert copied.nonneg and not copied.fit_intercept
        assert_allclose(copied.predict(X), model.predict(X))


def test_nonnegative_scalar_fast_path_correct_at_boundary():
    assert 'nonneg' in inspect.signature(fitters.StreamingWeightedLasso).parameters
    X = np.arange(1., 11.)[:, None]; y = -X[:,0]
    m = fitters.StreamingWeightedLasso(1, .98, 0., nonneg=True, store_history=True).fit(X, y)
    assert m.intercept_ == 0. and m.coef[0] == 0. and m.converged_
    assert np.array_equal(m.get_coefs(), np.zeros_like(X))


def test_selected_nonnegative_stream_keeps_constraint_and_zero_intercepts():
    assert 'nonneg' in inspect.signature(fitters.BatchLasso).parameters
    X, y, w = design()
    folds = [dict(train_start=0,train_stop=49,predict_start=50,predict_stop=100),
             dict(train_start=50,train_stop=99,predict_start=100,predict_stop=150)]
    candidates = {a:fitters.BatchLasso(4,.97,a,nonneg=True) for a in [.3,.03,.003]}
    chosen = selection.walk_forward_select(candidates,X,y,W=w,folds=folds)
    result = selection.stream_selected_folds(chosen,X,y,W=w,tol=1e-11,max_iter=30000)
    assert (result['coefs'] >= 0).all() and (result['intercepts'] == 0).all()
    assert_allclose(result['frozen'],chosen.prediction_,atol=4e-6, equal_nan=True)


def test_notebook_declares_variance_preprocessing_and_only_zero_intercept_estimators():
    cells = nbformat.read(ROOT/'notebooks/takehome.ipynb',4).cells
    source = '\n'.join(c.source for c in cells if c.cell_type == 'code')
    assert 'variance_scaled_design' in source
    assert 'fit_intercept=True' not in source.replace(' ', '')
    assert '[False, True]' not in source
    assert 'Ridge nonneg' in source and 'Lasso nonneg' in source
    for tag in ('regression_setup','submission_design'):
        cell = next(c for c in cells if tag in c.metadata.get('tags', []))
        assert 'variance_scaled_design' in cell.source
    config = next(c for c in cells if 'submission_config' in c.metadata.get('tags', []))
    assert 'fit_intercept=False' in config.source.replace(' ', '')


@pytest.mark.parametrize('name', ['BatchLasso', 'CvxpyWeightedLasso'])
@pytest.mark.parametrize('nonneg', [False, True])
def test_certified_zero_lasso_is_exact_zero_not_amplifiable_solver_noise(name, nonneg):
    X, y, w = design()
    model = getattr(fitters, name)(4, .97, 100., fit_intercept=False, nonneg=nonneg).fit(X,y,W=w)
    assert np.array_equal(model.coef, np.zeros(4))
    assert np.array_equal(model.predict(X), np.zeros(len(X)))
    assert model.intercept_ == 0.

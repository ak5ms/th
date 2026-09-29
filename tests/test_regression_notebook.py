"""Exercise the actual notebook prediction cell so chart alignment cannot drift."""
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]


def tagged(tag):
    nb = nbformat.read(ROOT / 'notebooks/01_eda.ipynb', as_version=4)
    cells = [c for c in nb.cells if tag in c.metadata.get('tags', [])]
    assert len(cells) == 1, f'Expected one cell tagged {tag}'
    return cells[0].source


def run_predictions(X, beta, bias, valid):
    df = pd.DataFrame(X, columns=['x1', 'x2'], index=pd.date_range('2020-01-01', periods=len(X)))
    ns = dict(pd=pd, np=np, df=df, fit_columns=['x1', 'x2'], valid_fit=np.asarray(valid),
              fitter=SimpleNamespace(get_coefs=lambda: np.asarray(beta, float),
                                     get_intercepts=lambda: np.asarray(bias, float)))
    exec(tagged('lagged_regression_predictions'), ns)
    return ns['yhat'].to_numpy()


def test_notebook_lags_both_coefficients_and_intercept():
    X = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
    beta = np.array([[1., 2.], [10., 20.], [100., 200.], [1000., 2000.]])
    bias = np.array([1., 10., 100., 1000.])
    actual = run_predictions(X, beta, bias, [True] * 4)
    expected = np.r_[np.nan, np.einsum('ij,ij->i', X[1:], beta[:-1]) + bias[:-1]]
    assert_allclose(actual, expected, equal_nan=True)
    beta[2:] *= 100
    bias[2:] *= 100
    assert_allclose(run_predictions(X, beta, bias, [True] * 4)[:3], actual[:3], equal_nan=True)


def test_notebook_zero_imputes_predictors_but_masks_prefit_rows():
    X = np.array([[1., 2.], [3., 4.], [5., np.nan], [7., 8.]])
    actual = run_predictions(X, np.ones((4, 2)), np.zeros(4), [False, True, False, True])
    assert_allclose(actual, [np.nan, np.nan, 5., 15.], equal_nan=True)


def test_every_feature_family_requests_its_own_overlay():
    for family, data in [('raw', 'raw'), ('dszl', 'dszl'), ('interaction', 'interaction'), ('residual', 'residual')]:
        source = tagged(f'family_overlay_{family}')
        assert f"meta_daily={data}_meta.resample('D').sum()" in source
    nb = nbformat.read(ROOT / 'notebooks/01_eda.ipynb', as_version=4)
    real = next(c for c in nb.cells if 'all_predictor_fit' in c.metadata.get('tags', []))
    assert 'store_history=True' in real.source

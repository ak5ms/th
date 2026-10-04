"""Exercise the actual notebook's OOS snapshot reconstruction, not raw inputs."""
from pathlib import Path
from types import SimpleNamespace
import nbformat
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

ROOT=Path(__file__).resolve().parents[1]

def tagged(tag):
    nb=nbformat.read(ROOT/'notebooks/takehome.ipynb',4)
    cells=[c for c in nb.cells if tag in c.metadata.get('tags',[])]
    assert len(cells)==1
    return cells[0].source


def run_predictions(X,folds,coefs,biases):
    X=np.nan_to_num(np.asarray(X,float),nan=0.,posinf=0.,neginf=0.)
    expected=np.full(len(X),np.nan)
    for f,b,bias in zip(folds,coefs,biases):
        a,z=f['predict_start'],f['predict_stop']
        expected[a:z]=X[a:z]@b+bias
    path=SimpleNamespace(folds_=folds,snapshots_=coefs,offsets_=biases)
    df=pd.DataFrame(X*1000,index=pd.date_range('2020-01-01',periods=len(X)))
    ns=dict(np=np,pd=pd,df=df,X_fit=X,ridge_selected=SimpleNamespace(path_=path,prediction_=expected))
    exec(tagged('lagged_regression_predictions'),ns)
    return ns['yhat'].to_numpy()


def test_notebook_snapshots_do_not_shift_again_or_apply_backward():
    X=np.arange(16,dtype=float).reshape(8,2)
    folds=[dict(predict_start=2,predict_stop=5),dict(predict_start=5,predict_stop=8)]
    coefs=[np.array([1.,2.]),np.array([10.,20.])];biases=[3.,30.]
    y=run_predictions(X,folds,coefs,biases)
    expected=np.r_[np.full(2,np.nan),X[2:5]@coefs[0]+3,X[5:]@coefs[1]+30]
    assert_allclose(y,expected,equal_nan=True)
    coefs[1]*=100;biases[1]*=100
    assert_allclose(run_predictions(X,folds,coefs,biases)[:5],y[:5],equal_nan=True)


def test_notebook_uses_finite_combined_design_and_excludes_initial_tuning_rows():
    X=np.array([[1.,2.],[3.,4.],[5.,np.nan],[7.,8.]])
    f=[dict(predict_start=2,predict_stop=4)]
    actual=run_predictions(X,f,[np.ones(2)],[1.])
    assert_allclose(actual,[np.nan,np.nan,6.,16.],equal_nan=True)


def test_every_feature_family_requests_its_own_overlay():
    for family,data in [('raw','raw'),('dszl','dszl'),('interaction','interaction'),('residual','residual')]:
        assert f"meta_daily={data}_meta.resample('D').sum()" in tagged(f'family_overlay_{family}')
    assert 'stream_selected_folds' in tagged('all_predictor_fit')

"""Align the first-selected-fold mask to rows in a DataFrame control."""
from pathlib import Path
import nbformat

p=Path('notebooks/takehome.ipynb');nb=nbformat.read(p,4)
for c in nb.cells:
    if 'alpha_forecast_controls' in c.metadata.get('tags',[]):
        old='ready=ready.where(rows >= alpha_first, False, axis=0)'
        assert old in c.source
        c.source=c.source.replace(old,'ready=ready.where(pd.Series(rows >= alpha_first, index=df.index), False, axis=0)')
        c.outputs=[];c.execution_count=None;c.metadata.pop('execution',None)
nbformat.write(nb,p)
p=Path('tests/test_prequential_selection.py')
p.write_text(p.read_text()+'''


def test_notebook_alpha_control_masks_rows_before_first_selected_fold():
    import ast,nbformat
    nb=nbformat.read(Path(__file__).resolve().parents[1]/'notebooks/takehome.ipynb',4)
    src=next(c.source for c in nb.cells if 'alpha_forecast_controls' in c.metadata.get('tags',[]))
    definition=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef))
    ix=pd.date_range('2020',periods=20,freq='D')
    frame=pd.DataFrame({'x1':np.arange(20,dtype=float)+1, 'x2':np.arange(20,dtype=float)+3},index=ix)
    ns=dict(np=np,pd=pd,df=frame,x_cols=frame.columns,rows=np.arange(20),alpha_first=10,AR_MIN_TRAIN=3)
    exec(compile(ast.Module(body=[definition],type_ignores=[]),'<controls>','exec'),ns)
    out=pd.concat(list(ns['matched_alpha_controls']()),axis=1)
    assert out.iloc[:10].isna().all().all()
    assert_frame_equal(out.iloc[10:],frame.iloc[10:])
''')

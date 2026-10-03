"""Refresh window-dependent outputs without rerunning unchanged feature-family EDA."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path.cwd()
base_path=root/'notebooks/takehome.ipynb'
asset_path=root/'notebooks/takehome_asset_vol.ipynb'
base=nbformat.read(base_path,4)
asset=nbformat.read(asset_path,4)


def tagged(nb, tag):
    return next(c for c in nb.cells if tag in c.metadata.get('tags',[]))


setup=nbformat.v4.new_code_cell('''from takehome.forecast import _model_data
from takehome.features import ts_std, sharpe
labels=pd.read_parquet(DATA_PATH,columns=['msgStamp','ret_5m'])
if 'msgStamp' in labels: labels=labels.set_index('msgStamp')
_,split=training_data(labels.sort_index())
df=_model_data(DATA_PATH,split['first_label'],split['cutoff'])
del labels
assert len(df)==split['train_rows']==640734
x_cols=df.head(0).filter(like='x').columns
HL,META_HL=6048,252*288
''')
tags=['regression_setup','batch_oos_sweep','batch_forecast_export','ridge_sizing_diagnostic',
      'all_predictor_fit','implementation_audit','lagged_regression_predictions',
      'coefficient_paths','regression_calibration_plot']
summary=next(c for c in base.cells if c.cell_type=='code' and "print('Fold OOS scoring starts:'" in c.source)
summary.metadata['tags']=list(dict.fromkeys([*summary.metadata.get('tags',[]),'two_year_summary']))
selected=[nbformat.v4.new_code_cell("get_ipython().run_line_magic('matplotlib','inline')"),
          deepcopy(base.cells[1]), setup]
selected += [deepcopy(tagged(base,tag)) for tag in tags]
selected += [deepcopy(tagged(asset,'asset_vol_comparison')),deepcopy(summary)]
for c in selected:
    c.outputs=[];c.execution_count=None;c.metadata.pop('execution',None)
nb=nbformat.v4.new_notebook(cells=selected,metadata=deepcopy(base.metadata))


def progress(cell,cell_index):
    print('EXECUTING',cell_index,cell.metadata.get('tags',[]),flush=True)


try:
    NotebookClient(nb,timeout=1800,kernel_name='python3',on_cell_start=progress,
        resources={'metadata':{'path':str(root)}}).execute()
except BaseException:
    nbformat.write(nb,'/tmp/two-year-failed.ipynb')
    raise
for c in nb.cells:
    for output in c.get('outputs',[]):
        if output.output_type=='stream': print(output.text,end='',flush=True)
assert not any(o.output_type=='error' for c in nb.cells for o in c.get('outputs',[]))

for destination in (base,asset):
    for tag in tags+['two_year_summary']+(['asset_vol_comparison'] if destination is asset else []):
        source=tagged(nb,tag)
        if tag=='two_year_summary' and destination is asset:
            target=next(c for c in destination.cells if c.cell_type=='code' and "print('Fold OOS scoring starts:'" in c.source)
        else:
            target=tagged(destination,tag)
        assert target.source==source.source
        target.outputs=deepcopy(source.outputs)
        target.execution_count=source.execution_count
        target.metadata['execution']=deepcopy(source.metadata.get('execution',{}))
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    destination.cells.append(nbformat.v4.new_markdown_cell(
        '## Execution provenance\n\n'
        f'On **{stamp}**, all window-dependent Ridge/Lasso penalty sweeps, frozen-stream '
        'comparisons, numerical audits, coefficient/calibration plots and the batch forecast export '
        'were freshly executed using all 640,734 research rows with rolling two-calendar-year '
        'training and two-calendar-year forecast windows. The asset-volatility comparison was also refreshed. '
        'Unchanged EDA, feature-family and alpha-forecast outputs are retained from their prior executions; '
        'this is not a new full Run All of unrelated feature experiments. The separate forecast runner '
        'generates the final two-year withheld predictions without using blackout labels. '
        'The h=1-only oracle and display-only first-HL coefficient exclusion are unchanged.',
        metadata={'tags':['execution_summary']}))
    nbformat.validate(destination)
nbformat.write(base,base_path);nbformat.write(asset,asset_path)

# This smaller notebook is re-executed completely under the same two-year folds.
p=root/'notebooks/02_normalization.ipynb'
normal=nbformat.read(p,4)
NotebookClient(normal,timeout=1800,kernel_name='python3',on_cell_start=progress,
    resources={'metadata':{'path':str(root)}}).execute()
nbformat.write(normal,p)
for p in (base_path,asset_path,root/'notebooks/02_normalization.ipynb'):
    result=nbformat.read(p,4)
    for cell in result.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is not None, (p,cell.source[:60])
            assert not any(o.output_type=='error' for o in cell.outputs)
    print('VALIDATED_EXECUTION',p,flush=True)

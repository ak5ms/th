"""Run actual model cells before the separate complete notebook execution."""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import nbformat
from nbclient import NotebookClient

root = Path.cwd()
original = nbformat.read(root / 'notebooks/takehome.ipynb', 4)
tags = ['submission_config', 'research_data', 'regression_setup', 'batch_oos_sweep',
        'ridge_sizing_diagnostic', 'all_predictor_fit', 'implementation_audit',
        'lagged_regression_predictions', 'coefficient_paths', 'regression_calibration_plot',
        'selected_ridge_backtest', 'directional_bias', 'two_year_summary',
        'submission_design', 'submission_ridge_fit', 'submission_export']
cells = []
for tag in tags:
    cells.append(deepcopy(next(c for c in original.cells if tag in c.metadata.get('tags', []))))
    if tag == 'submission_config':
        cells.append(nbformat.v4.new_code_cell('from takehome.features import sharpe'))
cells.append(nbformat.v4.new_code_cell('''import json
report = dict(
    bias=bias_metrics.reset_index().to_dict(orient='records'),
    batch_metrics=batch_metrics.reset_index().to_dict(orient='records'),
    choices={name: chosen.choices_ for name, chosen in selections.items()},
    streaming={name: result['audit'].to_dict(orient='records') for name, result in stream_results.items()},
    holdout=dict(alpha=submission_alpha, rows=len(submission_forecast),
                 positive_fraction=float(submission_forecast.forecast.gt(0).mean()),
                 mean=float(submission_forecast.forecast.mean()),
                 std=float(submission_forecast.forecast.std())))
Path('/tmp/zero-intercept-preflight-metrics.json').write_text(json.dumps(report, indent=2, default=str))
'''))
notebook = nbformat.v4.new_notebook(cells=cells, metadata=deepcopy(original.metadata))
started_at = perf_counter()

def started(cell, cell_index):
    print('PREFLIGHT CELL', cell_index, cell.metadata.get('tags', []),
          'seconds', round(perf_counter()-started_at, 1), flush=True)

def completed(cell, cell_index, execute_reply):
    for output in cell.get('outputs', []):
        if output.output_type == 'stream':
            print(output.text, end='', flush=True)
    nbformat.write(notebook, '/tmp/zero-intercept-preflight.ipynb')

try:
    NotebookClient(notebook, timeout=5400, kernel_name='python3',
                   on_cell_start=started, on_cell_executed=completed,
                   resources={'metadata': {'path': str(root)}}).execute()
finally:
    nbformat.write(notebook, '/tmp/zero-intercept-preflight.ipynb')
assert not any(o.output_type == 'error' for c in notebook.cells for o in c.get('outputs', []))
print('PREFLIGHT PASSED', perf_counter()-started_at, flush=True)

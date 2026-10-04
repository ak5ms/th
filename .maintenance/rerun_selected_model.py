"""Refresh the selected-model dependency chain, keeping unrelated executed research intact."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient

ROOT = Path.cwd()
path = ROOT/'notebooks/takehome.ipynb'
nb = nbformat.read(path, 4)
evidence = Path(os.environ.get('NONNEG_EVIDENCE_DIR', '/tmp/nonneg-ridge-evidence'))
evidence.mkdir(parents=True, exist_ok=True)


def tagged(tag):
    found = [c for c in nb.cells if tag in c.metadata.get('tags', [])]
    assert len(found) == 1, tag
    return found[0]


# Load every research row, but only fields needed by this model/plot dependency chain.
loader = nbformat.v4.new_code_cell('''from takehome.features import with_cashflow_feature, sharpe
labels = read_parquet_window(DATA_PATH, columns=['ret_5m'])
_, split = training_data(labels)
file_last = labels.index[-1]
del labels
needed = [f'x{i}' for i in range(1,100)] + ['cashflow','volume','ret_5m']
df = with_cashflow_feature(read_parquet_window(DATA_PATH, start=split['first_label'],
    stop=split['cutoff'], columns=needed))
x_cols = pd.Index([f'x{i}' for i in range(1,101)])
rows = np.arange(len(df))
assert len(df) == split['train_rows'] == 640734
print('ALL_RESEARCH_ROWS', len(df), 'ALL_MODEL_INPUTS', len(x_cols))
''')
refresh = ['submission_config', 'regression_setup', 'batch_oos_sweep',
           'ridge_sizing_diagnostic', 'lagged_regression_predictions', 'coefficient_paths',
           'regression_calibration_plot', 'oos_model_consistency', 'selected_ridge_backtest',
           'directional_bias', 'two_year_summary', 'submission_design',
           'submission_ridge_fit', 'submission_export']
selected = [deepcopy(tagged(refresh[0])), loader] + [deepcopy(tagged(t)) for t in refresh[1:]]
verify_source = '''from pathlib import Path
import hashlib, json
assert SUBMISSION_FAMILY == 'Ridge nonneg'
assert ridge_selected is selections['Ridge nonneg']
assert all(m.nonneg and not m.fit_intercept for m in submission_candidates.values())
assert (np.asarray(ridge_selected.path_.snapshots_) >= 0).all()
assert (np.asarray(submission_selection.path_.snapshots_) >= 0).all()
assert np.array_equal(submission_selection.path_.offsets_, np.zeros(len(submission_selection.path_.offsets_)))
assert all((np.asarray(p.offsets_) == 0).all() for p in submission_selection.candidates_.values())
assert len(submission_forecast) == 150048
assert {p.name for p in forecast_dir.iterdir()} == {'predictions.parquet'}
assert (submission_selection.selection_.validation_stop < submission_selection.selection_.predict_start).all()
# Retain a concise verification record outside forecasts/ and outside the repository.
report = dict(family=SUBMISSION_FAMILY, alpha=float(submission_alpha),
    research_choices=[None if x is None else float(x) for x in ridge_selected.choices_],
    holdout_rows=len(submission_forecast), holdout_first=str(submission_forecast.index[0]),
    holdout_last=str(submission_forecast.index[-1]), holdout_positive_fraction=float(submission_forecast.forecast.gt(0).mean()),
    holdout_mean=float(submission_forecast.forecast.mean()),
    min_selected_coefficient=float(np.asarray(submission_selection.path_.snapshots_).min()),
    nonzero_coefficients=int(np.count_nonzero(submission_selection.path_.snapshots_[-1])),
    max_abs_intercept=float(np.max(np.abs(submission_selection.path_.offsets_))),
    sha256=hashlib.sha256(forecast_path.read_bytes()).hexdigest(),
    metrics={str(k):float(v) for k,v in selected_stats.items()},
    consistency=consistency_summary.reset_index().to_dict('records'),
    applied_blocks=oos_consistency.to_dict('records'))
print('NONNEGATIVE_RIDGE_RELEASE_CHECK'); print(json.dumps(report, indent=2))
Path(os.environ['NONNEG_EVIDENCE_DIR']).joinpath('verification.json').write_text(json.dumps(report,indent=2)+'\\n')
'''
selected.append(nbformat.v4.new_code_cell(verify_source))
for c in selected:
    c.outputs = []; c.execution_count = None; c.metadata.pop('execution', None)
work = nbformat.v4.new_notebook(cells=selected, metadata=deepcopy(nb.metadata))
started = perf_counter()


def progress(cell, cell_index):
    print('EXECUTING', cell_index, cell.metadata.get('tags', []),
          f'{perf_counter()-started:.1f}s', flush=True)


os.environ['NONNEG_EVIDENCE_DIR'] = str(evidence)
try:
    NotebookClient(work, timeout=1800, kernel_name=os.environ.get('RERUN_KERNEL', 'python3'),
                   on_cell_start=progress, resources={'metadata': {'path': str(ROOT)}}).execute()
except BaseException:
    nbformat.write(work, evidence/'failed.ipynb')
    raise

streams = []
for c in work.cells:
    for output in c.get('outputs', []):
        assert output.output_type != 'error'
        if output.output_type == 'stream':
            streams.append(output.text)
(evidence/'cell-output.txt').write_text('\n'.join(streams))
for tag in refresh:
    src = next(c for c in work.cells if tag in c.metadata.get('tags', []))
    dst = tagged(tag)
    assert src.source == dst.source
    dst.outputs = deepcopy(src.outputs)
    dst.execution_count = src.execution_count
    dst.metadata['execution'] = deepcopy(src.metadata.get('execution', {}))

nb.cells = [c for c in nb.cells if 'execution_summary' not in c.metadata.get('tags', [])]
stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
nb.cells.append(nbformat.v4.new_markdown_cell(
    '## Execution provenance\n\n'
    f'**{stamp}: selected-model refresh from a fresh kernel.** '
    'All 640,734 research rows were reprocessed through the unchanged variance-scaled 200-column design; '
    'all four batch candidate grids and their previous-test choices were recomputed. '
    'The selected nonnegative Ridge sizing, coefficient, calibration, OOS-consistency, backtest and '
    'directional-exposure plots were regenerated, followed by final pre-holdout tuning, refitting and '
    'the single holdout Parquet export. '
    'Unrelated EDA, transformation, AR and streaming-Lasso outputs were retained from the '
    '**2026-10-04 19:24 UTC complete top-to-bottom execution** because their code, inputs and '
    'model settings are unchanged. This refresh is not a claim of a new full Run All of those sections. '
    '`python run_eda.py` reproduces the entire notebook in a fresh kernel.',
    metadata={'tags':['execution_summary']}))
nbformat.validate(nb)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
assert not any(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code' for o in c.outputs)
for tag in ('batch_oos_sweep','ridge_sizing_diagnostic','coefficient_paths',
            'regression_calibration_plot','oos_model_consistency','selected_ridge_backtest','directional_bias'):
    assert any('image/png' in o.get('data', {}) for o in tagged(tag).outputs), tag
nbformat.write(nb, path)
print('REFRESHED', len(refresh), 'cells;', f'{perf_counter()-started:.1f}s', flush=True)
print((evidence/'verification.json').read_text(), flush=True)

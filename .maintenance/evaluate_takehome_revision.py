from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path.cwd()
base_path = ROOT / 'notebooks/takehome.ipynb'
experiment_path = ROOT / 'notebooks/takehome_asset_vol.ipynb'
base = nbformat.read(base_path, as_version=4)
experiment = nbformat.read(experiment_path, as_version=4)


def tagged(nb, tag):
    return next(c for c in nb.cells if tag in c.metadata.get('tags', []))


def code(source, tags=()):
    return nbformat.v4.new_code_cell(source, metadata={'tags': list(tags)})


setup = '''x_cols = df.head(0).filter(like='x').columns
HL, META_HL = 288*21, 252*288
from takehome.features import ts_std, sharpe, evaluate_features
from takehome.plots import display_results
'''
fit = '''models = {
    f'Ridge {RIDGE_ALPHA:g}': BatchRidge(len(fit_columns), alpha=RIDGE_ALPHA, decay=DECAY),
    f'Lasso {DEFAULT_ALPHA:g}': BatchLasso(len(fit_columns), DECAY, DEFAULT_ALPHA,
                                       fit_intercept=True, tol=1e-11),
}
started = perf_counter()
batch_paths = walk_forward_sweep(models, X_fit, y_fit, W=W_fit, **FOLD)
print(f'Fixed batch models: {perf_counter()-started:.2f}s', flush=True)
fitter = StreamingWeightedLasso(len(fit_columns), DECAY, DEFAULT_ALPHA,
    max_iter=20000, tol=1e-10, fit_intercept=True, store_history=True)
started = perf_counter()
default_result = stream_at_folds(fitter, X_fit, y_fit, folds, W=W_fit)
online_yhat = pd.Series(default_result['live'], index=df.index)
print(f'Fixed streaming model: {perf_counter()-started:.2f}s; failed updates={fitter.n_failed_}', flush=True)
assert fitter.n_failed_ == 0
print('Frozen stream/batch prediction RMSE:',np.sqrt(np.nanmean((default_result['frozen']-batch_paths[f'Lasso {DEFAULT_ALPHA:g}'].prediction_)**2)))
del X_fit
'''
load = r'''# Batched loading avoids materializing the full feature file before the split.
# Recompute the same split using only timestamp/target, then verify training rows.
import gc
import pyarrow as pa
import pyarrow.parquet as pq
labels = pd.read_parquet(DATA_PATH, columns=['msgStamp', 'ret_5m']).set_index('msgStamp').sort_index()
training_labels, split = training_data(labels)
first, cutoff = split['first_label'], split['cutoff']
parts = []
for batch in pq.ParquetFile(DATA_PATH).iter_batches(batch_size=16384, use_threads=False):
    frame = batch.to_pandas().set_index('msgStamp')
    keep = (frame.index >= first) & (frame.index < cutoff)
    if keep.any():
        parts.append(frame.loc[keep].copy())
df = pd.concat(parts).sort_index()
assert df.index.equals(training_labels.index)
assert df.ret_5m.equals(training_labels.ret_5m)
assert len(df) == split['train_rows'] == 640734
print(split.to_string())
del labels, training_labels, parts, frame, batch
pa.default_memory_pool().release_unused(); gc.collect()
from takehome.features import with_cashflow_feature
df = with_cashflow_feature(df)
print(f'Training shape: {df.shape}', flush=True)
'''
code_cells = [deepcopy(base.cells[1]), code(load), code(setup),
              deepcopy(tagged(base, 'regression_setup')), code(fit),
              deepcopy(tagged(base, 'lagged_regression_predictions')),
              deepcopy(tagged(base, 'coefficient_paths')),
              deepcopy(tagged(base, 'oracle_lead_0')),
              deepcopy(tagged(base, 'oracle_lead_1')),
              deepcopy(tagged(base, 'oracle_summary')),
              deepcopy(tagged(experiment, 'asset_vol_comparison')),
              code('''from pathlib import Path
out = Path('/tmp/takehome-results'); out.mkdir(exist_ok=True)
for rule, days in sizing_daily.items():
    days.to_pickle(out / ('asset_daily.pkl' if rule == 'Lagged asset volatility' else 'original_daily.pkl'))
sizing_comparison.to_pickle(out / 'comparison.pkl')
asset_vol_events.to_pickle(out / 'events.pkl')
fixed_forecasts.to_pickle(out / 'fixed_forecasts.pkl')
ts_std(df.ret_5m.shift(1), HL).to_pickle(out / 'asset_sigma.pkl')
ts_std(fixed_forecasts, HL).to_pickle(out / 'forecast_sigma.pkl')
print('FITTED_TRAINING_ROWS', len(df)); print('HL_OMITTED', HL)
''')]
nb = nbformat.v4.new_notebook(cells=code_cells, metadata=deepcopy(base.metadata))
for c in nb.cells:
    c.outputs = []; c.execution_count = None; c.metadata.pop('execution', None)


def progress(cell, cell_index):
    print('EXECUTING', cell_index, cell.metadata.get('tags', []), flush=True)


try:
    NotebookClient(nb, timeout=1800, kernel_name='python3', on_cell_start=progress,
                   resources={'metadata': {'path': str(ROOT)}}).execute()
except BaseException:
    nbformat.write(nb, '/tmp/takehome-diagnostic-failed.ipynb')
    raise
nbformat.write(nb, '/tmp/takehome-diagnostic-executed.ipynb')
for c in nb.cells:
    for o in c.get('outputs', []):
        if o.output_type == 'stream':
            print(o.text, end='', flush=True)
assert not any(o.output_type == 'error' for c in nb.cells for o in c.outputs)

changed = ('coefficient_paths', 'oracle_lead_0', 'oracle_lead_1', 'oracle_summary')
for destination in (base, experiment):
    for tag in changed:
        original = tagged(destination, tag)
        refreshed = tagged(nb, tag)
        assert original.source == refreshed.source
        original.outputs = deepcopy(refreshed.outputs)
        original.execution_count = refreshed.execution_count
        original.metadata['execution'] = deepcopy(refreshed.metadata.get('execution', {}))
    destination.cells = [c for c in destination.cells if 'execution_summary' not in c.metadata.get('tags', [])]
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    destination.cells.append(nbformat.v4.new_markdown_cell(
        '## Execution provenance\n\n'
        'Unchanged EDA, feature-family and hyperparameter-sweep outputs are retained from the '
        '**2026-09-30 full execution**. '
        f'On **{stamp}**, the two fixed batch models and fixed streaming model were refitted '
        'with the original settings and all training rows; the truncated coefficient chart and '
        'h=0/h=1 oracle cells and summary were re-executed successfully. '
        'Only these requested diagnostics were refreshed; this is not a claim of a new full Run All. '
        + ('The additional asset-volatility comparison was executed on those same freshly fitted forecasts. '
           if destination is experiment else '')
        + 'No reserved-window observations were used. `python run_eda.py` performs a fresh full '
        'execution of the main notebook; Jupyter Run All also reproduces the comparison copy.',
        metadata={'tags': ['execution_summary']},
    ))

src, dst = tagged(nb, 'asset_vol_comparison'), tagged(experiment, 'asset_vol_comparison')
dst.outputs = deepcopy(src.outputs); dst.execution_count = src.execution_count
dst.metadata['execution'] = deepcopy(src.metadata.get('execution', {}))
nbformat.validate(base); nbformat.validate(experiment)
nbformat.write(base, base_path)
nbformat.write(experiment, experiment_path)
print('SUCCESS: requested cells refreshed in both notebooks; comparison executed.', flush=True)

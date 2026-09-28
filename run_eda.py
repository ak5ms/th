"""Run from the repository root: python run_eda.py."""
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    path = root / 'notebooks/01_eda.ipynb'
    nb = nbformat.read(path, as_version=4)
    started = perf_counter()
    NotebookClient(nb, timeout=1800, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}}).execute()
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    images = sum('image/png' in o.get('data', {}) for c in code for o in c.outputs)
    nbformat.write(nb, path)
    reports = root / 'reports'
    reports.mkdir(exist_ok=True)
    (reports / 'eda.html').write_text(HTMLExporter().from_notebook_node(nb)[0], encoding='utf-8')
    summary = ''.join(o.text for o in code[-1].outputs if o.output_type == 'stream')
    duration = perf_counter() - started
    (root / 'EDA_SUMMARY.md').write_text(
        f'# Training-only EDA\n\nExecuted in {duration:.1f} seconds; {len(code)} code cells; {images} figures.\n\n'
        'Only the first 80% of the labeled time span is analyzed. Earlier notebook versions used the full data; '
        'this is a holdout for subsequent work, not a retrospectively untouched sample.\n\n'
        f'```text\n{summary}\n```\n', encoding='utf-8')
    print(summary)
    print(f'EXECUTED: {len(code)} code cells, {images} PNG figures, {duration:.1f} seconds')

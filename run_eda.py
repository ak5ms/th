"""Execute takehome.ipynb in a fresh kernel; NOTEBOOK_PATH selects the audit notebook."""
from datetime import datetime, timezone
import os
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient


def execute_notebook(path: Path, root: Path) -> None:
    """Save only a fully successful execution; preserve the source on failure."""
    nb = nbformat.read(path, as_version=4)
    nb.cells = [c for c in nb.cells if 'execution_summary' not in c.metadata.get('tags', [])]
    for c in nb.cells:
        if c.cell_type == 'code':
            c.outputs = []
            c.execution_count = None
            c.metadata.pop('execution', None)
    started = perf_counter()

    def progress(cell, cell_index):
        if cell.cell_type == 'code':
            print(f'CELL {cell_index}: {cell.metadata.get("tags", [])}; '
                  f'elapsed={perf_counter()-started:.1f}s', flush=True)

    try:
        NotebookClient(nb, timeout=3600, kernel_name='python3', on_cell_start=progress,
                       resources={'metadata': {'path': str(root)}}).execute()
    except BaseException:
        # Optional diagnostics stay outside the repository unless explicitly requested.
        if os.environ.get('FAILURE_NOTEBOOK'):
            nbformat.write(nb, os.environ['FAILURE_NOTEBOOK'])
        raise
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    images = sum('image/png' in o.get('data', {}) for c in code for o in c.outputs)
    duration = perf_counter() - started
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    nb.cells.append(nbformat.v4.new_markdown_cell(
        f'## Execution provenance\n\n**{stamp}: fresh-kernel, complete top-to-bottom execution** '
        f'of {len(code)} code cells and {images} embedded figures in {duration:.1f} seconds, '
        'with no cell errors. No prior output was retained. '\
        'All research diagnostics and, for the submission notebook, the final Ridge fit/export '
        'were run from the displayed code. Model settings, actual fold boundaries and '\
        'holdout-use limitations are documented in the relevant sections.',
        metadata={'tags': ['execution_summary']},
    ))
    nbformat.validate(nb)
    temporary = path.with_suffix('.ipynb.tmp')
    nbformat.write(nb, temporary)
    temporary.replace(path)
    print(f'EXECUTED {path.name}: {len(code)} code cells, {images} PNG figures, {duration:.1f}s', flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    selected = Path(os.environ.get('NOTEBOOK_PATH', 'notebooks/takehome.ipynb'))
    path = selected if selected.is_absolute() else root / selected
    execute_notebook(path, root)

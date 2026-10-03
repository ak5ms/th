"""Execute and save the notebook in place: python run_eda.py."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    path = root / 'notebooks/takehome.ipynb'
    nb = nbformat.read(path, as_version=4)
    nb.cells = [c for c in nb.cells if 'execution_summary' not in c.metadata.get('tags', [])]
    started = perf_counter()
    NotebookClient(nb, timeout=1800, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}}).execute()
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    images = sum('image/png' in o.get('data', {}) for c in code for o in c.outputs)
    duration = perf_counter() - started
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    nb.cells.append(nbformat.v4.new_markdown_cell(
        f'## Last scripted execution\n\n{stamp}: completed {len(code)} code cells '
        f'and {images} figures in {duration:.1f} seconds, with no cell errors.\n\n'
        'Tables, figures, package versions and interpretation are embedded above. '
        'The only separate diagnostic file is `splits.json` at the repository root.',
        metadata={'tags': ['execution_summary']},
    ))
    nbformat.write(nb, path)
    print(f'EXECUTED: {len(code)} code cells, {images} PNG figures, {duration:.1f} seconds')

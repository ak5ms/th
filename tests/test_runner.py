"""The runner must save notebook outputs without generating external reports."""
from pathlib import Path
import runpy

import nbformat
import pytest
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[1]


def stage(tmp_path, source):
    script = tmp_path / 'run_eda.py'
    script.write_text((ROOT / 'run_eda.py').read_text(), encoding='utf-8')
    path = tmp_path / 'notebooks/takehome.ipynb'
    path.parent.mkdir()
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)]), path)
    return script, path


def test_runner_only_writes_notebook_and_splits(tmp_path):
    script, path = stage(tmp_path, "from pathlib import Path\nPath('splits.json').write_text('{}')\nprint('training-only diagnostic')")
    runpy.run_path(str(script), run_name='__main__')
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*') if p.is_file()} == {
        'run_eda.py', 'notebooks/takehome.ipynb', 'splits.json',
    }
    nb = nbformat.read(path, as_version=4)
    assert nb.cells[0].execution_count == 1
    assert 'training-only diagnostic' in nb.cells[0].outputs[0].text
    assert len([c for c in nb.cells if 'execution_summary' in c.metadata.get('tags', [])]) == 1
    runpy.run_path(str(script), run_name='__main__')
    rerun = nbformat.read(path, as_version=4)
    assert len(rerun.cells) == len(nb.cells)  # Update, do not duplicate the execution note.
    assert not (tmp_path / 'reports').exists()


def test_failed_execution_does_not_overwrite_notebook(tmp_path):
    script, path = stage(tmp_path, "raise ValueError('intentional failure')")
    original = path.read_bytes()
    with pytest.raises(CellExecutionError, match='intentional failure'):
        runpy.run_path(str(script), run_name='__main__')
    assert path.read_bytes() == original
    assert not (tmp_path / 'reports').exists()

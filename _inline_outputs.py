"""One-time migration; removed after the evaluated notebook is committed."""
from pathlib import Path
import nbformat as nbf

path = Path('notebooks/01_eda.ipynb')
nb = nbf.read(path, as_version=4)


def replace_once(old, new):
    matches = [c for c in nb.cells if old in c.source]
    assert len(matches) == 1, f'Expected one notebook occurrence: {old}'
    matches[0].source = matches[0].source.replace(old, new)


replace_once("REPORTS = ROOT / 'reports'\nREPORTS.mkdir(exist_ok=True)\n", '')
replace_once("(REPORTS / 'split.json')", "(ROOT / 'splits.json')")
replace_once("\ncoverage.to_csv(REPORTS / 'column_coverage.csv')", '')
replace_once("\ncashflow_outliers.to_csv(REPORTS / 'cashflow_outliers.csv')", '')
replace_once(
    'display(cashflow_outliers)',
    "with pd.option_context('display.max_rows', None, 'display.max_columns', None):\n    display(cashflow_outliers)",
)
replace_once(
    'The complete named-variable exception table is saved to CSV; pandas may shorten only its display.',
    'The complete named-variable exception table is displayed below, without row truncation. No CSV is exported.',
)
nb.cells[0].source += '\n\nAll diagnostics, full coverage and exception tables, package versions, and interpretation are kept in this notebook. Only the split boundaries are saved separately, in `splits.json` at the repository root.'
summary = next(i for i, c in enumerate(nb.cells) if c.source.startswith('## Training-only summary'))
nb.cells[summary:summary] = [
    nbf.v4.new_markdown_cell('## Execution environment\n\nVersions used for this evaluation are shown here, rather than exported to a separate file. The figures and statistical results above remain training-only.'),
    nbf.v4.new_code_cell('''from importlib.metadata import version
import platform

print(f'Python {platform.python_version()}')
packages = ['numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib', 'pyarrow',
            'diptest', 'nbformat', 'nbclient', 'ipykernel']
display(pd.Series({p: version(p) for p in packages}, name='version').to_frame())'''),
]
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
        cell.metadata = {}
source = '\n'.join(c.source for c in nb.cells)
assert all(x not in source for x in ['REPORTS', 'reports/', '.to_csv(', 'EDA_SUMMARY.md'])
assert "(ROOT / 'splits.json').write_text" in source
nbf.validate(nb)
nbf.write(nb, path)
print('Notebook migrated: all diagnostics inline; only splits.json is exported.')

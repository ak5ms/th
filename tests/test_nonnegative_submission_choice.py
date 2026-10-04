"""The displayed selected model and the sole holdout export must be the same family."""
import ast
from pathlib import Path

import nbformat
import numpy as np

from test_submission_revision import export_namespace, tagged

ROOT = Path(__file__).resolve().parents[1]


def test_research_selected_alias_uses_nonnegative_ridge():
    sentinel = {name: object() for name in ('Ridge', 'Ridge nonneg', 'Lasso')}
    assignments = [stmt for stmt in ast.parse(tagged('batch_oos_sweep')).body
                   if isinstance(stmt, ast.Assign) and any(
                       isinstance(node, ast.Name) and node.id == 'ridge_selected'
                       for target in stmt.targets for node in ast.walk(target))]
    assert len(assignments) == 1
    ns = {'selections': sentinel, 'SUBMISSION_FAMILY': 'Ridge nonneg'}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), '<selection>', 'exec'), ns)
    assert ns['ridge_selected'] is sentinel['Ridge nonneg']
    assert ns['lasso_selected'] is sentinel['Lasso']


def test_actual_final_candidates_have_nonnegative_coefficients_and_zero_intercepts(tmp_path):
    ns, _, _ = export_namespace(tmp_path)
    ns['SUBMISSION_FAMILY'] = 'Ridge nonneg'
    for tag in ('submission_design', 'submission_ridge_fit'):
        exec(tagged(tag), ns)
    assert all(model.nonneg for model in ns['submission_candidates'].values())
    assert all(not model.fit_intercept for model in ns['submission_candidates'].values())
    for path in ns['submission_selection'].candidates_.values():
        assert (np.asarray(path.snapshots_) >= 0).all()
        assert (np.asarray(path.offsets_) == 0).all()
    assert ns['submission_alpha'] in ns['RIDGE_ALPHAS']
    assert ns['submission_selection'].choices_[0] is None


def test_rationale_distinguishes_sign_prior_consistency_and_compute_budget():
    rationale = tagged('selection_rationale')
    for phrase in ('nonnegative Ridge', 'sign', 'OOS', 'five-minute', 'latency', 'Lasso'):
        assert phrase in rationale
    nb = nbformat.read(ROOT/'notebooks/takehome.ipynb', 4)
    assert any('oos_model_consistency' in c.metadata.get('tags', []) for c in nb.cells)

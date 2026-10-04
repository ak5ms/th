"""Select the existing nonnegative Ridge family; keep preprocessing and tuning intact."""
from pathlib import Path
import nbformat

root = Path.cwd()
p = root/'notebooks/takehome.ipynb'
nb = nbformat.read(p, 4)

def cell(tag):
    matches = [c for c in nb.cells if tag in c.metadata.get('tags', [])]
    assert len(matches) == 1, tag
    return matches[0]

def replace(tag, old, new):
    c = cell(tag)
    assert old in c.source, (tag, old)
    c.source = c.source.replace(old, new)
    if c.cell_type == 'code':
        c.outputs = []; c.execution_count = None; c.metadata.pop('execution', None)

replace('submission_config', "MODEL_FAMILIES = ('Ridge', 'Ridge nonneg', 'Lasso', 'Lasso nonneg')",
        "MODEL_FAMILIES = ('Ridge', 'Ridge nonneg', 'Lasso', 'Lasso nonneg')\nSUBMISSION_FAMILY = 'Ridge nonneg'  # Research plots and final candidate constraint share this choice.")
nb.cells[0].source += '\n**Submission choice:** nonnegative Ridge (`beta >= 0`, zero intercept), using its own previous-test-selected regularization schedule. Unconstrained Ridge and both Lasso families remain comparison models.\n'
replace('batch_oos_sweep', "ridge_selected, lasso_selected = selections['Ridge'], selections['Lasso']",
        "ridge_selected, lasso_selected = selections[SUBMISSION_FAMILY], selections['Lasso']")
replace('ridge_sizing_diagnostic', 'Same prior-test-selected Ridge forecasts: bounded sizing alternatives',
        'Same prior-test-selected nonnegative Ridge forecasts: bounded sizing alternatives')
replace('ridge_sizing_diagnostic', "print('ADAPTIVE_RIDGE_SIZING')", "print('ADAPTIVE_NONNEGATIVE_RIDGE_SIZING')")
replace('coefficient_paths', 'Prior-test-selected Ridge: all 200 coefficient paths; first HL prediction rows omitted',
        'Prior-test-selected nonnegative Ridge: all 200 coefficient paths; first HL prediction rows omitted')
replace('regression_calibration_plot', 'Combined raw + dszl Ridge: prior-test-selected next-fold calibration',
        'Variance-scaled raw + dszl, nonnegative Ridge: next-fold OOS calibration')
replace('selected_ridge_backtest', 'Raw + dszl Ridge: prior-test-selected alpha, next-fold OOS backtest',
        'Nonnegative Ridge: variance-scaled raw + dszl, next-fold OOS backtest')
replace('selected_ridge_backtest', "name='research_validation'", "name='nonnegative_ridge_research_validation'")
replace('selected_ridge_backtest', "print('ADAPTIVE_RIDGE_VALIDATION')", "print('ADAPTIVE_NONNEGATIVE_RIDGE_VALIDATION')")
replace('submission_ridge_fit',
        "submission_candidates = {float(a): BatchRidge(len(fit_columns), alpha=float(a), **RIDGE_CONFIG)",
        "assert SUBMISSION_FAMILY == 'Ridge nonneg'\nsubmission_candidates = {float(a): BatchRidge(len(fit_columns), alpha=float(a),\n                         nonneg=SUBMISSION_FAMILY.endswith('nonneg'), **RIDGE_CONFIG)")
replace('submission_ridge_fit', "print('HOLDOUT_SELECTED_ALPHA', submission_alpha)",
        "print('HOLDOUT_MODEL_FAMILY', SUBMISSION_FAMILY)\nprint('HOLDOUT_SELECTED_ALPHA', submission_alpha)")
replace('submission_ridge_fit', "print('HOLDOUT_ZERO_INTERCEPT_CHECK: all candidate and selected intercepts are zero')",
        "assert all(model.nonneg for model in submission_candidates.values())\nassert (np.asarray(submission_selection.path_.snapshots_) >= 0).all()\nassert all((np.asarray(path.snapshots_) >= 0).all()\n           for path in submission_selection.candidates_.values())\nprint('HOLDOUT_CONSTRAINT_CHECK: all candidate and selected coefficients are nonnegative; intercepts are zero')")

for c in nb.cells:
    if c.cell_type == 'markdown' and c.source.startswith('## Selected Ridge coefficient path'):
        c.source = c.source.replace('Selected Ridge', 'Selected nonnegative Ridge')
    if c.cell_type == 'markdown' and c.source.startswith('### Why can Ridge backtests'):
        c.source += '\nThe sizing alternatives below now use the selected nonnegative Ridge path, matching the final submission family.\n'

cell('selection_rationale').source = r'''## Final choice: nonnegative Ridge and its non-holdout backtest

We choose **nonnegative Ridge** for the final submission: coefficient constraints $\beta_j\geq0$, **no intercept**, and the same `F = [X, dszl(X, 10)]` followed by `F / EWMVar(F)` on all 200 columns. The fitting half-life remains 6,048 observations. Training and frozen forecast windows remain two calendar years. The regularization parameter is selected by **preceding-test raw-return MSE**, never from the block in which the chosen model is evaluated.

**Sign discipline.** Our alpha research screens candidates for the correct predictive sign first. Nonnegative weights preserve that intended direction in the combination rather than allowing the joint fit to reverse a screened alpha to exploit sample-specific correlations. This is the upstream sign-screening convention for the supplied alphas, not a new full-sample sign flip in this notebook. Such screening must use information available before the evaluated period. Nonnegative coefficients do not force forecasts or positions to be long: the features themselves are signed.

**OOS consistency.** We prefer this family for the most consistent OOS performance in the displayed research comparison, rather than for one favorable episode such as COVID. The block-by-block table below makes the basis of that judgment explicit: it reports gross P&L, prediction MSE, inactive-signal coverage and unannualized daily mean/std for every applied research forecast block and all four families. The last block is partial. The family choice reflects inspection of this research evidence; it is not a claim of an additional untouched model-family test. Only the penalty is selected mechanically using each preceding test block.

**Feature-computation budget.** At our **five-minute** decision frequency, computing the full feature set is not a binding constraint. Ridge lets us retain complementary raw and time-of-day representations and regularizes the correlated design without requiring sparsity. With a strict feature-computation or **latency** budget, **Lasso** would be useful: its zero coefficients can support pruning unused features from inference. Computational savings would require actually omitting those feature computations, not merely multiplying already-computed features by zero.

The selected curve, coefficient path and raw-forecast calibration now refer to this same nonnegative Ridge procedure. The holdout fit below reruns the same nonnegative candidate grid on the latest completed two-year validation interval, refits its winner, and freezes it for the true holdout. It does not reuse the unconstrained Ridge winner. Signal sizing remains a separate diagnostic policy and is never applied to the exported raw forecasts. Gross backtests exclude transaction costs; holdout labels are unavailable and are never used in this choice.
'''
replace('submission_export_description', 'First train every Ridge candidate', 'First train every nonnegative Ridge candidate')
replace('submission_export_description', 'The final model remains unconstrained Ridge with **no intercept**.',
        'The final model is **nonnegative Ridge with no intercept**, matching `SUBMISSION_FAMILY` and the selected-model plots.')

consistency = nbformat.v4.new_code_cell('''# Compare the applied blocks only: fold zero is initial tuning, not selected performance.
# Build each normalizer once across the whole selected history; never reset at a fold.
consistency_rows = []
for family, chosen in selections.items():
    forecast = oos_series(chosen.prediction_)
    exposure = signal_weights(forecast, HL, **SIGNAL_RULE).where(rows >= SCORE_FIRST)
    per_bar = exposure * df.ret_5m
    for fold_id, fold in enumerate(folds):
        if chosen.choices_[fold_id] is None:
            continue
        start, stop = max(fold['predict_start'], SCORE_FIRST), fold['predict_stop']
        if start >= stop:
            continue
        eligible = forecast.iloc[start:stop].notna() & np.isfinite(df.ret_5m.iloc[start:stop])
        actual = df.ret_5m.iloc[start:stop][eligible]
        pred = forecast.iloc[start:stop][eligible]
        block = per_bar.iloc[start:stop].where(eligible)
        daily = block.resample('D').sum()
        consistency_rows.append(dict(model=family, fold=fold_id,
            period=f'{df.index[start].date()} to {df.index[stop-1].date()}',
            selected_alpha=chosen.choices_[fold_id], forecast_rows=int(eligible.sum()),
            pnl_observations=int(block.count()), zero_forecast_fraction=float(pred.eq(0).mean()),
            gross_pnl=float(daily.sum()), daily_mean_over_std=float(sharpe(daily)),
            prediction_mse=float(np.mean((pred-actual)**2))))
oos_consistency = pd.DataFrame(consistency_rows)
assert set(oos_consistency['fold']) == set(range(1, len(folds)))
display(oos_consistency.set_index(['model','period']))
consistency_summary = oos_consistency.groupby('model', sort=False).agg(
    applied_blocks=('fold','count'), profitable_blocks=('gross_pnl',lambda x:int(x.gt(0).sum())),
    total_gross_pnl=('gross_pnl','sum'), worst_block_gross_pnl=('gross_pnl','min'),
    min_block_daily_mean_over_std=('daily_mean_over_std','min'))
display(consistency_summary)
print('OOS_MODEL_CONSISTENCY'); print(oos_consistency.to_string(index=False))
print('OOS_CONSISTENCY_SUMMARY'); print(consistency_summary.to_string())
oos_consistency.pivot(index='period', columns='model', values='gross_pnl').plot.bar(
    figsize=(12,4), title='Applied OOS blocks: nonnegative Ridge versus comparison families', rot=15)
plt.axhline(0, linestyle=':', linewidth=1)
plt.ylabel('Cumulative gross diagnostic P&L per block'); plt.xlabel('Forecast block (last partial)')
plt.tight_layout(); plt.show(); plt.close('all')
''', metadata={'tags':['oos_model_consistency']})
where = nb.cells.index(cell('selection_rationale')) + 1
nb.cells[where:where] = [nbformat.v4.new_markdown_cell(
    '### Applied-fold consistency audit\n\nThe tables use the same forecast periods and scoring rule. '
    'Unavailable exposures—including exact-zero Lasso signals before a usable scale exists—'
    'contribute no P&L under the existing daily aggregation; their observation counts and zero-forecast '
    'fractions are shown, not hidden. A flat block is not counted as a profitable block. '
    'Daily mean/std is unannualized and undefined for a fully flat block. '
    'The table is descriptive and is not used to retune any penalty.'), consistency]
nbformat.validate(nb); nbformat.write(nb,p)

p = root/'README.md'
s = p.read_text().replace('train unconstrained Ridge candidates', 'train nonnegative Ridge candidates')
s = s.replace("The four-family comparisons do not silently change the submission's established Ridge family.",
              "The final submission explicitly selects the nonnegative Ridge family; the research selected-model plots and final candidates share `SUBMISSION_FAMILY = 'Ridge nonneg'`.")
s += '''\n## Why nonnegative Ridge is the submission family\n\nWe screen alphas for their intended predictive sign in upstream research, so nonnegative coefficients preserve that sign discipline. The notebook does not introduce a full-sample sign flip. Its applied-fold consistency table and selected-model plots document the research OOS comparison with unconstrained Ridge and both Lasso variants. This is an explicit research-based model-family choice, not a new untouched model-family test; each family still chooses regularization only on its preceding test block.\n\nAt the five-minute decision frequency the full feature-computation budget is not binding, so a distributed, regularized combination is preferred to sparsity for its own sake. Lasso remains useful under strict latency or feature-count limits when unused feature computations can actually be pruned. Nonnegative coefficients do not imply nonnegative predictions because the input signals are signed. The export uses nonnegative Ridge, zero intercept, the existing variance-scaled 200-column design, and independently rerun pre-holdout tuning. Only `forecasts/predictions.parquet` is written.\n'''
p.write_text(s)

p = root/'tests/test_submission_revision.py'
s = p.read_text().replace('SELECTION_EMBARGO=1,display=lambda *a:None)',
                         "SELECTION_EMBARGO=1,SUBMISSION_FAMILY='Ridge nonneg',display=lambda *a:None)")
s = s.replace("alpha=alpha,**ns['RIDGE_CONFIG']", "alpha=alpha,nonneg=True,**ns['RIDGE_CONFIG']")
s = s.replace("alpha=best,**ns['RIDGE_CONFIG']", "alpha=best,nonneg=True,**ns['RIDGE_CONFIG']")
p.write_text(s)
print('Updated selected family, candidate constraints, rationale and selected-model plots.')

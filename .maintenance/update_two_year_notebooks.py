from pathlib import Path
import nbformat

root = Path.cwd()
for filename in ('takehome.ipynb', 'takehome_asset_vol.ipynb'):
    path=root/'notebooks'/filename
    nb=nbformat.read(path,4)
    nb.cells=[c for c in nb.cells if 'execution_summary' not in c.metadata.get('tags',[])]
    refresh={'regression_setup','batch_oos_sweep','ridge_sizing_diagnostic','all_predictor_fit',
             'implementation_audit','lagged_regression_predictions','coefficient_paths',
             'regression_calibration_plot','asset_vol_comparison'}
    for c in nb.cells:
        tags=set(c.metadata.get('tags',[]))
        if 'regression_setup' in tags:
            c.source=c.source.replace('walk_forward_folds, stream_at_folds', 'calendar_walk_forward_folds, stream_at_folds')
            c.source=c.source.replace('FOLD = dict(min_train_size=252*288, step=21*288, train_size=None, gap=0)\nfolds = walk_forward_folds(len(df), **FOLD)',
                'folds = calendar_walk_forward_folds(df.index, train_years=2, test_years=2)\nFOLD = dict(folds=folds)')
            c.source=c.source.replace("print(f'WF_PROTOCOL: {len(folds)} folds; {len(df)} training rows; {len(fit_columns)} predictors')",
                "print(f'WF_PROTOCOL: rolling 2-calendar-year train / 2-calendar-year forecast; {len(folds)} folds; {len(df)} training rows; {len(fit_columns)} predictors')")
        if c.cell_type=='markdown' and 'Use an expanding training window, first fit after' in c.source:
            start=c.source.index('Use an expanding training window, first fit after')
            stop=c.source.index('\n\nEvery sweep point',start)
            c.source=c.source[:start]+'''Use a **rolling two-calendar-year training window**, then hold coefficients fixed over the **next two calendar years**, and advance by two years. `calendar_walk_forward_folds` uses timestamps and `DateOffset(years=2)`, not a fixed bars-per-year approximation. The final forecast window is clipped to the available research data. The fold table retains both exact nominal calendar boundaries and actual row bounds.

Both batch models and the frozen streaming comparison use only that fold's preceding two years, with the existing decay $d=2^{-1/6048}$ inside the window. The continuous `live` streaming series remains an explicitly separate online EWM baseline; it is not a frozen walk-forward fit. The current prediction row's target is excluded. The prior row's label is assumed available, as in the existing beta.shift convention; add an embargo if target maturity requires it.''' +c.source[stop:]
        if c.cell_type=='markdown' and c.source.startswith('## Streaming lasso:'):
            c.source='''## Streaming lasso: continuous live baseline versus two-year frozen fits

For each alpha, the `live` estimator predicts before every row update in one continuous replay. It is an online EWM baseline, not a rolling/frozen estimator. For the `frozen` comparison, a fresh streaming estimator replays **only the same two-year training interval** used by the batched model, then holds its coefficients fixed throughout the next two-year prediction interval. Observations before that training interval cannot enter the frozen fit.

Compare batch with these window-matched frozen fits to audit numerical agreement. Comparing either with `live` also changes the update schedule and, after the training cutoff, available label history. The main coefficient history remains the continuous live baseline. It uses fixed alpha=1e-5, tolerance 1e-10 and 20,000 sweeps; both live and frozen fitting failures are reported.
'''
        if 'all_predictor_fit' in tags:
            c.source=c.source.replace("failed_updates=m.n_failed_, final_KKT=m.kkt_violation_, seconds=seconds,", 
                "failed_updates=m.n_failed_, frozen_failed_updates=int(np.sum(result.get('frozen_failed_updates', []))),\n        final_KKT=m.kkt_violation_, seconds=seconds,")
        if c.cell_type=='markdown' and c.source.startswith('### Investigating a performance gap'):
            c.source=c.source.replace('identical full-history cutoffs','identical two-year training windows').replace('full-history','two-year window').replace('full-prefix','two-year window')
        if 'implementation_audit' in tags:
            c.source=c.source.replace("    stop = folds[i]['train_stop']\n    moments = batch_moments(X_fit[:stop], y_fit[:stop], W_fit[:stop], DECAY)",
                "    start, stop = folds[i]['train_start'], folds[i]['train_stop']\n    moments = batch_moments(X_fit[start:stop], y_fit[start:stop], W_fit[start:stop], DECAY)")
            c.source=c.source.replace("dict(fold=i, training_stop=stop,", "dict(fold=i, training_start=start, training_stop=stop,")
            c.source=c.source.replace('FULL_PREFIX_AUDIT','TWO_YEAR_WINDOW_AUDIT')
            c.source=c.source.replace("stop=folds[worst]['train_stop']; a,b=folds[worst]['predict_start'],folds[worst]['predict_stop']\nstrict_moments=batch_moments(X_fit[:stop],y_fit[:stop],W_fit[:stop],DECAY)",
                "start,stop=folds[worst]['train_start'],folds[worst]['train_stop']\na,b=folds[worst]['predict_start'],folds[worst]['predict_stop']\nstrict_moments=batch_moments(X_fit[start:stop],y_fit[start:stop],W_fit[start:stop],DECAY)")
        if tags & refresh:
            c.outputs=[];c.execution_count=None;c.metadata.pop('execution',None)
        if c.cell_type=='code' and "print('Fold OOS scoring starts:'" in c.source:
            c.outputs=[];c.execution_count=None;c.metadata.pop('execution',None)
        if c.cell_type=='markdown':
            c.source=c.source.replace('Only the split boundaries are saved separately, in `splits.json` at the repository root.',
                'Split boundaries are saved in `splits.json`. The fixed batched-Lasso walk-forward forecasts are also exported under `forecasts/` in raw return units. The separate `run_forecast.py` writes predictions for the assignment\'s terminal two-year label blackout.')
    position=next(i for i,c in enumerate(nb.cells) if 'batch_oos_sweep' in c.metadata.get('tags',[]))+1
    nb.cells[position:position]=[
        nbformat.v4.new_markdown_cell('''## Export the batched walk-forward Lasso forecast

Export the **fixed alpha=1e-5** batch model, not the best point chosen after inspecting the sweep. `forecast` is the predicted `ret_5m` in its original units: no variance sizing, asset volatility, oracle, or extra shift is applied. Each row uses the coefficients and intercept from its preceding two-year training window, held fixed for its two-year forecast window. Initial pre-fit rows are excluded; all timestamps with a forecast are retained even when that row's realized target is missing.

These files contain walk-forward validation **inside the research training split**. `python run_forecast.py` separately writes `batched_lasso_oos.csv`/`.parquet` for the assignment's final two-year blackout, using a final two-year fit at that blackout's boundary. That final fit uses available labels from the research-reserved interval, but does not score them or choose a new penalty. No blackout label or asset-return history is required to generate forecasts.
'''),
        nbformat.v4.new_code_cell('''batch_lasso_forecast = pd.Series(
    batch_paths[f'Lasso {DEFAULT_ALPHA:g}'].prediction_, index=df.index, name='forecast'
).dropna().to_frame()
forecast_dir = Path(os.environ.get('FORECAST_OUTPUT_DIR', ROOT / 'forecasts'))
forecast_dir.mkdir(parents=True, exist_ok=True)
batch_lasso_forecast.to_csv(forecast_dir / 'batched_lasso_walk_forward.csv',
                           index_label='msgStamp', float_format='%.17g')
batch_lasso_forecast.to_parquet(forecast_dir / 'batched_lasso_walk_forward.parquet')
pd.DataFrame(batch_paths[f'Lasso {DEFAULT_ALPHA:g}'].folds_).to_csv(
    forecast_dir / 'batched_lasso_research_folds.csv', index=False)
display(batch_lasso_forecast)
print(f'Exported {len(batch_lasso_forecast):,} raw-return forecasts; alpha={DEFAULT_ALPHA:g}')
print('First forecast:', batch_lasso_forecast.index[0], '; last:', batch_lasso_forecast.index[-1])
print('Forecast directory:', forecast_dir.resolve())
assert batch_lasso_forecast.index.max() < split['cutoff']
''',metadata={'tags':['batch_forecast_export']})]
    nbformat.write(nb,path)

path=root/'notebooks/02_normalization.ipynb'
nb=nbformat.read(path,4)
nb.cells=[c for c in nb.cells if 'execution_summary' not in c.metadata.get('tags',[])]
for c in nb.cells:
    c.source=c.source.replace('from takehome.fitters import BatchRidge, BatchLasso, walk_forward_sweep',
        'from takehome.fitters import BatchRidge, BatchLasso, walk_forward_sweep, calendar_walk_forward_folds')
    c.source=c.source.replace('FOLD=dict(min_train_size=252*288,step=HL,train_size=None,gap=0)',
        'folds=calendar_walk_forward_folds(df.index,train_years=2,test_years=2)\nFOLD=dict(folds=folds)')
    c.source=c.source.replace("freeze_row=int(df.index.searchsorted(freeze_date))", 
        "freeze_row=int(df.index.searchsorted(freeze_date))\nfreeze_train_start=int(df.index.searchsorted(freeze_date-pd.DateOffset(years=2)))")
    c.source=c.source.replace('fit(X[:freeze_row],y[:freeze_row],W=W[:freeze_row])',
        'fit(X[freeze_train_start:freeze_row],y[freeze_train_start:freeze_row],W=W[freeze_train_start:freeze_row])')
    c.source=c.source.replace("SCORE_FIRST=FOLD['min_train_size']+HL", "SCORE_FIRST=folds[0]['predict_start']+HL")
    if c.cell_type=='code':c.outputs=[];c.execution_count=None;c.metadata.pop('execution',None)
nb.cells[0].source+='\n\nAll frozen Ridge/Lasso comparisons now use **two calendar years of rolling training followed by two calendar years of forecasts**. The simulated two-year label blackouts also fit only the preceding two years. The continuous signal-only normalization estimators keep their existing half-lives.\n'
nbformat.write(nb,path)

p=root/'README.md'
s=p.read_text()
s=s.replace('No CSV, HTML, Markdown report, or environment file is exported; there is no `reports/` folder. The only separate diagnostic file is `splits.json` at the repository root.',
    'The notebook additionally exports timestamped batched-Lasso validation forecasts under `forecasts/`. `python run_forecast.py` also generates the separate final withheld-period predictions. No standalone EDA report is generated.')
s=s.replace('no extra reports are exported.', 'forecast exports are described below.')
s+='''\n## Two-year walk-forward forecasts\n\nAll three notebooks use rolling **two-calendar-year training / two-calendar-year prediction** windows, stepped by two calendar years; the last research block may be partial. `calendar_walk_forward_folds` resolves dates to row bounds, and `walk_forward_sweep(..., folds=folds)` shares the same windows across Ridge/Lasso penalties. Frozen streaming fits replay only the matching two-year training window; the live streaming series is a separately labeled continuous online baseline. Existing EWM half-lives are unchanged.\n\nRun `python run_forecast.py` to write `forecasts/batched_lasso_walk_forward.csv`/`.parquet` for validation inside the original research split, and `forecasts/batched_lasso_oos.csv`/`.parquet` for the assignment's terminal two-year blackout. CSV columns are `msgStamp,forecast`; Parquet retains the timezone-aware index. The fixed model is BatchLasso with alpha=1e-5, intercept, unit weights on observed labels, the existing 6,048-row decay half-life and all 100 raw predictors including x100=cashflow/volume. Forecasts are raw ret_5m predictions: no signal/asset-volatility sizing, oracle or extra shift. Nonfinite predictors retain the original explicit zero-imputation; targets are never filled.\n\nThe final blackout fit uses the two years immediately before the blackout, including available labels from the research-reserved block, without evaluating those labels or selecting a new alpha. It is frozen for the complete two-year blackout and asserts that all blackout targets are missing. Internal missing labels are not mistaken for the blackout. The original research-only EDA boundary remains unchanged. `batched_lasso_fold_windows.csv` and `batched_lasso_metadata.json` record exact boundaries and coverage. `DATA_PATH` and `FORECAST_OUTPUT_DIR` may override the input and output locations.\n'''
p.write_text(s)
p=root/'run_eda.py'
s=p.read_text().replace("'The only separate diagnostic file is `splits.json` at the repository root.'",
    "'Split boundaries are in `splits.json`; batch forecast exports are under `forecasts/`.'")
p.write_text(s)

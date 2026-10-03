"""Apply the requested notebook edits without rerunning unchanged diagnostics."""
from copy import deepcopy
from pathlib import Path
import nbformat

ROOT = Path.cwd()
path = ROOT / 'src/takehome/features.py'
s = path.read_text()
a = s.index('def standalone_pnl(')
b = s.index('\n\ndef _time_index', a)
s = s[:a] + '''def _pnl_with_asset_sigma(X, returns, hl, asset_sigma=None):
    feature_sigma = ts_std(X, hl)
    denominator = (feature_sigma.pow(2) if asset_sigma is None else
                   feature_sigma.mul(asset_sigma, axis=0))
    return X.div(denominator.replace(0, np.nan)).mul(returns, axis=0)


def standalone_pnl(X: pd.DataFrame, returns: pd.Series, hl: int = 288 * 21, *,
                   asset_vol: bool = False):
    """Default: signal / feature_variance * return, unchanged.

    Experimental asset_vol=True uses signal / (feature_std * lagged_return_std).
    Both scales use ts_std's existing missing/zero and warm-up conventions; the
    asset scale sees returns.shift(1), never the current realized return. This
    diagnostic requires observed return history and is not a blackout solution.
    No variance floor, leverage cap or filling is added.
    """
    if not X.index.equals(returns.index):
        raise ValueError('Signal and return indexes must match.')
    asset_sigma = ts_std(returns.shift(1), hl) if asset_vol else None
    return _pnl_with_asset_sigma(X, returns, hl, asset_sigma)
''' + s[b:]
s = s.replace('meta=False, meta_hl=252*288, positive_only=False):',
              'meta=False, meta_hl=252*288, positive_only=False, asset_vol=False):', 1)
s = s.replace('    daily, summaries = [], []\n',
              '    daily, summaries = [], []\n    # Compute the common asset scale once, not once per feature block.\n    asset_sigma = ts_std(returns.shift(1), hl) if asset_vol else None\n', 1)
s = s.replace('        pnl = standalone_pnl(X, returns, hl)\n',
              '        pnl = _pnl_with_asset_sigma(X, returns, hl, asset_sigma)\n', 1)
s = s.replace('def backtest(signal, returns, hl=288*21):\n    """signal / EWMstd(signal)^2 * return; forecasts must already be aligned."""',
              'def backtest(signal, returns, hl=288*21, *, asset_vol=False):\n    """Standalone sizing, with optional lagged asset-vol diagnostic; align forecasts first."""')
s = s.replace('return standalone_pnl(signal, returns, hl).replace',
              'return standalone_pnl(signal, returns, hl, asset_vol=asset_vol).replace', 1)
path.write_text(s)

old_path = ROOT / 'notebooks/01_eda.ipynb'
nb = nbformat.read(old_path, as_version=4)
remove_tags = {f'oracle_lead_{h}' for h in (2, 6, 12, 24)}
remove_headings = {f'## Predictor lead h={h} — LOOK-AHEAD' for h in (2, 6, 12, 24)}
nb.cells = [c for c in nb.cells
            if not (remove_tags & set(c.metadata.get('tags', [])))
            and c.source.strip() not in remove_headings
            and 'execution_summary' not in c.metadata.get('tags', [])]
for c in nb.cells:
    tags = c.metadata.get('tags', [])
    changed = False
    if 'oracle_lead_0' in tags or 'oracle_lead_1' in tags:
        c.source = c.source.replace('len(df)-24', 'len(df)-1')
        changed = True
    if 'oracle_summary' in tags:
        c.source = c.source.replace('for h in [1,2,6,12,24]:', 'for h in [1]:')
        changed = True
    if c.cell_type == 'markdown' and c.source.startswith('# Optimistic future-predictor diagnostic'):
        c.source = '''# Optimistic future-predictor diagnostic — h=1 only

Use `df[x_cols].shift(-1)` as the only look-ahead experiment. Keep **h=0 as a causal-input control**, not as another oracle horizon. This asks whether knowing the next predictor value could help; it is deliberately noncausal and is not an achievable backtest or a guarantee of a forecastable upper bound.

Shifts are applied only after the training split. Both control and oracle exclude the **last one target row** for a common endpoint, rather than discarding 24 rows. Leads count dataframe rows, not elapsed minutes across gaps. Feature missingness is reported separately. Both use the existing `evaluate_features` sizing and lagged EWM-Sharpe combination; no reserved-window targets enter the experiment.
'''
    if c.cell_type == 'markdown' and c.source.startswith('## Coefficient paths and lagged OOS calibration'):
        c.source += '\n\nThe coefficient chart omits the first `HL` input rows (`HL = 6048`). This is display-only: the full coefficient/intercept histories, their one-row lag, forecasts and fits are unchanged.\n'
    if c.cell_type == 'code' and 'ax.plot(clock,' in c.source:
        c.source = c.source.replace("    clock = beta.index.as_unit('ns').asi8 / 86_400_000_000_000\n    ax.plot(clock, beta.to_numpy(),",
                                    "    beta_plot = beta.iloc[HL:]  # Display warm-up only; retain the full history.\n    clock = beta_plot.index.as_unit('ns').asi8 / 86_400_000_000_000\n    ax.plot(clock, beta_plot.to_numpy(),")
        c.source = c.source.replace('all 100 post-update coefficient paths', 'all 100 post-update coefficient paths (first HL rows omitted)')
        c.metadata['tags'] = list(dict.fromkeys([*tags, 'coefficient_paths']))
        changed = True
    if changed:
        c.outputs = []; c.execution_count = None; c.metadata.pop('execution', None)

nb.cells[0].source = nb.cells[0].source.replace('# ESc1 signal EDA — training data only', '# ESc1 take-home — training data only')
nbformat.write(nb, ROOT / 'notebooks/takehome.ipynb')
old_path.unlink()

for path in [ROOT / 'run_eda.py', ROOT / 'README.md', ROOT / 'tests/test_regression_notebook.py', ROOT / 'tests/test_runner.py']:
    path.write_text(path.read_text().replace('notebooks/01_eda.ipynb', 'notebooks/takehome.ipynb'))
path = ROOT / 'notebooks/02_normalization.ipynb'
other = nbformat.read(path, as_version=4)
for c in other.cells:
    c.source = c.source.replace('01_eda.ipynb', 'takehome.ipynb')
nbformat.write(other, path)

experiment = deepcopy(nb)
experiment.cells[0].source = '''# ESc1 take-home — isolated asset-volatility comparison

This is a copy of `takehome.ipynb`. **The original EDA, feature-family diagnostics, fits, sweeps and baseline P&Ls are retained unchanged.** The additional section immediately after the fixed-penalty OOS comparison evaluates the requested alternative on those same four forecast series:

```python
X.div(
    ts_std(X, HL).mul(ts_std(returns.shift(1), HL), axis=0).replace(0, np.nan)
).mul(returns, axis=0)
```

`axis=0` on both multiplications is intentional: asset volatility and returns align with rows, not feature-column labels. Only this added diagnostic opts into `asset_vol=True`; it does not change module defaults or later cells. Both rules are scored on the same finite observations for each model. No forecasts are refitted or capped, and no return is filled.

This needs historical asset returns. It is a labeled-training diagnostic, **not a solution for the assignment's OOS price/return blackout**, and is deliberately excluded from the main notebook's scoring rule.

The main notebook's h=1-only oracle (plus h=0 control), display-only coefficient warm-up exclusion, and training/test split are preserved here.
'''
intro = nbformat.v4.new_markdown_cell('''## Asset-volatility experiment: same forecasts, different sizing

The original rule divides by **feature variance**; the experiment divides by **feature standard deviation times lagged asset-return standard deviation**. The next two panels show unrescaled cumulative P&L for each rule. They have different units/scales: a smaller numerical curve alone is not evidence that tail concentration improved.

The table reports each rule's largest absolute bar/day and largest absolute day divided by its own daily P&L standard deviation. The latter is an **ex-post diagnostic only**, not a multiplier used in trading. Additional 2018 and 2020 event-window panels report cumulative P&L in the same ex-post daily-volatility units, so a pure rescaling cannot make a spike appear to disappear. Calendar-day sums and the common OOS scoring start follow the main comparison.

No asset volatility enters a model fit, target, oracle shift or hyperparameter selection. The one-row lag applies to return volatility; feature volatility retains the requested current-feature convention.
''', metadata={'tags': ['asset_vol_experiment_description']})
source = '''# Diagnostic-only: preserve the main notebook's default sizing everywhere else.
fixed_forecasts = pd.DataFrame({
    'Streaming lasso, live': oos_series(default_result['live']),
    'Streaming lasso, frozen': oos_series(default_result['frozen']),
    'Batch lasso, frozen': oos_series(batch_paths[f'Lasso {DEFAULT_ALPHA:g}'].prediction_),
    'Ridge, frozen': oos_series(batch_paths[f'Ridge {RIDGE_ALPHA:g}'].prediction_),
})
sizing_pnl = {
    'Feature variance (original)': backtest(fixed_forecasts, df.ret_5m, HL),
    'Lagged asset volatility': backtest(fixed_forecasts, df.ret_5m, HL, asset_vol=True),
}
common_valid = sizing_pnl['Feature variance (original)'].notna() & sizing_pnl['Lagged asset volatility'].notna()
scoring_rows = pd.Series(rows >= SCORE_FIRST, index=df.index)
sizing_daily, sizing_records = {}, []
for rule, per_bar in sizing_pnl.items():
    per_bar = per_bar.where(common_valid).where(scoring_rows, axis=0)
    sizing_pnl[rule] = per_bar
    days = per_bar.resample('D').sum().loc[score_day:]
    sizing_daily[rule] = days
    days.cumsum().plot(figsize=(13, 5), title=f'Fixed penalties: {rule}')
    plt.ylabel('Cumulative diagnostic P&L (unrescaled)'); plt.show(); plt.close('all')
    for model in days:
        series = days[model]
        daily_sigma = series.std()
        sizing_records.append(dict(rule=rule, model=model,
            daily_mean_over_std=sharpe(series), observations=int(per_bar[model].count()),
            max_abs_bar=per_bar[model].abs().max(), max_abs_day=series.abs().max(),
            max_abs_day_over_daily_std=series.abs().max() / daily_sigma,
            largest_abs_day=series.abs().idxmax()))
sizing_comparison = pd.DataFrame(sizing_records).set_index(['model', 'rule'])
display(sizing_comparison)
print('ASSET_VOL_COMPARISON'); print(sizing_comparison.to_string())

# Event panels use each curve's own full-scoring-period daily std for display only.
event_records = []
for year in (2018, 2020):
    for model in ('Streaming lasso, live', 'Batch lasso, frozen'):
        panel = {}
        for rule, days in sizing_daily.items():
            event = days[model].loc[f'{year}-01-01':f'{year}-04-30']
            daily_sigma = days[model].std()
            panel[rule] = event.cumsum() / daily_sigma
            event_records.append(dict(year=year, model=model, rule=rule,
                max_abs_day_over_daily_std=event.abs().max() / daily_sigma,
                largest_abs_day=event.abs().idxmax()))
        pd.DataFrame(panel).plot(figsize=(12, 4), title=f'{year} event window: {model}')
        plt.ylabel('Cumulative P&L / full-span daily P&L std (display only)')
        plt.show(); plt.close('all')
asset_vol_events = pd.DataFrame(event_records).set_index(['year', 'model', 'rule'])
display(asset_vol_events)
print('ASSET_VOL_EVENTS'); print(asset_vol_events.to_string())
assert df.index.max() < split['cutoff']
'''
cell = nbformat.v4.new_code_cell(source, metadata={'tags': ['asset_vol_comparison']})
where = next(i for i, c in enumerate(experiment.cells) if 'all_predictor_backtests' in c.metadata.get('tags', []))
experiment.cells[where+1:where+1] = [intro, cell]
nbformat.write(experiment, ROOT / 'notebooks/takehome_asset_vol.ipynb')

readme = ROOT / 'README.md'
s = readme.read_text()
s += '''\n## Isolated asset-volatility diagnostic\n\n`notebooks/takehome_asset_vol.ipynb` is a copy of the main notebook with an additional comparison immediately after the fixed-penalty OOS model chart. It evaluates `X / (ts_std(X, hl) * ts_std(returns.shift(1), hl)) * returns` on the same fitted forecasts, with row alignment and matched scoring observations. Original feature analyses and all fitting/sweep logic are unchanged. The helper opt-in is `backtest(..., asset_vol=True)` (also supported by `standalone_pnl` and `evaluate_features`); the default remains feature-variance sizing. The copy shows unrescaled curves plus scale-independent spike diagnostics. This needs historical asset returns and is not intended for the OOS label blackout. Open this notebook in Jupyter and Run All to reproduce the full analysis; `python run_eda.py` continues to execute only the main notebook.\n\nBoth notebooks retain only the one-row oracle lookahead, plus an h=0 causal control, with a common endpoint excluding one target row. Coefficient plots omit the first `HL` rows; estimation, full histories and one-row prediction lags are unchanged.\n'''
readme.write_text(s)

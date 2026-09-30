"""One-time notebook revision; removed after evaluated outputs are committed."""
from pathlib import Path
import nbformat as nbf

path=Path('notebooks/01_eda.ipynb')
nb=nbf.read(path,as_version=4)
nb.cells=[c for c in nb.cells if 'execution_summary' not in c.metadata.get('tags',[])]
M,C=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell

def tagged(tag):
    cells=[c for c in nb.cells if tag in c.metadata.get('tags',[])]
    assert len(cells)==1,(tag,len(cells))
    return cells[0]

nb.cells[1].source+='\nfrom takehome.features import ts_std, ewm_observed\n'
nb.cells.insert(2,M(r'''## Shared volatility convention

Every production EWM standard deviation now treats **zero and nonfinite inputs as missing**, sets `ignore_na=True`, and maps an exactly zero standard deviation back to missing:

```python
x.replace([0, np.inf, -np.inf], np.nan).ewm(
    halflife=hl, min_periods=hl, ignore_na=True
).std().replace(0, np.nan)
```

`ts_std` is the common helper; `ewm_observed` supplies matching moments for z-scores and EWM-Sharpe combinations. The cashflow diagnostic keeps its original minimal warm-up via `min_periods=0`. Raw zero signals still produce zero exposure when a variance estimate exists: this changes **moment estimation**, not the raw dataframe or regression observations. EWM means paired with these standard deviations use the same cleaned observations. Daily sums still include flat days as zero P&L.

This is a requested modeling convention, not a universal property of financial data: an actual zero return can be economically meaningful. With `ignore_na=True`, half-life counts nonzero observations within each series/group; the regression fitter's decay clock remains one step per input row. A long missing/zero run no longer collapses a variance estimate. Small **nonzero** variance can still produce large inverse-variance exposures; there is no hidden variance floor or leverage cap. See the Ridge sizing diagnostic below.

Reference: [pandas EWM missing-value semantics](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html).
'''))
for cell in nb.cells:
    if cell.cell_type=='code':
        cell.source=cell.source.replace("df['ret_5m'].ewm(halflife=288*21).std().shift()", "ts_std(df['ret_5m'], 288*21, min_periods=0).shift()")
        cell.source=cell.source.replace('df.ret_5m.ewm(halflife=288*21).std().shift()', 'ts_std(df.ret_5m, 288*21, min_periods=0).shift()')
    else:
        cell.source=cell.source.replace('moments = pnls.ewm(halflife=252*288)', 'moments = ewm_observed(pnls, 252*288)')

# Keep the useful plots, not hundreds of printed hyperparameter rows.
c=tagged('batch_oos_sweep')
c.source=c.source.replace('display(batch_sweep)', '')
c.source=c.source.replace("print('BATCH_OOS_SWEEP'); print(batch_sweep.to_string()); print('Batch grid seconds:',batch_seconds)",
                          "print(f'Batch grid: {len(models)} fixed candidates, {len(folds)} chronological folds; {batch_seconds:.1f} seconds.')")
c=tagged('all_predictor_fit')
c.source=c.source.replace('display(online_sweep)', '')
c.source=c.source.replace("print('STREAMING_VS_BATCH'); print(online_sweep.to_string())", '')
c.source=c.source.replace("print('FIXED_OOS_SHARPES'); print(comparison_daily.pipe(sharpe).to_string())", '''comparison_daily.pipe(sharpe).plot.barh(figsize=(9,4),title='Fixed penalties: OOS backtest Sharpe')
plt.xlabel('Calendar-day mean/std'); plt.show(); plt.close('all')
for metric in ['failed_updates', 'frozen_prediction_rmse']:
    online_sweep[metric].plot(logx=True,marker='o',figsize=(9,4),title=f'Streaming lasso: {metric}')
    plt.xlabel('Lasso alpha'); plt.ylabel(metric); plt.show(); plt.close('all')''')

# Avoid 100 repeated pandas date-axis conversions; every beta observation remains in the chart.
c=tagged('regression_beta_plot')
c.source='''np.testing.assert_allclose(yhat, online_yhat, rtol=1e-9, atol=1e-10, equal_nan=True)
import matplotlib.dates as mdates
with plt.rc_context({'path.simplify': True, 'path.simplify_threshold': 1., 'agg.path.chunksize': 10000}):
    fig, ax = plt.subplots(figsize=(14,5))
    clock = beta.index.as_unit('ns').asi8 / 86_400_000_000_000
    ax.plot(clock, beta.to_numpy(), linewidth=.6, alpha=.5, rasterized=True)
    locator=mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=beta.index.tz))
    ax.set(title='Streaming lasso: all 100 post-update coefficient paths', ylabel='Raw-feature coefficient')
    fig.tight_layout(); plt.show(); plt.close(fig)
'''

# Root-cause audit, with the legacy volatility explicitly confined to a before/after diagnostic.
insert=nb.cells.index(tagged('batch_oos_sweep'))+1
nb.cells[insert:insert]=[
M(r'''### Why can Ridge backtest values be very large?

Ridge predicts returns, but this backtest divides those small returns by the **variance of the predictions**, not the asset's variance. Thus the forecast and the position are different objects:

$$w_t=\hat y_t/\sigma_{\hat y,t}^{2}.$$

Multiplying a forecast by c multiplies its variance by c-squared, so the same backtest exposure is divided by c. **Shrinking a prediction can therefore increase its position.** A nearly constant nonzero intercept is another problem: subtracting the mean when estimating variance does not remove that intercept from the numerator. Abrupt changes in data/market conditions can combine a large forecast or return with a slow-moving, previously small variance estimate.

The audit below uses the actual frozen Ridge predictions across the fixed penalty grid. The previous zero-counting EWM is reconstructed only here to quantify this change; every production backtest uses the new shared helper. For Ridge, removing zero observations need not help: a small prediction is usually not exactly zero. No artificial leverage cap or variance floor is imposed, so these remain diagnostic exposure units rather than dollar P&L or portfolio returns.
'''),
C('''ridge_risk_records=[]
for a in RIDGE_ALPHAS:
    pred=oos_series(batch_paths[f'Ridge {a:g}'].prediction_)
    sigma=ts_std(pred,HL)
    # Historical comparison only, not used by any model/backtest result.
    previous_sigma=pred.ewm(halflife=HL,min_periods=HL,ignore_na=False).std().replace(0,np.nan)
    weight=pred.div(sigma.pow(2))
    contribution=weight.mul(df.ret_5m)
    ridge_risk_records.append(dict(alpha=a,zero_forecasts=int(pred.eq(0).sum()),
        min_sigma=sigma.min(),median_sigma=sigma.median(),
        max_abs_exposure=weight.abs().max(),p99_abs_exposure=weight.abs().quantile(.99),
        max_abs_pnl=contribution.abs().max(),
        max_sigma_change=(sigma-previous_sigma).abs().max()))
    if a==RIDGE_ALPHA:
        largest=contribution.abs().idxmax()
        ridge_spike=dict(timestamp=str(largest),forecast=float(pred.loc[largest]),
            sigma=float(sigma.loc[largest]),exposure=float(weight.loc[largest]),
            target_return=float(df.ret_5m.loc[largest]),pnl=float(contribution.loc[largest]))
ridge_sizing=pd.DataFrame(ridge_risk_records).set_index('alpha')
for metrics,title in [(['min_sigma','median_sigma'],'Ridge: small forecast volatility'),
                      (['max_abs_exposure','p99_abs_exposure'],'Ridge: inverse-variance exposure amplification')]:
    ax=ridge_sizing[metrics].plot(logx=True,logy=True,marker='o',figsize=(10,4),title=title)
    ax.set_xlabel('Ridge alpha'); plt.show(); plt.close('all')
print('Largest default-Ridge bar:',ridge_spike)
print('Exact-zero forecasts across the Ridge grid:',int(ridge_sizing.zero_forecasts.sum()),
      '; maximum change in EWM std from the zero/missing convention:',ridge_sizing.max_sigma_change.max())
''',metadata={'tags':['ridge_sizing_diagnostic']})]

new=[M(r'''# Forecast each alpha one period ahead with Streaming Lasso

Fit one **AR(2)** model for each of x1 through x100. “Previous two predictors” means two time lags of that same alpha, not two unrelated columns. Once row t is observed, learn the relation from past examples and use the two currently available values:

$$x_s=a_t+b_{1,t}x_{s-1}+b_{2,t}x_{s-2}+e_s,\quad s\le t,$$
$$\widehat x_{t+1\mid t}=a_t+b_{1,t}x_t+b_{2,t}x_{t-1}.$$

Unlike the return-regression diagnostic, **x_t itself is already observed** at this decision. Updating the alpha model through x_t before forecasting x_{t+1} is causal; using x_{t+1} in fitting would not be. The forecast is stored at its **decision timestamp t**, ready to replace the deliberately look-ahead `x.shift(-1)` input. It is not shifted backward from a future fitted value. Tests alter future feature values and verify earlier forecasts are unchanged, and compare prefix fits with the direct CVXPY loss.

Use W=1 on finite target/lag triples, an intercept, half-life 6048, and at least 6048 complete examples before publishing a forecast. Missing lags are not zero-filled or backfilled. Missing examples advance the fitting decay clock. No return labels are used to fit these alpha forecasts, so their online update can continue during the price blackout, provided the alphas themselves remain available.

For comparable regularization across different alpha units, divide each alpha and its lags by a fixed standard deviation estimated from its **initial warm-up examples only**, excluding zero values. The scale is frozen thereafter; a constant warm-up uses an RMS/one fallback. Fit with the existing feature-scaled lasso penalty, then restore original units. The grid's alpha is therefore in initial-standard-deviation units, not directly comparable with the return-regression alpha. Earlier warm-up fits are never scored. Hyperparameters are fixed candidates, not chosen using the reserved test period.

Backtest each forecast using the unchanged `forecast / ts_std(forecast)**2 * ret_5m`, with the new zero/missing volatility convention. A persistence control uses x_t on the same causal availability mask. The oracle control uses the actual x_{t+1}, explicitly noncausal. The realized next alpha is used **only to evaluate forecast error**, never to construct the forecast or to filter the causal P&L. The existing ret_5m alignment is retained; a deployable trading claim still requires confirming that its return interval starts after the decision.
'''),
M('''## AR(2) regularization sweep

Present candidate performance as plots, not tables. Every candidate fits all 100 alpha series and all their training observations. The fixed illustrative penalty is 0.01; it is not replaced by the best point after viewing the charts. For this comparison, forecast, persistence and oracle combinations all use **signed** lagged EWM-Sharpe weights, so allocation constraints do not confound the comparison. The earlier raw-family plot retains its original nonnegative scores.
'''),
C('''from takehome.fitters import forecast_alpha_blocks

AR_ALPHAS = [1e-3,1e-2,1e-1,1.]
AR_DEFAULT, AR_MIN_TRAIN = .01, HL
alpha_returns = df.ret_5m.where(rows < len(df)-1)
ar_records, ar_diagnostics, ar_meta_paths = [], {}, {}

def measured_alpha_blocks(penalty, diagnostics):
    for block in forecast_alpha_blocks(df[x_cols],hl=HL,alpha=penalty,
                                      min_train=AR_MIN_TRAIN,batch_size=8,diagnostics=diagnostics):
        for j,col in enumerate(block):
            current=df[col]
            actual=current.shift(-1)  # Evaluation only. Not supplied to the fitter.
            valid=np.isfinite(block[col]) & np.isfinite(current) & np.isfinite(actual)
            error=(block.loc[valid,col]-actual[valid]).pow(2).mean()
            persistence_error=(current[valid]-actual[valid]).pow(2).mean()
            diagnostics[-len(block.columns)+j].update(
                forecast_pairs=int(valid.sum()),
                rmse_vs_persistence=np.sqrt(error/persistence_error) if persistence_error>0 else np.nan)
        yield block

for penalty in AR_ALPHAS:
    info=[]; started=perf_counter()
    days,summary,meta=evaluate_features(measured_alpha_blocks(penalty,info),alpha_returns,
                                      hl=HL,meta=True,meta_hl=META_HL,positive_only=False)
    fit_info=pd.DataFrame(info).set_index('feature')
    ratio=fit_info.rmse_vs_persistence
    scores=summary.daily_mean_over_std
    ar_records.append(dict(alpha=penalty,meta_sharpe=sharpe(meta.resample('D').sum()),
        median_member_sharpe=scores.median(),p95_member_sharpe=scores.quantile(.95),
        median_rmse_vs_persistence=ratio.median(),fraction_rmse_improved=ratio.dropna().lt(1).mean(),
        failed_updates=int(fit_info.failed_updates.sum()),seconds=perf_counter()-started))
    ar_diagnostics[penalty]=fit_info
    ar_meta_paths[penalty]=meta
    if penalty==AR_DEFAULT:
        ar_default_daily,ar_default_summary,ar_default_meta=days,summary,meta
    print(f'AR(2) alpha={penalty:g}: {len(info)} signals; {ar_records[-1]["failed_updates"]} nonconverged updates; {ar_records[-1]["seconds"]:.1f}s')
ar_sweep=pd.DataFrame(ar_records).set_index('alpha')
for metrics,title in [(['meta_sharpe','median_member_sharpe','p95_member_sharpe'],'AR(2): backtest daily Sharpe'),
                      (['median_rmse_vs_persistence'],'AR(2): next-alpha RMSE / persistence RMSE'),
                      (['failed_updates'],'AR(2): solver convergence audit')]:
    ax=ar_sweep[metrics].plot(logx=True,marker='o',figsize=(10,4),title=title)
    if 'median_rmse_vs_persistence' in metrics:
        ax.axhline(1,linestyle='--',label='Persistence'); ax.legend()
    ax.set_xlabel('Lasso alpha (initial-alpha-scale units)'); plt.show(); plt.close('all')
''',metadata={'tags':['alpha_forecast_sweep']}),
M('''## Backtest the fixed AR(2) forecasts

All lines use the alpha forecast made at that decision timestamp. The purple secondary-axis line combines those forecasts' individual backtests using only earlier P&Ls. Histogram entries are individual alpha strategies, not model-fitting correlations.
'''),
C('''display_results(ar_default_daily,f'Forecast next alpha: AR(2) lasso, alpha={AR_DEFAULT:g}',
                meta_daily=ar_default_meta.resample('D').sum())
''',metadata={'tags':['alpha_forecast_backtest']}),
M('''## Matched persistence and realized-future controls

The persistence input is the current alpha, which is the forecast x_{t+1}=x_t. It uses exactly the same **causal** warm-up and current-lag availability rule as AR(2). The oracle uses the actual next value; it may additionally be missing when that future value is unavailable. No future-availability condition is imposed on the AR(2) or persistence backtests. The final training target row is omitted from all three. Volatility warm-ups can still differ when a forecast is exactly zero, so coverage is reported by the underlying summaries.
'''),
C('''def matched_alpha_controls(oracle=False):
    for start in range(0,len(x_cols),8):
        X=df[x_cols[start:start+8]].replace([np.inf,-np.inf],np.nan)
        triples=X.notna() & X.shift(1).notna() & X.shift(2).notna()
        ready=triples.cumsum().ge(AR_MIN_TRAIN) & X.notna() & X.shift(1).notna()
        yield (X.shift(-1) if oracle else X).where(ready)

control_records=[]
for oracle,label in [(False,'Persistence: current alpha'),(True,'Oracle: realized next alpha (LOOK-AHEAD)')]:
    days,summary,meta=evaluate_features(matched_alpha_controls(oracle),alpha_returns,
                                      hl=HL,meta=True,meta_hl=META_HL,positive_only=False)
    display_results(days,label,meta_daily=meta.resample('D').sum())
    control_records.append(dict(model=label,meta_sharpe=sharpe(meta.resample('D').sum()),
        median_member_sharpe=summary.daily_mean_over_std.median(),
        median_pnl_observations=summary.pnl_observations.median()))
control_records.append(dict(model='AR(2) forecast, alpha=0.01',meta_sharpe=sharpe(ar_default_meta.resample('D').sum()),
    median_member_sharpe=ar_default_summary.daily_mean_over_std.median(),
    median_pnl_observations=ar_default_summary.pnl_observations.median()))
ar_comparison=pd.DataFrame(control_records).set_index('model')
ar_comparison[['meta_sharpe','median_member_sharpe']].plot.barh(figsize=(11,4),title='Causal alpha forecast versus persistence and oracle')
plt.xlabel('Unannualized calendar-day mean/std'); plt.tight_layout(); plt.show(); plt.close('all')
ratios=ar_diagnostics[AR_DEFAULT].rmse_vs_persistence.dropna()
plt.figure(figsize=(9,4)); plt.hist(ratios,bins=30)
plt.axvline(1,linestyle='--'); plt.xlabel('Next-alpha RMSE / persistence RMSE')
plt.ylabel('Alpha count'); plt.title('AR(2) forecast accuracy across all alphas'); plt.show(); plt.close('all')
print(f'Default AR(2): {ratios.lt(1).sum()}/{len(ratios)} alphas beat persistence on forecast RMSE; median ratio={ratios.median():.4f}.')
assert df.index.max()<split['cutoff']
''',metadata={'tags':['alpha_forecast_controls']})]
insert=next(i for i,c in enumerate(nb.cells) if c.source.startswith('# Execution environment'))
nb.cells[insert:insert]=new
for c in nb.cells:
    if c.cell_type=='code':
        c.outputs=[];c.execution_count=None;c.metadata.pop('execution',None)
source='\n'.join(c.source for c in nb.cells)
assert 'literal_daily_sharpe' not in source
assert 'display(batch_sweep)' not in source and 'display(online_sweep)' not in source
assert source.count('def matched_alpha_controls')==1
nbf.validate(nb);nbf.write(nb,path)
p=Path('README.md')
p.write_text(p.read_text()+'''\n## One-step alpha forecasts\n\n`fitters.forecast_alpha` learns each alpha from its own two lagged values using the streaming jitclass; forecasts are indexed by decision time. `forecast_alpha_blocks` limits column memory. The notebook compares these causal forecasts against matched persistence and explicitly noncausal future-value controls, and plots the fixed hyperparameter grid. All production EWM standard deviations use zero/nonfinite-as-missing inputs and `ignore_na=True` through `features.ts_std` / `ewm_observed`. The Ridge diagnostic separates forecast scale from inverse-variance exposure; zero handling is not a leverage cap.\n''')
print('Notebook revised: AR(2) forecasts, plotted sweeps, shared volatility policy and Ridge sizing audit.')

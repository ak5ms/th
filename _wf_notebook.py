"""One-time notebook migration, removed after execution."""
from pathlib import Path
import nbformat as nbf

path = Path('notebooks/01_eda.ipynb')
nb = nbf.read(path, as_version=4)
nb.cells = [c for c in nb.cells if 'execution_summary' not in c.metadata.get('tags', [])]
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
load = next(c for c in nb.cells if c.cell_type == 'code' and 'df, split = training_data(raw)' in c.source)
load.source = load.source.replace('feature_cols = infer_feature_cols(df)',
    "from takehome.features import with_cashflow_feature\ndf = with_cashflow_feature(df)  # x100: raw ratio; undefined ratios remain missing.\nfeature_cols = infer_feature_cols(df)")
nb.cells.insert(nb.cells.index(load)+1, M('''### Added predictor: x100

`x100 = cashflow / volume`, appended after x99 and included in all feature families and fits. This is the raw ratio, **not clipped** to [-1, 1]. Zero-volume and nonfinite ratios remain missing. Only the regression design later replaces missing predictors with zero; the EDA frame is not imputed. The feature is formed after the training/test split.'''))
start = next(i for i,c in enumerate(nb.cells) if c.source.startswith('# All-predictor regression backtests'))
end = next(i for i,c in enumerate(nb.cells) if c.source.startswith('# Execution environment'))
reference_cell = next(c for c in nb.cells if c.cell_type=='code' and 'CVXPY_RECONCILIATION' in c.source)
old_predict = next(c for c in nb.cells if 'lagged_regression_predictions' in c.metadata.get('tags', []))
new = [
M(r'''# Walk-forward regression backtests

Use all **100** raw predictors (including x100), W=1 on available targets, and the existing explicit zero-imputation of missing predictors. No target is imputed and every row advances decay. Keep the reserved final 20% untouched.

The only backtest is $p_t=\hat y_t/\mathrm{EWMstd}_{6048}(\hat y)_t^2\times\mathrm{ret\_5m}_t$. There is no extra forecast shift, cost model or leverage cap. Undefined divisions stay missing. Daily Sharpe means **unannualized calendar-day mean/std**.

## Objective and shared interface

With $a_{i,t}=W_i d^{t-i}$, $A=\sum a_i$ and weighted population feature scales $s_j$, the lasso loss is
$$\frac{1}{2A}\sum_i a_i(y_i-b-x_i^\top\beta)^2+\alpha\sum_j s_j|\beta_j|.$$
Ridge replaces the last term by $\alpha\sum_j(s_j\beta_j)^2/2$. Both leave the intercept unpenalized. All scales/means are fitted inside the previous training window.

`StreamingWeightedLasso_` is a real Numba jitclass owning state and compiled methods. `StreamingWeightedLasso` validates inputs, warns on nonconvergence, and exposes the familiar history and prediction interface. Welford/coordinate-descent mathematics is unchanged; forecasts are emitted directly in the compiled row loop before incorporating that row's label.

`BatchLasso` solves an independent CVXPY quadratic program from batch-recomputed moments. `CvxpyWeightedLasso` retains the original-row residual loss as a separate reference. `BatchRidge` solves the normal equations. `walk_forward_sweep` shares batch moments across penalties, not fitted parameters across evaluation rows. `BatchedFitters.get_coefs(lag=1)` and `get_intercepts(lag=1)` reconstruct its OOS `prediction_`.

## Fold and hyperparameter contract

Use an expanding training window, first fit after **72,576 input rows**, and hold its coefficients fixed over the next **6,048 rows**, then refit. These are row counts, not exact calendar months. Both batch models and streaming lasso use decay $d=2^{-1/6048}$, so matching cutoff fits have the same effective weighted history. `gap=0` still excludes the current prediction row; the prior row's label is assumed available, as in the requested beta.shift convention. Increase gap if the actual target maturity requires it.

Every sweep point is scored on forecasts from a previous fold, not fitted values. Grid values are fixed in advance; no best-alpha selection is claimed to be unbiased after viewing these validation scores. These are **walk-forward validation results inside the training 80%**, not results on the reserved test block. The first 6,048 OOS forecasts warm the common variance estimator; all models use the same subsequent evaluation span.'''),
C('''from takehome.fitters import (StreamingWeightedLasso, BatchRidge, BatchLasso,
                              CvxpyWeightedLasso, BatchedFitters, walk_forward_sweep,
                              walk_forward_folds, stream_at_folds, batch_moments, _scaled_moments)
from takehome.features import backtest, forecast_metrics
from takehome.plots import plot_calibration
from time import perf_counter

fit_columns = list(x_cols)
assert len(fit_columns) == 100 and fit_columns[-1] == 'x100'
X_fit = df[fit_columns].replace([np.inf,-np.inf], np.nan).fillna(0).to_numpy(dtype=float)
X_fit = np.ascontiguousarray(X_fit)
y_fit = df.ret_5m.to_numpy(dtype=float)
valid_fit = np.isfinite(y_fit)
W_fit = valid_fit.astype(float)
DECAY, DEFAULT_ALPHA, RIDGE_ALPHA = 2**(-1/HL), 1e-5, 1e-2
LASSO_ALPHAS = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4])
RIDGE_ALPHAS = np.logspace(-6, 2, 9)
FOLD = dict(min_train_size=252*288, step=21*288, train_size=None, gap=0)
folds = walk_forward_folds(len(df), **FOLD)
OOS_FIRST = folds[0]['predict_start']
SCORE_FIRST = OOS_FIRST + HL - 1
rows = np.arange(len(df))
score_day = df.index[SCORE_FIRST].normalize()

def oos_series(values):
    return pd.Series(values, index=df.index).where(rows >= OOS_FIRST)

def score_prediction(prediction):
    per_bar = backtest(prediction, df.ret_5m, HL).where(rows >= SCORE_FIRST)
    daily = per_bar.resample('D').sum().loc[score_day:]
    stats = forecast_metrics(prediction.iloc[SCORE_FIRST:], df.ret_5m.iloc[SCORE_FIRST:])
    return daily, dict(daily_sharpe=sharpe(daily), pnl_observations=int(per_bar.count()), **stats)

fold_table = pd.DataFrame(folds)
for key in ['train_start','train_stop','predict_start','predict_stop']:
    fold_table[key+'_msgStamp'] = [df.index[min(int(v),len(df)-1)] for v in fold_table[key]]
display(fold_table.head(8))
print(f'WF_PROTOCOL: {len(folds)} folds; {len(df)} training rows; {len(fit_columns)} predictors')
print('First OOS forecast:', df.index[OOS_FIRST], '; first scoring row:', df.index[SCORE_FIRST])
print('Valid target rows:', int(W_fit.sum()), '; x100 finite rows:', int(df.x100.notna().sum()))
''', metadata={'tags':['regression_setup']}),
M('''## Numerical reference: direct CVXPY residual loss

This synthetic check uses exactly matched prefix observations, decay, feature-scaled penalty and intercept convention. The later real-data checks separately compare whole-history batch moments and frozen forecasts.'''),
reference_cell,
M('''## Batch Ridge and batch lasso: next-fold OOS sweeps

One batch moment calculation per cutoff and decay is reused across the penalty grid. No future rows enter that calculation. Each fitted model is frozen over its next prediction fold; an expanding-window fit is never applied backward to its own training rows.'''),
C('''models = {f'Ridge {a:g}': BatchRidge(len(fit_columns), alpha=float(a), decay=DECAY) for a in RIDGE_ALPHAS}
models.update({f'Lasso {a:g}': BatchLasso(len(fit_columns), DECAY, float(a),
                                      fit_intercept=True, tol=1e-11) for a in LASSO_ALPHAS})
started = perf_counter()
batch_paths = walk_forward_sweep(models, X_fit, y_fit, W=W_fit, **FOLD)
batch_seconds = perf_counter()-started
batch_daily, batch_rows = {}, []
for name, path in batch_paths.items():
    daily, stats = score_prediction(oos_series(path.prediction_))
    batch_daily[name] = daily
    batch_rows.append(dict(model=name.split()[0], alpha=models[name].alpha, **stats))
batch_sweep = pd.DataFrame(batch_rows).set_index(['model','alpha'])
for family in ['Ridge', 'Lasso']:
    batch_sweep.loc[family, 'daily_sharpe'].plot(logx=True, marker='o', figsize=(10,4),
        title=f'{family}: previous-fold OOS backtest sweep')
    plt.xlabel('Regularization coefficient (log scale)'); plt.ylabel('OOS calendar-day mean/std')
    plt.show(); plt.close('all')
fixed_batch = pd.DataFrame({
    'Ridge OOS, alpha=0.01': batch_daily[f'Ridge {RIDGE_ALPHA:g}'],
    'Batch lasso OOS, alpha=1e-5': batch_daily[f'Lasso {DEFAULT_ALPHA:g}'],
})
fixed_batch.cumsum().plot(figsize=(13,5),title='Fixed penalties: frozen next-fold OOS P&L')
plt.ylabel('Cumulative backtest P&L'); plt.show(); plt.close('all')
display(batch_sweep)
print('BATCH_OOS_SWEEP'); print(batch_sweep.to_string()); print('Batch grid seconds:',batch_seconds)
''',metadata={'tags':['ridge_backtest_sweep','batch_oos_sweep']}),
M('''## Streaming lasso: live updates versus frozen checkpoint fits

For each alpha, replay all rows once. `live` predicts before every row update. `frozen` takes the very same estimator's coefficients at the batch cutoffs and holds them until the next fold. This separates **update frequency** from **implementation correctness**: compare batch with frozen streaming first, then compare frozen with live streaming.

The main history is retained at fixed alpha=1e-5. Solver tolerance is tightened to 1e-10 with 20,000 sweeps; failed updates are counted, not suppressed. Only requested audit checkpoints retain moment snapshots.'''),
C('''online_results, online_rows, stream_daily = {}, [], {}
audit_ids = np.unique(np.linspace(0, len(folds)-1, 6, dtype=int)).tolist()
for alpha in LASSO_ALPHAS:
    m = StreamingWeightedLasso(len(fit_columns), DECAY, float(alpha), max_iter=20000, tol=1e-10,
                              fit_intercept=True, store_history=True if alpha==DEFAULT_ALPHA else False)
    started = perf_counter()
    result = stream_at_folds(m, X_fit, y_fit, folds, W=W_fit,
                            audit_folds=audit_ids if alpha==DEFAULT_ALPHA else ())
    seconds = perf_counter()-started
    live_daily, live_stats = score_prediction(oos_series(result['live']))
    frozen_daily, frozen_stats = score_prediction(oos_series(result['frozen']))
    reference = batch_paths[f'Lasso {alpha:g}'].prediction_
    difference = result['frozen']-reference
    online_rows.append(dict(alpha=alpha, live_sharpe=live_stats['daily_sharpe'],
        frozen_sharpe=frozen_stats['daily_sharpe'],
        batch_sharpe=batch_sweep.loc[('Lasso',alpha),'daily_sharpe'],
        frozen_prediction_rmse=np.sqrt(np.nanmean(difference**2)),
        frozen_max_abs_error=np.nanmax(abs(difference)),
        failed_updates=m.n_failed_, final_KKT=m.kkt_violation_, seconds=seconds,
        slope=live_stats['slope'], correlation=live_stats['correlation']))
    online_results[float(alpha)] = result
    stream_daily[float(alpha)] = live_daily
    if alpha==DEFAULT_ALPHA:
        fitter, online_yhat, default_result = m, pd.Series(result['live'], index=df.index), result
        default_live_daily, default_frozen_daily = live_daily, frozen_daily
    print(f'ONLINE alpha={alpha:g}: {seconds:.2f}s; failed updates={m.n_failed_}', flush=True)
online_sweep = pd.DataFrame(online_rows).set_index('alpha')
online_sweep[['live_sharpe','frozen_sharpe','batch_sharpe']].plot(
    logx=True,marker='o',figsize=(11,5),title='OOS lasso sweep: live updates, frozen streaming, independent batch')
plt.xlabel('Lasso alpha (log scale)'); plt.ylabel('OOS calendar-day mean/std')
plt.show(); plt.close('all')
comparison_daily = pd.DataFrame({
    'Streaming lasso, live': default_live_daily,
    'Streaming lasso, frozen': default_frozen_daily,
    'Batch lasso, frozen': batch_daily[f'Lasso {DEFAULT_ALPHA:g}'],
    'Ridge, frozen': batch_daily[f'Ridge {RIDGE_ALPHA:g}'],
})
comparison_daily.cumsum().plot(figsize=(13,5),title='Fixed penalties: OOS model P&Ls on a common scoring span')
plt.ylabel('Cumulative backtest P&L'); plt.show(); plt.close('all')
display(online_sweep)
print('STREAMING_VS_BATCH'); print(online_sweep.to_string())
print('FIXED_OOS_SHARPES'); print(comparison_daily.pipe(sharpe).to_string())
''',metadata={'tags':['all_predictor_fit','lasso_backtest_sweep','all_predictor_backtests']}),
M('''### Investigating a performance gap

Compare the frozen prediction vectors, not just Sharpe. At selected identical full-history cutoffs, compare weighted means, standardized covariance/cross-moments and original-unit loss. Then repeat the row-residual CVXPY reference on real 4,096-row windows. This tests the actual feature scales, imputation and returns rather than relying only on synthetic arrays.

The zero-filled design, W=1, decay, intercept, penalty, warm-up and test timestamps are identical in the frozen comparison. Coefficients may be non-unique for redundant predictors, so prediction and objective agreement are more informative than raw coefficient differences. A live-versus-frozen gap can instead reflect refitting frequency or estimation noise. The inverse-variance backtest can also amplify very small numerical differences when forecast variance is near zero; the shared-denominator comparison below tests that separately.'''),
C('''audit_rows = []
reference_path = batch_paths[f'Lasso {DEFAULT_ALPHA:g}']
for i in audit_ids:
    stop = folds[i]['train_stop']
    moments = batch_moments(X_fit[:stop], y_fit[:stop], W_fit[:stop], DECAY)
    G, h, scale, v = _scaled_moments(moments, True)
    state = default_result['states'][i]
    theta = default_result['coefs'][i]*scale
    objective = .5*(v-2*h@theta+theta@G@theta)+DEFAULT_ALPHA*abs(theta).sum()
    audit_rows.append(dict(fold=i, training_stop=stop,
        max_scaled_mean_error=np.max(abs(state[0]-moments[0])/scale),
        max_standardized_cov_error=np.max(abs(state[2]-moments[2])/scale[:,None]/scale),
        max_scaled_cross_error=np.max(abs(state[3]-moments[3])/scale),
        objective_gap=abs(objective-reference_path.objectives_[i]),
        streaming_KKT=default_result['kkt_at_folds'][i]))
implementation_audit = pd.DataFrame(audit_rows)
display(implementation_audit)
print('FULL_PREFIX_AUDIT'); print(implementation_audit.to_string(index=False))

real_checks=[]
for stop in [folds[0]['train_stop'], folds[len(folds)//2]['train_stop'], len(df)]:
    start=max(0,stop-4096)
    xx, yy, ww=X_fit[start:stop], y_fit[start:stop], W_fit[start:stop]
    direct=CvxpyWeightedLasso(len(fit_columns),DECAY,DEFAULT_ALPHA,fit_intercept=True,tol=1e-12).fit(xx,yy,W=ww)
    compiled=StreamingWeightedLasso(len(fit_columns),DECAY,DEFAULT_ALPHA,fit_intercept=True,
                                  tol=1e-11,max_iter=50000).fit(xx,yy,W=ww)
    fast_batch=BatchLasso(len(fit_columns),DECAY,DEFAULT_ALPHA,fit_intercept=True,tol=1e-12).fit(xx,yy,W=ww)
    real_checks.append(dict(training_stop=stop, rows=stop-start,
        stream_vs_row_cvxpy_max_error=np.max(abs(compiled.predict(xx)-direct.predict(xx))),
        batch_vs_row_cvxpy_max_error=np.max(abs(fast_batch.predict(xx)-direct.predict(xx))),
        final_stream_KKT=compiled.kkt_violation_, final_converged=compiled.converged_))
real_checks=pd.DataFrame(real_checks); display(real_checks)
print('REAL_ROW_CVXPY_CHECKS'); print(real_checks.to_string(index=False))

# Strictly solve the SAME moments without appending another observation.
from takehome.fitters import _coordinate_descent
worst=int(implementation_audit.loc[implementation_audit.objective_gap.idxmax(),'fold'])
stop=folds[worst]['train_stop']; a,b=folds[worst]['predict_start'],folds[worst]['predict_stop']
strict_moments=batch_moments(X_fit[:stop],y_fit[:stop],W_fit[:stop],DECAY)
G,h,scale,v=_scaled_moments(strict_moments,True)
theta=default_result['coefs'][worst]*scale
iterations,kkt,tolerance=_coordinate_descent(G,h,theta,DEFAULT_ALPHA,500000,1e-13)
mx,my=strict_moments[:2]
strict_prediction=X_fit[a:b]@(theta/scale)+(my-mx@(theta/scale))
strict_check=pd.Series(dict(fold=worst, strict_iterations=iterations, strict_KKT=kkt,
    max_next_fold_error=np.max(abs(strict_prediction-reference_path.prediction_[a:b]))))
display(strict_check.to_frame('same_objective_strict_solve'))
print('STRICT_CHECK'); print(strict_check.to_string())

ref=oos_series(reference_path.prediction_)
frozen=oos_series(default_result['frozen'])
shared_scale=ts_std(ref,HL).pow(2).replace(0,np.nan)
shared_pnl=pd.concat({'Batch':ref.div(shared_scale).mul(df.ret_5m),
                     'Frozen stream':frozen.div(shared_scale).mul(df.ret_5m)},axis=1)
shared_daily=shared_pnl.where(pd.Series(rows>=SCORE_FIRST,index=df.index),axis=0).resample('D').sum().loc[score_day:]
print('SHARED_DENOMINATOR_SHARPES'); print(shared_daily.pipe(sharpe).to_string())
relative_prediction_error=online_sweep.loc[DEFAULT_ALPHA,'frozen_prediction_rmse']/df.ret_5m.std()
print('Frozen prediction RMSE / target standard deviation:',relative_prediction_error)
print('Numerical agreement and refit-frequency effects must be assessed separately; inspect the diagnostics above.')
''',metadata={'tags':['implementation_audit']}),
M('''## Coefficient paths and lagged OOS calibration

The coefficient chart shows the post-update history of the default unit-weight lasso. The scatter uses the **common OOS scoring span**, not the initial fitting period. Both coefficients and intercept are shifted by one row. Calibration is y on yhat, with an intercept, and is not used to rescale predictions.'''),
old_predict,
C('''np.testing.assert_allclose(yhat, online_yhat, rtol=1e-9, atol=1e-10, equal_nan=True)
beta.plot(figsize=(14,5),legend=False,linewidth=.6,alpha=.5,
          title='StreamingWeightedLasso_ jitclass: all 100 post-update coefficient paths')
plt.ylabel('Raw-feature coefficient'); plt.show(); plt.close('all')
''',metadata={'tags':['regression_beta_plot']}),
C('''calibration=plot_calibration(yhat.where(rows>=SCORE_FIRST),df.ret_5m,
    title='Unit-weight streaming lasso: lagged OOS predictions vs targets')
display(calibration)
print('OOS_CALIBRATION'); print(calibration.to_string())
''',metadata={'tags':['regression_calibration_plot']}),
M(r'''# Optimistic future-predictor diagnostic

For h=1,2,6,12,24, feed `df[x_cols].shift(-h)` into the unchanged `evaluate_features` workflow. Include h=0 as the control. These are **intentionally look-ahead oracle features**, not tradable signals and not a measured AR(1) forecast. Knowing a realized future feature may include information from the target's return interval. This is an optimistic experiment, not a mathematical upper bound for every feasible forecasting/position-sizing rule.

Shift only the already-reserved training frame: the trailing h rows become missing, and no values are pulled from the test block. All horizons exclude the last 24 target rows for a common endpoint. Feature missingness still differs by horizon and is reported. Leads count dataframe rows, not elapsed minutes across gaps. Each horizon gets its own P&L plot, Sharpe histogram and lagged EWM-Sharpe combination; the combination remains on the secondary axis. The meta estimator itself sees only its past oracle P&L, but the underlying signal is deliberately noncausal.'''),
]
for horizon in [0,1,2,6,12,24]:
    new += [M(f'## Predictor lead h={horizon}'+(' — causal-input control' if horizon==0 else ' — LOOK-AHEAD')),
C(f'''h = {horizon}
oracle_returns = df.ret_5m.where(rows < len(df)-24)
blocks = (df[x_cols[start:start+8]].shift(-h) for start in range(0,len(x_cols),8))
oracle_daily, oracle_summary, oracle_meta = evaluate_features(
    blocks, oracle_returns, hl=HL, meta=True, meta_hl=META_HL, positive_only=True)
display_results(oracle_daily, f'Predictor lead h={{h}}: '+('control' if h==0 else 'LOOK-AHEAD'),
                meta_daily=oracle_meta.resample('D').sum())
if h==0:
    oracle_records, oracle_score_columns = [], {{}}
scores=oracle_summary.daily_mean_over_std
oracle_score_columns[h]=scores
oracle_records.append(dict(h=h, candidates=len(scores), defined=int(scores.notna().sum()),
    median_sharpe=scores.median(), p95_sharpe=scores.quantile(.95), best_sharpe=scores.max(),
    meta_sharpe=sharpe(oracle_meta.resample('D').sum()),
    median_pnl_observations=oracle_summary.pnl_observations.median()))
display(oracle_summary.head(10))
''', metadata={'tags':[f'oracle_lead_{horizon}']})]
new += [C('''oracle_comparison=pd.DataFrame(oracle_records).set_index('h')
per_feature_oracle=pd.DataFrame(oracle_score_columns)
for h in [1,2,6,12,24]:
    delta=per_feature_oracle[h]-per_feature_oracle[0]
    oracle_comparison.loc[h,'median_sharpe_change_vs_h0']=delta.median()
    oracle_comparison.loc[h,'fraction_improved_vs_h0']=delta.gt(0).where(delta.notna()).mean()
display(oracle_comparison)
print('ORACLE_SHIFT_SUMMARY'); print(oracle_comparison.to_string())
assert df.index.max()<split['cutoff'] and fitter.n_seen_==len(df)
''',metadata={'tags':['oracle_summary']})]
nb.cells[start:end] = new
summary_index=next(i for i,c in enumerate(nb.cells) if c.source.startswith('## Training-only summary and checks'))
nb.cells[summary_index+1].source = '''assert df.index.max() < split['cutoff']
assert len(fit_columns) == 100 and fit_columns[-1] == 'x100'
assert fitter.n_seen_ == len(df) == split['train_rows']
print('Training-only rows:',len(df),'; predictors:',len(fit_columns))
print('Fold OOS scoring starts:',df.index[SCORE_FIRST])
print('All displays and diagnostics are inline; the reserved test block remains unused.')'''
for c in nb.cells:
    if c.cell_type=='markdown':
        c.source=c.source.replace('4,851','4,950').replace('9,702','9,900').replace('99 raw','100 raw')
    if c.cell_type=='code':
        c.outputs,c.execution_count=[],None
        c.metadata.pop('execution',None)
source='\n'.join(c.source for c in nb.cells)
assert 'literal_' not in source and "normalization='zscore'" not in source
assert source.count('df = with_cashflow_feature(df)')==1
nbf.validate(nb); nbf.write(nb,path)
p=Path('README.md'); lines=p.read_text().splitlines()
lines=[line for line in lines if not any(key in line.lower() for key in ['literal','two backtest','in-sample ridge','full-training ridge'])]
p.write_text('\n'.join(lines)+'''\n\n## Causal regression comparison\n\n`StreamingWeightedLasso_` is the Numba jitclass; `StreamingWeightedLasso` is its Python interface. `BatchRidge` and CVXPY `BatchLasso` share the `BatchedFitters`/`walk_forward_sweep` interface, storing fold snapshots and next-fold OOS predictions. `stream_at_folds` isolates the effect of live versus frozen refitting. The notebook includes x100 = cashflow/volume, matched-loss audits, OOS penalty sweeps, and explicitly noncausal predictor-lead experiments. All use the original training split; no extra reports are exported.\n''')
print('Notebook rebuilt for jitclass, 100 predictors, walk-forward sweeps and oracle leads.')

"""One-time migration; removed after the evaluated notebook is committed."""
from pathlib import Path
import nbformat as nbf

p = Path('src/takehome/plots.py'); s = p.read_text()
old = """        ax.plot(meta_daily.index, meta_daily.cumsum(), color='purple', linewidth=3,
                zorder=10, label='Lagged EWM-Sharpe combination')
        ax.legend(loc='upper left')"""
new = """        right = meta_daily.rename('Lagged EWM-Sharpe combination').cumsum().plot(
            ax=ax, secondary_y=True, x_compat=True, color='purple', linewidth=3, zorder=10)
        right.set_ylabel('Combined P&L (right axis; independent scale)')
        right.legend(loc='upper left')"""
assert old in s
p.write_text(s.replace(old,new).replace('Members and their unrescaled combination, then the member Sharpe histogram.',
                                       'Members on the left, unrescaled combination on the right, then histogram.'))
p=Path('tests/test_plot_diagnostics.py'); s=p.read_text()
a=s.index('    lines = shown[0].axes[0].lines'); b=s.index('    assert_allclose(scores,',a)
s=s[:a]+'''    assert len(shown[0].axes) == 2
    lines = shown[0].axes[0].lines
    overlay = shown[0].axes[1].lines[0]
    assert len(lines) == len(daily.columns)
    for j, line in enumerate(lines):
        assert_allclose(line.get_ydata(), daily.iloc[:, j].cumsum())
    assert_allclose(overlay.get_ydata(), combined.cumsum())
    assert overlay.get_linewidth() > 4 * lines[0].get_linewidth()
    assert 'Lagged EWM-Sharpe' in overlay.get_label()
    assert 'independent scale' in shown[0].axes[1].get_ylabel()
'''+s[b:]
p.write_text(s)

p=Path('src/takehome/features.py')
p.write_text(p.read_text()+'''

def forecast_metrics(prediction, target):
    """Unweighted y-on-yhat calibration and errors on their finite overlap."""
    if not prediction.index.equals(target.index):
        raise ValueError('Forecast and target indexes must match.')
    values = pd.concat([prediction, target], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    p, y = values.iloc[:, 0], values.iloc[:, 1]
    slope = p.cov(y) / p.var() if len(p) > 1 and p.var() > 0 else np.nan
    return dict(n=len(p), slope=slope, intercept=y.mean()-slope*p.mean(),
                correlation=p.corr(y) if len(p) > 1 and p.std() > 0 and y.std() > 0 else np.nan,
                rmse=np.sqrt((y-p).pow(2).mean()), zero_forecast_rmse=np.sqrt(y.pow(2).mean()))
''')

path=Path('notebooks/01_eda.ipynb'); nb=nbf.read(path,as_version=4)
nb.cells=[c for c in nb.cells if 'execution_summary' not in c.metadata.get('tags',[])]
M,C=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell

def containing(text):
    cells=[c for c in nb.cells if text in c.source]
    assert len(cells)==1, (text,len(cells))
    return cells[0]

for c in nb.cells:
    if c.cell_type=='markdown':
        c.source=c.source.replace('on the same axis and in the same exposure units as its components',
                                  'on a secondary right y-axis; its axis is independent of the component axis')
        c.source=c.source.replace('without display rescaling', 'without multiplying its values (the right axis autoscales separately)')

prep=[
M(r'''# All-predictor regression backtests

Use **all 99 raw `x` predictors** and only the reserved training frame. A complete-case intersection can discard most rows because different signals have different coverage. To retain every predictor and the full row clock, this baseline **replaces missing/nonfinite predictors by zero** in a separate modeling array; the original `df` is unchanged. Zero is a simple modeling assumption, not a claim that missing signals are observed zeros. The fitted intercept remains enabled. No target is filled: missing `ret_5m` gets observation weight zero, while every row still advances decay. All usable target rows get W=1 in the main experiment.

This differs from the previous three-predictor complete-case demonstration. We therefore also refit the SAME 99-column zero-imputed design with squared-notional weights at the fixed default lasso alpha, so the effect of W can be assessed without also changing feature coverage. No new calendar filter, target shift, clipping or feature selection is silently introduced.

## What `backtest` means here

The main comparison retains the standalone-alpha convention:
$$p_t=\frac{\hat y_t}{\operatorname{EWMstd}_{6048}(\hat y)_t^2}\,\mathrm{ret\_5m}_t.$$

The literal newly requested expression is also evaluated separately:
$$p_t^{\rm literal}=\frac{\hat y_t}{\operatorname{ts\_zscore}_{6048}(\hat y)_t}\,\mathrm{ret\_5m}_t.$$

They are **not the same normalization**. Since $\hat y/z(\hat y)=\hat y\sigma_{\hat y}/(\hat y-\mu_{\hat y})$, the literal expression can lose the signal direction when the mean is near zero and explode when the z-score is near zero. We do not force either result to look better: exact zero/nonfinite divisions remain missing; finite large exposures are not capped. `backtest(..., normalization='variance')` is the main convention; `normalization='zscore'` is the literal sensitivity check.

EWM uses half-life=min_periods=6048 and the current forecast. Forecasts already use previous-row coefficients; the backtest does not shift them a second time. Daily `sum()` and unannualized calendar-day mean/std are unchanged. No trading costs are included. Return-maturity and execution assumptions still need confirmation. These sweeps use training data only and are exploratory, not a test-set estimate or a final model-selection decision.'''),
C('''from takehome.features import backtest, forecast_metrics
from takehome.fitters import BatchRidge, StreamingWeightedLasso
from time import perf_counter

fit_columns = list(x_cols)
assert len(fit_columns) == 99
X_fit = df[fit_columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
X_fit = np.ascontiguousarray(X_fit)
y_fit = df['ret_5m'].to_numpy(dtype=float)
valid_fit = np.isfinite(y_fit)
W_unit = valid_fit.astype(float)
notional_weights = (df['wmid'] * df['volume']).pow(2).to_numpy(dtype=float)
W_notional = np.where(valid_fit & np.isfinite(notional_weights) & (notional_weights > 0), notional_weights, 0.)
DEFAULT_ALPHA, RIDGE_ALPHA = 1e-5, 1e-2
LASSO_ALPHAS = np.array([1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4])
RIDGE_ALPHAS = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100.])
display(pd.Series({
    'training_rows': len(df), 'predictors': len(fit_columns),
    'unit_weight_rows': int(W_unit.sum()), 'notional_weight_rows': int((W_notional > 0).sum()),
    'raw_complete_case_rows': int(df[fit_columns].notna().all(axis=1).sum()),
    'zero_imputed_predictor_cells': int(df[fit_columns].isna().sum().sum()),
}, name='modeling_data').to_frame())

def run_online(alpha, weights, store_history=False):
    model = StreamingWeightedLasso(len(fit_columns), decay=2**(-1/HL), alpha=float(alpha),
                                  max_iter=5000, tol=1e-8, fit_intercept=True,
                                  store_history=store_history)
    start = perf_counter()
    pred = model.fit_predict(X_fit, y_fit, W=weights)
    return model, pd.Series(pred, index=df.index, name='yhat'), perf_counter()-start
''',metadata={'tags':['regression_setup']}),
M(r'''## Full-training Ridge baseline (IN-SAMPLE)

Before online lasso, fit Ridge to the entire **training** sample (never the held-out 20%). This deliberately uses future training labels when predicting earlier training rows; its backtest is an **in-sample diagnostic benchmark**, not a deployable strategy. A one-row shift cannot remove this full-sample lookahead.

`BatchRidge` minimizes $\frac{1}{2A}\sum_t W_t(y_t-b-X_t\beta)^2+\frac{\lambda}{2}\sum_j(s_j\beta_j)^2$, with training population feature scales $s_j$, W=1 on valid targets and an unpenalized intercept. The ridge normal equations are solved without forming an inverse and tested against an explicit CVXPY loss. L1 and L2 penalty numbers have different meanings; their magnitudes are not directly comparable.

The fixed baseline is lambda=0.01, not whichever training sweep point looks best. Ridge receives its own log-x sweep and P&L plot. Its fixed Sharpe is also drawn on the lasso plot only when that does not substantially expand the vertical scale.'''),
C('''ridge_predictions, ridge_records = {}, []
for alpha in RIDGE_ALPHAS:
    start = perf_counter()
    ridge = BatchRidge(len(fit_columns), alpha=float(alpha)).fit(X_fit, y_fit, W=W_unit)
    prediction = pd.Series(ridge.predict(X_fit), index=df.index, name=alpha)
    daily = backtest(prediction, df['ret_5m'], HL).resample('D').sum()
    literal = backtest(prediction, df['ret_5m'], HL, normalization='zscore').resample('D').sum()
    ridge_predictions[float(alpha)] = prediction
    ridge_records.append(dict(alpha=alpha, daily_sharpe=sharpe(daily),
                              literal_daily_sharpe=sharpe(literal), seconds=perf_counter()-start,
                              **forecast_metrics(prediction, df['ret_5m'])))
ridge_sweep = pd.DataFrame(ridge_records).set_index('alpha')
ridge_yhat = ridge_predictions[RIDGE_ALPHA].rename('Ridge IN-SAMPLE')
ridge_sweep[['daily_sharpe', 'literal_daily_sharpe']].plot(
    logx=True, marker='o', figsize=(10, 4), title='Full-training Ridge: IN-SAMPLE backtest sweep')
plt.ylabel('Unannualized calendar-day mean/std'); plt.xlabel('Ridge lambda (log scale)')
plt.show(); plt.close('all')
ridge_default_daily = pd.concat({
    'Variance normalization': backtest(ridge_yhat, df.ret_5m, HL),
    'Literal z-score division': backtest(ridge_yhat, df.ret_5m, HL, normalization='zscore'),
}, axis=1).resample('D').sum()
for name in ridge_default_daily:
    ridge_default_daily[name].cumsum().plot(figsize=(12, 4), title=f'Ridge lambda={RIDGE_ALPHA:g}: {name} (IN-SAMPLE)')
    plt.ylabel('Cumulative diagnostic P&L'); plt.show(); plt.close('all')
with pd.option_context('display.float_format', '{:.6g}'.format): display(ridge_sweep)
print('RIDGE_SWEEP'); print(ridge_sweep.to_string())
''',metadata={'tags':['ridge_backtest_sweep']}),
]
i=next(i for i,c in enumerate(nb.cells) if c.source.startswith('# Initial fitter:'))
nb.cells[i:i]=prep

real=containing('X_fit = df[fit_columns].to_numpy')
real.source='''# All 99 predictors; main experiment uses equal observation weights.
W_fit = W_unit
fitter, online_yhat, fit_seconds = run_online(DEFAULT_ALPHA, W_fit, store_history=True)
assert fitter.n_seen_ == len(df) and fitter.n_features == 99
weighted_fitter, weighted_yhat, weighted_seconds = run_online(DEFAULT_ALPHA, W_notional)
display(pd.Series(fitter.coef, index=fit_columns, name='unit_weight_coefficient').to_frame())
display(pd.DataFrame({
    'W=1': {'rows_advanced': fitter.n_seen_, 'nonconverged_updates': fitter.n_failed_,
            'final_KKT_residual': fitter.kkt_violation_, 'final_converged': fitter.converged_, 'seconds': fit_seconds},
    'Squared notional': {'rows_advanced': weighted_fitter.n_seen_, 'nonconverged_updates': weighted_fitter.n_failed_,
                         'final_KKT_residual': weighted_fitter.kkt_violation_, 'final_converged': weighted_fitter.converged_, 'seconds': weighted_seconds},
}))'''
real.metadata['tags']=['all_predictor_fit']
containing('## Initial fit on real training data').source='''## All-predictor online fit and weight comparison

Fit all 99 raw predictors, with the zero-imputation policy above. The main run uses unit weights on valid targets; the comparator uses squared-notional weights on the identical design. Both have half-life 6048, fixed alpha=1e-5 and an intercept. Predictions are emitted **before** every row update. Storage chunks do not change the fitting cadence: the model still updates at each row, including decay on zero-weight rows. History is retained only for the main coefficient chart.

Any iteration-limit failures are explicitly counted below and in the sweep. A finite answer is not automatically proof of convergence. No hyperparameter is selected using the held-out block.'''
c=containing('## Coefficient paths and lagged forecast calibration')
c.source=c.source.replace('on the fixed raw columns `x1`, `x2`, `x3`', 'on all 99 raw predictor columns')
c.source=c.source.replace('with the same half-life, penalty and `(wmid * volume)**2` training weights', 'with half-life 6048, alpha=1e-5 and unit training weights')
c.source=c.source.replace('`min_count=p` prevents a missing predictor from silently becoming a partial forecast',
                        'Missing predictors use the explicitly declared zero-imputed modeling design')
c.source=c.source.replace('The figure also shows a calibration slope under the same observation weights as the fitter, because unweighted and weighted calibration answer different questions.',
                        'The comparison below additionally evaluates the squared-notional fit on the same prediction/target rows.')
c=next(c for c in nb.cells if 'lagged_regression_predictions' in c.metadata.get('tags',[]))
c.source=c.source.replace('.mul(df[fit_columns])', ".mul(df[fit_columns].replace([np.inf, -np.inf], np.nan).fillna(0))")
c=next(c for c in nb.cells if 'regression_beta_plot' in c.metadata.get('tags',[]))
c.source=c.source.replace("beta.plot(figsize=(14, 5), title='Online weighted lasso: post-update coefficient paths')",
                        "beta.plot(figsize=(14, 5), legend=False, linewidth=.6, alpha=.5, title='All 99 predictor coefficients: online lasso, W=1')")
c=next(c for c in nb.cells if 'regression_calibration_plot' in c.metadata.get('tags',[]))
c.source=c.source.replace("yhat, df['ret_5m'], weights=pd.Series(W_fit, index=df.index),", "yhat, df['ret_5m'],")
c.source=c.source.replace('Online weighted lasso: previous-row forecasts vs training targets', 'All 99 predictors, W=1: previous-row forecasts vs training targets')
c.source+="\nnp.testing.assert_allclose(yhat, online_yhat, rtol=1e-10, atol=1e-10, equal_nan=True)"

post=[
M('''## Did unit weighting improve calibration? Backtest the predictions

The first table compares the two 99-predictor fits on their **common finite prediction/target sample**. It also reports the full-training Ridge baseline, clearly marked in-sample. Slope near one, small intercept and lower RMSE would be encouraging, but a negative or flat calibration slope is not repaired by a positive scale factor. No calibration multiplier is fitted back into these forecasts.

Each backtest below uses every training timestamp and reports both normalizations separately. The main forecast has already been lagged through its coefficients; do not shift its predictions again. Initial undefined exposures and missing targets stay missing until the requested daily aggregation.'''),
C('''comparison_predictions = pd.concat({'Lasso W=1': yhat, 'Lasso squared notional': weighted_yhat,
                                    'Ridge IN-SAMPLE': ridge_yhat}, axis=1)
common = comparison_predictions.notna().all(axis=1) & df.ret_5m.notna()
weight_comparison = pd.DataFrame({name: forecast_metrics(pred.where(common), df.ret_5m.where(common))
                                  for name, pred in comparison_predictions.items()}).T
with pd.option_context('display.float_format', '{:.6g}'.format): display(weight_comparison)
print('WEIGHT_CALIBRATION_COMMON_SAMPLE'); print(weight_comparison.to_string())
regression_daily = {}
for normalization in ['variance', 'zscore']:
    per_bar = backtest(comparison_predictions.iloc[:, :2], df.ret_5m, HL, normalization=normalization)
    daily = per_bar.resample('D').sum()
    regression_daily[normalization] = daily
    daily.cumsum().plot(figsize=(12, 5), title=f'All-predictor online backtest: {normalization} normalization')
    plt.ylabel('Cumulative diagnostic P&L'); plt.show(); plt.close('all')
    display(pd.DataFrame({'daily_sharpe': daily.pipe(sharpe), 'pnl_observations': per_bar.count(),
                          'max_absolute_bar_pnl': per_bar.abs().max()}))
    print('DEFAULT_BACKTEST', normalization); print(daily.pipe(sharpe).to_string())
''',metadata={'tags':['all_predictor_backtests']}),
M('''## Online-lasso regularization sweep

The six positive alphas are fixed on a log-spaced grid, including the previous 1e-5 default. Every point is a fresh `StreamingWeightedLasso` on all 99 columns, W=1, the entire training row clock, and prior-coefficient predictions. There is no parameter-dependent row sample. The plot shows Sharpe of daily **backtest P&L**, not Sharpe of predictions or returns. Undefined ratios remain undefined.

The fixed default, not the best training point, supplies the coefficient/scatter plots. Ridge is an in-sample benchmark; its horizontal reference is omitted when it would expand the lasso vertical range by more than three lasso-range widths (with a minimum width of 0.01). The separate Ridge sweep/P&L above always remains visible. Any nonconverged lasso updates are counted and marked; do not treat such a point as an accurately solved optimum.'''),
C('''lasso_records, lasso_predictions = [], {}
for alpha in LASSO_ALPHAS:
    if alpha == DEFAULT_ALPHA:
        model, prediction, seconds = fitter, yhat, fit_seconds
    else:
        model, prediction, seconds = run_online(alpha, W_unit)
    lasso_predictions[float(alpha)] = prediction
    per_bar = backtest(prediction, df.ret_5m, HL)
    literal = backtest(prediction, df.ret_5m, HL, normalization='zscore')
    lasso_records.append(dict(alpha=alpha,
        daily_sharpe=sharpe(per_bar.resample('D').sum()),
        literal_daily_sharpe=sharpe(literal.resample('D').sum()),
        pnl_observations=int(per_bar.count()), final_active_coefficients=int(np.count_nonzero(model.coef)),
        nonconverged_updates=model.n_failed_, final_converged=model.converged_, seconds=seconds,
        **forecast_metrics(prediction, df.ret_5m)))
    print(f'alpha={alpha:g}: {seconds:.2f}s; nonconverged updates={model.n_failed_}', flush=True)
lasso_sweep = pd.DataFrame(lasso_records).set_index('alpha')
for metric, label in [('daily_sharpe', 'Standalone variance normalization'),
                      ('literal_daily_sharpe', 'Literal yhat / ts_zscore(yhat)')]:
    ax = lasso_sweep[metric].plot(logx=True, marker='o', figsize=(10, 5), label='Online lasso W=1')
    reference = ridge_sweep.loc[RIDGE_ALPHA, metric]
    finite = lasso_sweep[metric].dropna()
    width = max(float(finite.max()-finite.min()), .01) if len(finite) else .01
    if len(finite) and np.isfinite(reference) and finite.min()-3*width <= reference <= finite.max()+3*width:
        ax.axhline(reference, linestyle='--', label=f'Ridge lambda={RIDGE_ALPHA:g} (IN-SAMPLE)')
    else:
        ax.text(.02,.98,f'Ridge in-sample reference: {reference:.4g}; see separate Ridge plots',
                transform=ax.transAxes,va='top')
    failed = lasso_sweep.nonconverged_updates.gt(0)
    if failed.any():
        ax.scatter(lasso_sweep.index[failed], lasso_sweep.loc[failed,metric], marker='x', s=70,
                   label='Contains nonconverged updates')
    ax.set(xlabel='Lasso alpha (log scale)', ylabel='Unannualized calendar-day mean/std',
           title=f'99-predictor backtest penalty sweep: {label}')
    ax.legend(); plt.show(); plt.close('all')
with pd.option_context('display.float_format', '{:.6g}'.format): display(lasso_sweep)
print('LASSO_SWEEP'); print(lasso_sweep.to_string())
assert df.index.max() < split['cutoff'] and fitter.n_seen_ == split['train_rows']
''',metadata={'tags':['lasso_backtest_sweep']}),
]
i=next(i for i,c in enumerate(nb.cells) if 'regression_calibration_plot' in c.metadata.get('tags',[]))+1
nb.cells[i:i]=post

for c in nb.cells:
    if c.cell_type=='code':
        c.outputs=[]; c.execution_count=None; c.metadata.pop('execution',None)
nbf.validate(nb); nbf.write(nb,path)

p=Path('tests/test_regression_notebook.py'); s=p.read_text()
s=s.replace('def test_notebook_keeps_missing_predictor_and_prefit_rows_missing():',
            'def test_notebook_zero_imputes_predictors_but_masks_prefit_rows():')
s=s.replace('[np.nan, np.nan, np.nan, 15.]', '[np.nan, np.nan, 5., 15.]')
s=s.replace("real = next(c for c in nb.cells if 'X_fit = df[fit_columns]' in c.source)",
            "real = next(c for c in nb.cells if 'all_predictor_fit' in c.metadata.get('tags', []))")
p.write_text(s)
p=Path('README.md'); s=p.read_text().replace('in purple, without display rescaling',
'in purple on a secondary right y-axis, without multiplying its values')
p.write_text(s+'''\n## All-predictor regression backtests\n\nThe notebook fits all 99 raw predictors with unit and squared-notional weights on the same zero-imputed design, retains prior-row coefficient predictions, and plots the daily-Sharpe lasso penalty sweep. A full-training Ridge sweep is explicitly in-sample and never uses the holdout. `features.backtest` defaults to the existing variance normalization and separately supports the literal z-score-division sensitivity. `StreamingWeightedLasso.fit_predict` bounds temporary history with storage-only chunks; every row is still fitted online.\n''')
print('Patched notebook: all predictors, unit weights, ridge and lasso sweeps, two backtest conventions, secondary axes.')

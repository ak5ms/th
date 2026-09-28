"""One-time notebook migration, removed once the evaluated notebook is saved."""
from pathlib import Path
import nbformat as nbf

path = Path('notebooks/01_eda.ipynb')
nb = nbf.read(path, as_version=4)
nb.cells = [c for c in nb.cells if 'execution_summary' not in c.metadata.get('tags', [])]
assert not any(c.source.startswith('# Standalone feature importance') for c in nb.cells)
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
new = [
M(r'''# Standalone feature importance

Assess each signal on its own with the requested exposure $x/\sigma_x^2$. This is **standalone performance**, not its unique incremental contribution after controlling for the other, often correlated, features. No sign is flipped and no test data are accessed.

Use every training row and every `x` column. The half-life and minimum number of non-null observations are both `288*21 = 6048`. EWM uses pandas defaults (`adjust=True`, `ignore_na=False`, sample-standard-deviation estimate); the current signal is included in its own scale. Only zero-volatility divisions are changed to missing to avoid infinite positions. There is no fitted leverage cap.

Daily `sum()` is retained exactly as requested: an all-missing day (including warm-up) becomes zero. The histogram uses **unannualized calendar-day mean/std**, not annualized trading-day Sharpe. Unequal feature availability means these are not common-sample or execution-adjusted comparisons. The coverage table below makes the differing sample lengths visible.'''),
C('''from takehome.features import ts_std, ts_standardize, ts_zscore, sharpe, standalone_pnl

x_cols = df.head(0).filter(like='x').columns
HL = 288 * 21
# Equivalent to x.div(ts_std(x, HL)**2), then multiply returns by ROW.
pnl = standalone_pnl(df[x_cols], df['ret_5m'], hl=HL)
pnl_daily = pnl.resample('D').sum()
assert pnl.index.equals(df.index) and pnl.columns.equals(x_cols)
assert not np.isinf(pnl.to_numpy()).any()
pnl_daily.cumsum().plot(figsize=(14, 6), legend=False,
                       title='Standalone feature P&L: all training observations')
plt.ylabel('Cumulative diagnostic P&L (arbitrary exposure units)')
plt.show()
plt.close('all')
pnl_daily.pipe(sharpe).plot.hist(bins=25, figsize=(9, 4),
                               title='Standalone feature daily mean/std (not annualized)')
plt.xlabel('Daily mean / daily standard deviation')
plt.show()
plt.close('all')'''),
C('''standalone = pd.DataFrame({
    'daily_mean_over_std': pnl_daily.pipe(sharpe),
    'pnl_observations': pnl.count(),
    'days_with_observations': pnl.notna().resample('D').sum().gt(0).sum(),
    'first_pnl_msgStamp': pnl.apply(lambda x: x.first_valid_index()),
    'last_pnl_msgStamp': pnl.apply(lambda x: x.last_valid_index()),
}).sort_values('daily_mean_over_std', ascending=False)
with pd.option_context('display.max_rows', None):
    display(standalone)
print('Best/worst standalone DAILY mean/std:',
      standalone.daily_mean_over_std.max(), standalone.daily_mean_over_std.min())'''),
M(r'''## Why feature volatility can proxy asset volatility

For a signed standardized signal, the intended idealized exposure is

$$w_{j,t}=\frac{\mathrm{ts\_standardize}(x_j)_t}{\sigma_{r,t}}
=\frac{x_{j,t}}{\sigma_{x_j,t}\,\sigma_{r,t}},\qquad
\mathrm{PnL}_{j,t}=w_{j,t}r_{t\to t+h}.$$

`ts_std(signal) / asset_vol` would be a different, unsigned exposure: it removes the direction of the signal. The expression above uses **ts_standardize**, not **ts_std**, in the numerator. Use `ts_zscore` instead only when removing the local signal mean is intentional.

Prices/realized returns are withheld in the final deployment period, so realized asset volatility cannot be refreshed there. If $\sigma_{x_j,t}\approx k_j\sigma_{r,t}$ with reasonably stable $k_j$, then $w_{j,t}\approx k_j x_{j,t}/\sigma_{x_j,t}^2$. Dropping the constant gives the requested diagnostic. This is a proxy assumption, not an identity; a changing feature amplitude, family or regime can break it. Fixed positive scale does not change mean/std but changes P&L units.

Weighting can alternatively enter the fitter. For WLS, $H=X(X^\top W X)^{-1}X^\top W$ already includes the observation weights in $\widehat y=Hy$. However, weighting historical residuals does **not automatically implement the same forecast-time inverse-volatility exposure**. Lasso does not have one fixed, response-independent hat matrix: the active set depends on the target. The initial fitter below therefore documents its weighting objective explicitly.

Only the feature history enters the signal scale. Nevertheless, the alignment/availability of the supplied `ret_5m` must be established before these curves can be described as executable forward-return backtests. They contain no costs or execution lag.'''),
M(r'''# Initial fitter: online weighted lasso

Implementation: `src/takehome/fitters.py`. A small Python class wraps two Numba-compiled kernels rather than requiring a `jitclass`. It warm-starts each solve, stores $O(p^2)$ sufficient statistics, and retains coefficient history only when requested.

## Objective and W convention

At update $t$, define $a_{i,t}=d^{t-i}v_i$, $A_t=\sum_i a_{i,t}$ and weighted population feature standard deviation $s_{j,t}$. The exact objective is

$$\min_{b,\beta}\ \frac{1}{2A_t}\sum_{i\le t}a_{i,t}(y_i-b-x_i^\top\beta)^2
+\alpha\sum_j s_{j,t}|\beta_j|.$$

Default `fit_intercept=False` fixes $b=0$, preserving the supplied sketch's regression convention. `fit_intercept=True` instead centers the **loss as well as the scale** and estimates an unpenalized intercept. These are different models for features with nonzero means.

Here $v_i=(\mathrm{wmid}_i\,\mathrm{volume}_i)^2$ is the **diagonal entry of W**, not its square root; it is applied once. Pass a length-$n$ vector rather than constructing an $n\times n$ matrix. An explicitly diagonal matrix is also accepted; non-diagonal W is rejected. The latter requires cross-observation terms, not this streaming update. Multiplying all $v_i$ by a common positive constant leaves the objective unchanged because it is normalized by $A_t$.

This choice is **squared-notional weighting**, not automatically inverse-error-variance or asset-volatility weighting. It deliberately emphasizes high-notional observations; large weights can dominate the fit. A different weighting hypothesis should be tested separately.

## Sufficient statistics and coordinate update

In raw-moment notation, the updates are

$$A\leftarrow dA+v,\quad M\leftarrow dM+vx,\quad Y\leftarrow dY+vy,$$
$$S\leftarrow dS+vxx^\top,\qquad z\leftarrow dz+vxy.$$

Let $\mu_x=M/A$ and $\mu_y=Y/A$. Then $s_j^2=S_{jj}/A-\mu_{x,j}^2$.
Without an intercept use $G=S/A$, $h=z/A$; with one use $G=S/A-\mu_x\mu_x^\top$, $h=z/A-\mu_x\mu_y$, and recover $b=\mu_y-\mu_x^\top\beta$.

$$\beta_j\leftarrow\frac{\operatorname{soft}\!\left(h_j-\sum_{k\ne j}G_{jk}\beta_k,\ \alpha s_j\right)}{G_{jj}}.$$

Multiplying numerator/denominator by $A$ gives the supplied threshold $\alpha s_j A$. The implementation solves the same problem in standardized coefficient units, $\theta_j=s_j\beta_j$, to make stopping tolerances insensitive to feature units. Convergence checks **KKT residuals**, not just small coefficient changes; an iteration-budget failure is reported.

For numerical stability the code stores centered scatter $C$ and cross-scatter $c$ rather than subtracting large raw moments. With $A_0=dA$, $A'=A_0+v$, $\delta_x=x-\mu_x$, $\delta_y=y-\mu_y$, $k=A_0v/A'$, update

$$C\leftarrow dC+k\delta_x\delta_x^\top,\quad c\leftarrow dc+k\delta_x\delta_y,$$
$$\mu_x\leftarrow\mu_x+(v/A')\delta_x,\quad\mu_y\leftarrow\mu_y+(v/A')\delta_y.$$

These are algebraically the same sufficient statistics. A zero column gets coefficient zero. A nonzero constant column uses RMS scale when there is no intercept so that it is not accidentally unpenalized; with an intercept it is absorbed by the intercept. This fixes the zero-denominator/zero-penalty edge cases in the sketch.

## Timing, missingness and history

`fit` **appends** rows, matching repeated `partial_fit`; construct a fresh object to reset. Each call/row advances the decay clock. Positive-weight rows must have finite X and y. A zero-weight row adds no observation but still advances decay, so missing observations are not compressed out of time. There is no pairwise-missing covariance estimator or silent imputation.

For live forecasts, predict first and update only when the label has matured, using the features stored at the forecast's origin. The class cannot infer label availability from arrays. `get_coefs()` contains **post-update** coefficients and must not be multiplied by the just-fitted target to claim out-of-sample performance. During the price blackout supervised updates must stop; keep `store_history=False` for bounded memory.

## Mathematical verification

The tests compare weighted batch lasso at successive prefixes, the one-feature soft-threshold solution, and weighted least squares at alpha=0. They also cover weighted sufficient statistics, independent KKT conditions, diagonal-W/vector equivalence, common weight rescaling, feature-unit changes, intercept handling, constant/duplicate columns, chunking, zero-weight decay, future-data invariance, saved-state continuation and convergence failures.

References: [scikit-learn Lasso objective and sample-weight normalization](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html), [WLS weight convention](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.WLS.html), [pandas EWM conventions](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html).'''),
C('''# Independent batch reconciliation on a fixed synthetic example, not the holdout.
from sklearn.linear_model import Lasso
from takehome.fitters import StreamingWeightedLasso

rng = np.random.default_rng(42)
X_check = rng.normal(size=(256, 4)) * [1, 2, 3, 4] + .3
y_check = X_check @ np.array([.6, 0, -.3, .15]) + rng.normal(size=256) * .1
v_check = rng.lognormal(size=256)
decay, alpha = .99, .05
online = StreamingWeightedLasso(4, decay, alpha, max_iter=50000, tol=1e-11,
                               store_history=True).fit(X_check, y_check, W=v_check)
verification = []
for n in [32, 64, 128, 256]:
    a = v_check[:n] * decay ** np.arange(n-1, -1, -1)
    mean = np.average(X_check[:n], weights=a, axis=0)
    scale = np.sqrt(np.average((X_check[:n]-mean)**2, weights=a, axis=0))
    batch = Lasso(alpha=alpha, fit_intercept=False, max_iter=100000, tol=1e-12)
    batch.fit(X_check[:n] / scale, y_check[:n], sample_weight=a)
    difference = online.get_coefs()[n-1] - batch.coef_ / scale
    verification.append({'prefix_rows': n, 'max_abs_coefficient_error': abs(difference).max()})
verification = pd.DataFrame(verification)
display(verification)
assert verification.max_abs_coefficient_error.max() < 1e-7
assert online.n_failed_ == 0'''),
M('''## Initial fit on real training data (implementation demonstration)

Use the fixed first three signals, `x1`, `x2`, `x3`, to keep the initial fitter demonstration inspectable. This is not a feature-selected or optimized final model. Every training timestamp advances decay; rows with missing inputs/target/weight get zero observation weight. The many missing values across all 99 features require an explicit modeling policy before attempting the full joint fit.

The alpha below is a fixed demonstration value, not tuned on training P&L or the held-out block. We report coefficients and convergence, **not an in-sample Sharpe as a forecast result**. Fitter timing includes first-use compilation where applicable.'''),
C('''from time import perf_counter

fit_columns = ['x1', 'x2', 'x3']
X_fit = df[fit_columns].to_numpy(dtype=float)
y_fit = df['ret_5m'].to_numpy(dtype=float)
notional_weights = (df['wmid'] * df['volume']).pow(2).to_numpy(dtype=float)
valid_fit = (np.isfinite(X_fit).all(axis=1) & np.isfinite(y_fit)
             & np.isfinite(notional_weights) & (notional_weights > 0))
W_fit = np.where(valid_fit, notional_weights, 0.0)  # Preserve the full row clock.
fitter = StreamingWeightedLasso(len(fit_columns), decay=2**(-1/HL), alpha=1e-5,
                                max_iter=5000, tol=1e-8, fit_intercept=True)
started = perf_counter()
fitter.fit(X_fit, y_fit, W=W_fit)
fit_seconds = perf_counter() - started
assert fitter.n_seen_ == len(df) and df.index.max() < split['cutoff']
display(pd.Series(fitter.coef, index=fit_columns, name='coefficient').to_frame())
display(pd.Series({
    'training_rows_advanced': fitter.n_seen_, 'positive_weight_rows': int(valid_fit.sum()),
    'intercept': fitter.intercept_, 'final_KKT_residual': fitter.kkt_violation_,
    'final_KKT_tolerance': fitter.kkt_tolerance_, 'final_converged': fitter.converged_,
    'nonconverged_updates': fitter.n_failed_, 'fit_seconds': fit_seconds,
    'largest_raw_weight_share': W_fit.max() / W_fit.sum(),
}, name='initial_fitter_diagnostics').to_frame())'''),
]
i = next(i for i, c in enumerate(nb.cells) if c.source.startswith('## Execution environment'))
nb.cells[i:i] = new
for cell in nb.cells:
    if cell.source.startswith('## Execution environment'):
        cell.source = cell.source.replace('## Execution environment', '# Execution environment', 1)
    if "packages = ['numpy'" in cell.source:
        cell.source = cell.source.replace("'diptest', 'nbformat'", "'diptest', 'numba', 'scikit-learn', 'nbformat'")
    if cell.cell_type == 'code':
        cell.outputs, cell.execution_count, cell.metadata = [], None, {}
source = '\n'.join(c.source for c in nb.cells)
assert 'reports/' not in source and '.to_csv(' not in source
nbf.validate(nb)
nbf.write(nb, path)

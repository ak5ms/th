"""One-time notebook migration; removed after execution."""
from pathlib import Path
import nbformat as nbf

path = Path('notebooks/01_eda.ipynb')
nb = nbf.read(path, as_version=4)
nb.cells = [c for c in nb.cells if 'execution_summary' not in c.metadata.get('tags', [])]
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def cell_with(text):
    matches = [c for c in nb.cells if text in c.source]
    assert len(matches) == 1, (text, len(matches))
    return matches[0]


c = cell_with('cashflow_outliers =')
a = c.source.index('with pd.option_context(')
c.source = c.source[:a] + '''display(cashflow_outliers.sample(n=min(20, len(cashflow_outliers)), random_state=0).sort_index())
exception_dates = cashflow_outliers.groupby(cashflow_outliers.index.date).size().rename('exception_rows')
print(f'{len(exception_dates)} unique local dates; counts below cover ALL exceptions, not the sample.')
with pd.option_context('display.max_rows', None):
    display(exception_dates.to_frame())'''
for c in nb.cells:
    c.source = c.source.replace('complete cashflow-exception table', 'cashflow-exception sample and complete date counts')
    c.source = c.source.replace('full coverage and exception tables', 'full coverage, exception samples and date counts')
    c.source = c.source.replace('The complete named-variable exception table is displayed below, without row truncation. No CSV is exported.',
                                'Display a deterministic sample of at most 20 rows, followed by all distinct dates and their exception counts. Only the display is sampled; the predicate and counts use every training row. No CSV is exported.')

sessions = [
M('''## ES tradability: exchange schedule versus missing quotes

Estimate weekday/local-clock quote availability using **training wmid only**. Reindex onto a complete five-minute grid for this diagnostic so absent weekends count as absent, rather than disappearing from the denominator. This does not fill prices or alter the modeling frame.

The empirical gaps indicate the Sunday evening open, Friday evening close, daily maintenance and a historical afternoon pause. They also show boundary labels consistent with **end-labeled five-minute bars**: the first bar after 18:00 New York is 18:05. A closing-bar quote is not proof that a new trade can be entered after the close.

`is_tradable` is the **scheduled exchange-open instant** mask. `is_trading_bar` asks whether the entire trailing five-minute bar was within a scheduled open interval. `quote_available` is kept separate. Missing quotes are not automatically exchange closures, and the schedule alone does not capture outages, circuit breakers or contract-specific halts.

Schedule sources: [CME holiday calendar](https://www.cmegroup.com/trading-hours.html), [September 21, 2015 close change](https://www.cmegroup.com/tools-information/lookups/advisories/electronic-trading/20150817.html), [June 28, 2021 afternoon-pause removal](https://www.cmegroup.com/notices/electronic-trading/2021/06/20210621.html), and [December 5, 2018 abbreviated session](https://www.cmegroup.com/notices/clearing/2018/12/Chadv18-474.html). The implementation uses pinned `pandas_market_calendars` CME_Equity holiday rules with those dated corrections, not 2026 holiday dates projected backward. Remaining vendor/calendar discrepancies are shown rather than silently discarded; future holiday schedules should be checked against CME notices.

All times below are America/New_York. Before 2015-09-21, regular close was 17:15; thereafter 17:00. Before 2021-06-28, the 16:15–16:30 pause applied; thereafter it did not. Regular reopen is 18:00 Sunday through Thursday.'''),
C('''from takehome.sessions import is_tradable as es_is_tradable, es_schedule, availability_by_week

weekly_availability = availability_by_week(df['wmid'])
weekly = weekly_availability['has_wmid'].unstack('weekday').reindex(columns=range(7))
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(weekly.T, aspect='auto', origin='lower', extent=(0, 24, -.5, 6.5))
ax.set_yticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
ax.set_xlabel('New York clock hour'); ax.set_title('Training wmid availability on a complete five-minute grid')
fig.colorbar(im, ax=ax, label='Fraction of slots with wmid'); plt.show(); plt.close('all')

# A single pooled weekly rule would hide historical changes.
era_profiles = {}
for name, start, end in [('Before close change', '2014-01-01', '2015-09-20'),
                          ('After close change', '2015-09-21', '2021-06-27'),
                          ('After pause removal', '2021-06-28', '2022-07-14')]:
    q = df.loc[start:end, 'wmid']
    p = availability_by_week(q)
    era_profiles[name] = p.loc[p.index.get_level_values('weekday') < 5, 'has_wmid'].groupby('minute').mean()
boundaries = [16*60+15,16*60+20,16*60+30,16*60+35,17*60,17*60+5,17*60+15,17*60+20,18*60,18*60+5,19*60,20*60]
era_availability = pd.DataFrame(era_profiles).reindex(boundaries)
era_availability.index = [f'{m//60:02d}:{m%60:02d}' for m in boundaries]
display(era_availability)

is_tradable = es_is_tradable(df.index)
is_trading_bar = es_is_tradable(df.index, bar_end=True)
quote_available = df['wmid'].notna()
display(pd.crosstab(is_trading_bar.rename('scheduled_trading_bar'), quote_available.rename('wmid_present')))
session_diagnostics = pd.Series({
    'training_rows': len(df), 'scheduled_open_instants': int(is_tradable.sum()),
    'scheduled_trading_bars': int(is_trading_bar.sum()), 'quotes_present': int(quote_available.sum()),
    'missing_quotes_in_scheduled_bars': int((is_trading_bar & ~quote_available).sum()),
    'quotes_outside_scheduled_bars': int((~is_trading_bar & quote_available).sum()),
}, name='session_diagnostics')
display(session_diagnostics.to_frame())
calendar_mismatches = df.loc[~is_trading_bar & quote_available, ['wmid']]
print('Dates with the most quotes outside the schedule (audit, not automatic relabeling):')
display(calendar_mismatches.groupby(calendar_mismatches.index.date).size().nlargest(12).to_frame('rows'))
print('SESSION_DIAGNOSTICS'); print(session_diagnostics.to_string())
print('ERA_AVAILABILITY'); print(era_availability.to_string())'''),
M('''### Calendar audit and use

The recurring isolated missing slot around 19:00 winter / 20:00 summer New York corresponds to midnight UTC and looks like a data-production artifact, not a CME maintenance interval. Do not train an exchange calendar by treating every recurring NaN as a market closure.

The mask is defined for later fitting/execution workflows and can be generated without future prices. Existing standalone P&Ls and the ad hoc combination below are deliberately **not silently filtered** by this newly introduced mask, so their definitions remain comparable. To impose trading restrictions later, apply the schedule at the trade-decision/label interval, with explicit bar-label and fill conventions. The calendar is a scheduled-open model, not a guarantee of executable liquidity.'''),
]
i = next(i for i,c in enumerate(nb.cells) if c.source.startswith('## 1. Scatter matrix'))
nb.cells[i:i] = sessions

c = cell_with('del pnl')
c.source = c.source.replace('from takehome.features import dszl, pairwise_features, evaluate_features',
                            'from takehome.features import dszl, pairwise_features, evaluate_features, combine_pnls')
c.source = c.source.replace('del pnl', "META_HL = 252 * 288\nraw_meta = combine_pnls(pnl, hl=META_HL, positive_only=True)\ndel pnl")
for prefix in ['dszl', 'interaction', 'residual']:
    c = cell_with(f'{prefix}_daily, {prefix}_summary =')
    c.source = c.source.replace(f'{prefix}_daily, {prefix}_summary =',
                                f'{prefix}_daily, {prefix}_summary, {prefix}_meta =')
    c.source = c.source.replace("df['ret_5m'], hl=HL,", "df['ret_5m'], hl=HL, meta=True, meta_hl=META_HL,")

blend = [
M(r'''## Ad hoc combination of signals

Estimate each component's score from its **intrabar diagnostic P&L**, not daily P&L:

$$s_{j,t}=\frac{\mathrm{EWMmean}(p_j)_t}{\mathrm{EWMstd}(p_j)_t},\quad
m_{j,t}=\frac{s_{j,t}}{\sum_k |s_{k,t}|},\quad
p_t^{\rm meta}=\sum_j m_{j,t-1}p_{j,t}.$$

Use `halflife=252*288 = 72576` rows and pandas default EWM warm-up. Clip negative scores only for the **raw baseline**; keep signed scores for dszl, products and residuals. Scores with zero/undefined volatility get zero allocation. A zero gross denominator means no position. This fixes the sketch's invalid `hl=` EWM keyword and makes row-wise normalization and the final sum across signals explicit.

```python
moments = pnls.ewm(halflife=252*288)
score = moments.mean().div(moments.std().replace(0, np.nan)).fillna(0)
# Raw signals only: score = score.clip(lower=0)
meta_w = score.div(score.abs().sum(axis=1).replace(0, np.nan), axis=0)
combined = meta_w.shift().mul(pnls).sum(axis=1)
combined.resample('D').sum().cumsum().plot()
```

Pair computations normalize across **all candidates**, not separately inside each eight-column batch. The additive lagged numerator/gross calculation is algebraically identical to the dense formula and is tested against it. Missing current P&L contributes zero without redistributing its previously chosen allocation. Missing values do not count as artificial zero observations when estimating EWM moments.

These are training-only, ad hoc combinations. There is no alpha-sign selection using the full sample, no cost model, no covariance adjustment, and no holdout access. One-row lag is adequate only if the component's preceding P&L is actually observable then; `ret_5m` label availability remains a separate contract. An in-sample adaptive baseline is not a validated out-of-sample Sharpe. The half-life is in rows, not exactly a calendar year.'''),
C('''meta_pnls = pd.concat({
    'Raw (positive scores)': raw_meta, 'dszl(10) (signed)': dszl_meta,
    'Products (signed)': interaction_meta, 'Residuals (signed)': residual_meta,
}, axis=1)
assert meta_pnls.index.equals(df.index) and np.isfinite(meta_pnls.to_numpy()).all()
meta_daily = meta_pnls.resample('D').sum()
meta_daily.cumsum().plot(figsize=(12, 5), title='Ad hoc lagged EWM-Sharpe signal combinations')
plt.ylabel('Cumulative diagnostic P&L (arbitrary exposure units)'); plt.show(); plt.close('all')
meta_summary = pd.DataFrame({
    'daily_mean_over_std': sharpe(meta_daily), 'daily_mean': meta_daily.mean(),
    'daily_std': meta_daily.std(), 'nonzero_bars': meta_pnls.ne(0).sum(),
    'cumulative_pnl': meta_daily.sum(),
})
display(meta_summary)
print('META_COMBINATIONS'); print(meta_summary.to_string())'''),
]
i = next(i for i,c in enumerate(nb.cells) if c.source.startswith('# Initial fitter:'))
nb.cells[i:i] = blend

c = cell_with('X_check =')
c.source = '''# Independent CVXPY loss at identical prefixes; no Numba Gram/CD reuse.
from takehome.fitters import StreamingWeightedLasso, CvxpyWeightedLasso, BatchedFitters

rng = np.random.default_rng(42)
X_check = rng.normal(size=(256, 4)) * [1, 2, 3, 4] + .3
y_check = .2 + X_check @ np.array([.6, 0, -.3, .15]) + rng.normal(size=256) * .1
v_check = (rng.uniform(1800, 2200, 256) * rng.lognormal(4, 1, 256))**2
decay, alpha = .99, .05
verification = []
for with_intercept in [False, True]:
    online = StreamingWeightedLasso(4, decay, alpha, max_iter=50000, tol=1e-11,
                                    fit_intercept=with_intercept, store_history=True).fit(X_check, y_check, W=v_check)
    for n in [32, 64, 128, 256]:
        ref = CvxpyWeightedLasso(4, decay, alpha, fit_intercept=with_intercept, tol=1e-12)
        ref.fit(X_check[:n], y_check[:n], W=v_check[:n])
        beta, bias = online.get_coefs()[n-1], online.get_intercepts()[n-1]
        a = v_check[:n] * decay**np.arange(n-1, -1, -1); a /= a.sum()
        loss_online = .5 * (a @ (y_check[:n] - X_check[:n] @ beta - bias)**2)
        loss_online += alpha * np.sum(ref.scale_ * np.abs(beta))
        verification.append({
            'intercept': with_intercept, 'prefix_rows': n,
            'max_coefficient_error': np.max(np.abs(beta-ref.coef)),
            'max_prediction_error': np.max(np.abs(X_check[:n] @ beta+bias-ref.predict(X_check[:n]))),
            'objective_gap': abs(loss_online-ref.objective_), 'cvxpy_status': ref.status_,
        })
    assert online.n_failed_ == 0
verification = pd.DataFrame(verification)
with pd.option_context('display.float_format', '{:.3e}'.format):
    display(verification)
print('CVXPY_RECONCILIATION'); print(verification.to_string(index=False))
assert verification.max_coefficient_error.max() < 5e-7
assert verification.max_prediction_error.max() < 2e-6
assert verification.objective_gap.max() < 1e-9'''

batch = [
M(r'''## Harmonized batch and online fitting

`CvxpyWeightedLasso` constructs the loss from the original observations with an independent CVXPY/Clarabel solve. It does not call Numba or reuse its sufficient-statistic implementation. The exact weighted loss remains

$$\frac12\sum_i q_i(y_i-b-x_i^\top\beta)^2+\alpha\sum_j s_j|\beta_j|,
\qquad q_i=\frac{v_i d^{t-i}}{\sum_k v_kd^{t-k}}.$$

For conditioning, let $u$ be weighted response RMS (centered when fitting an intercept), $\theta_j=s_j\beta_j/u$, and use standardized design $Z$. Dividing the whole objective by $u^2$ gives the implemented expression:

```python
residual = target / unit - Z @ theta - offset
loss = cp.sum_squares(cp.multiply(np.sqrt(q), residual)) / 2
objective = loss + alpha / unit * cp.norm1(theta)
```

The original-unit coefficients, intercept and objective are returned. Zero-weight rows are excluded from the loss but retain their age in the exponential decay clock. Constant-column scale fallbacks match the declared Numba objective. Duplicate predictors may have nonunique coefficients, so their **predictions and objective values**, not coefficient identities, are compared. Solver statuses other than `optimal` raise rather than silently passing verification. The tests now use CVXPY instead of sklearn, including prefix, weight, scaling and constant-column tests.

`BatchedFitters` is a small schedule adapter around any fit-overwrites estimator exposing `n_features`, `fit(X,y,W)`, `coef`, and `intercept_`. `min_train_size` starts fitting, `step` is the refit/forecast-chunk size, `train_size=None` expands the window, and `gap` embargoes additional rows. `fit` rebuilds a batch path; the streaming class's `fit` still appends.

Both paths expose `get_coefs(lag=0)` / `get_intercepts(lag=0)` as **post-update histories**, preserving the old API. Use `lag=1` for next-row application, or a larger availability delay if required. Pre-fit rows are NaN. `predict(X_new)` uses the final fitted snapshot; do not use it to backfill a walk-forward forecast path. The example below shows explicit per-row prediction from the lagged histories.

Reference: [CVXPY lasso example](https://www.cvxpy.org/examples/machine_learning/lasso_regression.html). The reference solver is for verification and coarser refits, not for millions of per-row generic convex solves.'''),
C('''batch_model = BatchedFitters(
    CvxpyWeightedLasso(4, decay=.99, alpha=.05, fit_intercept=True),
    min_train_size=32, step=16, train_size=96, gap=1,
).fit(X_check, y_check, W=v_check)
beta_path = batch_model.get_coefs(lag=1)
bias_path = batch_model.get_intercepts(lag=1)
walk_forward_predictions = np.einsum('ij,ij->i', X_check, beta_path) + bias_path
folds = pd.DataFrame(batch_model.folds_)
display(folds.head(8))
print('BATCHED_FITTER:', len(folds), 'fits;', np.isfinite(walk_forward_predictions).sum(), 'prequential predictions')
assert (folds.train_stop + 1 <= folds.predict_start).all()
assert np.isnan(walk_forward_predictions[:33]).all()'''),
]
i = next(i for i,c in enumerate(nb.cells) if c.source.startswith('## Initial fit on real training'))
nb.cells[i:i] = batch
for c in nb.cells:
    c.source = c.source.replace('[scikit-learn Lasso objective and sample-weight normalization](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)',
                                '[CVXPY lasso loss](https://www.cvxpy.org/examples/machine_learning/lasso_regression.html)')
    c.source = c.source.replace("'scikit-learn'", "'cvxpy', 'clarabel', 'pandas_market_calendars'")
    if c.cell_type == 'code':
        c.outputs, c.execution_count, c.metadata = [], None, {}
source = '\n'.join(c.source for c in nb.cells)
assert 'from sklearn' not in source and 'import sklearn' not in source
assert 'reports/' not in source and '.to_csv(' not in source
nbf.validate(nb); nbf.write(nb, path)

readme = Path('README.md')
s = readme.read_text().replace('complete cashflow-exception table', 'cashflow-exception sample and complete date counts')
s += '''\n## Fitting and baseline interfaces\n\n`fitters.py` includes the Numba streaming estimator, independent `CvxpyWeightedLasso`, and the `BatchedFitters` walk-forward adapter. Both history interfaces accept `lag=1` for prior-estimate alignment. Tests reconcile original-unit objectives and predictions against CVXPY, not sklearn. `features.combine_pnls` computes a lagged, globally gross-normalized intrabar EWM-Sharpe blend, also inside column-batched pair evaluation. `sessions.py` separates exchange-open instants, trailing trading bars and empirical quote availability using CME holiday rules and dated historical-hour corrections. See the notebook for the calendar audit and limitations.\n'''
readme.write_text(s)
print('Notebook updated: CVXPY verification, batched fitting, blends, sessions, concise exceptions.')

"""One-time notebook migration; deleted after evaluation."""
from pathlib import Path
import nbformat as nbf

path = Path('notebooks/01_eda.ipynb')
nb = nbf.read(path, as_version=4)
nb.cells = [c for c in nb.cells if 'execution_summary' not in c.metadata.get('tags', [])]
assert not any('### Time-of-day standardization' in c.source for c in nb.cells)
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
new = [
M(r'''## Transform experiments

In this section **display results** means a separate cumulative daily P&L figure and a separate histogram of daily mean/std for each experiment. Each transformation is applied to the **raw** `x_cols`, independently of the others, before the same exposure rule:

$$z_t=\mathrm{transform}(x)_t,\qquad w_t=z_t/\widehat{\sigma}_{z,t}^{\,2},\qquad \mathrm{PnL}_t=w_t\,\mathrm{ret\_5m}_t.$$

The downstream EWM half-life and minimum observations remain `HL=288*21`. All training rows are used. Pairwise expansions are processed in small **column batches**, retaining only daily P&L and coverage, rather than materializing billions of intrabar values. No row or pair subsampling, sign flipping, or holdout access occurs. The input `ret_5m` is used only to evaluate P&L, never to build a transformation.

Warm-up/missing observations and zero variance stay missing before the same daily `sum()` convention. All-missing days become zero, undefined Sharpes are omitted from histograms and counted, and near-duplicate residuals can amplify numerical noise through the inverse-variance exposure. These are **standalone training diagnostics**, not estimates of a feature's unique contribution to a joint model. Many more pair candidates also create many more opportunities to obtain a good-looking training result by chance.'''),
C('''from time import perf_counter
from takehome.features import dszl, pairwise_features, evaluate_features
from takehome.plots import display_results

# The baseline table/daily series are retained; its large intrabar matrix is no longer needed.
del pnl
raw_features = df[x_cols]
experiment_seconds = {}
assert raw_features.index.equals(df.index) and df.index.max() < split['cutoff']'''),
M(r'''### Time-of-day standardization — dszl(hl=10)

Use `index.time` (local clock time in the supplied timezone), and standardize **the group**, not the outer DataFrame:

```python
X.groupby(X.index.time, group_keys=False).apply(
    lambda group: ts_standardize(group, hl=10)
).reindex(X.index)
```

There is no demeaning. Half-life 10 means **ten occurrences of the same clock time**, not ten adjacent five-minute bars; each group needs ten non-null values. A second `HL=6048` EWM standard deviation is then applied to the transformed signal by the existing P&L rule. The current feature observation is included in its feature-only scale, as for the raw baseline. Reindexing restores chronological order without filling missing values.

Reference: [pandas EWM weighting and missing-value conventions](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html).'''),
C('''started = perf_counter()
dszl_daily, dszl_summary = evaluate_features(
    [dszl(raw_features, hl=10)], df['ret_5m'], hl=HL,
)
experiment_seconds['dszl(10)'] = perf_counter() - started
assert dszl_daily.shape[1] == len(x_cols)
display_results(dszl_daily, 'Time-of-day standardization: hl=10')
print('Top 10 by training daily mean/std; coverage differs by feature.')
display(dszl_summary.head(10))'''),
M(r'''### Two-way interactions — products of distinct raw features

Use $z_{ij,t}=x_{i,t}x_{j,t}$ for every $i<j$: **4,851 products** for 99 predictors. Squares and duplicate reverse-order products are excluded. These are raw products, not products of the dszl signals and not demeaned products. Missing either input makes the product missing. The following P&L scaling uses the **product's own** EWM variance.'''),
C('''started = perf_counter()
interaction_daily, interaction_summary = evaluate_features(
    pairwise_features(raw_features, kind='product', batch_size=8),
    df['ret_5m'], hl=HL,
)
experiment_seconds['Raw products'] = perf_counter() - started
assert interaction_daily.shape[1] == len(x_cols) * (len(x_cols) - 1) // 2
display_results(interaction_daily, 'Two-way raw-feature interactions')
print('Top 10 of all distinct products; this is a training screen, not feature selection.')
display(interaction_summary.head(10))'''),
M(r'''### Pairwise residuals — online lasso with alpha=0, W=1

Use both directions for every pair: **9,702 residual series**. The name `x_i~x_j` means the residual of raw `x_i` regressed on raw `x_j`, with an intercept. Its reverse is a different experiment.

The existing `StreamingWeightedLasso` is used with `n_features=1`, `alpha=0` (the implementation's regularization parameter), `fit_intercept=True`, and `decay=2**(-1/6048)`. Each jointly finite pair receives unit observation weight; missing rows receive zero weight but still advance the original row clock. No price, return or notional weights enter these regressions.

For a residual at row $t$, fit only earlier jointly observed raw pairs:

$$ (a_{t-1},b_{t-1})=\arg\min_{a,b}\sum_{s<t}d^{t-1-s}I_s(x_{i,s}-a-bx_{j,s})^2,$$
$$z_{i\mid j,t}=x_{i,t}-a_{t-1}-b_{t-1}x_{j,t}.$$

Require two prior joint observations; the downstream P&L still needs 6,048 non-null residual observations. The shifted coefficient/intercept histories ensure the current response does not fit away its own residual. This differs from using post-update/in-sample regression residuals. It removes the historical relationship with **one** other predictor, not all predictors jointly.

For one predictor and `alpha=0`, the fitter uses the exact scalar solution $b=C_{xy}/C_{xx}$, $a=\bar y-b\bar x$, from the same weighted centered sufficient statistics. This is a fast path inside the existing fitter, not an approximate substitute or a separate estimator. Tests reconcile it with direct weighted least squares and the previous coordinate-descent path, including skipped rows, intercepts, and pre-update timing.'''),
C('''started = perf_counter()
residual_daily, residual_summary = evaluate_features(
    pairwise_features(raw_features, kind='residual', hl=HL, batch_size=8),
    df['ret_5m'], hl=HL,
)
experiment_seconds['Pair residuals'] = perf_counter() - started
assert residual_daily.shape[1] == len(x_cols) * (len(x_cols) - 1)
display_results(residual_daily, 'Pairwise residuals: alpha=0, W=1, hl=6048')
print('Top 10 of both directed residuals; near-collinear pairs require numerical caution.')
display(residual_summary.head(10))'''),
M('''### Training-only comparison

Compare the distributions, not only their maxima: raw predictors have 99 candidates, whereas products/residuals have thousands. These are unannualized calendar-day mean/std ratios under the requested missing-day convention. Availability differs; cumulative P&L units also differ after nonlinear transformations. This table does not establish out-of-sample improvement. No test data have been used.'''),
C('''experiment_summaries = {
    'Raw baseline': standalone, 'dszl(10)': dszl_summary,
    'Raw products': interaction_summary, 'Pair residuals': residual_summary,
}
comparison_rows = {}
for name, table in experiment_summaries.items():
    s = table['daily_mean_over_std'].dropna()
    comparison_rows[name] = {
        'candidates': len(table), 'defined_sharpes': len(s),
        'median_daily_sharpe': s.median(), 'p95_daily_sharpe': s.quantile(.95),
        'max_daily_sharpe': s.max(), 'min_daily_sharpe': s.min(),
        'positive_fraction': s.gt(0).mean(), 'compute_seconds': experiment_seconds.get(name, np.nan),
    }
transformation_comparison = pd.DataFrame.from_dict(comparison_rows, orient='index')
display(transformation_comparison)
print(transformation_comparison.to_string())
print('Train-only rows:', len(df), '; final training timestamp:', df.index.max())
for name, table in experiment_summaries.items():
    print('\\n', name, 'top 5 (training diagnostics)')
    print(table.head(5).to_string())
assert len(df) == split['train_rows'] and df.index.max() < split['cutoff']'''),
]
i = next(i for i, c in enumerate(nb.cells) if c.source.startswith('# Initial fitter:'))
nb.cells[i:i] = new
for c in nb.cells:
    c.source = c.source.replace('A small Python class wraps two Numba-compiled kernels',
                                'A small Python class wraps Numba-compiled kernels')
    if c.cell_type == 'code':
        c.outputs, c.execution_count, c.metadata = [], None, {}
source = '\n'.join(c.source for c in nb.cells)
assert 'reports/' not in source and '.to_csv(' not in source
nbf.validate(nb)
nbf.write(nb, path)
readme = Path('README.md')
readme.write_text(readme.read_text() + '''\n## Standalone transformation experiments\n\nThe notebook separately displays cumulative P&L and a daily-Sharpe histogram for time-of-day standardization, every distinct raw pair product, and both directed pair residuals. `features.py` holds `dszl`, `pair_residual`, `pairwise_features` and `evaluate_features`; `plots.py` provides `display_results`. Pairs are column-batched, not row-sampled. Residuals use the existing lasso at `alpha=0`, `W=1`, with an intercept and pre-update predictions. All output stays inside the evaluated notebook.\n''')
print('Added three independent transformation experiments; no external reports.')

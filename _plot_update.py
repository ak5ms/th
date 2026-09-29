"""One-time notebook patch; deleted after evaluated outputs are committed."""
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


# Calculate the raw combination before its panel, instead of after the panel.
raw = cell_with('pnl = standalone_pnl')
plot_start = raw.source.index('pnl_daily.cumsum().plot(')
raw.source = raw.source[:plot_start] + '''from takehome.features import combine_pnls
from takehome.plots import display_results

META_HL = 252 * 288
raw_meta = combine_pnls(pnl, hl=META_HL, positive_only=True)
display_results(pnl_daily, 'Raw standalone features (nonnegative meta scores)',
                meta_daily=raw_meta.resample('D').sum())'''
raw.metadata['tags'] = ['family_overlay_raw']
setup = cell_with('del pnl')
old = "META_HL = 252 * 288\nraw_meta = combine_pnls(pnl, hl=META_HL, positive_only=True)\n"
assert old in setup.source
setup.source = setup.source.replace(old, '')

for prefix, title in [
    ('dszl', 'Time-of-day standardization: hl=10'),
    ('interaction', 'Two-way raw-feature interactions'),
    ('residual', 'Pairwise residuals: alpha=0, W=1, hl=6048'),
]:
    c = cell_with(f'display_results({prefix}_daily,')
    old = f"display_results({prefix}_daily, '{title}')"
    assert old in c.source
    c.source = c.source.replace(old, f"display_results({prefix}_daily, '{title}',\n                meta_daily={prefix}_meta.resample('D').sum())")
    c.metadata['tags'] = [f'family_overlay_{prefix}']

cell_with('# Standalone feature importance').source += r'''

**Purple overlay:** each feature-family P&L panel includes its own lagged EWM-Sharpe combination, on the same axis and in the same exposure units as its components. Thin lines are individual portfolios; the thicker purple line is the actual combination, not a smoothed or rescaled illustration. Meta scores use intrabar P&L, half-life `252*288`, gross normalization across that whole family, and a one-row lag. Clip negative scores only for raw signals; allow signed scores for all transformed families. A normalized combination is not a sum of all member P&Ls and need not sit near the best individual line. No future performance is used to resize the overlay.'''
cell_with('## Transform experiments').source += '\n\nEach panel overlays the combination of that family only. The histogram continues to describe individual feature Sharpes; it does not add the combined portfolio as another feature.'

# Retain the comparison table, not another cross-family P&L chart with unlike units.
c = cell_with('meta_pnls = pd.concat')
a, b = c.source.index('meta_daily.cumsum().plot('), c.source.index('meta_summary =')
c.source = c.source[:a] + c.source[b:]
c = cell_with('## Ad hoc combination of signals')
c.source = c.source.replace('## Ad hoc combination of signals', '## Ad hoc lagged EWM-Sharpe combination summary', 1)
c.source += '\n\nThe actual combined P&L is highlighted in purple in each corresponding family panel above. This section retains the numerical cross-family comparison; raw cumulative P&L magnitudes are not comparable across different exposure units.'

# Retaining three coefficient histories changes storage, not the fitted objective.
real = cell_with('X_fit = df[fit_columns]')
old = 'tol=1e-8, fit_intercept=True)'
assert old in real.source and 'store_history=True' not in real.source
real.source = real.source.replace(old, 'tol=1e-8, fit_intercept=True, store_history=True)')
i = nb.cells.index(real) + 1
nb.cells[i:i] = [
    M(r'''## Coefficient paths and lagged forecast calibration

This is the existing weighted online lasso on the fixed raw columns `x1`, `x2`, `x3`, with the same half-life, penalty and `(wmid * volume)**2` training weights. Enabling coefficient history does not change fitting. The coefficient chart shows the post-update estimates $\beta_t$; prediction at row $t$ uses the **previous row's coefficients and intercept**:

$$\hat y_t=b_{t-1}+\sum_j\beta_{j,t-1}x_{j,t}.$$

`beta.shift().mul(x)` is summed across feature columns. `min_count=p` prevents a missing predictor from silently becoming a partial forecast, and the intercept is shifted too. Rows before the first positive-weight update are unforecastable. There is no extra warm-up or P&L-based selection; the scatter includes **every finite prediction/target pair in training**, with no row subsampling. No held-out data are used.

The plotted calibration regression is **y on yhat**, with an intercept:

$$y_t=a_{\rm cal}+b_{\rm cal}\hat y_t+e_t,\qquad
b_{\rm cal}=\frac{\sum_t(\hat y_t-\overline{\hat y})(y_t-\bar y)}{\sum_t(\hat y_t-\overline{\hat y})^2}.$$

The ideal diagnostic has intercept 0 and slope 1. For a positive slope, values below 1 suggest predictions are too large; values above 1 suggest they are too small. A negative slope suggests reversed direction, not merely a scale error. A constant forecast has an undefined slope. The figure also shows a calibration slope under the same observation weights as the fitter, because unweighted and weighted calibration answer different questions.

A lagged, exponentially weighted **lasso** does not algebraically guarantee calibration slope 1. Shrinkage, changing coefficients, nonstationarity, unequal observation weights, and incorrect return alignment can all affect the result. These are diagnostics, not evidence of a solver bug by themselves; CVXPY reconciliation above tests numerical implementation separately. **No post-hoc rescaling is applied** to force slope 1. Using this calibration fit to adjust predictions would itself require a separate causal fit/validation design.

A one-row lag assumes the previous row's label is observable at the next decision. The actual `ret_5m` maturity and execution convention still need verification; these are training-period sequential diagnostics, not claimed deployable returns.'''),
    C('''beta = pd.DataFrame(fitter.get_coefs(), index=df.index, columns=fit_columns)
intercept = pd.Series(fitter.get_intercepts(), index=df.index, name='intercept')
# Post-update coefficients include y_t: shift BOTH coefficients and intercept.
yhat = beta.shift().mul(df[fit_columns]).sum(axis=1, min_count=len(fit_columns))
yhat = yhat.add(intercept.shift()).rename('yhat')
prior_updates = pd.Series(valid_fit, index=df.index).cumsum().shift(fill_value=0)
yhat = yhat.where(prior_updates.gt(0))''', metadata={'tags': ['lagged_regression_predictions']}),
    C('''beta.plot(figsize=(14, 5), title='Online weighted lasso: post-update coefficient paths')
plt.ylabel('Coefficient in raw-feature units')
plt.xlabel('msgStamp')
plt.show()
plt.close('all')''', metadata={'tags': ['regression_beta_plot']}),
    C('''from takehome.plots import plot_calibration

calibration = plot_calibration(
    yhat, df['ret_5m'], weights=pd.Series(W_fit, index=df.index),
    title='Online weighted lasso: previous-row forecasts vs training targets',
)
with pd.option_context('display.float_format', '{:.6g}'.format):
    display(calibration)
print('LAGGED_FORECAST_CALIBRATION'); print(calibration.to_string())
print('Rows with no complete lagged forecast/target pair:',
      len(df) - int(calibration.loc['OLS', 'n']))
assert yhat.index.equals(df.index) and df.index.max() < split['cutoff']''',
      metadata={'tags': ['regression_calibration_plot']}),
]

for c in nb.cells:
    if c.cell_type == 'code':
        c.outputs, c.execution_count = [], None
        c.metadata.pop('execution', None)
source = '\n'.join(c.source for c in nb.cells)
assert source.count('meta_daily=') == 4
assert source.count('raw_meta = combine_pnls') == 1
assert 'reports/' not in source and '.to_csv(' not in source
nbf.validate(nb)
nbf.write(nb, path)
readme = Path('README.md')
readme.write_text(readme.read_text() + '''\n## Overlay and calibration diagnostics\n\nEach feature-family P&L panel highlights its own lagged EWM-Sharpe combination in purple, without display rescaling. The real-data regression shows coefficient histories and a full-training scatter of prior-row forecasts against targets, including OLS and W-weighted calibration slopes. Both coefficient and intercept histories are lagged. Calibration is diagnostic only: predictions are not rescaled to force slope one.\n''')
print('Notebook updated: four family overlays and strictly lagged regression plots.')

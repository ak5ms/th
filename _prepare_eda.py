"""One-time notebook migration; removed after the evaluated notebook is saved."""
from pathlib import Path
import nbformat as nbf

path = Path('notebooks/01_eda.ipynb')
nb = nbf.read(path, as_version=4)
assert len(nb.cells) == 20 and 'Cashflow' in nb.cells[16].source
nb.cells[0].source = '''# ESc1 signal EDA — training data only

All diagnostics below use the complete training portion, indexed by `msgStamp`. The final 20% of the labeled **elapsed time span** is reserved before any EDA. There is no row subsampling.

Earlier notebook versions inspected the full dataset. This split is excluded from this run and future model selection, but is not a retrospectively pristine holdout. Correlations and serial-correlation tests are descriptive, not evidence of out-of-sample predictability.'''
nb.cells[1].source = '''from pathlib import Path
import json, os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy.cluster.hierarchy import dendrogram

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src' / 'takehome').exists())
sys.path.insert(0, str(ROOT / 'src'))
from takehome.data import column_coverage, training_data
from takehome.plots import heatmap
from takehome.eda import (
    acf_matrix, autocorrelation_table, bell_shape_table, correlation_clustering,
    correlation_to_column, infer_feature_cols, monthly_shift_scores, redundant_pairs,
)

DATA_PATH = Path(os.environ.get('DATA_PATH', ROOT / 'ESc1_signal_components_5min (6).parquet'))
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)
SCATTER_N_FEATURES = 9  # Limit displayed columns, never observations.
ACF_LAG, SEASONAL_NLAGS = 15, 288'''
nb.cells[2].source = '''## 0. Load, set msgStamp index, reserve the test span

Let `first` and `last` be the first and last non-null `ret_5m` timestamps. The boundary is `first + 0.8 * (last - first)`. Training is `[first, boundary)` and the reserved test block is `[boundary, last]`. This is a time split, so row percentages need not equal 80/20. Outside-span unlabeled rows are excluded; internal missing observations are retained.'''
nb.cells[3].source = '''raw = pd.read_parquet(DATA_PATH)
if raw.index.name and raw.index.name not in raw.columns:
    raw = raw.reset_index()
raw = raw.set_index('msgStamp')
if not isinstance(raw.index, pd.DatetimeIndex):
    if pd.api.types.is_numeric_dtype(raw.index.dtype):
        raise ValueError('Numeric msgStamp: specify its epoch unit explicitly before converting.')
    raw.index = pd.to_datetime(raw.index, errors='raise')
raw = raw.sort_index()
df, split = training_data(raw)
del raw  # Do not retain a full-sample frame for later cells to accidentally use.

assert df.index.name == 'msgStamp'
assert len(df) == split['train_rows'] and df.index.max() < split['cutoff']
print(split.to_string())
print(f'Training timestamps: {df.index.min()} through {df.index.max()}')
(REPORTS / 'split.json').write_text(json.dumps(split.to_dict(), default=str, indent=2))
feature_cols = infer_feature_cols(df)
price_col, cashflow_col, time_col = 'adjMid', 'cashflow', 'msgStamp'
print(f'Training shape: {df.shape}; features: {len(feature_cols)}')
display(df.head())'''
nb.cells[4].source = '''## 1. Scatter matrix

Nine representative feature columns are shown for readability, using **all their training observations**. Missing pairs are omitted by the plotting routine; rows are not sampled. The all-feature correlation matrix below covers all feature pairs.'''
nb.cells[5].source = '''scatter_features = [c for c in feature_cols if df[c].nunique() > 1][:SCATTER_N_FEATURES]
scatter_matrix(df[scatter_features], figsize=(14, 14), diagonal='hist',
               alpha=.08, s=.2, rasterized=True)
plt.suptitle(f'All {len(df):,} training rows; {len(scatter_features)} feature columns', y=.995)
plt.show()
plt.close('all')'''
nb.cells[6].source = '''## 2. Distribution-shape screen and return sanity check

No Gaussianity test. The full-sample screen flags a dip statistic above 0.01, absolute Bowley quartile skew above 0.25, an exact point mass above 10%, or fewer than 20 unique values. These are practical degeneracy thresholds, not calibrated hypothesis tests; a pass does not prove a bell shape. High kurtosis is deliberately allowed.

Using the [Hartigan dip statistic](https://github.com/RUrlus/diptest) rather than its iid p-value avoids both a sample cap and interpreting tiny large-sample departures as useful rejections. Serially dependent features are not iid.

`adjMid.pct_change(fill_method=None)` is a contemporaneous sanity check. The supplied `ret_5m` is also reported separately; its alignment is not assumed to match that price change. All values and sample counts below are training-only.'''
nb.cells[7].source = '''shape = bell_shape_table(df, feature_cols)
flagged_shape = shape.loc[~shape.bell_like]
display(shape)
print(f'Pass shape screen: {shape.bell_like.sum()}/{len(shape)}; flagged: {len(flagged_shape)}')
assert shape.set_index('feature')['n'].reindex(feature_cols).eq(df[feature_cols].count()).all()
for col in flagged_shape.feature.head(6):
    df[col].plot.hist(bins=80, density=True, figsize=(8, 4), title=f'{col}: all training observations')
    plt.show()
    plt.close('all')

df['_ret1'] = df.adjMid.pct_change(fill_method=None)
return_corr = correlation_to_column(df, '_ret1', feature_cols)
provided_corr = correlation_to_column(df, 'ret_5m', feature_cols)
strong_target = return_corr.loc[return_corr.max_abs_corr >= .10]
display(strong_target)
print('Top correlations to supplied ret_5m (not assumed contemporaneous):')
display(provided_corr.head(15))
return_corr.head(20).sort_values('max_abs_corr').plot.barh(
    x='feature', y=['pearson', 'spearman'], figsize=(9, 7),
    title='Training features vs adjMid.pct_change()')
plt.axvline(0, linewidth=1)
plt.show()
plt.close('all')'''
nb.cells[9].source = '''corr, z, order = correlation_clustering(df, feature_cols, method='spearman')
plt.figure(figsize=(14, 5))
dendrogram(z, labels=feature_cols, leaf_rotation=90, leaf_font_size=6)
plt.title('Full-training feature dendrogram: 1 - |Spearman correlation|')
plt.tight_layout()
plt.show()
plt.close('all')
heatmap(corr.loc[order, order], 'Clustered training correlation matrix', labels=False, limits=(-1, 1))
plt.close('all')
upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
pairs = redundant_pairs(corr, threshold=.8)
display(pairs.head(30))
gap_pairs = pairs.dropna(subset=['index_gap']).copy()
gap_pairs['index_gap'] = gap_pairs.index_gap.astype(int)
all_pairs = redundant_pairs(corr, threshold=0).dropna(subset=['index_gap'])
all_pairs['index_gap'] = all_pairs.index_gap.astype(int)
gap_summary = all_pairs.groupby('index_gap').agg(
    n_pairs=('abs_corr', 'size'), median_abs_corr=('abs_corr', 'median'),
    p90_abs_corr=('abs_corr', lambda x: x.quantile(.9)),
    frac_gt_080=('abs_corr', lambda x: (x >= .8).mean()),
).sort_values('median_abs_corr', ascending=False)
all_pairs['same_mod3'] = all_pairs.index_gap.mod(3).eq(0)
mod3_summary = all_pairs.groupby('same_mod3').abs_corr.agg(['count', 'mean', 'median'])
display(gap_summary.head(20), mod3_summary)'''
nb.cells[10].source = '''## 4. Full-training autocorrelation

Ljung–Box(15) is applied directly to each feature, not regression residuals. Display and rank raw statistics, ACF(1), and per-feature observation counts; Q also grows with sample size. Missing values are omitted per feature: these are lags between successive observed values, not guaranteed five-minute clock intervals.'''
nb.cells[11].source = '''auto = autocorrelation_table(df, feature_cols, lag=ACF_LAG)
assert auto.set_index('feature')['n'].reindex(feature_cols).eq(df[feature_cols].count()).all()
display(auto.head(20))
print(f'Usable features: {auto.lb_stat.notna().sum()}/{len(auto)}')
print(f'Ljung-Box({ACF_LAG}), FDR q<5%: {(auto.lb_qvalue < .05).sum()}/{len(auto)}')
auto.head(20).sort_values('lb_stat').plot.barh(
    x='feature', y='lb_stat', figsize=(9, 7), legend=False,
    title=f'Largest full-training Ljung-Box({ACF_LAG}) statistics')
plt.xlabel('Ljung-Box statistic')
plt.show()
plt.close('all')'''
nb.cells[13].source = '''season_acf = acf_matrix(df, feature_cols, nlags=SEASONAL_NLAGS)
long_lag = season_acf.loc[ACF_LAG + 1:]
valid = long_lag.columns[long_lag.notna().any()]
peak_lag = long_lag[valid].abs().idxmax()
peak = pd.DataFrame({
    'peak_lag': peak_lag,
    'peak_acf': [long_lag.loc[peak_lag[c], c] for c in valid],
}).sort_values('peak_acf', key=lambda s: s.abs(), ascending=False)
display(peak.head(20))
heatmap(season_acf.loc[1:].T, 'Full-training ACF: observed-row lags 1–288', labels=False, limits=(-1, 1))
plt.close('all')'''
nb.cells[14].source = '''## 5. Training-period distribution shifts

Adjacent-month mean changes are scaled by the **training** standard deviation. The log standard-deviation ratio compares adjacent months. These are descriptive changes, not formal break-date estimates. All grouping now uses `msgStamp`; partial boundary months can have fewer observations.'''
nb.cells[15].source = '''mean_shift, vol_shift = monthly_shift_scores(df.reset_index(), 'msgStamp', feature_cols)
breaks = pd.DataFrame({
    'max_abs_monthly_mean_shift_z': mean_shift.abs().max(),
    'max_abs_monthly_log_std_ratio': vol_shift.abs().max(),
}).sort_values('max_abs_monthly_mean_shift_z', ascending=False)
display(breaks.head(20))
heatmap(mean_shift[breaks.head(20).index].T, 'Monthly training mean shifts', figsize=(14, 8))
plt.close('all')'''
nb.cells[16].source = '''## 6. Cashflow: infer the definition from named fields

Only named numeric market variables are compared; `x*` features are deliberately excluded. Examine cashflow relative to volume as well as unconditional level correlations.'''
nb.cells[17].source = '''named_numeric = [c for c in df.select_dtypes(include=np.number).columns
                 if c not in feature_cols and not c.startswith('_')]
named_corr = df[named_numeric].corr(method='spearman')
display(named_corr)
heatmap(named_corr, 'Training correlations: named variables', figsize=(8, 6), limits=(-1, 1))
plt.close('all')
cash_named = named_corr['cashflow'].drop('cashflow').sort_values(key=lambda x: x.abs()).to_frame('spearman')
display(cash_named)'''
nb.cells[18].source = '''## Training-only summary and checks'''
nb.cells[19].source = '''assert df.index.max() < split['cutoff']
assert len(df) == split['train_rows']
print('TIME-SPAN SPLIT (msgStamp)')
print(split.to_string())
print(f'Training rows actually analyzed: {len(df):,}; feature columns: {len(feature_cols)}')
print(f'Training index: {df.index.min()} through {df.index.max()}')
print('No row subsampling in plots, shape screen, correlations or ACF.')
print(f'Pass shape screen: {shape.bell_like.sum()}/{len(shape)}')
print(f'Usable Ljung-Box features: {auto.lb_stat.notna().sum()}/{len(auto)}')
print(f'Largest LB({ACF_LAG}): {auto.lb_stat.max():.1f} ({auto.iloc[0].feature})')
print(f'Strong feature pairs (|rho| >= .8): {len(pairs)}')
print(f'Cashflow/volume absolute value > 1: {len(cashflow_outliers):,} rows')
print(f'Zero-volume training rows: {df.volume.eq(0).sum():,}')
print(f'Non-null flow diagnostic contributions: {flow_contribution.count():,}')
print(f'Last cumulative flow diagnostic value: {flow_cumulative.iloc[-1]:.6g}')
print('Cashflow is likely signed trade flow or quote imbalance (hypothesis, not verified).')
print('The cashflow curve is descriptive; return alignment/executability remain unverified.')
print('\\nTop contemporaneous price-return correlations')
print(return_corr[['feature', 'n', 'pearson', 'spearman']].head(10).to_string(index=False))
print('\\nTop supplied ret_5m correlations')
print(provided_corr[['feature', 'n', 'pearson', 'spearman']].head(10).to_string(index=False))
print('\\nLargest Ljung-Box statistics')
print(auto[['feature', 'n', 'acf1', 'lb_stat']].head(10).to_string(index=False))
print('\\nStrong redundant pairs')
print(pairs.head(10).to_string(index=False))
print('\\nLargest monthly shifts (training only)')
print(breaks.head(10).to_string())'''
coverage = [
    nbf.v4.new_markdown_cell('''### Per-column coverage — training data only

First and last **non-null msgStamp** for every column, non-null count, and percentage of training rows that are null. Entirely missing columns remain visible with `NaT` bounds. No test-period distribution statistics are computed.'''),
    nbf.v4.new_code_cell('''coverage = column_coverage(df)
with pd.option_context('display.max_rows', None):
    display(coverage)
coverage.to_csv(REPORTS / 'column_coverage.csv')'''),
]
flow = [
    nbf.v4.new_markdown_cell('''### Cashflow-to-volume exceptions

Show rows where `abs(cashflow / volume) > 1`. Zero volume is not silently discarded: nonzero cashflow divided by zero appears as infinity and is included by this predicate. The complete named-variable exception table is saved to CSV; pandas may shorten only its display.'''),
    nbf.v4.new_code_cell('''flow_ratio = df.cashflow / df.volume
cashflow_outliers = df.loc[flow_ratio.abs().gt(1), named_numeric].assign(cashflow_per_volume=flow_ratio)
print(f'{len(cashflow_outliers):,} / {len(df):,} training rows have |cashflow / volume| > 1')
print(f'Zero-volume rows: {df.volume.eq(0).sum():,}')
display(cashflow_outliers)
cashflow_outliers.to_csv(REPORTS / 'cashflow_outliers.csv')'''),
    nbf.v4.new_markdown_cell('''### Volatility-scaled signed-flow diagnostic

**Working hypothesis:** cashflow is likely signed trade flow or quote imbalance, rather than an unsigned price-times-volume cash amount. This is a hypothesis, not a documented definition. Out-of-range ratios can also reflect unit, aggregation, or timestamp mismatches.

The expression below is the requested calculation. It clips the ratio to [-1, 1], scales by the previous row's EWM return standard deviation (`halflife=288*21`, pandas defaults), multiplies by the supplied return, then sums by calendar day and cumulatively. All state is learned inside training. Initial undefined volatility/contributions remain missing; pandas daily `sum()` uses its default behavior.

This curve is exploratory, not a validated strategy: the alignment of `ret_5m` with contemporaneous cashflow is unverified, and there are no execution delays or costs. Only the volatility denominator is lagged.'''),
    nbf.v4.new_code_cell('''(df['cashflow'] / df['volume']).clip(-1, 1).div(
    df['ret_5m'].ewm(halflife=288*21).std().shift()
).mul(df['ret_5m']).resample('D').sum().cumsum().plot(
    figsize=(12, 4), title='Cashflow / volume: training-only cumulative diagnostic'
)
plt.show()
plt.close('all')

# Retain the same calculation for the summary and reproducibility checks.
flow_contribution = flow_ratio.clip(-1, 1).div(
    df.ret_5m.ewm(halflife=288*21).std().shift()
).mul(df.ret_5m)
flow_cumulative = flow_contribution.resample('D').sum().cumsum()'''),
]
nb.cells = nb.cells[:4] + coverage + nb.cells[4:18] + flow + nb.cells[18:]
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
        cell.metadata = {}
source = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
assert all(s not in source for s in ['.sample(', 'CROSS_SECTION_N', 'ACF_N', 'acf_source', 'sns.'])
nbf.validate(nb)
nbf.write(nb, path)

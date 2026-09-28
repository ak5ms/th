from __future__ import annotations

import re

import numpy as np
import pandas as pd
from diptest import dipstat
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf


def infer_feature_cols(df: pd.DataFrame) -> list[str]:
    numeric = list(df.select_dtypes(include=np.number).columns)
    signal = [c for c in numeric if re.search(r"(^|_)(signal|feature|x)[_ ]?\d+", c, re.I)]
    if signal:
        return signal
    market = re.compile(r"adjmid|cashflow|price|volume|open|high|low|close|bid|ask|time|date|timestamp", re.I)
    return [c for c in numeric if not market.search(c)]


def bell_shape_table(df: pd.DataFrame, cols: list[str], dip_limit: float = .01,
                     skew_limit: float = .25, max_mass_limit: float = .10) -> pd.DataFrame:
    """Full-sample degeneracy screen; thresholds are effect sizes, not p-values."""
    rows = []
    for col in cols:
        values = df[col].dropna().to_numpy()
        unique, counts = np.unique(values, return_counts=True)
        if len(values) < 8:
            rows.append((col, len(values), len(unique), np.nan, np.nan, np.nan, False))
            continue
        q1, med, q3 = np.quantile(values, [.25, .5, .75])
        skew = (q3 + q1 - 2 * med) / (q3 - q1) if q3 > q1 else np.nan
        dip = dipstat(values) if len(unique) >= 3 else np.nan
        mass = counts.max() / len(values)
        ok = len(unique) >= 20 and mass <= max_mass_limit and abs(skew) <= skew_limit and dip <= dip_limit
        rows.append((col, len(values), len(unique), mass, skew, dip, ok))
    return pd.DataFrame(rows, columns=[
        'feature', 'n', 'nunique', 'max_mass', 'robust_skew', 'dip_stat', 'bell_like',
    ]).sort_values(['bell_like', 'dip_stat'], ascending=[True, False])


def _corr_row(x: pd.Series, y: pd.Series) -> tuple[float, float, float, float, int]:
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 3 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan, len(z)
    pr, pp = pearsonr(z.iloc[:, 0], z.iloc[:, 1])
    sr, sp = spearmanr(z.iloc[:, 0], z.iloc[:, 1])
    return pr, pp, sr, sp, len(z)


def correlation_to_column(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    rows = [(f, *_corr_row(df[f], df[target])) for f in features]
    out = pd.DataFrame(rows, columns=['feature', 'pearson', 'pearson_p', 'spearman', 'spearman_p', 'n'])
    for pcol in ['pearson_p', 'spearman_p']:
        out[pcol.replace('_p', '_q')] = multipletests(out[pcol].fillna(1), method='fdr_bh')[1]
    score = out[['pearson', 'spearman']].abs().max(axis=1)
    return out.assign(max_abs_corr=score).sort_values('max_abs_corr', ascending=False)


def correlation_to_returns(df: pd.DataFrame, features: list[str], price_col: str = 'adjMid') -> pd.DataFrame:
    x = df.copy()
    x['_ret1'] = x[price_col].pct_change(fill_method=None)
    return correlation_to_column(x, '_ret1', features)


def correlation_clustering(df: pd.DataFrame, features: list[str], method: str = 'spearman'):
    corr = df[features].corr(method=method).fillna(0.0)
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 1.0)
    corr = pd.DataFrame(values, index=corr.index, columns=corr.columns)
    dist = (1 - corr.abs()).clip(0, 1)
    z = linkage(squareform(dist.values, checks=False), method='average')
    order = corr.index[leaves_list(z)].tolist()
    return corr, z, order


def redundant_pairs(corr: pd.DataFrame, threshold: float = .8) -> pd.DataFrame:
    """Return strongly correlated pairs and their numeric x-name separation."""
    rows = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            value = corr.loc[a, b]
            if abs(value) < threshold:
                continue
            ia, ib = re.fullmatch(r'x(\d+)', str(a), re.I), re.fullmatch(r'x(\d+)', str(b), re.I)
            gap = abs(int(ia.group(1)) - int(ib.group(1))) if ia and ib else np.nan
            rows.append((a, b, value, abs(value), gap))
    return pd.DataFrame(rows, columns=[
        'feature1', 'feature2', 'corr', 'abs_corr', 'index_gap',
    ]).sort_values('abs_corr', ascending=False, ignore_index=True)


def autocorrelation_table(df: pd.DataFrame, features: list[str], lag: int = 15) -> pd.DataFrame:
    rows = []
    for col in features:
        x = df[col].dropna().to_numpy()
        if len(x) <= lag or np.nanstd(x) == 0:
            rows.append((col, len(x), np.nan, np.nan, np.nan, np.nan))
            continue
        a = acf(x, nlags=lag, fft=True)
        k = np.arange(1, lag + 1)
        q = len(x) * (len(x) + 2) * np.sum(a[1:] ** 2 / (len(x) - k))
        rows.append((col, len(x), a[1], float(np.max(np.abs(a[1:]))), q, float(chi2.sf(q, lag))))
    out = pd.DataFrame(rows, columns=[
        'feature', 'n', 'acf1', 'max_abs_acf_1_to_lag', 'lb_stat', 'lb_pvalue',
    ])
    out['lb_qvalue'] = multipletests(out['lb_pvalue'].fillna(1), method='fdr_bh')[1]
    return out.sort_values('lb_stat', ascending=False)


def acf_matrix(df: pd.DataFrame, features: list[str], nlags: int = 96) -> pd.DataFrame:
    out = {}
    for col in features:
        x = df[col].dropna().to_numpy()
        out[col] = acf(x, nlags=nlags, fft=True) if len(x) > nlags and np.nanstd(x) > 0 else np.full(nlags + 1, np.nan)
    return pd.DataFrame(out, index=pd.RangeIndex(nlags + 1, name='lag'))


def monthly_shift_scores(df: pd.DataFrame, time_col: str, features: list[str]):
    x = df[[time_col, *features]].copy()
    x[time_col] = pd.to_datetime(x[time_col])
    x = x.set_index(time_col).sort_index()[features]
    monthly_mean, monthly_std = x.resample('MS').mean(), x.resample('MS').std()
    scale = x.std().replace(0, np.nan)
    return monthly_mean.diff().div(scale), np.log(monthly_std.div(monthly_std.shift()))

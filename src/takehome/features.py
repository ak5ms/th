"""Small pandas helpers for standalone, training-only feature diagnostics."""
import numpy as np
import pandas as pd


def ts_std(x, hl: int):
    return x.ewm(halflife=hl, min_periods=hl).std()


def ts_standardize(x, hl: int):
    return x.div(ts_std(x, hl).replace(0, np.nan))


def ts_zscore(x, hl: int):
    return x.sub(x.ewm(halflife=hl, min_periods=hl).mean()).div(ts_std(x, hl).replace(0, np.nan))


def sharpe(x):
    """Unannualized mean/std (sample std); no risk-free-rate subtraction."""
    std = x.std()
    return x.mean() / np.where(std == 0, np.nan, std)


def standalone_pnl(X: pd.DataFrame, returns: pd.Series, hl: int = 288 * 21):
    """signal / feature_variance * return; current feature is in EWM state.

    Return timing must be checked separately; this is not an execution model.
    Zero variance and warm-up stay missing, with no artificial leverage cap.
    """
    return X.div(ts_std(X, hl).pow(2).replace(0, np.nan)).mul(returns, axis=0)


def _time_index(x):
    if (not isinstance(x.index, pd.DatetimeIndex) or not x.index.is_monotonic_increasing
            or not x.index.is_unique or x.index.hasnans):
        raise ValueError('Require a sorted, unique DatetimeIndex without NaT.')


def dszl(X, hl: int = 10):
    """Divide by EWM std within each local clock-time group; no demeaning."""
    _time_index(X)
    return X.groupby(X.index.time, sort=False, group_keys=False).apply(
        lambda group: ts_standardize(group, hl)
    ).reindex(X.index)


def pair_residual(y: pd.Series, x: pd.Series, hl: int = 288 * 21):
    """y_t - (b_{t-1} + beta_{t-1} x_t), using alpha=0 and unit joint-row weights."""
    from .fitters import StreamingWeightedLasso

    _time_index(y)
    if not y.index.equals(x.index) or not np.isfinite(hl) or hl <= 0:
        raise ValueError('Require aligned indexes and a positive finite half-life.')
    valid = np.isfinite(x) & np.isfinite(y)
    model = StreamingWeightedLasso(1, decay=2**(-1/hl), alpha=0,
                                  fit_intercept=True, store_history=True)
    model.fit(x.to_numpy()[:, None], y.to_numpy(), W=valid.to_numpy(dtype=float))
    beta = pd.Series(model.get_coefs()[:, 0], index=y.index).shift()
    intercept = pd.Series(model.get_intercepts(), index=y.index).shift()
    ready = valid.cumsum().shift(fill_value=0).ge(2)
    return (y - intercept - beta * x).where(valid & ready)


def pairwise_features(X: pd.DataFrame, kind='product', hl: int = 288 * 21, batch_size=8):
    """All raw i<j products or both directed residuals; batch columns, never rows."""
    from itertools import combinations, permutations, islice

    _time_index(X)
    if kind not in {'product', 'residual'} or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Use product/residual and a positive integer batch_size.')
    if not X.columns.is_unique:
        raise ValueError('Feature names must be unique.')
    pairs = (combinations if kind == 'product' else permutations)(X.columns, 2)
    while batch := list(islice(pairs, batch_size)):
        block = {f'{a}*{b}' if kind == 'product' else f'{a}~{b}':
                 X[a] * X[b] if kind == 'product' else pair_residual(X[a], X[b], hl)
                 for a, b in batch}
        yield pd.DataFrame(block, index=X.index)


def evaluate_features(blocks, returns: pd.Series, hl: int = 288 * 21, *,
                      meta=False, meta_hl=252*288, positive_only=False):
    """Same standalone P&L per block; retain only daily P&L and coverage."""
    daily, summaries = [], []
    numerator = pd.Series(0., index=returns.index) if meta else None
    gross = numerator.copy() if meta else None
    for X in blocks:
        if not X.index.equals(returns.index):
            raise ValueError('Feature and return indexes must match exactly.')
        pnl = standalone_pnl(X, returns, hl)
        if np.isinf(pnl.to_numpy()).any():
            raise ValueError('Infinite P&L: inspect transformed features and their variance.')
        if meta:
            n, g = _blend_totals(pnl, meta_hl, positive_only)
            numerator += n
            gross += g
        days = pnl.resample('D').sum()  # Preserve the existing all-missing-day = 0 convention.
        summaries.append(pd.DataFrame({
            'daily_mean_over_std': sharpe(days), 'pnl_observations': pnl.count(),
            'days_with_observations': pnl.notna().resample('D').sum().gt(0).sum(),
            'first_pnl_msgStamp': pnl.apply(lambda x: x.first_valid_index()),
            'last_pnl_msgStamp': pnl.apply(lambda x: x.last_valid_index()),
        }))
        daily.append(days)
    if not daily:
        raise ValueError('At least one feature block is required.')
    result = (pd.concat(daily, axis=1), pd.concat(summaries).sort_values('daily_mean_over_std', ascending=False))
    if meta:
        return (*result, numerator.div(gross.replace(0, np.nan)).fillna(0).rename('meta_pnl'))
    return result


def _blend_totals(pnl, hl=252*288, positive_only=False):
    """Unnormalized lagged contribution/gross; additive across column batches."""
    if not np.isfinite(hl) or hl <= 0:
        raise ValueError('Require a positive half-life.')
    ewm = pnl.ewm(halflife=hl)
    score = ewm.mean().div(ewm.std().replace(0, np.nan))
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
    if positive_only:
        score = score.clip(lower=0)
    lagged = score.shift(1).fillna(0)
    return lagged.mul(pnl).sum(axis=1), lagged.abs().sum(axis=1)


def combine_pnls(pnl, hl=252*288, positive_only=False):
    """Lagged EWM-Sharpe weights with unit gross, on intrabar P&Ls.

    NaNs do not become zero observations in EWM estimation. Missing current
    P&L contributes zero without reallocating the previously chosen weights.
    """
    numerator, gross = _blend_totals(pnl, hl, positive_only)
    return numerator.div(gross.replace(0, np.nan)).fillna(0).rename('meta_pnl')

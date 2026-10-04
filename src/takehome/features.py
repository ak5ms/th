"""Small pandas helpers for standalone, training-only feature diagnostics."""
import numpy as np
import pandas as pd


def ewm_observed(x, hl, min_periods=0):
    """Zero/nonfinite values are missing; decay advances on observed values."""
    return x.replace([0, np.inf, -np.inf], np.nan).ewm(
        halflife=hl, min_periods=min_periods, ignore_na=True)


def ts_std(x, hl: int, min_periods=None):
    """Nonzero-observation EWM std; exactly zero output stays undefined."""
    return ewm_observed(x, hl, hl if min_periods is None else min_periods).std().replace(0, np.nan)


def ts_standardize(x, hl: int):
    return x.div(ts_std(x, hl).replace(0, np.nan))


def ts_zscore(x, hl: int):
    return x.sub(ewm_observed(x, hl, min_periods=hl).mean()).div(ts_std(x, hl).replace(0, np.nan))


def sharpe(x):
    """Unannualized mean/std (sample std); no risk-free-rate subtraction."""
    std = x.std()
    return x.mean() / np.where(std == 0, np.nan, std)


def _pnl_with_asset_sigma(X, returns, hl, asset_sigma=None):
    feature_sigma = ts_std(X, hl)
    denominator = (feature_sigma.pow(2) if asset_sigma is None else
                   feature_sigma.mul(asset_sigma, axis=0))
    return X.div(denominator.replace(0, np.nan)).mul(returns, axis=0)


def standalone_pnl(X: pd.DataFrame, returns: pd.Series, hl: int = 288 * 21, *,
                   asset_vol: bool = False, normalization: dict | None = None):
    """Default: signal / feature_variance * return, unchanged.

    Experimental asset_vol=True uses signal / (feature_std * lagged_return_std).
    Both scales use ts_std's existing missing/zero and warm-up conventions; the
    asset scale sees returns.shift(1), never the current realized return. This
    diagnostic requires observed return history and is not a blackout solution.
    The legacy default adds no floor or cap. Pass normalization=dict(...) to
    opt into signal_weights (signal history only), independently of scoring returns.
    """
    if not X.index.equals(returns.index):
        raise ValueError('Signal and return indexes must match.')
    if normalization is not None:
        if asset_vol:
            raise ValueError('Choose normalization or asset_vol, not both.')
        from .normalization import signal_weights
        return signal_weights(X, hl, **normalization).mul(returns, axis=0)
    asset_sigma = ts_std(returns.shift(1), hl) if asset_vol else None
    return _pnl_with_asset_sigma(X, returns, hl, asset_sigma)


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


def dszl_design(X: pd.DataFrame, hl: int = 10, batch_size: int = 8) -> pd.DataFrame:
    """Causal dszl features, then explicit zero-imputation, in bounded column batches.

    State uses the entire supplied feature prefix and must not reset at folds.
    No targets or asset-return scale enter this transformation. The returned
    float64 C-contiguous matrix can be shared by all regression candidates.
    """
    _time_index(X)
    if (not isinstance(X, pd.DataFrame) or not X.columns.is_unique or not len(X.columns)
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or isinstance(hl, bool) or not isinstance(hl, int) or hl < 2):
        raise ValueError('Require unique feature columns, integer hl >= 2 and positive batch_size.')
    matrix = np.empty(X.shape, dtype=np.float64, order='C')
    for start in range(0, X.shape[1], batch_size):
        block = dszl(X.iloc[:, start:start+batch_size], hl=hl).to_numpy(dtype=float, copy=True)
        np.nan_to_num(block, copy=False, nan=0., posinf=0., neginf=0.)
        matrix[:, start:start+batch_size] = block
    return pd.DataFrame(matrix, index=X.index, columns=X.columns, copy=False)


def pair_residual(y: pd.Series, x: pd.Series, hl: int = 288 * 21):
    """y_t - beta_{t-1} x_t, with zero intercept and unit joint-row weights."""
    from .fitters import StreamingWeightedLasso

    _time_index(y)
    if not y.index.equals(x.index) or not np.isfinite(hl) or hl <= 0:
        raise ValueError('Require aligned indexes and a positive finite half-life.')
    valid = np.isfinite(x) & np.isfinite(y)
    model = StreamingWeightedLasso(1, decay=2**(-1/hl), alpha=0,
                                  fit_intercept=False, store_history=True)
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
                      meta=False, meta_hl=252*288, positive_only=False, asset_vol=False,
                      normalization: dict | None = None):
    """Same standalone P&L per block; retain daily P&L, coverage and sizing audit.

    normalization explicitly opts into signal-only sizing; the legacy default
    remains unchanged. Column batching never resets the time-series state.
    """
    if asset_vol and normalization is not None:
        raise ValueError('Choose normalization or asset_vol, not both.')
    daily, summaries = [], []
    # Compute the common asset scale once, not once per feature block.
    asset_sigma = ts_std(returns.shift(1), hl) if asset_vol else None
    numerator = pd.Series(0., index=returns.index) if meta else None
    gross = numerator.copy() if meta else None
    for X in blocks:
        if not X.index.equals(returns.index):
            raise ValueError('Feature and return indexes must match exactly.')
        weights = None
        if normalization is None:
            pnl = _pnl_with_asset_sigma(X, returns, hl, asset_sigma)
        else:
            from .normalization import signal_weights
            weights = signal_weights(X, hl, **normalization)
            pnl = weights.mul(returns, axis=0)
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
        if weights is not None:
            summaries[-1]['max_abs_weight'] = weights.abs().max()
            cap = normalization.get('cap', 3.)
            summaries[-1]['capped_fraction'] = (
                weights.abs().ge(cap - 1e-12).sum().div(weights.count().replace(0, np.nan))
                if cap is not None else 0.)
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
    ewm = ewm_observed(pnl, hl)
    score = ewm.mean().div(ewm.std().replace(0, np.nan))
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
    if positive_only:
        score = score.clip(lower=0)
    lagged = score.shift(1).fillna(0)
    return lagged.mul(pnl).sum(axis=1), lagged.abs().sum(axis=1)


def combine_pnls(pnl, hl=252*288, positive_only=False):
    """Lagged EWM-Sharpe weights with unit gross, on intrabar P&Ls.

    Zeros and nonfinite values do not become observations in EWM estimation. Missing current
    P&L contributes zero without reallocating the previously chosen weights.
    """
    numerator, gross = _blend_totals(pnl, hl, positive_only)
    return numerator.div(gross.replace(0, np.nan)).fillna(0).rename('meta_pnl')


def backtest(signal, returns, hl=288*21, *, asset_vol=False, normalization: dict | None = None):
    """Standalone sizing, with optional lagged asset-vol diagnostic; align forecasts first."""
    if not signal.index.equals(returns.index):
        raise ValueError('Signal and return indexes must match.')
    return standalone_pnl(signal, returns, hl, asset_vol=asset_vol, normalization=normalization).replace([np.inf, -np.inf], np.nan)


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


def with_cashflow_feature(df):
    """Append the next x-number: raw cashflow/volume; undefined ratios stay missing."""
    import re
    numbers = [int(m.group(1)) for c in df if (m := re.fullmatch(r'x(\d+)', str(c)))]
    name = f'x{max(numbers, default=0)+1}'
    ratio = df['cashflow'].div(df['volume'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return df.assign(**{name: ratio})


def raw_dszl_design(X: pd.DataFrame, hl: int = 10, batch_size: int = 8) -> pd.DataFrame:
    """Concatenate [raw X, dszl(X, hl)] in that order, then zero-impute.

    Raw observations remain in their original units; the transformed half uses
    the entire supplied feature prefix and no target. No state resets at folds.
    A single C-contiguous float64 output is shared across the model grid.
    """
    _time_index(X)
    if (not isinstance(X, pd.DataFrame) or not X.columns.is_unique or not len(X.columns)
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or isinstance(hl, bool) or not isinstance(hl, int) or hl < 2):
        raise ValueError('Require unique feature columns, integer hl >= 2 and positive batch_size.')
    p = X.shape[1]
    columns = [f'raw:{c}' for c in X.columns] + [f'dszl:{c}' for c in X.columns]
    if len(set(columns)) != len(columns):
        raise ValueError('String representations of feature names must be unique.')
    matrix = np.empty((len(X), 2*p), dtype=np.float64, order='C')
    for start in range(0, p, batch_size):
        stop = min(start+batch_size, p)
        block = X.iloc[:, start:stop].to_numpy(dtype=float, copy=True)
        np.nan_to_num(block, copy=False, nan=0., posinf=0., neginf=0.)
        matrix[:, start:stop] = block
        block = dszl(X.iloc[:, start:stop], hl=hl).to_numpy(dtype=float, copy=True)
        np.nan_to_num(block, copy=False, nan=0., posinf=0., neginf=0.)
        matrix[:, p+start:p+stop] = block
    return pd.DataFrame(matrix, index=X.index, columns=columns, copy=False)



def variance_scaled_design(X: pd.DataFrame, hl: int = 10, *,
                           variance_hl: int = 6048, batch_size: int = 8) -> pd.DataFrame:
    """Build F=[X, dszl(X, hl)], divide each column by its EWM variance.

    This is F / EWMVar(F), not F / EWMStd(F) and not a demeaned z-score.
    Both transforms see only the feature prefix through the current row. The
    current feature is assumed observable before predicting the forward return.
    EWM variance uses the existing adjust=True, ignore_na=True, unbiased sample
    convention; zero/nonfinite observations do not update it. Require
    variance_hl observations. Zero/undefined variances produce unavailable
    transformed inputs, which are zero-imputed only AFTER division. There is no
    variance floor, input clipping, return scaling, or reset at fold boundaries.

    Supply the same original feature-history prefix for research and holdout
    inference. Column batches bound memory without changing the time-series
    state. The owned output is C-contiguous float64 in [raw, dszl] column order.
    """
    _time_index(X)
    if (not isinstance(X, pd.DataFrame) or not X.columns.is_unique or not len(X.columns)
            or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
            or isinstance(hl, bool) or not isinstance(hl, int) or hl < 2
            or isinstance(variance_hl, bool) or not isinstance(variance_hl, int) or variance_hl < 2):
        raise ValueError('Require unique columns, integer half-lives >= 2 and positive batch_size.')
    p = len(X.columns)
    columns = [f'raw:{c}' for c in X.columns] + [f'dszl:{c}' for c in X.columns]
    if len(set(columns)) != len(columns):
        raise ValueError('String representations of feature names must be unique.')
    matrix = np.empty((len(X), 2*p), dtype=np.float64, order='C')
    for start in range(0, p, batch_size):
        stop = min(start + batch_size, p)
        raw = X.iloc[:, start:stop]
        for offset, block in ((0, raw), (p, dszl(raw, hl=hl))):
            variance = ewm_observed(block, variance_hl, min_periods=variance_hl).var()
            variance = variance.where(variance > 0)
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                values = block.div(variance).to_numpy(dtype=float, copy=True)
            np.nan_to_num(values, copy=False, nan=0., posinf=0., neginf=0.)
            matrix[:, offset+start:offset+stop] = values
    return pd.DataFrame(matrix, index=X.index, columns=columns, copy=False)

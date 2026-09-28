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

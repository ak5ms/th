"""Causal, signal-only exposures; no returns or test-period fitted constants."""
import numpy as np
import pandas as pd
from .features import ewm_observed


def signal_weights(signal, hl=288*21, *, method='std', floor_fraction=.25,
                   cap=3., min_periods=None, lag=1, reference=None):
    """Return dimensionless signed exposures, using only signal history.

    std: x / max(prior_std, floor_fraction * initial_std).
    variance: initial_std * x / max(prior_std, floor_fraction * initial_std)^2.
    rms: replace std by sqrt(EWM(x^2)); do not subtract the signal mean.
    cap bounds absolute exposure, not P&L. No asset-return input is required.

    The reference is frozen at the FIRST available scale, never a full-sample
    statistic. An explicit positive reference must come from earlier data
    (a scalar or a Series indexed by DataFrame columns). Replay signal history
    through test dates to keep EWM state; never restart the normalizer at a fold.
    lag=1 uses prior-row statistics; lag=0 is for the existing-rule control.
    Zeros are missing for scale estimation but give zero exposure once ready.
    Missing/nonfinite current signals and the initial warm-up stay missing.
    """
    nmin = int(np.ceil(hl)) if min_periods is None and np.isfinite(hl) else min_periods
    if (method not in {'std', 'variance', 'rms'} or not np.isfinite(hl) or hl <= 0
            or not isinstance(nmin, (int, np.integer)) or nmin < 2
            or not np.isfinite(floor_fraction) or floor_fraction < 0
            or not isinstance(lag, (int, np.integer)) or lag < 0
            or (cap is not None and (not np.isfinite(cap) or cap <= 0))):
        raise ValueError('Invalid method, half-life, observation count, floor, cap or lag.')
    if (not isinstance(signal, (pd.Series, pd.DataFrame))
            or not isinstance(signal.index, pd.DatetimeIndex)
            or not signal.index.is_monotonic_increasing or not signal.index.is_unique
            or signal.index.hasnans):
        raise ValueError('Require a Series/DataFrame with sorted unique timestamps.')
    x = pd.DataFrame(signal).replace([np.inf, -np.inf], np.nan)
    if not x.columns.is_unique:
        raise ValueError('Feature names must be unique.')
    scale = (ewm_observed(x.pow(2), hl, nmin).mean().pow(.5) if method == 'rms'
             else ewm_observed(x, hl, nmin).std())
    scale = scale.replace([0, np.inf, -np.inf], np.nan).shift(lag)
    if reference is None:
        anchor = scale.where(scale.notna().cumsum().eq(1)).ffill()
    else:
        anchor = pd.Series(reference, index=x.columns, dtype=float)
        if not np.isfinite(anchor).all() or (anchor <= 0).any():
            raise ValueError('Every reference scale must be positive and finite.')
    denominator = scale.clip(lower=floor_fraction*anchor, axis='columns')
    out = x/denominator
    if method == 'variance':
        out = out.mul(anchor, axis='columns')/denominator
    if cap is not None:
        out = out.clip(-cap, cap)
    return out.iloc[:, 0].rename(signal.name) if isinstance(signal, pd.Series) else out

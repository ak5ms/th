

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

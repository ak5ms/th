"""Small plotting helper shared by notebook diagnostics."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def heatmap(frame, title, figsize=(12, 8), labels=True, limits=None):
    fig, ax = plt.subplots(figsize=figsize)
    bounds = {} if limits is None else dict(vmin=limits[0], vmax=limits[1])
    image = ax.imshow(frame.to_numpy(), aspect='auto', interpolation='nearest', **bounds)
    if labels:
        ax.set_xticks(range(frame.shape[1]), frame.columns.astype(str), rotation=90, fontsize=7)
        ax.set_yticks(range(frame.shape[0]), frame.index.astype(str), fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    plt.show()


def display_results(pnl_daily, title, *, meta_daily=None):
    """Members on the left, unrescaled combination on the right, then histogram."""
    from .features import sharpe

    if meta_daily is not None and not meta_daily.index.equals(pnl_daily.index):
        raise ValueError('Combined and standalone daily P&L must have the same index.')
    fig, ax = plt.subplots(figsize=(14, 6))
    for start in range(0, len(pnl_daily.columns), 32):
        cumulative = pnl_daily.iloc[:, start:start + 32].cumsum()
        ax.plot(cumulative.index, cumulative.to_numpy(), linewidth=.4, alpha=.3)
    if meta_daily is not None:
        right = meta_daily.rename('Lagged EWM-Sharpe combination').cumsum().plot(
            ax=ax, secondary_y=True, x_compat=True, color='purple', linewidth=3, zorder=10)
        right.set_ylabel('Combined P&L (right axis; independent scale)')
        right.legend(loc='upper left')
    ax.set_title(f'{title}: {len(pnl_daily.columns):,} features, all training days')
    ax.set_ylabel('Cumulative diagnostic P&L (arbitrary exposure units)')
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    scores = sharpe(pnl_daily)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(scores.dropna(), bins=30)
    ax.set_title(f'{title}: daily mean/std ({scores.notna().sum():,} defined)')
    ax.set_xlabel('Calendar-day mean / standard deviation (not annualized)')
    ax.set_ylabel('Feature count')
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return scores


def plot_calibration(yhat, y, *, weights=None, title='Prior-row forecasts: training calibration'):
    """Scatter every finite pair; regress y on yhat with an intercept, not the reverse.

    Return OLS and optional WLS diagnostics; never rescale the input forecasts.
    """
    if not yhat.index.equals(y.index) or (weights is not None and not weights.index.equals(y.index)):
        raise ValueError('Prediction, target and weight indexes must match.')
    if weights is not None and (not np.isfinite(weights).all() or (weights < 0).any()):
        raise ValueError('Calibration weights must be finite and nonnegative.')
    valid = np.isfinite(yhat) & np.isfinite(y)
    p, target = yhat[valid].to_numpy(), y[valid].to_numpy()
    if len(p) < 2:
        raise ValueError('Calibration needs at least two finite prediction/target pairs.')
    choices = {'OLS': np.ones(len(p))}
    if weights is not None:
        choices['WLS'] = weights[valid].to_numpy()
    rows = {}
    for name, w in choices.items():
        keep = w > 0
        n = int(keep.sum())
        slope = bias = r2 = np.nan
        if n >= 2:
            q = w[keep] / w[keep].max()
            q /= q.sum()
            px, py = p[keep], target[keep]
            mx, my = q @ px, q @ py
            dx, dy = px - mx, py - my
            xx, yy, xy = q @ (dx*dx), q @ (dy*dy), q @ (dx*dy)
            if xx > 0:
                slope, bias = xy / xx, my - (xy / xx) * mx
                r2 = xy**2 / (xx * yy) if yy > 0 else np.nan
        rows[name] = dict(n=n, slope=slope, intercept=bias, r_squared=r2)
    stats = pd.DataFrame.from_dict(rows, orient='index')
    slope, bias = stats.loc['OLS', ['slope', 'intercept']]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(p, target, s=2, alpha=.05, linewidths=0, rasterized=True)
    ax.axline((0, 0), slope=1, linestyle=':', linewidth=1.5, label='Ideal: y = yhat')
    if np.isfinite(slope):
        limits = np.array([p.min(), p.max()])
        ax.plot(limits, bias + slope * limits, linestyle='--', linewidth=2, label='OLS: y ~ yhat')
        note = rf'$\beta(y \sim \hat y)$ = {slope:.4f}' + f'\nIntercept = {bias:.3g}; n = {len(p):,}'
    else:
        note = f'Calibration slope undefined (constant forecast); n = {len(p):,}'
    if weights is not None:
        note += f"\nW-weighted slope = {stats.loc['WLS', 'slope']:.4f}"
    ax.text(.03, .97, note, transform=ax.transAxes, va='top', fontsize=11,
            bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
    ax.set(xlabel='Prediction yhat (return units)', ylabel='Target y = ret_5m (return units)', title=title)
    ax.legend(loc='lower right')
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return stats

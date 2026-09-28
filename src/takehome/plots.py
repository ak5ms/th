"""Small plotting helper shared by notebook diagnostics."""
import matplotlib.pyplot as plt


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


def display_results(pnl_daily, title):
    """One cumulative P&L figure and one unannualized daily Sharpe histogram."""
    from .features import sharpe

    fig, ax = plt.subplots(figsize=(14, 6))
    for start in range(0, len(pnl_daily.columns), 32):
        cumulative = pnl_daily.iloc[:, start:start + 32].cumsum()
        ax.plot(cumulative.index, cumulative.to_numpy(), linewidth=.4, alpha=.3)
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

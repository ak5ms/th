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

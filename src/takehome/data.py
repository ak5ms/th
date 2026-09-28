"""Time-span holdout and column coverage; no imputation or row sampling."""
import pandas as pd


def column_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """First/last non-null timestamps and missingness, including empty columns."""
    return pd.DataFrame({
        'first_valid': df.apply(pd.Series.first_valid_index),
        'last_valid': df.apply(pd.Series.last_valid_index),
        'non_null': df.count(),
        'null_pct': df.isna().mean().mul(100),
    }).rename_axis('column')


def training_data(df: pd.DataFrame, target: str = 'ret_5m', fraction: float = 0.8):
    """Reserve the final time fraction of the labeled span, not of its row count."""
    index = df.index
    if not isinstance(index, pd.DatetimeIndex) or not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError('The time index must be a sorted, unique DatetimeIndex.')
    if not 0 < fraction < 1:
        raise ValueError('fraction must lie strictly between 0 and 1.')
    first, last = df[target].first_valid_index(), df[target].last_valid_index()
    if first is None or last <= first:
        raise ValueError('At least two distinct labeled timestamps are required.')
    cutoff = first + fraction * (last - first)
    train_mask = (index >= first) & (index < cutoff)
    test_mask = (index >= cutoff) & (index <= last)
    split = pd.Series({
        'first_label': first, 'cutoff': cutoff, 'last_label': last,
        'train_rows': int(train_mask.sum()), 'test_rows': int(test_mask.sum()),
        'train_fraction_of_time': fraction,
    }, name='split')
    return df.loc[train_mask].copy(), split

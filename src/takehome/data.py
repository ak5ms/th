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


def read_parquet_window(path, *, start=None, stop=None, columns=None, batch_size=16384):
    """Read a timestamp window [start, stop) without materializing the full file.

    Supports msgStamp stored as a column or named pandas index. Column filtering
    happens in Arrow before conversion; chronological ordering is restored once.
    An omitted stop includes the final timestamp. No values are imputed.
    """
    import pyarrow.parquet as pq

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer.')
    if start is not None and stop is not None and start >= stop:
        raise ValueError('Require start < stop.')
    requested = None if columns is None else list(columns)
    read_columns = None if requested is None else list(dict.fromkeys(['msgStamp', *requested]))
    parts, empty = [], None
    for batch in pq.ParquetFile(path).iter_batches(columns=read_columns,
                                                  batch_size=batch_size, use_threads=False):
        frame = batch.to_pandas()
        if 'msgStamp' in frame:
            frame = frame.set_index('msgStamp')
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError('Require a DatetimeIndex or datetime msgStamp column.')
        if requested is not None:
            frame = frame[requested]
        if empty is None:
            empty = frame.iloc[:0].copy()
        keep = ~frame.index.isna()
        if frame.index.hasnans:
            raise ValueError('msgStamp must not contain NaT.')
        if start is not None:
            keep &= frame.index >= start
        if stop is not None:
            keep &= frame.index < stop
        if keep.any():
            parts.append(frame.loc[keep].copy())
    if empty is None:
        raise ValueError('The parquet file contains no rows.')
    result = pd.concat(parts).sort_index() if parts else empty
    if not result.index.is_unique:
        raise ValueError('msgStamp must be unique.')
    return result

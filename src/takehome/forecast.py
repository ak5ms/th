"""Timestamped, unscaled batched-Lasso forecasts for validation and withheld data."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import training_data
from .features import with_cashflow_feature
from .fitters import BatchLasso, calendar_walk_forward_folds, walk_forward_sweep


def _model_data(path, first, stop=None):
    """Read only model columns and requested timestamps in bounded row batches."""
    parquet = pq.ParquetFile(path)
    columns = ['msgStamp', 'ret_5m', 'cashflow', 'volume']
    columns += sorted((c for c in parquet.schema.names if re.fullmatch(r'x\d+', c)),
                      key=lambda c: int(c[1:]))
    parts = []
    for batch in parquet.iter_batches(columns=columns, batch_size=16384, use_threads=False):
        frame = batch.to_pandas()
        if 'msgStamp' in frame:
            frame = frame.set_index('msgStamp')
        keep = frame.index >= first
        if stop is not None:
            keep &= frame.index < stop
        if keep.any():
            parts.append(frame.loc[keep].copy())
    if not parts:
        raise ValueError('No model rows in the requested time interval.')
    result = with_cashflow_feature(pd.concat(parts).sort_index())
    del parts, frame, batch
    import pyarrow as pa
    pa.default_memory_pool().release_unused()
    return result


def _fit(frame, *, alpha, hl, folds):
    columns = sorted((c for c in frame if re.fullmatch(r'x\d+', c)), key=lambda c: int(c[1:]))
    # One owned design buffer: avoid several whole-frame replace/fillna copies.
    X = np.array(frame[columns].to_numpy(dtype=float), dtype=float, order='C', copy=True)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y = frame.ret_5m.to_numpy(dtype=float)
    weights = np.isfinite(y).astype(float)
    for fold in folds:
        if not weights[fold['train_start']:fold['train_stop']].any():
            raise ValueError('Each training window needs observed targets.')
    model = BatchLasso(len(columns), decay=2**(-1/hl), alpha=alpha, fit_intercept=True, tol=1e-11)
    path = walk_forward_sweep({'lasso': model}, X, y, W=weights, folds=folds)['lasso']
    prediction = pd.Series(path.prediction_, index=frame.index, name='forecast').dropna().to_frame()
    return prediction, columns


def _write_prediction(prediction, directory, stem):
    csv, parquet = directory/f'{stem}.csv', directory/f'{stem}.parquet'
    prediction.to_csv(csv, index_label='msgStamp', float_format='%.17g')
    prediction.to_parquet(parquet)
    return csv, parquet


def write_lasso_forecasts(data_path, output_dir, *, years=2, alpha=1e-5, hl=6048):
    """Export research-only WF predictions and the final two-year blackout.

    The existing 80%-of-labeled-time research split is preserved. Its validation
    fits use rolling `years` training followed by `years` prediction windows.
    Separately, a final fit uses only the `years` immediately preceding the
    assignment's last-`years` data span and is frozen throughout that span. This
    final fitting step may use labels from the research-reserved block, but
    never evaluates or selects a penalty using them. All blackout targets must
    be missing. Internal missing targets do not define the blackout boundary.

    Features are the existing raw x columns plus cashflow/volume. Nonfinite
    features are explicitly zero-imputed; labels are never filled. No feature
    forecast, oracle, return volatility, position sizing or rescaling is used.
    """
    if (isinstance(years, bool) or not isinstance(years, (int, np.integer)) or years < 1
            or not np.isfinite([alpha, hl]).all() or alpha < 0 or hl <= 0):
        raise ValueError('Require positive integer years, alpha >= 0 and hl > 0.')
    data_path, directory = Path(data_path), Path(output_dir)
    labels = pd.read_parquet(data_path, columns=['msgStamp', 'ret_5m'])
    if 'msgStamp' in labels:
        labels = labels.set_index('msgStamp')
    labels = labels.sort_index()
    # training_data validates a sorted unique DatetimeIndex and a usable labeled span.
    _, split = training_data(labels)
    if labels.index.hasnans:
        raise ValueError('Timestamps must not contain NaT.')
    boundary = labels.index[-1] - pd.DateOffset(years=int(years))
    blackout = labels.loc[labels.index >= boundary]
    if blackout.empty or blackout.ret_5m.notna().any():
        raise ValueError('The final calendar-year span must be a withheld blackout with all targets missing.')
    if boundary - pd.DateOffset(years=int(years)) < split['first_label']:
        raise ValueError('Insufficient labeled history for the final training window.')

    research = _model_data(data_path, split['first_label'], split['cutoff'])
    research_folds = calendar_walk_forward_folds(research.index, train_years=years, test_years=years)
    if not research_folds:
        raise ValueError('Research span is too short for a two-window walk-forward evaluation.')
    validation, features = _fit(research, alpha=alpha, hl=hl, folds=research_folds)
    del research

    final = _model_data(data_path, boundary - pd.DateOffset(years=int(years)))
    final_folds = calendar_walk_forward_folds(final.index, train_years=years,
        test_years=years, first_prediction=boundary)
    # The terminal timestamp is inclusive; no empty-data refit is allowed there.
    final_folds = final_folds[:1]
    final_folds[0]['predict_stop'] = len(final)
    submitted, final_features = _fit(final, alpha=alpha, hl=hl, folds=final_folds)
    if features != final_features or not submitted.index.equals(blackout.index):
        raise ValueError('Feature layout or withheld forecast coverage changed.')
    if not np.isfinite(submitted.forecast).all():
        raise ValueError('All withheld rows must receive finite forecasts.')
    del final

    directory.mkdir(parents=True, exist_ok=True)
    vc, vp = _write_prediction(validation, directory, 'batched_lasso_walk_forward')
    oc, op = _write_prediction(submitted, directory, 'batched_lasso_oos')
    windows = pd.DataFrame([dict(kind='research_validation', fold=i, **f) for i,f in enumerate(research_folds)]
                          + [dict(kind='final_withheld', fold=i, **f) for i,f in enumerate(final_folds)])
    windows.to_csv(directory/'batched_lasso_fold_windows.csv', index=False)
    metadata = dict(model='BatchLasso', target='ret_5m', alpha=float(alpha), hl=float(hl),
        decay=float(2**(-1/hl)), fit_intercept=True, train_years=int(years), test_years=int(years),
        features=features, predictor_imputation='nonfinite -> 0', forecast_units='raw ret_5m units',
        research_cutoff=split['cutoff'].isoformat(), research_folds=len(research_folds),
        validation_rows=len(validation), validation_first=validation.index[0].isoformat(),
        validation_last=validation.index[-1].isoformat(),
        withheld_boundary=boundary.isoformat(), final_train_start=(boundary-pd.DateOffset(years=int(years))).isoformat(),
        oos_rows=len(submitted), oos_first=submitted.index[0].isoformat(), oos_last=submitted.index[-1].isoformat(),
        final_fit_uses_research_reserved_labels=True, blackout_targets_used=False,
        note='Final supplied timestamp included in the frozen withheld block; no refit on missing labels.')
    (directory/'batched_lasso_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return dict(validation_csv=vc, validation_parquet=vp, oos_csv=oc, oos_parquet=op, metadata=metadata)

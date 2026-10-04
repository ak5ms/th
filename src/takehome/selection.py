"""Previous-test-block hyperparameter selection; no same-fold tuning or static default."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .fitters import (
    BatchedFitters, StreamingWeightedLasso, _fit_input, _validated_folds,
    forecast_alpha, walk_forward_sweep,
)


@dataclass
class PredictionSelection:
    prediction_: np.ndarray
    choices_: list[Any]
    selection_: pd.DataFrame
    validation_scores_: pd.DataFrame


@dataclass
class SelectedWalkForward(PredictionSelection):
    path_: BatchedFitters
    candidates_: dict[Any, BatchedFitters]
    all_folds_: list[dict]


def select_previous_test(predictions: Mapping[Any, np.ndarray], y, folds, *,
                         weights=None, embargo: int = 1, allow_missing_history: bool = False) -> PredictionSelection:
    """Select by MSE in test fold k-1 and apply to test fold k, never k-1.

    Candidate order breaks exact ties (supply largest penalties first for a
    stronger-shrinkage tie break). Every candidate is scored on the same finite,
    positive-weight target rows. Any nonfinite prediction on those rows makes
    that candidate ineligible, not artificially better through reduced coverage.
    The last `embargo` validation rows are excluded for target maturity. No
    prediction is published in the first fold, which is initial tuning only.
    A later all-missing/invalid validation block raises instead of silently
    falling back to a fixed regularization parameter. Terminal missing labels
    are never used to select their own fold's candidate. Sparse feature diagnostics
    can opt into allow_missing_history: a block with no past scoring examples
    leaves the next forecast missing. Invalid predictions on observed validation
    targets still raise. Return-model selection retains the strict default.
    """
    if not predictions:
        raise ValueError('Supply candidate predictions.')
    target = np.asarray(y, dtype=float)
    if target.ndim != 1:
        raise ValueError('Targets must be a one-dimensional array.')
    if isinstance(embargo, bool) or not isinstance(embargo, (int, np.integer)) or embargo < 0:
        raise ValueError('embargo must be a nonnegative integer.')
    folds = _validated_folds(folds, len(target))
    w = np.asarray(1. if weights is None else weights, dtype=float)
    if w.ndim == 0:
        w = np.full(len(target), w.item())
    if w.shape != target.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Validation weights must be finite, nonnegative, and length n.')
    arrays = {key: np.asarray(value, dtype=float) for key, value in predictions.items()}
    if any(a.shape != target.shape for a in arrays.values()):
        raise ValueError('Every candidate must predict exactly the target shape.')
    selected = np.full(len(target), np.nan)
    choices, decisions, scores = [], [], []
    previous = None
    for i, fold in enumerate(folds):
        first, end = fold['predict_start'], fold['predict_stop']
        if i == 0:
            choices.append(None)
        else:
            # This uses only the PREVIOUS block's scores; current labels are not read yet.
            valid = [r for r in previous if np.isfinite(r['mse'])]
            if not valid:
                if not allow_missing_history or any(r['n'] > 0 for r in previous):
                    raise ValueError(f'No eligible candidate in preceding validation fold {i-1}.')
                # No eligible historical examples is not a license for a static
                # alpha or a future-data choice. Publish no forecast in this block.
                best = previous[0]
                choices.append(None)
                decisions.append(dict(fold=i, validation_fold=i-1, candidate=None,
                    validation_start=best['validation_start'], validation_stop=best['validation_stop'],
                    validation_rows=0, validation_mse=np.nan, predict_start=first,
                    predict_stop=end, status='waiting_for_validation'))
            else:
                best = min(valid, key=lambda r: r['mse'])
                key = best['candidate']
                choices.append(key)
                selected[first:end] = arrays[key][first:end]
                decisions.append(dict(fold=i, validation_fold=i-1, candidate=key,
                    validation_start=best['validation_start'], validation_stop=best['validation_stop'],
                    validation_rows=best['n'], validation_mse=best['mse'],
                    predict_start=first, predict_stop=end, status='selected'))
        # Scores are held for the NEXT decision and displayed as validation only.
        stop = max(first, end-int(embargo))
        active = np.isfinite(target[first:stop]) & (w[first:stop] > 0)
        n = int(active.sum())
        v = w[first:stop][active]
        if n:
            v = v/v.max()
            v /= v.sum()
        current = []
        for key, values in arrays.items():
            pred = values[first:stop][active]
            eligible = n > 0 and np.isfinite(pred).all()
            mse = np.nan
            if eligible:
                with np.errstate(over='ignore', invalid='ignore'):
                    mse = float(v @ np.square(pred-target[first:stop][active]))
                if not np.isfinite(mse):
                    mse = np.nan
            current.append(dict(fold=i, candidate=key, validation_start=first,
                validation_stop=stop, n=n, mse=mse))
        scores.extend(current)
        previous = current
    decision_columns = ['fold', 'validation_fold', 'candidate', 'validation_start',
                        'validation_stop', 'validation_rows', 'validation_mse',
                        'predict_start', 'predict_stop', 'status']
    score_columns = ['fold', 'candidate', 'validation_start', 'validation_stop', 'n', 'mse']
    return PredictionSelection(selected, choices, pd.DataFrame(decisions, columns=decision_columns),
                               pd.DataFrame(scores, columns=score_columns))


def select_walk_forward_paths(paths: Mapping[Any, BatchedFitters], y, *,
                              weights=None, embargo: int = 1) -> SelectedWalkForward:
    """Select already-fitted candidate paths and retain only selected fold snapshots."""
    if not paths:
        raise ValueError('Supply candidate paths.')
    from copy import deepcopy

    first = next(iter(paths.values()))
    folds = first.folds_
    if any(path.folds_ != folds or path.n_features != first.n_features for path in paths.values()):
        raise ValueError('Candidates must have the same training/prediction folds and feature layout.')
    if any(path.n_rows_ != len(y) for path in paths.values()):
        raise ValueError('Candidate path length must match y.')
    picked = select_previous_test({key: path.prediction_ for key, path in paths.items()},
                                  y, folds, weights=weights, embargo=embargo)
    chosen = BatchedFitters(deepcopy(first.fitter))._reset(len(y))
    for i, key in enumerate(picked.choices_):
        if key is None:
            continue
        path = paths[key]
        chosen.folds_.append(folds[i].copy())
        chosen.snapshots_.append(path.snapshots_[i].copy())
        chosen.offsets_.append(path.offsets_[i])
        chosen.objectives_.append(path.objectives_[i])
        chosen.coef = path.snapshots_[i].copy()
        chosen.intercept_ = path.offsets_[i]
    chosen.prediction_ = picked.prediction_
    # The fitted selection is not a constant-alpha estimator to be refit.
    return SelectedWalkForward(picked.prediction_, picked.choices_, picked.selection_,
                               picked.validation_scores_, chosen, dict(paths),
                               [f.copy() for f in folds])


def walk_forward_select(fitters, X, y, W=None, *, folds, embargo: int = 1):
    """Fit candidate windows, then select each window using the previous OOS test."""
    paths = walk_forward_sweep(fitters, X, y, W=W, folds=folds)
    return select_walk_forward_paths(paths, y, weights=W, embargo=embargo)


def stream_selected_folds(selected: SelectedWalkForward, X, y, W=None, *,
                          tol=1e-10, max_iter=20000):
    """Audit the selected batch-Lasso schedule with independent streaming replays.

    Each selected fold starts from exactly its two-year training data and uses
    that fold's previously chosen alpha. The frozen snapshot is an implementation
    comparator to batch Lasso; live then learns test labels strictly after each
    prediction, with the SAME validation-selected alpha until the next fold.
    No grid is picked by looking at a live curve after the fact. Training history
    is not retained, and no dense T-by-feature coefficient history is allocated.
    """
    p = selected.path_.n_features
    X, y, w = _fit_input(X, y, W, p)
    live, frozen = np.full(len(y), np.nan), np.full(len(y), np.nan)
    audits, coefs, offsets = [], [], []
    for i, key in enumerate(selected.choices_):
        if key is None:
            continue
        f = selected.all_folds_[i]
        a, b, c, d = (f[k] for k in ('train_start','train_stop','predict_start','predict_stop'))
        reference = selected.candidates_[key].fitter
        model = StreamingWeightedLasso(p, reference.decay, reference.alpha,
            fit_intercept=reference.fit_intercept, tol=tol, max_iter=max_iter)
        model.fit(X[a:b], y[a:b], W=w[a:b])
        terminal_converged = bool(model.converged_)
        terminal_error = model.kkt_violation_
        training_failures = model.n_failed_
        frozen[c:d] = model.predict(X[c:d])
        coefs.append(model.coef.copy()); offsets.append(float(model.intercept_))
        if b < c:
            # Advance forgetting through embargoed rows, without their labels.
            model.fit(X[b:c], np.full(c-b, np.nan), W=np.zeros(c-b))
        live[c:d] = model.fit_predict(X[c:d], y[c:d], W=w[c:d])
        errors = frozen[c:d]-selected.prediction_[c:d]
        audits.append(dict(fold=i, alpha=reference.alpha, train_start=a, train_stop=b,
            terminal_converged=terminal_converged, terminal_kkt=terminal_error,
            training_nonconverged_updates=training_failures,
            live_nonconverged_updates=model.n_failed_-training_failures,
            frozen_rmse=float(np.sqrt(np.mean(errors**2))), frozen_max_error=float(np.max(np.abs(errors)))))
    return dict(live=live, frozen=frozen, audit=pd.DataFrame(audits),
                coefs=np.array(coefs), intercepts=np.array(offsets))


def forecast_alpha_selected(x: pd.Series, *, folds, alphas, hl=6048,
                            min_train=6048, tol=1e-8, max_iter=5000):
    """Online AR(2) candidates; prior two-year next-feature test MSE chooses alpha.

    The last validation decision is excluded: its next-feature target may not
    yet be known. No return label is used. Candidate state is causal continuous
    EWM as in forecast_alpha; only the hyperparameter decisions use test folds.
    """
    alphas = list(alphas)
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError('Provide distinct AR regularization candidates.')
    candidates, diagnostics = {}, []
    for alpha in alphas:
        pred, info = forecast_alpha(x, hl=hl, alpha=alpha, min_train=min_train,
                                    tol=tol, max_iter=max_iter)
        candidates[alpha] = pred.to_numpy()
        diagnostics.append(info)
    values = x.replace([np.inf, -np.inf], np.nan)
    complete = values.notna() & values.shift(1).notna() & values.shift(2).notna()
    ready = complete.cumsum().ge(min_train) & values.notna() & values.shift(1).notna()
    result = select_previous_test(candidates, values.shift(-1).to_numpy(dtype=float), folds,
                                  weights=ready.to_numpy(dtype=float), embargo=1, allow_missing_history=True)
    return pd.Series(result.prediction_, index=x.index, name=x.name), dict(
        selection=result.selection_, scores=result.validation_scores_, choices=result.choices_,
        candidates=diagnostics)

from pathlib import Path
p=Path('src/takehome/fitters.py')
s=p.read_text()
pos=s.index('\n\nclass BatchedFitters:')
s=s[:pos]+'''

def calendar_walk_forward_folds(index, *, train_years=2, test_years=2,
                                first_prediction=None, gap=0):
    """Rolling calendar-year training and nonoverlapping prediction windows.

    Bounds are left-closed/right-open, located with searchsorted (not bars/year).
    Calendar boundaries stay anchored to the original timestamp, including leap
    years. The final prediction window is clipped to available rows. Optional
    first_prediction anchors a final withheld-period fit; gap embargoes extra
    rows before that boundary. Time metadata retains the nominal calendar bounds.
    """
    import pandas as pd

    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans
            or not index.is_monotonic_increasing or not index.is_unique):
        raise ValueError('Require a sorted, unique DatetimeIndex without NaT.')
    for value in (train_years, test_years):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError('Calendar window lengths must be positive integer years.')
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or gap < 0:
        raise ValueError('gap must be a nonnegative integer number of rows.')
    if len(index) == 0:
        return []
    origin = index[0] if first_prediction is None else pd.Timestamp(first_prediction)
    if pd.isna(origin) or (origin.tz is None) != (index.tz is None):
        raise ValueError('first_prediction must have the same timezone awareness as the index.')
    if index.tz is not None:
        origin = origin.tz_convert(index.tz)
    offset = train_years if first_prediction is None else 0
    folds, i = [], 0
    while True:
        boundary = origin + pd.DateOffset(years=int(offset + i*test_years))
        end = origin + pd.DateOffset(years=int(offset + (i+1)*test_years))
        if boundary > index[-1]:
            break
        lower = boundary - pd.DateOffset(years=int(train_years))
        start = int(index.searchsorted(lower, side='left'))
        first = int(index.searchsorted(boundary, side='left'))
        stop = first - gap
        last = int(index.searchsorted(end, side='left'))
        if first < last:
            if stop <= start:
                raise ValueError('A calendar fold has no training rows before its prediction window.')
            folds.append(dict(train_start=start, train_stop=stop,
                predict_start=first, predict_stop=last,
                train_start_time=lower, train_stop_time=boundary,
                predict_start_time=boundary, predict_stop_time=end))
        i += 1
    return folds


def _validated_folds(folds, n):
    """Copy and validate explicit row bounds before any estimator is fitted."""
    result, previous_stop = [], None
    keys = ('train_start', 'train_stop', 'predict_start', 'predict_stop')
    for fold in folds:
        if not isinstance(fold, dict) or any(k not in fold for k in keys):
            raise ValueError('Every fold requires train/predict start and stop bounds.')
        bounds = [fold[k] for k in keys]
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in bounds):
            raise ValueError('Explicit fold row bounds must be integers.')
        a, b, c, d = bounds
        if not 0 <= a < b <= c <= d <= n:
            raise ValueError('Require 0 <= train_start < train_stop <= predict_start <= predict_stop <= n.')
        if previous_stop is not None and c != previous_stop:
            raise ValueError('Explicit prediction folds must be contiguous and nonoverlapping.')
        result.append(fold.copy())
        previous_stop = d
    return result
''' + s[pos:]
s=s.replace('def __init__(self, fitter, *, min_train_size=1, step=1, train_size=None, gap=0):',
            'def __init__(self, fitter, *, min_train_size=1, step=1, train_size=None, gap=0, folds=None):',1)
s=s.replace('        self.fitter, self.n_features = fitter, fitter.n_features\n',
'''        self.fitter, self.n_features = fitter, fitter.n_features
        if folds is not None and (min_train_size != 1 or step != 1 or train_size is not None or gap != 0):
            raise ValueError('Use explicit folds or row-window parameters, not both.')
        self.explicit_folds = None if folds is None else [f.copy() for f in folds]
''',1)
s=s.replace("result = walk_forward_sweep({'model': self.fitter}, X, y, W=W, **self.fold_parameters)['model']",
            "result = walk_forward_sweep({'model': self.fitter}, X, y, W=W,\n                                    folds=self.explicit_folds, **self.fold_parameters)['model']",1)
s=s.replace('def walk_forward_sweep(fitters, X, y, W=None, **fold_parameters):',
            'def walk_forward_sweep(fitters, X, y, W=None, *, folds=None, **fold_parameters):',1)
s=s.replace("paths = {name: BatchedFitters(model, **fold_parameters)._reset(len(y)) for name, model in fitters.items()}\n    models = deepcopy(fitters)\n    for fold in walk_forward_folds(len(y), **fold_parameters):",
'''selected = (walk_forward_folds(len(y), **fold_parameters) if folds is None
                else _validated_folds(folds, len(y)))
    paths = {name: BatchedFitters(model, folds=folds, **fold_parameters)._reset(len(y))
             for name, model in fitters.items()}
    models = deepcopy(fitters)
    for fold in selected:''',1)
s=s.replace('''    """Return live and cutoff-frozen forecasts from ONE chronological replay.

    The frozen path respects each train_stop; live forecasts update each row.
    audit_folds optionally captures centered moments at selected checkpoints.
    """''','''    """Continuous live EWM forecasts and exactly window-matched frozen fits.

    Expanding folds reuse one chronological replay. Rolling folds independently
    replay only each training window for frozen fits; the live comparator and
    its full histories remain a continuous online estimator, not a rolling one.
    audit_folds captures the frozen training window's centered moments.
    """''',1)
anchor='''    if model.n_seen_:
        raise ValueError('Use a fresh streaming model for a full walk-forward replay.')
'''
replacement=anchor+'''    folds = _validated_folds(folds, len(y))
    if any(f['train_start'] != 0 for f in folds):
        live = model.fit_predict(X, y, W=w)
        frozen = np.full(len(y), np.nan)
        coefs, offsets, states, errors, failures = [], [], {}, [], []
        for i, fold in enumerate(folds):
            a, b, c, d = (fold[k] for k in ('train_start','train_stop','predict_start','predict_stop'))
            checkpoint = StreamingWeightedLasso(model.n_features, model.decay, model.alpha,
                max_iter=model.max_iter, tol=model.tol, fit_intercept=model.fit_intercept,
                store_history=False)
            checkpoint.fit(X[a:b], y[a:b], W=w[a:b])
            frozen[c:d] = checkpoint.predict(X[c:d])
            coefs.append(checkpoint.coef.copy()); offsets.append(checkpoint.intercept_)
            errors.append(checkpoint.kkt_violation_); failures.append(checkpoint.n_failed_)
            if i in audit_folds and checkpoint.Wsum > 0:
                states[i] = (checkpoint.mean_x_.copy(), checkpoint.mean_y_,
                             checkpoint.C_.copy()/checkpoint.Wsum,
                             checkpoint.c_.copy()/checkpoint.Wsum)
        return dict(live=live, frozen=frozen, coefs=np.array(coefs),
                    intercepts=np.array(offsets), states=states,
                    kkt_at_folds=np.array(errors), frozen_failed_updates=np.array(failures))
'''
assert anchor in s
s=s.replace(anchor,replacement,1)
p.write_text(s)

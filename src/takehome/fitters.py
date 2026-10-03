"""Online, feature-scaled lasso with exponential forgetting and diagonal W.

At step t, a_i = W_i * decay**(t-i). Minimize
    sum(a_i * (y_i - b - x_i @ beta)**2) / (2 * sum(a_i))
    + alpha * sum(sigma_j * abs(beta_j)).
Sigma is the weighted population feature standard deviation, not its RMS.
No intercept by default, as in the original sketch; fit_intercept=True centers
both the Gram matrix and cross-product and leaves the intercept unpenalized.
"""
from __future__ import annotations

import warnings
import numpy as np
from numba import njit, types
from numba.experimental import jitclass
from numba.typed import List


@njit(cache=True)
def _coordinate_descent(G, h, theta, alpha, max_iter, tol):
    """Warm-started cyclic CD; maintain correlations, verify full KKT each sweep."""
    threshold = tol * (1.0 + np.max(np.abs(h)))
    residual = h - G @ theta
    for iteration in range(max_iter):
        for j in range(len(theta)):
            rho = residual[j] + G[j, j] * theta[j]
            updated = (np.sign(rho) * max(abs(rho) - alpha, 0.0) / G[j, j]
                       if G[j, j] > 0 else 0.0)
            delta = updated - theta[j]
            if delta != 0:
                theta[j] = updated
                for k in range(len(theta)):
                    residual[k] -= G[k, j] * delta
        # Recompute to prevent accumulated roundoff from passing a false KKT test.
        residual = h - G @ theta
        error = 0.0
        for j in range(len(theta)):
            v = (abs(residual[j] - alpha * np.sign(theta[j])) if theta[j] != 0
                 else max(abs(residual[j]) - alpha, 0.0))
            error = max(error, v)
        if error <= threshold:
            break
    return iteration + 1, error, threshold


@njit(cache=True)
def _stream_scalar_ols(X, y, weights, C, c, mx, my, total, coef, intercept,
                       decay, tol, fit_intercept, record):
    """Exact one-coordinate alpha=0 solve; the same weighted centered statistics."""
    n = len(y)
    history = np.empty((n if record else 0, 1))
    intercepts = np.empty(n if record else 0)
    xx, xy, mean, beta = C[0, 0], c[0], mx[0], coef[0]
    failed, iterations, error, threshold = 0, 0, 0.0, tol
    predictions = np.full(n, np.nan)
    for t in range(n):
        if total > 0:
            predictions[t] = X[t, 0]*beta + intercept
        old = decay * total
        total = old + weights[t]
        xx *= decay
        xy *= decay
        if weights[t] > 0:
            dx, dy = X[t, 0] - mean, y[t] - my
            fraction = weights[t] / total
            xx += old * fraction * dx * dx
            xy += old * fraction * dx * dy
            mean += fraction * dx
            my += fraction * dy
            g = xx / total + (0.0 if fit_intercept else mean * mean)
            h = xy / total + (0.0 if fit_intercept else mean * my)
            beta = h / g if g > 0 else 0.0
            intercept = my - mean * beta if fit_intercept else 0.0
            scale = np.sqrt(max(xx / total, 0.0))
            if scale == 0:
                scale = abs(mean) if not fit_intercept and mean != 0 else 1.0
            error, threshold = abs(g * beta - h) / scale, tol * (1 + abs(h / scale))
            failed += error > threshold
            iterations = 1
        if record:
            history[t, 0], intercepts[t] = beta, intercept
    C[0, 0], c[0], mx[0], coef[0] = xx, xy, mean, beta
    return total, my, intercept, history, intercepts, failed, iterations, error, threshold, predictions


@njit(cache=True)
def _stream(X, y, weights, C, c, mx, my, total, coef, intercept,
            decay, alpha, max_iter, tol, fit_intercept, record):
    """Weighted Welford updates avoid subtracting two large raw moments."""
    if X.shape[1] == 1 and alpha == 0:
        return _stream_scalar_ols(X, y, weights, C, c, mx, my, total, coef, intercept,
                                  decay, tol, fit_intercept, record)
    n, p = X.shape
    history = np.empty((n if record else 0, p))
    intercepts = np.empty(n if record else 0)
    G, h, scale, theta = np.empty((p, p)), np.empty(p), np.empty(p), np.empty(p)
    failed, iterations, error, threshold = 0, 0, 0.0, tol
    predictions = np.full(n, np.nan)
    for t in range(n):
        if total > 0:
            predictions[t] = X[t] @ coef + intercept
        old = decay * total
        total = old + weights[t]
        C *= decay
        c *= decay
        if weights[t] > 0:
            dx, dy = X[t] - mx, y[t] - my
            fraction = weights[t] / total
            correction = old * fraction
            for j in range(p):
                c[j] += correction * dx[j] * dy
                for k in range(p):
                    C[j, k] += correction * dx[j] * dx[k]
            mx += fraction * dx
            my += fraction * dy
            for j in range(p):
                variance = max(C[j, j] / total, 0.0)
                scale[j] = np.sqrt(variance)
                if scale[j] == 0:
                    # Do not leave a nonzero constant column unpenalized.
                    scale[j] = abs(mx[j]) if not fit_intercept and mx[j] != 0 else 1.0
                theta[j] = coef[j] * scale[j]
            for j in range(p):
                h[j] = (c[j] / total + (0.0 if fit_intercept else mx[j] * my)) / scale[j]
                for k in range(p):
                    value = C[j, k] / total + (0.0 if fit_intercept else mx[j] * mx[k])
                    G[j, k] = value / scale[j] / scale[k]
            iterations, error, threshold = _coordinate_descent(G, h, theta, alpha, max_iter, tol)
            failed += error > threshold
            coef[:] = theta / scale
            intercept = my - mx @ coef if fit_intercept else 0.0
        if record:
            history[t] = coef
            intercepts[t] = intercept
    return total, my, intercept, history, intercepts, failed, iterations, error, threshold, predictions


_CORE_SPEC = (
    [(k, types.int64) for k in ('n_features', 'max_iter', 'n_seen_', 'n_failed_', 'n_iter_')]
    + [(k, types.float64) for k in ('decay', 'alpha', 'tol', 'Wsum', 'mean_y_', 'intercept_',
                                   'kkt_violation_', 'kkt_tolerance_')]
    + [(k, types.boolean) for k in ('fit_intercept', 'store_history', 'converged_')]
    + [('C_', types.float64[:, ::1])]
    + [(k, types.float64[::1]) for k in ('c_', 'mean_x_', 'coef')]
    + [('history', types.ListType(types.float64[:, ::1])),
       ('intercepts', types.ListType(types.float64[::1]))]
)


@jitclass(_CORE_SPEC)
class StreamingWeightedLasso_:
    """Compiled state and methods. Python validation lives in the public wrapper."""
    def __init__(self, n_features, decay, alpha, max_iter, tol, fit_intercept, store_history):
        self.n_features, self.decay, self.alpha = n_features, decay, alpha
        self.max_iter, self.tol = max_iter, tol
        self.fit_intercept, self.store_history = fit_intercept, store_history
        self.C_ = np.zeros((n_features, n_features))
        self.c_, self.mean_x_, self.coef = np.zeros(n_features), np.zeros(n_features), np.zeros(n_features)
        self.Wsum = self.mean_y_ = self.intercept_ = 0.0
        self.n_seen_ = self.n_failed_ = self.n_iter_ = 0
        self.kkt_violation_, self.kkt_tolerance_, self.converged_ = 0.0, tol, True
        self.history = List.empty_list(types.float64[:, ::1])
        self.intercepts = List.empty_list(types.float64[::1])

    def fit(self, X, y, weights):
        """Append rows and emit predictions before their updates."""
        result = _stream(X, y, weights, self.C_, self.c_, self.mean_x_, self.mean_y_,
                         self.Wsum, self.coef, self.intercept_, self.decay, self.alpha,
                         self.max_iter, self.tol, self.fit_intercept, self.store_history)
        self.Wsum, self.mean_y_, self.intercept_ = result[0], result[1], result[2]
        self.n_seen_ += len(y)
        self.n_failed_ += result[5]
        if np.any(weights > 0):
            self.n_iter_, self.kkt_violation_, self.kkt_tolerance_ = result[6], result[7], result[8]
            self.converged_ = result[7] <= result[8]
        if self.store_history:
            self.history.append(result[3])
            self.intercepts.append(result[4])
        return result[9]

    def partial_fit(self, x, y, weight=1.0):
        self.fit(x.reshape((1, self.n_features)), np.array([y]), np.array([weight]))
        return self

    def predict(self, X):
        return X @ self.coef + self.intercept_

    def get_coefs(self):
        n = 0
        for block in self.history:
            n += len(block)
        out = np.empty((n, self.n_features))
        start = 0
        for block in self.history:
            out[start:start+len(block)] = block
            start += len(block)
        return out

    def get_intercepts(self):
        n = 0
        for block in self.intercepts:
            n += len(block)
        out = np.empty(n)
        start = 0
        for block in self.intercepts:
            out[start:start+len(block)] = block
            start += len(block)
        return out


class StreamingWeightedLasso:
    """Validated Python facade; `_core` owns the compiled estimator state.

    fit appends. fit_predict emits beta[t-1] @ X[t] + b[t-1]. W=0 advances
    decay without adding an observation. History is optional and post-update.
    """
    def __init__(self, n_features, decay, alpha, max_iter=1000, tol=1e-8,
                 *, fit_intercept=False, store_history=False):
        if (not isinstance(n_features, (int, np.integer)) or n_features < 1
                or not isinstance(max_iter, (int, np.integer)) or max_iter < 1
                or not np.isfinite([decay, alpha, tol]).all()
                or not 0 < decay <= 1 or alpha < 0 or tol <= 0):
            raise ValueError('Require p>=1, max_iter>=1, 0<decay<=1, alpha>=0 and tol>0.')
        self._core = StreamingWeightedLasso_(int(n_features), float(decay), float(alpha),
                         int(max_iter), float(tol), bool(fit_intercept), bool(store_history))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_core'), name)

    @property
    def history(self):
        return list(self._core.history)

    def _run(self, X, y, w):
        failed = self._core.n_failed_
        result = self._core.fit(X, y, w)
        failed = self._core.n_failed_ - failed
        if failed:
            warnings.warn(f'{failed} updates did not converge; inspect KKT errors or raise max_iter.',
                          RuntimeWarning, stacklevel=3)
        return result

    def fit(self, X, y, W=None):
        X, y, w = _fit_input(X, y, W, self.n_features)
        if len(y):
            self._run(X, y, w)
        return self

    def fit_predict(self, X, y, W=None, *, chunk_size=8192):
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer.')
        X, y, w = _fit_input(X, y, W, self.n_features)
        out = np.full(len(y), np.nan)
        for start in range(0, len(y), chunk_size):
            stop = min(start+chunk_size, len(y))
            out[start:stop] = self._run(X[start:stop], y[start:stop], w[start:stop])
        return out

    def partial_fit(self, x, y, weight=1.0):
        return self.fit(np.asarray(x, dtype=float).reshape(1, -1), np.array([y]), W=weight)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim not in (1, 2) or X.shape[-1] != self.n_features or not np.isfinite(X).all():
            raise ValueError('Prediction input must be finite with p columns.')
        return X @ self.coef + self.intercept_

    def get_coefs(self, lag=0):
        if not self.store_history:
            raise ValueError('Enable store_history=True to retain coefficient snapshots.')
        return _lagged(self._core.get_coefs(), lag)

    def get_intercepts(self, lag=0):
        if not self.store_history:
            raise ValueError('Enable store_history=True to retain intercept snapshots.')
        return _lagged(self._core.get_intercepts(), lag)

    def __getstate__(self):
        state = {name: getattr(self._core, name) for name, _ in _CORE_SPEC}
        state['history'], state['intercepts'] = list(self._core.history), list(self._core.intercepts)
        return state

    def __setstate__(self, state):
        self.__init__(*(state[k] for k in ('n_features','decay','alpha','max_iter','tol')),
                      fit_intercept=state['fit_intercept'], store_history=state['store_history'])
        for name, value in state.items():
            if name in ('history', 'intercepts'):
                for block in value:
                    getattr(self._core, name).append(block)
            else:
                setattr(self._core, name, value)


def _fit_input(X, y, W, p):
    X, y = np.ascontiguousarray(X, dtype=float), np.ascontiguousarray(y, dtype=float)
    if X.ndim != 2 or X.shape[1] != p or y.shape != (len(X),):
        raise ValueError('Expected X shape (n, p) and y shape (n,).')
    w = np.asarray(1.0 if W is None else W, dtype=float)
    if w.ndim == 0:
        w = np.full(len(y), w.item())
    elif w.ndim == 2:
        if w.shape != (len(y), len(y)) or np.count_nonzero(w) != np.count_nonzero(np.diag(w)):
            raise ValueError('Only diagonal W is supported; prefer a weight vector.')
        w = np.diag(w)
    if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Weights must be finite, nonnegative, and length n.')
    if not np.isfinite(X[w > 0]).all() or not np.isfinite(y[w > 0]).all():
        raise ValueError('Positive-weight rows must be finite; use W=0 to skip.')
    return X, y, np.ascontiguousarray(w)


def _lagged(values, lag):
    if not isinstance(lag, (int, np.integer)) or lag < 0:
        raise ValueError('lag must be a nonnegative integer.')
    result = np.full_like(values, np.nan)
    if lag == 0:
        return values.copy()
    if lag < len(values):
        result[lag:] = values[:-lag]
    return result


class CvxpyWeightedLasso:
    """Independent batch solution of the module's loss, using original rows.

    fit REPLACES the model; no streaming moments/Numba solver are reused.
    Exponential ages include zero-weight rows. W is a diagonal/vector weight.
    """
    def __init__(self, n_features, decay=1.0, alpha=.01, *, fit_intercept=False,
                 tol=1e-11, max_iter=2000):
        if (not isinstance(n_features, (int, np.integer)) or n_features < 1
                or not isinstance(max_iter, (int, np.integer)) or max_iter < 1
                or not np.isfinite([decay, alpha, tol]).all()
                or not 0 < decay <= 1 or alpha < 0 or tol <= 0):
            raise ValueError('Invalid feature count, decay, penalty, tolerance or iteration limit.')
        self.n_features, self.decay, self.alpha = n_features, decay, alpha
        self.fit_intercept, self.tol, self.max_iter = fit_intercept, tol, max_iter
        self.coef, self.intercept_ = np.zeros(n_features), 0.0

    def fit(self, X, y, W=None):
        import cvxpy as cp

        X, y, w = _fit_input(X, y, W, self.n_features)
        keep = w > 0
        self.coef, self.intercept_, self.objective_ = np.zeros(self.n_features), 0.0, 0.0
        if not keep.any():
            self.status_ = 'no_observations'
            return self
        # Stable normalization; common weight units cannot change the fit.
        log_a = np.log(w[keep]) + np.log(self.decay) * np.arange(len(y)-1, -1, -1)[keep]
        a = np.exp(log_a - log_a.max())
        a /= a.sum()
        X, y = X[keep], y[keep]
        mx, my = X[0] + a @ (X - X[0]), y[0] + a @ (y - y[0])
        scale = np.sqrt(a @ (X - mx)**2)
        scale = np.where(scale > 0, scale, np.where(self.fit_intercept, 1., np.abs(mx)))
        self.scale_ = np.where(scale > 0, scale, 1.)
        Z = (X - mx if self.fit_intercept else X) / self.scale_
        target = y - my if self.fit_intercept else y
        unit = np.sqrt(a @ target**2) or 1.0
        theta = cp.Variable(self.n_features)
        offset = cp.Variable() if self.fit_intercept else 0.0
        residual = target / unit - Z @ theta - offset
        loss = cp.sum_squares(cp.multiply(np.sqrt(a), residual)) / 2
        problem = cp.Problem(cp.Minimize(loss + self.alpha / unit * cp.norm1(theta)))
        problem.solve(solver='CLARABEL', tol_gap_abs=self.tol, tol_gap_rel=self.tol,
                      tol_feas=self.tol, max_iter=self.max_iter)
        self.status_ = problem.status
        if self.status_ != cp.OPTIMAL:
            raise RuntimeError(f'CVXPY reference did not solve accurately: {self.status_}')
        self.coef = np.asarray(theta.value).ravel() * unit / self.scale_
        self.intercept_ = (float(offset.value) * unit + my - mx @ self.coef
                           if self.fit_intercept else 0.0)
        self.objective_ = .5 * (a @ (y - X @ self.coef - self.intercept_)**2)
        self.objective_ += self.alpha * np.sum(self.scale_ * np.abs(self.coef))
        return self

    predict = StreamingWeightedLasso.predict


def batch_moments(X, y, weights, decay):
    """Independent, row-based weighted moments; ages include zero-weight rows."""
    keep = weights > 0
    if not keep.any():
        return None
    log_a = np.log(weights[keep]) + np.log(decay) * np.arange(len(y)-1, -1, -1)[keep]
    a = np.exp(log_a-log_a.max()); a /= a.sum()
    X, y = X[keep], y[keep]
    mx, my = X[0] + a @ (X-X[0]), y[0] + a @ (y-y[0])
    dx, dy = X-mx, y-my
    C = dx.T @ (a[:, None]*dx)
    return mx, my, (C+C.T)/2, dx.T @ (a*dy), float(a @ dy**2)


def _scaled_moments(moments, intercept):
    mx, my, C, c, vy = moments
    scale = np.sqrt(np.maximum(np.diag(C), 0))
    scale = np.where(scale > 0, scale, np.where(intercept, 1., np.abs(mx)))
    scale = np.where(scale > 0, scale, 1.)
    G = C if intercept else C + np.outer(mx, mx)
    h = c if intercept else c + mx*my
    return G/scale[:, None]/scale, h/scale, scale, vy+(0 if intercept else my**2)


class BatchLasso(CvxpyWeightedLasso):
    """CVXPY quadratic loss from independently recomputed batch moments.

    The row-residual CvxpyWeightedLasso remains the independent reference.
    This p-variable form makes whole-history walk-forward refits inexpensive.
    """
    def fit(self, X, y, W=None):
        X, y, w = _fit_input(X, y, W, self.n_features)
        return self.fit_moments(batch_moments(X, y, w, self.decay))

    def fit_moments(self, moments):
        import cvxpy as cp

        if moments is None:
            self.coef, self.intercept_, self.objective_ = np.zeros(self.n_features), 0., 0.
            self.status_ = 'no_observations'
            return self
        G, h, self.scale_, v = _scaled_moments(moments, self.fit_intercept)
        unit = np.sqrt(max(v, 0)) or 1.
        theta = cp.Variable(self.n_features)
        loss = .5*cp.quad_form(theta, cp.psd_wrap(G)) - (h/unit) @ theta
        problem = cp.Problem(cp.Minimize(loss + self.alpha/unit*cp.norm1(theta)))
        problem.solve(solver='CLARABEL', tol_gap_abs=self.tol, tol_gap_rel=self.tol,
                      tol_feas=self.tol, max_iter=self.max_iter)
        self.status_ = problem.status
        if problem.status != cp.OPTIMAL:
            raise RuntimeError(f'Batch lasso did not solve accurately: {problem.status}')
        scaled = np.asarray(theta.value).ravel()*unit
        self.coef = scaled/self.scale_
        self.intercept_ = float(moments[1]-moments[0]@self.coef) if self.fit_intercept else 0.
        self.objective_ = .5*(v-2*h@scaled+scaled@G@scaled) + self.alpha*np.abs(scaled).sum()
        gradient = G@scaled-h
        active = np.abs(scaled) > 1e-8*unit
        violation = np.where(active, np.abs(gradient+self.alpha*np.sign(scaled)),
                              np.maximum(np.abs(gradient)-self.alpha, 0))
        self.kkt_violation_ = float(violation.max())
        return self


class BatchRidge:
    """Weighted loss/2 + alpha*||feature_std*beta||^2/2; fit overwrites."""
    def __init__(self, n_features, alpha=.01, *, decay=1., fit_intercept=True):
        if (not isinstance(n_features, (int, np.integer)) or n_features < 1
                or not np.isfinite([alpha, decay]).all() or alpha <= 0 or not 0 < decay <= 1):
            raise ValueError('Require p>=1, alpha>0 and 0<decay<=1.')
        self.n_features, self.alpha, self.decay = n_features, float(alpha), float(decay)
        self.fit_intercept = fit_intercept
        self.coef, self.intercept_ = np.zeros(n_features), 0.

    def fit(self, X, y, W=None):
        X, y, w = _fit_input(X, y, W, self.n_features)
        return self.fit_moments(batch_moments(X, y, w, self.decay))

    def fit_moments(self, moments):
        if moments is None:
            raise ValueError('Ridge requires at least one positive-weight row.')
        G, h, self.scale_, v = _scaled_moments(moments, self.fit_intercept)
        theta = np.linalg.solve(G+self.alpha*np.eye(self.n_features), h)
        self.coef = theta/self.scale_
        self.intercept_ = float(moments[1]-moments[0]@self.coef) if self.fit_intercept else 0.
        self.objective_ = .5*(v-2*h@theta+theta@G@theta+self.alpha*(theta@theta))
        return self

    predict = StreamingWeightedLasso.predict


def walk_forward_folds(n, *, min_train_size=1, step=1, train_size=None, gap=0):
    """Exclusive train_stop <= predict_start; gap counts extra embargo rows."""
    for value, minimum in [(min_train_size, 1), (step, 1), (gap, 0)]:
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError('Use positive integer sizes and a nonnegative integer gap.')
    if train_size is not None and (not isinstance(train_size, (int, np.integer))
                                   or train_size < min_train_size):
        raise ValueError('train_size must be None or >= min_train_size.')
    return [dict(train_start=0 if train_size is None else max(0, first-gap-train_size),
                 train_stop=first-gap, predict_start=first, predict_stop=min(first+step, n))
            for first in range(min_train_size+gap, n+1, step)]


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


class BatchedFitters:
    """Frozen next-fold predictions from a batch estimator; fit rebuilds the path.

    prediction_ is OOS. get_coefs(lag=1) reproduces it; default history is
    post-fit like StreamingWeightedLasso. Only fold snapshots are stored;
    requesting a row-level coefficient history explicitly expands it.
    """
    def __init__(self, fitter, *, min_train_size=1, step=1, train_size=None, gap=0, folds=None):
        self.fitter, self.n_features = fitter, fitter.n_features
        if folds is not None and (min_train_size != 1 or step != 1 or train_size is not None or gap != 0):
            raise ValueError('Use explicit folds or row-window parameters, not both.')
        self.explicit_folds = None if folds is None else [f.copy() for f in folds]
        self.fold_parameters = dict(min_train_size=min_train_size, step=step,
                                    train_size=train_size, gap=gap)
        walk_forward_folds(0, **self.fold_parameters)
        self._reset(0)

    def _reset(self, n):
        self.n_rows_ = n
        self.coef, self.intercept_ = np.zeros(self.n_features), 0.
        self.folds_, self.snapshots_, self.offsets_, self.objectives_ = [], [], [], []
        self.prediction_ = np.full(n, np.nan)
        return self

    def _record(self, fold, model, X):
        self.coef, self.intercept_ = model.coef.copy(), float(model.intercept_)
        self.folds_.append(fold.copy()); self.snapshots_.append(self.coef.copy())
        self.offsets_.append(self.intercept_)
        self.objectives_.append(getattr(model, 'objective_', np.nan))
        start, stop = fold['predict_start'], fold['predict_stop']
        self.prediction_[start:stop] = X[start:stop]@self.coef+self.intercept_

    def fit(self, X, y, W=None):
        result = walk_forward_sweep({'model': self.fitter}, X, y, W=W,
                                    folds=self.explicit_folds, **self.fold_parameters)['model']
        self.__dict__.update(result.__dict__)
        return self

    predict = StreamingWeightedLasso.predict

    def get_coefs(self, lag=0):
        out = np.full((self.n_rows_, self.n_features), np.nan)
        for i, fold in enumerate(self.folds_):
            stop = fold['predict_stop'] - 1
            if fold['predict_stop'] == self.n_rows_:
                stop = self.n_rows_  # Retain the final post-update snapshot.
            out[fold['predict_start']-1:stop] = self.snapshots_[i]
        return _lagged(out, lag)

    def get_intercepts(self, lag=0):
        out = np.full(self.n_rows_, np.nan)
        for i, fold in enumerate(self.folds_):
            stop = fold['predict_stop'] - 1
            if fold['predict_stop'] == self.n_rows_:
                stop = self.n_rows_  # Retain the final post-update snapshot.
            out[fold['predict_start']-1:stop] = self.offsets_[i]
        return _lagged(out, lag)


def walk_forward_sweep(fitters, X, y, W=None, *, folds=None, **fold_parameters):
    """Named estimators -> BatchedFitters paths; share moments across penalties.

    Scales, means and moments use prior training rows only. Generic estimators
    need fit(X,y,W), coef, intercept_, n_features; moment support is optional.
    No best-alpha choice is made using the returned evaluation-period scores.
    """
    from copy import deepcopy

    if not fitters:
        raise ValueError('Supply at least one named fitter.')
    p = next(iter(fitters.values())).n_features
    if any(model.n_features != p for model in fitters.values()):
        raise ValueError('All fitters must use the same design.')
    X, y, w = _fit_input(X, y, W, p)
    selected = (walk_forward_folds(len(y), **fold_parameters) if folds is None
                else _validated_folds(folds, len(y)))
    paths = {name: BatchedFitters(model, folds=selected if folds is not None else None,
                                 **fold_parameters)._reset(len(y))
             for name, model in fitters.items()}
    models = deepcopy(fitters)
    for fold in selected:
        start, stop = fold['train_start'], fold['train_stop']
        moments = {}
        for name, model in models.items():
            if hasattr(model, 'fit_moments'):
                if model.decay not in moments:
                    moments[model.decay] = batch_moments(X[start:stop], y[start:stop], w[start:stop], model.decay)
                model.fit_moments(moments[model.decay])
            else:
                model.fit(X[start:stop], y[start:stop], W=w[start:stop])
            paths[name]._record(fold, model, X)
    return paths


def stream_at_folds(model, X, y, folds, W=None, *, audit_folds=()):
    """Continuous live EWM forecasts and exactly window-matched frozen fits.

    Expanding folds reuse one chronological replay. Rolling folds independently
    replay only each training window for frozen fits; the live comparator and
    its full histories remain a continuous online estimator, not a rolling one.
    audit_folds captures the frozen training window's centered moments.
    """
    X, y, w = _fit_input(X, y, W, model.n_features)
    if model.n_seen_:
        raise ValueError('Use a fresh streaming model for a full walk-forward replay.')
    folds = _validated_folds(folds, len(y))
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
    live, frozen = np.full(len(y), np.nan), np.full(len(y), np.nan)
    coefs, offsets, states, errors = [], [], {}, []
    last = 0
    for i, fold in enumerate(folds):
        stop, first, end = (fold[k] for k in ('train_stop','predict_start','predict_stop'))
        if fold['train_start'] != 0 or not last <= stop <= first <= end <= len(y):
            raise ValueError('Streaming comparison requires ordered expanding-window folds.')
        live[last:stop] = model.fit_predict(X[last:stop], y[last:stop], W=w[last:stop])
        last = stop
        frozen[first:end] = X[first:end]@model.coef+model.intercept_
        coefs.append(model.coef.copy()); offsets.append(model.intercept_)
        errors.append(model.kkt_violation_)
        if i in audit_folds and model.Wsum > 0:
            states[i] = (model.mean_x_.copy(), model.mean_y_, model.C_.copy()/model.Wsum,
                         model.c_.copy()/model.Wsum)
    live[last:] = model.fit_predict(X[last:], y[last:], W=w[last:])
    return dict(live=live, frozen=frozen, coefs=np.array(coefs), intercepts=np.array(offsets),
                states=states, kkt_at_folds=np.array(errors))


def forecast_alpha(x, *, hl=288*21, alpha=.01, min_train=288*21,
                   tol=1e-8, max_iter=5000):
    """At row t forecast x[t+1] from x[t], x[t-1], after learning x[t].

    Fit x[s] ~ x[s-1] + x[s-2], with an intercept and unit complete-row weights.
    A fixed scale from the initial min_train complete examples makes alpha
    comparable across signal units. No forecast is published before that
    warm-up. Missing lags are never filled; zero-weight rows advance decay.
    Returns a same-index Series (decision timestamps) and fitting diagnostics.
    """
    import pandas as pd
    from time import perf_counter
    from .features import _time_index

    _time_index(x)
    if (not isinstance(min_train, (int, np.integer)) or min_train < 2
            or not np.isfinite([hl, alpha, tol]).all() or hl <= 0 or alpha < 0 or tol <= 0):
        raise ValueError('Require min_train>=2, hl>0, alpha>=0 and tol>0.')
    values = x.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    lags = np.full((len(x), 2), np.nan)
    lags[1:, 0], lags[2:, 1] = values[:-1], values[:-2]
    valid = np.isfinite(values) & np.isfinite(lags).all(axis=1)
    locations = np.flatnonzero(valid)
    info = dict(feature=x.name, alpha=alpha, n_updates=int(valid.sum()),
                failed_updates=0, scale=np.nan, seconds=0.)
    if len(locations) < min_train:
        return pd.Series(np.nan, index=x.index, name=x.name), info
    warm = values[locations[:min_train]]
    warm = warm[warm != 0]  # Same zero-as-missing scale convention as ts_std.
    scale = (float(warm.std()) or float(np.sqrt(np.mean(warm**2))) or 1.) if len(warm) else 1.
    model = StreamingWeightedLasso(2, 2**(-1/hl), alpha, max_iter=max_iter, tol=tol,
                                   fit_intercept=True, store_history=True)
    started = perf_counter()
    model.fit(lags/scale, values/scale, W=valid.astype(float))
    # Post-update coefficients now know x_t, but never x_{t+1}.
    current = np.column_stack([values, lags[:, 0]])/scale
    prediction = (np.einsum('ij,ij->i', model.get_coefs(), current)
                  + model.get_intercepts())*scale
    ready = (np.cumsum(valid) >= min_train) & np.isfinite(current).all(axis=1)
    prediction[~ready] = np.nan
    info.update(failed_updates=model.n_failed_, scale=scale, seconds=perf_counter()-started)
    return pd.Series(prediction, index=x.index, name=x.name), info


def forecast_alpha_blocks(X, *, batch_size=8, diagnostics=None, **parameters):
    """Forecast every column independently; batch columns, never sample rows."""
    import pandas as pd

    if not isinstance(batch_size, int) or batch_size < 1 or not X.columns.is_unique:
        raise ValueError('Require positive batch_size and unique feature names.')
    for start in range(0, len(X.columns), batch_size):
        block = {}
        for name in X.columns[start:start+batch_size]:
            block[name], info = forecast_alpha(X[name], **parameters)
            if diagnostics is not None:
                diagnostics.append(info)
        yield pd.DataFrame(block, index=X.index)

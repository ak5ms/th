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
from numba import njit


@njit(cache=True)
def _coordinate_descent(G, h, theta, alpha, max_iter, tol):
    """Solve in standardized coefficient units; stop on the KKT residual."""
    threshold = tol * (1.0 + np.max(np.abs(h)))
    error = np.inf
    for iteration in range(max_iter):
        for j in range(len(theta)):
            rho = h[j] - G[j] @ theta + G[j, j] * theta[j]
            theta[j] = (np.sign(rho) * max(abs(rho) - alpha, 0.0) / G[j, j]
                        if G[j, j] > 0 else 0.0)
        gradient = G @ theta - h
        error = 0.0
        for j in range(len(theta)):
            v = (abs(gradient[j] + alpha * np.sign(theta[j])) if theta[j] != 0
                 else max(abs(gradient[j]) - alpha, 0.0))
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
    for t in range(n):
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
    return total, my, intercept, history, intercepts, failed, iterations, error, threshold


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
    for t in range(n):
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
    return total, my, intercept, history, intercepts, failed, iterations, error, threshold


class StreamingWeightedLasso:
    """fit appends observations, just like repeated partial_fit calls.

    W is a scalar, an observation-weight vector, or an explicitly diagonal
    matrix. Prefer the vector: no n-by-n matrix is allocated internally.
    Positive-weight rows must be finite. A zero-weight row adds no observation
    but still advances the forgetting clock, including when x/y are missing.
    Supply a label only after it is known; predict BEFORE partial_fit online.
    History is post-update and optional (O(n*p)); default state is O(p*p).
    """
    def __init__(self, n_features, decay, alpha, max_iter=1000, tol=1e-8,
                 *, fit_intercept=False, store_history=False):
        if (not isinstance(n_features, (int, np.integer)) or n_features < 1
                or not isinstance(max_iter, (int, np.integer)) or max_iter < 1
                or not np.isfinite([decay, alpha, tol]).all()
                or not 0 < decay <= 1 or alpha < 0 or tol <= 0):
            raise ValueError('Require p>=1, max_iter>=1, 0<decay<=1, alpha>=0 and tol>0.')
        self.n_features, self.decay, self.alpha = n_features, float(decay), float(alpha)
        self.max_iter, self.tol = max_iter, float(tol)
        self.fit_intercept, self.store_history = bool(fit_intercept), bool(store_history)
        self.C_ = np.zeros((n_features, n_features))
        self.c_, self.mean_x_, self.coef = (np.zeros(n_features) for _ in range(3))
        self.Wsum = self.mean_y_ = self.intercept_ = 0.0
        self.history, self._intercepts = [], []
        self.n_seen_ = self.n_failed_ = self.n_iter_ = 0
        self.kkt_violation_, self.kkt_tolerance_, self.converged_ = 0.0, self.tol, True

    def fit(self, X, y, W=None):
        """Append rows; W_i multiplies squared loss ONCE, not W_i squared."""
        X, y, weights = _fit_input(X, y, W, self.n_features)
        valid = weights > 0
        if not len(y):
            return self
        result = _stream(X, y, np.ascontiguousarray(weights), self.C_, self.c_, self.mean_x_,
                         self.mean_y_, self.Wsum, self.coef, self.intercept_, self.decay,
                         self.alpha, self.max_iter, self.tol, self.fit_intercept, self.store_history)
        (self.Wsum, self.mean_y_, self.intercept_, history, intercepts, failed,
         iterations, error, threshold) = result
        self.n_seen_ += len(y)
        self.n_failed_ += failed
        if valid.any():
            self.n_iter_, self.kkt_violation_, self.kkt_tolerance_ = iterations, error, threshold
            self.converged_ = error <= threshold
        if self.store_history:
            self.history.append(history)
            self._intercepts.append(intercepts)
        if failed:
            warnings.warn(f'{failed} updates did not converge; raise max_iter or inspect the design.',
                          RuntimeWarning, stacklevel=2)
        return self

    def partial_fit(self, x, y, weight=1.0):
        """Incorporate one newly available label and its observation weight."""
        return self.fit(np.asarray(x, dtype=float).reshape(1, -1), np.array([y]), W=weight)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim not in (1, 2) or X.shape[-1] != self.n_features or not np.isfinite(X).all():
            raise ValueError('Prediction input must be finite with p columns.')
        return X @ self.coef + self.intercept_

    def get_coefs(self, lag=0):
        """Post-update by default; lag=1 aligns the previous estimate with the next row."""
        if not self.store_history:
            raise ValueError('Enable store_history=True to retain coefficient snapshots.')
        values = np.concatenate(self.history) if self.history else np.empty((0, self.n_features))
        return _lagged(values, lag)

    def get_intercepts(self, lag=0):
        if not self.store_history:
            raise ValueError('Enable store_history=True to retain intercept snapshots.')
        values = np.concatenate(self._intercepts) if self._intercepts else np.empty(0)
        return _lagged(values, lag)


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


class BatchedFitters:
    """Wrap a fit-overwrites estimator with a row-clock walk-forward schedule.

    After row t, fit [max(0, t-gap+1-train_size), t-gap+1), every step rows.
    train_size=None expands; min_train_size controls the first fit. get_coefs()
    is POST-update like the online fitter; get_coefs(lag=1) is for the next row.
    gap is an additional label embargo. fit rebuilds the path, never appends.
    """
    def __init__(self, fitter, *, min_train_size=1, step=1, train_size=None, gap=0):
        for value, minimum in [(min_train_size, 1), (step, 1), (gap, 0)]:
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError('Use positive integer sizes and a nonnegative integer gap.')
        if train_size is not None and (not isinstance(train_size, (int, np.integer))
                                       or train_size < min_train_size):
            raise ValueError('train_size must be None or >= min_train_size.')
        self.fitter, self.n_features = fitter, fitter.n_features
        self.min_train_size, self.step, self.train_size, self.gap = min_train_size, step, train_size, gap
        self.coef, self.intercept_ = np.zeros(self.n_features), 0.0
        self._coefs, self._bias = np.empty((0, self.n_features)), np.empty(0)
        self.folds_ = []

    def fit(self, X, y, W=None):
        from copy import deepcopy

        X, y, w = _fit_input(X, y, W, self.n_features)
        self._coefs, self._bias = np.full_like(X, np.nan), np.full(len(y), np.nan)
        self.folds_ = []
        self.coef, self.intercept_ = np.zeros(self.n_features), 0.0
        for t in range(self.min_train_size + self.gap - 1, len(y), self.step):
            end = t - self.gap + 1
            start = 0 if self.train_size is None else max(0, end - self.train_size)
            model = deepcopy(self.fitter).fit(X[start:end], y[start:end], W=w[start:end])
            self.coef, self.intercept_ = model.coef.copy(), float(model.intercept_)
            self._coefs[t:t+self.step], self._bias[t:t+self.step] = self.coef, self.intercept_
            self.folds_.append(dict(train_start=start, train_stop=end,
                                    predict_start=t+1, predict_stop=min(t+self.step+1, len(y))))
        return self

    predict = StreamingWeightedLasso.predict

    def get_coefs(self, lag=0):
        return _lagged(self._coefs, lag)

    def get_intercepts(self, lag=0):
        return _lagged(self._bias, lag)

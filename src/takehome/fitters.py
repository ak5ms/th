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
        X, y = np.ascontiguousarray(X, dtype=float), np.ascontiguousarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features or y.shape != (len(X),):
            raise ValueError('Expected X shape (n, p) and y shape (n,).')
        weights = np.asarray(1.0 if W is None else W, dtype=float)
        if weights.ndim == 0:
            weights = np.full(len(y), weights.item())
        elif weights.ndim == 2:
            if (weights.shape != (len(y), len(y))
                    or np.count_nonzero(weights) != np.count_nonzero(np.diag(weights))):
                raise ValueError('Only diagonal W is supported; prefer a weight vector.')
            weights = np.diag(weights)
        if weights.shape != y.shape or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError('Weights must be finite, nonnegative, and length n.')
        valid = weights > 0
        if not np.isfinite(X[valid]).all() or not np.isfinite(y[valid]).all():
            raise ValueError('Positive-weight rows must have finite X and y; use W=0 to skip.')
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

    def get_coefs(self):
        """Return a copy of post-update coefficients, never pre-update forecasts."""
        if not self.store_history:
            raise ValueError('Enable store_history=True to retain coefficient snapshots.')
        return np.concatenate(self.history) if self.history else np.empty((0, self.n_features))

    def get_intercepts(self):
        if not self.store_history:
            raise ValueError('Enable store_history=True to retain intercept snapshots.')
        return np.concatenate(self._intercepts) if self._intercepts else np.empty(0)

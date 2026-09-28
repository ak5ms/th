

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

from pathlib import Path

cd = '''@njit(cache=True)
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


'''

method = '''    def fit_predict(self, X, y, W=None, *, chunk_size=8192):
        """Predict BEFORE each update; chunks bound temporary coefficient storage only.

        Exactly beta[t-1] @ X[t] + intercept[t-1], including across chunks/calls.
        Missing-target rows may use W=0 but still advance decay. With history
        disabled, only O(chunk_size*p) temporary coefficients are retained.
        """
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer.')
        X, y, w = _fit_input(X, y, W, self.n_features)
        result = np.full(len(y), np.nan)
        record, self.store_history = self.store_history, True
        try:
            for start in range(0, len(y), chunk_size):
                stop = min(start + chunk_size, len(y))
                prior, bias, seen = self.coef.copy(), self.intercept_, self.Wsum > 0
                self.fit(X[start:stop], y[start:stop], W=w[start:stop])
                coefs, offsets = self.history[-1], self._intercepts[-1]
                result[start] = X[start] @ prior + bias
                result[start+1:stop] = np.einsum('ij,ij->i', X[start+1:stop], coefs[:-1]) + offsets[:-1]
                ready = seen | ((np.cumsum(w[start:stop] > 0) - (w[start:stop] > 0)) > 0)
                result[start:stop] = np.where(ready, result[start:stop], np.nan)
                if not record:
                    self.history.pop()
                    self._intercepts.pop()
        finally:
            self.store_history = record
        return result

'''

ridge = '''

class BatchRidge:
    """Full supplied-sample ridge: weighted mean squared loss / 2 + alpha*||s*beta||^2/2.

    Population feature scaling and an unpenalized intercept. fit overwrites.
    Predicting its training rows is IN-SAMPLE, not a walk-forward forecast.
    """
    def __init__(self, n_features, alpha=.01):
        if not isinstance(n_features, (int, np.integer)) or n_features < 1 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('Require a positive feature count and ridge alpha > 0.')
        self.n_features, self.alpha = n_features, float(alpha)
        self.coef, self.intercept_ = np.zeros(n_features), 0.0

    def fit(self, X, y, W=None):
        X, y, w = _fit_input(X, y, W, self.n_features)
        keep = w > 0
        if not keep.any():
            raise ValueError('Ridge requires at least one positive-weight row.')
        X, y, w = X[keep], y[keep], w[keep]
        w = w / w.max()
        w /= w.sum()
        mx, my = X[0] + w @ (X-X[0]), y[0] + w @ (y-y[0])
        centered = X - mx
        self.scale_ = np.sqrt(w @ centered**2)
        self.scale_[self.scale_ == 0] = 1.0
        Z = centered / self.scale_
        G, h = Z.T @ (w[:, None] * Z), Z.T @ (w * (y-my))
        theta = np.linalg.solve(G + self.alpha*np.eye(self.n_features), h)
        self.coef = theta / self.scale_
        self.intercept_ = float(my - mx @ self.coef)
        return self

    predict = StreamingWeightedLasso.predict
'''

backtest = '''

def backtest(signal, returns, hl=288*21, *, normalization='variance'):
    """Per-bar diagnostic P&L. Forecasts must already use prior-row coefficients.

    variance preserves standalone alpha / EWMstd(alpha)^2; zscore implements
    the separately requested alpha / ts_zscore(alpha) literally. No costs,
    leverage cap, extra shift, or filtering of near-zero denominators.
    """
    if not signal.index.equals(returns.index):
        raise ValueError('Signal and return indexes must match.')
    if normalization == 'variance':
        denominator = ts_std(signal, hl).pow(2)
    elif normalization == 'zscore':
        denominator = ts_zscore(signal, hl)
    else:
        raise ValueError('normalization must be variance or zscore.')
    return signal.div(denominator.replace(0, np.nan)).mul(returns, axis=0).replace([np.inf, -np.inf], np.nan)
'''

if __name__ == '__main__':
    p=Path('src/takehome/fitters.py'); s=p.read_text()
    start=s.index('@njit(cache=True)\ndef _coordinate_descent')
    end=s.index('@njit(cache=True)', start+10)
    s=s[:start]+cd+s[end:]
    assert 'def fit_predict' not in s and 'class BatchRidge' not in s
    s=s.replace('    def partial_fit(self, x, y, weight=1.0):', method+'    def partial_fit(self, x, y, weight=1.0):',1)
    p.write_text(s+ridge)
    p=Path('src/takehome/features.py'); p.write_text(p.read_text()+backtest)

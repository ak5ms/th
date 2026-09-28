"""One-time source migration, removed after tests and notebook execution."""
from pathlib import Path

path=Path('src/takehome/fitters.py')
s=path.read_text()
assert 'class CvxpyWeightedLasso' not in s
start=s.index('        X, y = np.ascontiguousarray',s.index('    def fit(self, X, y, W=None):'))
stop=s.index('        if not len(y):',start)
s=s[:start]+"        X, y, weights = _fit_input(X, y, W, self.n_features)\n        valid = weights > 0\n"+s[stop:]
s=s.replace('    def get_coefs(self):','    def get_coefs(self, lag=0):').replace('    def get_intercepts(self):','    def get_intercepts(self, lag=0):')
s=s.replace('Return a copy of post-update coefficients, never pre-update forecasts.', 'Post-update by default; lag=1 aligns the previous estimate with the next row.')
s=s.replace('        return np.concatenate(self.history) if self.history else np.empty((0, self.n_features))',
            '        values = np.concatenate(self.history) if self.history else np.empty((0, self.n_features))\n        return _lagged(values, lag)')
s=s.replace('        return np.concatenate(self._intercepts) if self._intercepts else np.empty(0)',
            '        values = np.concatenate(self._intercepts) if self._intercepts else np.empty(0)\n        return _lagged(values, lag)')
path.write_text(s+Path('_fitter_extension.py').read_text())
path=Path('src/takehome/features.py')
s=path.read_text()
s=s.replace('def evaluate_features(blocks, returns: pd.Series, hl: int = 288 * 21):',
            'def evaluate_features(blocks, returns: pd.Series, hl: int = 288 * 21, *,\n                      meta=False, meta_hl=252*288, positive_only=False):')
s=s.replace('    daily, summaries = [], []\n', '    daily, summaries = [], []\n    numerator = pd.Series(0., index=returns.index) if meta else None\n    gross = numerator.copy() if meta else None\n')
s=s.replace("        days = pnl.resample('D').sum()", "        if meta:\n            n, g = _blend_totals(pnl, meta_hl, positive_only)\n            numerator += n\n            gross += g\n        days = pnl.resample('D').sum()")
s=s.replace("    return pd.concat(daily, axis=1), pd.concat(summaries).sort_values('daily_mean_over_std', ascending=False)",
            "    result = (pd.concat(daily, axis=1), pd.concat(summaries).sort_values('daily_mean_over_std', ascending=False))\n    if meta:\n        return (*result, numerator.div(gross.replace(0, np.nan)).fillna(0).rename('meta_pnl'))\n    return result")
path.write_text(s+Path('_blend_extension.py').read_text())
path=Path('tests/test_fitters.py')
s=path.read_text().replace('from sklearn.linear_model import Lasso\n','')
s=s.replace('from takehome.fitters import StreamingWeightedLasso', 'from takehome.fitters import StreamingWeightedLasso, CvxpyWeightedLasso')
start=s.index('def reference(');stop=s.index('\n\n@pytest',start)
s=s[:start]+'''def reference(X, y, weights, decay, alpha, intercept):
    ref = CvxpyWeightedLasso(X.shape[1], decay, alpha, fit_intercept=intercept,
                            tol=1e-12, max_iter=5000).fit(X, y, W=weights)
    return ref.coef, ref.intercept_
'''+s[stop:]
path.write_text(s)
path=Path('requirements.txt')
lines=[line for line in path.read_text().splitlines() if 'scikit-learn' not in line]
path.write_text('\n'.join(lines)+'\ncvxpy>=1.6,<2\npandas_market_calendars==5.4.0\n')
print('Replaced sklearn reference with direct CVXPY loss; added batched fitting and global meta-normalization.')

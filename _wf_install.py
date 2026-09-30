"""One-time source migration; removed after the executed notebook is saved."""
from pathlib import Path

p=Path('src/takehome/fitters.py'); source=p.read_text()
assert 'class StreamingWeightedLasso_:' not in source
kernels=source[:source.index('class StreamingWeightedLasso:')]
kernels=kernels.replace('from numba import njit', 'from numba import njit, types\nfrom numba.experimental import jitclass\nfrom numba.typed import List')
a=kernels.index('def _stream_scalar_ols'); b=kernels.index('\n\n@njit', a)
scalar=kernels[a:b]
scalar=scalar.replace('    for t in range(n):\n        old =', '    predictions = np.full(n, np.nan)\n    for t in range(n):\n        if total > 0:\n            predictions[t] = X[t, 0]*beta + intercept\n        old =')
kernels=kernels[:a]+scalar+kernels[b:]
a=kernels.index('def _stream(')
generic=kernels[a:].replace('    for t in range(n):\n        old =', '    predictions = np.full(n, np.nan)\n    for t in range(n):\n        if total > 0:\n            predictions[t] = X[t] @ coef + intercept\n        old =')
kernels=kernels[:a]+generic
old='return total, my, intercept, history, intercepts, failed, iterations, error, threshold'
assert kernels.count(old)==2
kernels=kernels.replace(old, old+', predictions')
reference=source[source.index('def _fit_input'):source.index('class BatchedFitters:')]
p.write_text(kernels+Path('_wf_core.txt').read_text()+'\n\n'+reference+Path('_wf_batch.txt').read_text())

p=Path('src/takehome/features.py'); s=p.read_text()
a=s.index('def backtest('); b=s.index('\ndef forecast_metrics',a)
s=s[:a]+'''def backtest(signal, returns, hl=288*21):
    """signal / EWMstd(signal)^2 * return; forecasts must already be aligned."""
    if not signal.index.equals(returns.index):
        raise ValueError('Signal and return indexes must match.')
    return standalone_pnl(signal, returns, hl).replace([np.inf, -np.inf], np.nan)

'''+s[b:]
s+='''

def with_cashflow_feature(df):
    """Append the next x-number: raw cashflow/volume; undefined ratios stay missing."""
    import re
    numbers = [int(m.group(1)) for c in df if (m := re.fullmatch(r'x(\\d+)', str(c)))]
    name = f'x{max(numbers, default=0)+1}'
    ratio = df['cashflow'].div(df['volume'].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return df.assign(**{name: ratio})
'''
p.write_text(s)

p=Path('tests/test_regression_sweeps.py'); s=p.read_text()
a=s.index('def test_backtest_matches_standalone'); b=s.index('\ndef test_ridge_constant',a)
s=s[:a]+'''def test_backtest_matches_standalone_variance_formula():
    from takehome import features
    idx=pd.date_range('2020-01-01', periods=120, freq='1h')
    p=pd.Series(np.sin(np.arange(120)*.3)+.2,index=idx)
    r=pd.Series(np.cos(np.arange(120)*.3),index=idx)
    expected=p.div(features.ts_std(p,10).pow(2).replace(0,np.nan)).mul(r)
    assert_allclose(features.backtest(p,r,10),expected,equal_nan=True)
    assert_allclose(features.backtest(p,r,10)[:70],features.backtest(p[:70],r[:70],10),equal_nan=True)
    with pytest.raises(ValueError): features.backtest(p,r.iloc[:-1],10)

'''+s[b:]
s=s.replace('both backtest conventions', 'the standalone backtest convention')
p.write_text(s)
print('Installed compiled core, independent batch objectives, fold paths, and x100 helper.')

"""One-time source migration; removed after verified notebook execution."""
from pathlib import Path
p=Path('src/takehome/fitters.py')
s=p.read_text()
assert 'def forecast_alpha(' not in s
p.write_text(s+Path('_ar2_methods.txt').read_text())
p=Path('src/takehome/features.py'); s=p.read_text()
a=s.index('def ts_std('); b=s.index('\ndef ts_standardize',a)
s=s[:a]+'''def ewm_observed(x, hl, min_periods=0):
    """Zero/nonfinite values are missing; decay advances on observed values."""
    return x.replace([0, np.inf, -np.inf], np.nan).ewm(
        halflife=hl, min_periods=min_periods, ignore_na=True)


def ts_std(x, hl: int, min_periods=None):
    """Nonzero-observation EWM std; exactly zero output stays undefined."""
    return ewm_observed(x, hl, hl if min_periods is None else min_periods).std().replace(0, np.nan)

''' + s[b:]
s=s.replace('x.sub(x.ewm(halflife=hl, min_periods=hl).mean())',
            'x.sub(ewm_observed(x, hl, min_periods=hl).mean())')
s=s.replace('ewm = pnl.ewm(halflife=hl)', 'ewm = ewm_observed(pnl, hl)')
s=s.replace('NaNs do not become zero observations in EWM estimation.',
            'Zeros and nonfinite values do not become observations in EWM estimation.')
p.write_text(s)
p=Path('tests/test_blends.py'); s=p.read_text()
s=s.replace('pnl.ewm(halflife=10).mean()/pnl.ewm(halflife=10).std()',
            'pnl.replace(0,np.nan).ewm(halflife=10,ignore_na=True).mean()/pnl.replace(0,np.nan).ewm(halflife=10,ignore_na=True).std()')
p.write_text(s)
print('Installed causal AR(2) forecast helpers and shared zero-aware EWM moments.')

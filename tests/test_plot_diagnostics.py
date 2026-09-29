"""Actual plot data and calibration math, not only successful rendering."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose
from takehome import plots


@pytest.fixture
def shown(monkeypatch):
    figures = []
    monkeypatch.setattr(plt, 'show', lambda: figures.append(plt.gcf()))
    yield figures
    plt.close('all')


def test_family_overlay_uses_exact_cumulative_pnl(shown):
    index = pd.date_range('2020-01-01', periods=4)
    daily = pd.DataFrame({'a': [1., 2., -1., .5], 'b': [-1., 3., 2., .5]}, index=index)
    combined = pd.Series([.1, .2, -.1, .05], index=index)
    scores = plots.display_results(daily, 'Family', meta_daily=combined)
    assert len(shown) == 2
    lines = shown[0].axes[0].lines
    assert len(lines) == 3
    for j, line in enumerate(lines[:-1]):
        assert_allclose(line.get_ydata(), daily.iloc[:, j].cumsum())
    assert_allclose(lines[-1].get_ydata(), combined.cumsum())
    assert lines[-1].get_linewidth() > 4 * lines[0].get_linewidth()
    assert lines[-1].get_zorder() > lines[0].get_zorder()
    assert 'Lagged EWM-Sharpe' in lines[-1].get_label()
    assert_allclose(scores, daily.mean() / daily.std())
    assert sum(p.get_height() for p in shown[1].axes[0].patches) == len(daily.columns)
    assert not plt.get_fignums()


def test_overlay_rejects_misaligned_daily_index(shown):
    daily = pd.DataFrame({'x': [1., 2.]}, index=pd.date_range('2020-01-01', periods=2))
    wrong = pd.Series([1., 2.], index=daily.index + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match='index'):
        plots.display_results(daily, 'Family', meta_daily=wrong)
    assert not shown


def test_calibration_direction_and_all_finite_observations(shown):
    assert hasattr(plots, 'plot_calibration'), 'Missing calibration scatter helper'
    yhat = pd.Series(np.r_[np.linspace(-2, 3, 10001), np.nan, np.inf])
    y = 4 + 2 * yhat
    result = plots.plot_calibration(yhat, y)
    assert result.loc['OLS', 'n'] == 10001
    assert result.loc['OLS', 'slope'] == pytest.approx(2.)
    assert result.loc['OLS', 'intercept'] == pytest.approx(4.)
    assert result.loc['OLS', 'r_squared'] == pytest.approx(1.)
    ax = shown[0].axes[0]
    assert len(ax.collections[0].get_offsets()) == 10001
    assert_allclose(ax.collections[0].get_offsets()[:, 0], yhat.iloc[:10001])
    assert_allclose(ax.collections[0].get_offsets()[:, 1], y.iloc[:10001])
    assert any('2.0000' in t.get_text() for t in ax.texts)
    assert 'prediction' in ax.get_xlabel().lower()
    assert 'target' in ax.get_ylabel().lower()
    assert not plt.get_fignums()


def test_weighted_calibration_matches_direct_wls(shown):
    assert hasattr(plots, 'plot_calibration'), 'Missing calibration scatter helper'
    yhat = pd.Series([-2., -1., 0., 1., 2.])
    y = pd.Series([1., 0., 2., 2., 10.])
    w = pd.Series([1., 2., 0., 3., 12.])
    result = plots.plot_calibration(yhat, y, weights=w)
    design = np.column_stack([np.ones(len(yhat)), yhat])
    expected = np.linalg.lstsq(design * np.sqrt(w.to_numpy())[:, None], y * np.sqrt(w), rcond=None)[0]
    assert_allclose(result.loc['WLS', ['intercept', 'slope']], expected, atol=1e-12)
    assert result.loc['WLS', 'n'] == 4
    assert len(shown[0].axes[0].collections[0].get_offsets()) == 5
    scaled = plots.plot_calibration(yhat, y, weights=w * 1e20)
    assert_allclose(scaled, result, atol=1e-12)


def test_constant_forecasts_are_not_claimed_calibrated(shown):
    assert hasattr(plots, 'plot_calibration'), 'Missing calibration scatter helper'
    result = plots.plot_calibration(pd.Series(np.ones(4)), pd.Series(np.arange(4)))
    assert np.isnan(result.loc['OLS', 'slope'])
    assert any('undefined' in t.get_text().lower() for t in shown[0].axes[0].texts)


def test_calibration_rejects_misalignment_and_bad_weights(shown):
    assert hasattr(plots, 'plot_calibration'), 'Missing calibration scatter helper'
    yhat, y = pd.Series([1., 2., 3.]), pd.Series([2., 3., 4.])
    with pytest.raises(ValueError, match='index'):
        plots.plot_calibration(yhat, y.set_axis([3, 4, 5]))
    with pytest.raises(ValueError, match='weight'):
        plots.plot_calibration(yhat, y, weights=pd.Series([1., -1., 3.]))
    with pytest.raises(ValueError, match='finite'):
        plots.plot_calibration(yhat * np.nan, y)

import numpy as np
import pandas as pd

from takehome.eda import (
    acf_matrix, autocorrelation_table, bell_shape_table, correlation_clustering,
    correlation_to_column, correlation_to_returns, infer_feature_cols,
    monthly_shift_scores, redundant_pairs,
)


def test_infer_feature_cols_prefers_signal_columns():
    df = pd.DataFrame({"adjMid": [1, 2], "cashflow": [0, 1], "signal_0": [1, 2], "signal_1": [2, 3]})
    assert infer_feature_cols(df) == ["signal_0", "signal_1"]


def test_bell_shape_accepts_symmetric_heavy_tails_and_flags_degenerate_shapes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "heavy_tail": rng.standard_t(3, 5000),
        "bimodal": np.r_[rng.normal(-3, 0.25, 2500), rng.normal(3, 0.25, 2500)],
        "point_mass": np.r_[np.zeros(3000), rng.normal(size=2000)],
    })
    out = bell_shape_table(df, list(df)).set_index("feature")
    assert bool(out.loc["heavy_tail", "bell_like"])
    assert not bool(out.loc["bimodal", "bell_like"])
    assert not bool(out.loc["point_mass", "bell_like"])
    assert abs(out.loc["heavy_tail", "robust_skew"]) < 0.1


def test_return_correlation_finds_known_signal():
    rng = np.random.default_rng(1)
    ret = rng.normal(scale=0.01, size=2000)
    price = 100 * np.cumprod(1 + ret)
    df = pd.DataFrame({"adjMid": price, "signal_0": np.r_[0, ret[1:]], "signal_1": rng.normal(size=2000)})
    assert correlation_to_returns(df, ["signal_0", "signal_1"]).iloc[0]["feature"] == "signal_0"


def test_correlation_clustering_returns_all_features():
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    df = pd.DataFrame({"a": x, "b": -x + rng.normal(scale=0.01, size=500), "c": rng.normal(size=500)})
    corr, link, order = correlation_clustering(df, ["a", "b", "c"])
    assert corr.shape == (3, 3)
    assert link.shape == (2, 4)
    assert set(order) == {"a", "b", "c"}


def test_autocorrelation_table_flags_persistent_series():
    rng = np.random.default_rng(3)
    eps = rng.normal(size=1500)
    x = np.empty_like(eps)
    x[0] = eps[0]
    for i in range(1, len(x)):
        x[i] = 0.9 * x[i - 1] + eps[i]
    out = autocorrelation_table(pd.DataFrame({"x": x}), ["x"], lag=15)
    assert out.loc[0, "lb_stat"] > 100
    assert out.loc[0, "lb_pvalue"] < 1e-6
    assert out.loc[0, "acf1"] > 0.7


def test_monthly_shift_scores_localizes_level_break():
    t = pd.date_range("2020-01-01", periods=24 * 30 * 8, freq="h")
    rng = np.random.default_rng(4)
    stable = rng.normal(size=len(t))
    shifted = rng.normal(size=len(t))
    shifted[len(t) // 2:] += 5
    df = pd.DataFrame({"time": t, "stable": stable, "shifted": shifted})
    mean_shift, vol_shift = monthly_shift_scores(df, "time", ["stable", "shifted"])
    assert mean_shift["shifted"].abs().max() > mean_shift["stable"].abs().max()
    assert mean_shift.shape == vol_shift.shape


def test_correlation_to_column_finds_related_feature():
    rng = np.random.default_rng(5)
    cash = rng.normal(size=1000)
    df = pd.DataFrame({"cashflow": cash, "x": 2 * cash + rng.normal(scale=0.1, size=1000), "y": rng.normal(size=1000)})
    assert correlation_to_column(df, "cashflow", ["x", "y"]).iloc[0]["feature"] == "x"


def test_acf_matrix_recovers_periodic_lag():
    t = np.arange(1200)
    x = np.sin(2 * np.pi * t / 12)
    assert acf_matrix(pd.DataFrame({"x": x}), ["x"], nlags=24).loc[12, "x"] > 0.9


def test_redundant_pairs_reports_numeric_name_gap():
    corr = pd.DataFrame(
        [[1, .95, 0], [.95, 1, 0], [0, 0, 1]],
        index=["x1", "x4", "x2"], columns=["x1", "x4", "x2"],
    )
    out = redundant_pairs(corr, threshold=.9)
    assert out.loc[0, "feature1"] == "x1"
    assert out.loc[0, "feature2"] == "x4"
    assert out.loc[0, "index_gap"] == 3

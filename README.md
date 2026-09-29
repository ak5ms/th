# ESc1 take-home

## Setup and execution

```bash
git lfs pull
python -m pip install -r requirements.txt
python -m pytest -q
python run_eda.py
```

Or open `notebooks/01_eda.ipynb` in Jupyter and Run All. `DATA_PATH` optionally overrides the Parquet path.

## Outputs

The evaluated notebook contains the figures, complete column-coverage table, cashflow-exception sample and complete date counts, package versions, interpretation notes, and training-only summary. The runner saves it in place and records its execution time in a final notebook note. No CSV, HTML, Markdown report, or environment file is exported; there is no `reports/` folder. The only separate diagnostic file is `splits.json` at the repository root.

## Data contract

- Index: sorted, unique `msgStamp` timestamps, preserving their supplied timezone.
- Labeled span: first through last non-null `ret_5m`.
- Cutoff: `first + 0.8 * (last - first)`; elapsed time, not row count.
- Training: `[first, cutoff)`; reserved test: `[cutoff, last]`. Rows outside the labeled span are excluded.
- Only training rows enter diagnostics, normalization, plots and the cashflow calculation. Internal nulls remain null; there is no forward filling or row subsampling.
- The scatter matrix limits displayed feature columns, not observations. Other diagnostics cover all 99 features where defined.
- Earlier EDA versions inspected the complete dataset. The reserved block is excluded from subsequent research, but is not retrospectively an untouched holdout.

`src/takehome/data.py` holds the split and coverage helpers, `eda.py` the statistics, and `plots.py` a small heatmap helper. The notebook shows the cashflow expression directly.

The distribution screen uses the full-sample Hartigan dip statistic, quartile skew and point-mass checks, not a normality test. Its thresholds are descriptive. Cashflow is hypothesized to be signed trade flow or quote imbalance; the cumulative curve is exploratory, not a validated executable backtest.

## Standalone transformation experiments

The notebook separately displays cumulative P&L and a daily-Sharpe histogram for time-of-day standardization, every distinct raw pair product, and both directed pair residuals. `features.py` holds `dszl`, `pair_residual`, `pairwise_features` and `evaluate_features`; `plots.py` provides `display_results`. Pairs are column-batched, not row-sampled. Residuals use the existing lasso at `alpha=0`, `W=1`, with an intercept and pre-update predictions. All output stays inside the evaluated notebook.

## Fitting and baseline interfaces

`fitters.py` includes the Numba streaming estimator, independent `CvxpyWeightedLasso`, and the `BatchedFitters` walk-forward adapter. Both history interfaces accept `lag=1` for prior-estimate alignment. Tests reconcile original-unit objectives and predictions against CVXPY, not sklearn. `features.combine_pnls` computes a lagged, globally gross-normalized intrabar EWM-Sharpe blend, also inside column-batched pair evaluation. `sessions.py` separates exchange-open instants, trailing trading bars and empirical quote availability using CME holiday rules and dated historical-hour corrections. See the notebook for the calendar audit and limitations.

## Overlay and calibration diagnostics

Each feature-family P&L panel highlights its own lagged EWM-Sharpe combination in purple, without display rescaling. The real-data regression shows coefficient histories and a full-training scatter of prior-row forecasts against targets, including OLS and W-weighted calibration slopes. Both coefficient and intercept histories are lagged. Calibration is diagnostic only: predictions are not rescaled to force slope one.

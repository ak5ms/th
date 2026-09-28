# ESc1 take-home

## Setup and execution

```bash
git lfs pull
python -m pip install -r requirements.txt
python -m pytest -q
python run_eda.py
```

Or open `notebooks/01_eda.ipynb` in Jupyter and Run All. `DATA_PATH` optionally overrides the Parquet path. The committed notebook includes evaluated tables and figures. The runner also writes `reports/eda.html`, `reports/column_coverage.csv`, `reports/cashflow_outliers.csv`, `reports/split.json`, and `EDA_SUMMARY.md`.

## Data contract

- Index: sorted, unique `msgStamp` timestamps, preserving their supplied timezone.
- Labeled span: first through last non-null `ret_5m`.
- Cutoff: `first + 0.8 * (last - first)`; this is elapsed time, not row count.
- Training: `[first, cutoff)`; reserved test: `[cutoff, last]`. Rows outside the labeled span are excluded.
- Only training rows enter diagnostics, normalization, correlations, plots and the cashflow calculation. Internal nulls remain null; there is no forward filling.
- All training observations are used. A scatter matrix displays a subset of feature columns, but does not sample rows. Other diagnostics cover all 99 feature columns where defined.
- Column coverage is training-only and retains wholly null columns. Display truncation of large tables does not change calculations; full cashflow exceptions are saved to CSV.
- Previous EDA versions inspected the complete dataset. The reserved block is excluded from this run and future research, but is not a retrospectively untouched holdout.

`src/takehome/data.py` holds the split and coverage helpers, `eda.py` the statistics, and `plots.py` a small heatmap helper. The notebook shows the requested cashflow expression directly.

The distribution screen uses the full-sample Hartigan dip statistic plus quartile skew and point-mass checks, not a normality test or a large-sample p-value. Its effect-size thresholds are descriptive. Cashflow is hypothesized to be signed trade flow or quote imbalance; the cumulative curve is exploratory and not a validated, executable backtest.

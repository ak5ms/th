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

The evaluated notebook contains the figures, complete column-coverage table, complete cashflow-exception table, package versions, interpretation notes, and training-only summary. The runner saves it in place and records its execution time in a final notebook note. No CSV, HTML, Markdown report, or environment file is exported; there is no `reports/` folder. The only separate diagnostic file is `splits.json` at the repository root.

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

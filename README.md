# ESc1 take-home

`notebooks/takehome.ipynb` contains the feature analysis, out-of-sample selection audit, model comparisons, and holdout forecast export. `forecasts/` contains **one file only: `predictions.parquet`**. It has a timezone-aware `msgStamp` index and one `forecast` column with raw `ret_5m` predictions for the true final two-year holdout. No research forecasts, CSVs, coefficients, metadata or charts are exported there. Configuration and diagnostics stay inside the notebook.

## Reproduction

Install `requirements.txt` and run `python run_eda.py` from the repository root. The runner uses a fresh kernel, executes the entire notebook and saves outputs only on success. Place the supplied Parquet dataset at the default path displayed in the notebook, or set `DATA_PATH`. `FORECAST_OUTPUT_DIR` optionally changes the output directory; it must contain no files other than the single prediction file. `pytest -q` runs the tests.

## Combined design and model selection

The return model uses `[X, dszl(X, 10)]`: all 100 raw predictors, including x100=cashflow/volume, followed by all 100 same-clock-time standardized versions. Column labels are `raw:x1` through `raw:x100`, then `dszl:x1` through `dszl:x100`. The transform uses no target or future feature, carries state across folds, does not demean, and preserves the zero/nonfinite-as-missing EWM convention. Nonfinite model inputs are zero-imputed after concatenation; labels are never filled. Candidate loss and regularization scales use only their training windows.

Ridge and Lasso have separate regularization grids and **previous-test-block selection**, not a static reference alpha. Candidates train on two calendar years, predict the next two years with frozen coefficients, and are scored by raw-return mean squared error. The winner from test block k is refitted and applied only to block k+1. Initial test block zero is tuning only: there are no selected-strategy forecasts there. One row is embargoed at training cutoffs and at the end of validation blocks to avoid immature labels. Exact MSE ties prefer the stronger penalty through descending grid order. Missing/invalid preceding validation raises instead of falling back to a fixed parameter.

The frozen streaming-Lasso audit and its live-update comparator share the selected batch-Lasso penalty schedule. Their training intervals match the batch fits. The live comparator learns each test label only after its prediction; it does not independently choose a hindsight winner. Online AR(2) feature-forecast experiments also select their per-feature penalties using preceding two-year next-feature test MSE; they never use return labels. Pure unpenalized residual transformations and synthetic numerical-reference checks are not hyperparameter-selected return strategies.

## Research and final holdout

The initial 80% of labeled elapsed time remains the research sample. The following labeled interval is excluded from those charts but now explicitly used for final pre-holdout tuning; it is not called an untouched test after doing so. Earlier versions inspected broader samples, so research reservations are not retrospectively pristine.

At holdout boundary B, train Ridge candidates on `[B-4 years, B-2 years)`, validate on `[B-2 years, B)`, select by validation MSE, and refit the chosen alpha on `[B-2 years, B)`. Apply the same one-row embargo. Freeze coefficients throughout the entire blackout, including the last supplied timestamp. The combined-feature state is carried from the original feature-history start for identical research and inference semantics. Every true-holdout target must be missing, and every supplied holdout row must receive a finite prediction. Only that holdout prediction frame is written to Parquet.

## Forecasts versus positions

Raw forecasts are never scaled for export. Diagnostic backtests separately use lagged signal standard deviation, a floor at 25% of its first valid scale and an absolute exposure cap of 3, with no fold reset. This requires no asset-return volatility at inference. No P&L is clipped. The cap controls denominator-driven exposure explosions, not market-return shocks or dollar risk, and does not establish forecasting accuracy. MSE selection is on raw forecasts, not these capped P&L curves.

Red notebook TODOs retain the requested uncertainty-aware t-stat sizing and volume-weighted fitting ideas. Neither is claimed implemented. Fitting and validation weights remain unit on finite target rows. Transaction costs and exact decision-time feature/forward-label availability remain methodological limitations.

Sparse AR feature experiments leave a block unavailable when the previous test block contains no mature feature-target examples. They never substitute a static alpha or tune on the current block. The notebook reports pending choices and aligns the persistence/oracle control masks. This does not remove any raw or dszl column from the 200-input return models, whose regularization selection remains strict.

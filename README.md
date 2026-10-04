# ESc1 take-home

`notebooks/takehome.ipynb` contains feature analysis, out-of-sample model-selection audits, model and exposure comparisons, and the final holdout export. `forecasts/` contains **one file only: `predictions.parquet`**: a timezone-aware `msgStamp` index and one `forecast` column in raw `ret_5m` units for the true final two-year holdout. All parameters, selection tables and diagnostics remain inside the notebook; no research predictions, CSVs, coefficient dumps or metadata sidecars are exported.

## Reproduction

Install `requirements.txt`, then run `python run_eda.py` from the repository root. The runner clears all outputs, uses a fresh kernel, executes the entire notebook and saves the executed notebook only on success. The full run includes all 4,950 products, 9,900 directed residuals, 100 AR feature forecasts, four return-model families, both streaming-Lasso variants, and holdout inference. Set `DATA_PATH` to override the supplied input path. `FORECAST_OUTPUT_DIR` optionally changes the prediction directory; it must contain no unrelated files. Run `pytest -q` for tests.

## Model inputs: combined features divided by EWM variance

First form `F = [X, dszl(X, 10)]`: all 100 raw predictors (including x100=cashflow/volume) followed by all 100 same-clock-time standardized counterparts. Then form **`F_xform = F / EWMVar(F)`** before fitting any return model. This is division by **variance**, not standard deviation and not a demeaned z-score. Column order is `raw:x1` through `raw:x100`, then `dszl:x1` through `dszl:x100`.

`variance_scaled_design` implements this in bounded column batches. The outer EWM variance uses half-life 6,048, minimum 6,048 observed values, `adjust=True`, `ignore_na=True`, and the unbiased sample-variance convention. As in the existing EWM helpers, zero and nonfinite observations do not update the scale. It includes the currently observed feature, but no future feature or return. No feature centering, input clipping or variance floor is applied. A zero or undefined variance makes the transformed input unavailable; zero-imputation occurs only **after division**, never before estimating the scale. Targets are never filled.

Both time-of-day and outer variance state carry from the original feature-history start through research, final pre-holdout validation and inference. Neither state resets at a fold boundary. The notebook independently checks both halves against the literal variance formula on all research rows for representative columns. Tests verify every column, missing/zero inputs, current-feature timing, prefix invariance, column batching and exact research/holdout transform agreement.

The estimators still compute training-window feature scales for their regularization penalties. Those fixed, training-only scales are distinct from the time-varying EWM-variance division and do not undo it.

## Zero-intercept models and nonnegative variants

Every forecasting fit in the notebook runs with **`fit_intercept=False`**: batch Ridge, batch Lasso, the independent row-residual CVXPY reference, frozen and live streaming Lasso, nonnegative variants, AR(2) feature forecasts and pair-residual transformations. No constant column is added, and neither feature nor target training means are subtracted for the regression loss or restored in prediction. Mean/covariance statistics are maintained for numerical stability; for a zero-intercept fit the loss uses uncentered second moments (`C + mx mx'`, `c + mx my`), and every intercept is exactly zero.

`BatchRidge`, `BatchLasso`, `CvxpyWeightedLasso` and `StreamingWeightedLasso` default to no intercept and support `nonneg=True`, which constrains **coefficients**, not forecasts. Ridge uses a Cholesky-equivalent nonnegative least-squares solve; Lasso uses the constrained convex objective; streaming Lasso uses one-sided coordinate updates and the corresponding KKT conditions. Selected streaming checkpoints preserve the constraint. Both batch Lasso implementations certify the exact all-zero solution using its KKT condition, preventing tiny solver residuals from becoming amplified position signals; no arbitrary coefficient cutoff is used. The library retains an explicit opt-in intercept mode for compatibility and independent numerical tests, but no forecasting run in the submission notebook enables it. Calibration regressions can estimate a diagnostic intercept; it is never fed back into forecasts.

## Previous-test selection, not a static penalty

Ridge, nonnegative Ridge, Lasso and nonnegative Lasso each have their own preceding-test selection from the displayed regularization grids. Candidates train on two calendar years, predict the next two years with frozen coefficients, and are scored by raw-return mean squared error. The winner from test block k is refitted and applied only to block k+1. Initial test block zero is tuning only, not selected-strategy performance. One row is embargoed at training cutoffs and at validation-block ends for target maturity. Exact MSE ties favor the stronger penalty through descending grid order. Missing preceding validation cannot trigger a fixed-parameter fallback.

Frozen streaming-Lasso audits and their live-update comparators follow the selected schedule of the corresponding unconstrained/nonnegative batch-Lasso family. Live updates learn test labels only after their predictions. Terminal and intermediate convergence checks are shown separately, with failed diagnostic folds identified rather than represented as validated results. AR(2) experiments select each feature's penalty using preceding two-year next-feature validation MSE, without return labels. Unpenalized residual transformations and synthetic solver checks are not separately tuned return strategies.

## Exposure and long-bias assessment

The EWM-variance transform changes the inputs to regression; it does not by itself determine positions. Main backtests retain the separate signal-only lagged standard-deviation rule, 25%-of-first-scale floor and absolute exposure cap of 3. Raw forecasts are not capped, demeaned or converted to positions in the export. Legacy sizing controls are explicitly labeled diagnostics, not hidden smoothing of the active curves.

A zero intercept does **not** guarantee zero mean forecasts, market neutrality, or protection against a selloff. Nonzero feature means and regime shifts can still create directional exposure. The notebook reports positive-forecast fractions, forecast mean/std, average signed exposure, net/gross ratios and cap frequencies for all four selected model families, including a separately labeled 2020 stress diagnostic. The stress interval is not a tuning criterion. No new family is selected merely because it looks better during COVID.

## Research and true holdout

The initial 80% of labeled elapsed time remains the research sample. The following labeled interval stays out of research charts but is explicitly used for final pre-holdout validation. Earlier versions inspected broader samples; those reservations are not retrospectively pristine.

At blackout boundary B, train unconstrained Ridge candidates on `[B-4 years, B-2 years)`, validate on `[B-2 years, B)`, select the best validation MSE and refit that alpha on `[B-2 years, B)`. Use the same variance-divided features, zero intercept and one-row embargo as research. Freeze coefficients through the final supplied holdout timestamp. The four-family comparisons do not silently change the submission's established Ridge family. All actual holdout targets must be missing, all holdout predictions must be finite, and only that timestamped frame is written to Parquet.

Red TODOs retain the requested open questions about uncertainty-aware t-stat sizing and volume-weighted fitting. Neither is claimed implemented. All P&Ls are gross diagnostics; costs, decision-time feature availability and forward-return label timing still require review. Bounding exposure is not a guarantee of executable profitability or constant realized risk.

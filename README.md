# ESc1 take-home

The submission workflow is **`notebooks/takehome.ipynb`**. It runs from data checks and feature research through chronological validation to the final raw-return Ridge forecasts. `notebooks/asset_vol.ipynb` is a separate, smaller signal-only sizing audit. Despite its historical filename, it does not use asset-return volatility and does not generate submission forecasts.

## Reproduce

Use Python 3.11+ and install `requirements.txt`. Place `ESc1_signal_components_5min (6).parquet` in the repository root, or set `DATA_PATH` to its location. The source dataset is stored with Git LFS in the repository; a source archive without LFS materialization is not the data file.

```bash
python -m pip install -r requirements.txt
python run_eda.py
```

The runner starts a fresh Jupyter kernel, clears every old output, executes the complete main notebook, and atomically saves it only after all cells succeed. It prints progress by cell and records full-execution provenance. All rows and all feature pairs are used, so the complete research notebook is substantially more expensive than a single model fit. Arrow row batches and eight-column feature batches limit temporary memory; no observations or pairs are sampled.

Run the diagnostic separately:

```bash
NOTEBOOK_PATH=notebooks/asset_vol.ipynb python run_eda.py
```

Opening either notebook in Jupyter and choosing **Restart Kernel and Run All** also runs its displayed code. `FORECAST_OUTPUT_DIR` can change the main notebook's export directory. There is no separate forecast runner or model-specific export function.

## Configuration and inference contract

The main notebook declares and displays one fixed configuration: all 100 predictors, including x100=cashflow/volume; time-of-day `dszl(10)` features without demeaning; Ridge alpha=0.01 with an unpenalized intercept; unit weights on finite labels; a 6,048-row EWM fitting half-life; and rolling two-calendar-year training followed by two-calendar-year frozen forecasts, advanced by two years. Missing transformed model inputs are zero-imputed, not targets. The initial EWM and dszl state uses zero/nonfinite-as-missing, `ignore_na=True`; regression decay advances by input row.

Ridge/Lasso sensitivity sweeps share exactly the same transformed design and calendar windows. Frozen streaming Lasso replays only each corresponding two-year training window; live streaming Lasso is an explicitly separate continuous online baseline. The h=1 oracle and AR(2) feature-forecast experiments are grouped under Transform experiments, before joint model fitting. The coefficient plot omits its first `HL` rows only for display.

All active main backtest plots use the same explicit **lagged signal standard deviation / 25%-of-first-scale floor / absolute exposure cap of 3**. This policy receives signals only; returns enter afterward for scoring. The normalizer is not reset at folds. It bounds exposure, not asset returns or dollar losses; no P&L is clipped. The normalization audit reconstructs legacy raw-unit and past-anchored Original shape controls and reports jump concentration. The Original shape multiplier changes units, not concentration. Raw forecasts remain unscaled for calibration and export. No return-volatility model or t-stat sizing has been implemented.

## Validation and submission

The first 80% of the labeled elapsed-time span is the research sample; its last 20% is not scored or used to choose settings in this notebook. Earlier versions inspected a broader sample, so that reservation is not retrospectively pristine. Walk-forward validation and the selected configuration's backtest are displayed inside research only.

At the end, the notebook reuses **the same `RIDGE_CONFIG`** to fit on the two calendar years before the final two-year label blackout. This final fit includes available labels from the research-reserved interval, without scoring them or choosing a new penalty. Its coefficients stay frozen for the full blackout. The same-clock-time dszl state is carried from the original feature-history start through the final fit and inference; it uses no return labels. Every blackout target is checked to be missing, and every supplied blackout row must receive a finite prediction.

`forecasts/` contains:

- `ridge_oos.csv` / `.parquet`: submission predictions in raw `ret_5m` units.
- `ridge_walk_forward.csv` / `.parquet`: the exact research forecasts supporting the selected-model backtest.
- `ridge_metadata.json`, `ridge_research_windows.csv`, `ridge_coefficients.csv`: configuration, split/use policy, coverage, checksums, window boundaries and final parameters.

CSV columns are `msgStamp,forecast`; Parquet preserves the timezone-aware index. Neither export contains positions, future-return volatility, oracle inputs or an extra forecast shift. Old Lasso exports and the superseded normalization/comparison notebooks have been removed to avoid submitting an obsolete model.

## Open methodological items

Red TODOs remain visible in the notebook as requested: investigate a valid uncertainty-aware forecast t-stat rather than temporal forecast-volatility sizing, and assess volume-weighted fitting as a proxy for feasible trade size. Neither is claimed implemented. Backtests are gross diagnostics; transaction costs, decision-time feature availability and the supplied forward-return timing still need validation. A bounded exposure policy does not establish alpha, executable profitability or a constant realized risk target.

Run `pytest -q` for calendar-window, numerical fitter, prefix-causality, normalization, notebook alignment and actual inline-export tests. `run_eda.py` separately verifies every executed cell and embeds its run provenance.

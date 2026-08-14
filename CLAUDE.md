# wind-forecast-dashboard — agent context

## What this is
A small reproducible project that forecasts the **percentage of GB wind in the
national generation mix, day-ahead** and serves it through a Plotly Dash app
deployed on Render's free tier. Live demo: https://wind-forecast-dashboard.onrender.com

The target is `wind_perc` (0–100), **not** absolute MW. Errors quoted as
"%-points" are points on this 0–100 scale (RMSE 1.89 = ±1.9 percentage points).

## Two pipelines, sharply separated

**Training (manual, run when retraining):**
[src/etl_training.py](src/etl_training.py) →
[src/featurise.py](src/featurise.py) `--mode training` →
[src/train_model.py](src/train_model.py) →
[src/validate.py](src/validate.py) →
[src/backtest.py](src/backtest.py).
Produces [models/model.cbm](models/model.cbm),
[metrics.json](metrics.json), [cv_metrics.json](cv_metrics.json),
[models/catboost_full.parquet](models/catboost_full.parquet).

`backtest.py` is not optional. The Historical tab's chart and KPI cards read
`catboost_full.parquet`; filling it by predicting the training data made that
tab advertise an in-sample ~1.3 next to the hero card's honest ~5.8. It now
runs an expanding-window backtest, and rows before the first fold stay NaN
rather than being back-filled with an in-sample guess.

**Nightly inference (GitHub Actions):**
[src/pipeline.py](src/pipeline.py) orchestrates
[src/etl_inference.py](src/etl_inference.py) →
[src/featurise.py](src/featurise.py) (default inference mode) →
[src/predict.py](src/predict.py).
Produces [data/predictions/latest.parquet](data/predictions/latest.parquet) and
[data/features/history.parquet](data/features/history.parquet).

featurise (inference mode) writes `for_predict.parquet` directly;
`features.parquet` is a committed full-history snapshot the nightly run
never touches. `pipeline.py` then unions snapshot + prior history + new
rows into `history.parquet` (idempotent, keyed on `datetime`) and fails
loudly — any empty or missing input raises, so the Actions job goes red
instead of republishing stale forecasts as a fresh nightly update.

## The 48h embargo (read this before touching features)

The product publishes a **day-ahead** forecast, so a `wind_perc` observation is
only usable as a feature if it predates the target by >= 48h. An earlier
version took every column indiscriminately, which pulled in
`wind_perc_lag_{30m,1h,3h,24h}` — present while training, NaN in production.
They carried **88% of that model's feature importance**, so the advertised
accuracy (RMSE 1.39) described a one-hour nowcast while the thing actually
served scored ~12.1. Fixing it is what took the live forecast to ~5.4.

Consequences, all of which must be kept in sync:
- [src/featurise.py](src/featurise.py) `target_history_features()` builds every
  target-derived feature on a strict hourly grid and shifts it by
  `FORECAST_HORIZON_H = 48`. It is the *only* place `wind_perc` may be lagged.
- [src/train_model.py](src/train_model.py) and [src/validate.py](src/validate.py)
  both re-filter with `_is_admissible()` as a belt-and-braces guard.
- `tests/test_featurise.py::test_no_sub_horizon_target_lags` fails the build if
  a sub-horizon lag reappears.

Inference cannot compute a 365-day trailing mean from the nightly's short
fetch, so `featurise.main(mode="inference")` passes a **backbone** — the union
of `history.parquet`, `features.parquet` and the fresh pull — into
`engineer_features`. If `history.parquet` is thin, the capacity features go
NaN at serve time and the skew comes straight back.

## Data gaps matter now

The strongest features are trailing windows over the target, so holes in
`history.parquet` degrade the live forecast. The broken-nightly period left
gaps (56 days to 2026-07-03, 23 days to 2026-08-10) that put last-30-day
coverage at 20%. Repaired to ~97% with
[src/backfill_history.py](src/backfill_history.py) (`python -m
src.backfill_history --days 430`). `etl_inference.CI_HISTORY_DAYS` is now 14,
not 3, so gaps shorter than a fortnight heal on the next successful run.

## Uncertainty band

`model_p10/p90.cbm` are quantile companions, but quantile regression is not
calibrated by construction — raw coverage measured 68% against a nominal 80%.
`train_model.py` measures a **split-conformal delta** on a 60-day calibration
window the quantile models never saw, stores it as `conformal_delta_pts` in
[metrics.json](metrics.json), and `predict.py` widens (or narrows — the delta
can be negative) both bounds by it.

## Data sources
- **Carbon Intensity API** — GB wind generation as % of mix (the target).
- **Open-Meteo** — `temperature_2m`, `wind_speed_10m`, `wind_speed_100m`,
  `wind_gusts_10m`, `wind_direction_10m`, `surface_pressure`, combined into a
  **capacity-weighted composite of 5 GB wind regions** (Scotland onshore 30%,
  North Sea 30%, East Anglia 15%, Irish Sea 15%, Moray Firth 10%). Both ETLs
  carry the identical `WEIGHTED_LOCATIONS` list — keep them in sync. Wind
  direction uses a circular weighted mean; everything else is linear. Both
  archive (training/history) and forecast (next 48h) endpoints are used.

## Artefacts: committed vs gitignored
**Committed** (despite `.gitignore` patterns — these were tracked before they
were ignored, so git keeps tracking them):
- [models/model.cbm](models/model.cbm)
- [data/features/history.parquet](data/features/history.parquet),
  [data/features/features.parquet](data/features/features.parquet),
  [data/features/features_full_history.parquet](data/features/features_full_history.parquet),
  [data/features/for_predict.parquet](data/features/for_predict.parquet),
  [data/features/training_features.parquet](data/features/training_features.parquet)
- [data/predictions/latest.parquet](data/predictions/latest.parquet),
  [data/predictions/catboost_full.parquet](data/predictions/catboost_full.parquet)
- [data/raw/](data/raw/) parquets including the training set
- [metrics.json](metrics.json), [cv_metrics.json](cv_metrics.json)

**Gitignored** via `.gitignore` (now plain UTF-8 LF — old broken UTF-16 LE
version is fixed): `__pycache__/`, `.venv/`, `notebooks/`,
`catboost_info/`, `models/`, `data/`. The catboost_info/ directory was
previously tracked; it is no longer.

## Run locally
```bash
pip install -r requirements.txt
# (or `pip install -e ".[dev]"` to also pull ruff/black/pre-commit/pytest)

# Training (run once). Defaults to GPU; set CATBOOST_DEVICE=CPU to retrain
# without CUDA.
python src/etl_training.py
python src/featurise.py --mode training
CATBOOST_DEVICE=CPU python src/train_model.py
python src/validate.py

# One nightly cycle
python -m src.pipeline

# Dashboard
python dashboard/app.py          # http://127.0.0.1:8050
```

Optional dev tooling: `pre-commit install` enables ruff + black + hygiene
hooks on commit. Configuration lives in [pyproject.toml](pyproject.toml).

## Render deploy contract
- Web service, free tier. Build = `pip install -r requirements.txt`.
- Start = `gunicorn dashboard.app:server --bind 0.0.0.0:$PORT --workers 2
  --threads 4 --preload --timeout 60` (see [Procfile](Procfile)).
- [Dockerfile](Dockerfile) mirrors the Procfile and is suitable for Fly /
  Cloud Run; Render itself still uses the Procfile.
- Cold starts ~30s; the dashboard reads 5 parquet files at module import.

## CI workflows
- [.github/workflows/ci.yml](.github/workflows/ci.yml) — runs on every push
  and PR. Compiles all sources and imports `dashboard.app` to catch
  syntax/import-time breakage. No lint enforcement yet (waiting on a
  one-shot format pass).
- [.github/workflows/nightly.yml](.github/workflows/nightly.yml) — 01:30
  UTC daily. Runs `python -m src.pipeline`, commits every tracked file
  the pipeline rewrites (`latest.parquet`, `history.parquet`,
  `for_predict.parquet`, the raw ci/openmeteo parquets) as
  "Nightly forecast update: YYYY-MM-DD", then triggers the Render
  redeploy webhook only when data was actually pushed AND the
  `RENDER_DEPLOY_HOOK_URL` repo secret is set. A concurrency group
  prevents overlapping runs racing on git push.

## Working preferences
- Commit and push in stages — one logical change per commit, push after each.
  Don't bundle unrelated changes into a single commit even when fixing many
  small items in a row.

## Gotchas worth knowing
- The legacy `wind_mw` column is gone from the source tree but still
  appears in old git blame. The current target is `wind_perc` everywhere.
- Rolling windows on `wind_speed_*` are still row-based and still mislabeled
  (`_3h` is 6 rows, `_24h` is 48, `_48h` is 96). The *target* windows in
  `target_history_features` are time-based and correctly named — don't assume
  the two families behave alike.
- Weather inputs were upgraded (2026-08): both ETLs now fetch
  `temperature_2m, wind_speed_10m, wind_speed_100m, wind_gusts_10m,
  wind_direction_10m, surface_pressure`. The Open-Meteo **archive** only
  exposes `wind_speed_100m` — NOT 80m/120m, which the forecast API does
  have; mixing them silently yields all-NaN columns. The retrained model
  (holdout RMSE 1.45, was 1.89) uses hub-height v³ proxies, gust factor,
  direction sin/cos and 3h pressure tendency.
- `model.cbm` ships with quantile companions `model_p10.cbm` /
  `model_p90.cbm` (same hyper-params, `Quantile:alpha=` loss).
  `predict.py` emits `wind_perc_pred_p10/p90` when they exist and degrades
  gracefully when they don't; the dashboard NaN-fills missing band columns.
- Fast retrain: `OPTUNA_TRIALS=0 CATBOOST_DEVICE=CPU python src/train_model.py`
  reuses `best_params` from [metrics.json](metrics.json) instead of the
  100-trial search. Full retune still available via `OPTUNA_TRIALS=100`.
- [cv_metrics.json](cv_metrics.json) regenerated after the honest rebuild:
  walk-forward RMSE **6.86 ± 1.30 %-pts** against
  climatology 16.60 and 48h-persistence
  18.50. Hold-out (final 30 days,
  untouched by tuning) is 5.81. **Quote the
  walk-forward number** — its folds are ~485 days and the earliest trains on a
  fraction of the series, so it is the conservative one. Anything near 1.4 is
  a pre-fix leaked figure. validate.py uses SMAPE because unbounded MAPE
  explodes on calm hours.
- `tests/test_featurise.py` covers lag alignment, holiday seeding, the
  power-curve proxies and rolling-window leakage; CI runs it on every push.
- The Render deploy hook URL was previously committed in plaintext in
  `nightly.yml`. The workflow now reads it from a repo secret, but the
  old URL is still in git history forever — **rotate on Render** if it
  hasn't been done already.
- The nightly commit step MUST use `git add -f` for the data/ parquets.
  They are tracked-despite-gitignore, and `git add` of an ignored path
  exits non-zero even for a tracked file (git 2.54), which under the
  Actions `set -e` aborted the step. This silently broke every nightly
  from ~2026-05-07 (when the UTF-16 .gitignore was fixed to actually
  ignore data/) until the `-f` fix. That is why the live demo data went
  stale.
- `px.line` auto-upgrades to Scattergl on large frames; Scattergl rejects
  `line.shape="spline"`. `style_series_figure` only applies spline to
  real SVG `scatter` traces — don't reintroduce it unconditionally.

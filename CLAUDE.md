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
[src/validate.py](src/validate.py).
Produces [models/model.cbm](models/model.cbm),
[metrics.json](metrics.json), [cv_metrics.json](cv_metrics.json).

**Nightly inference (GitHub Actions):**
[src/pipeline.py](src/pipeline.py) orchestrates
[src/etl_inference.py](src/etl_inference.py) →
[src/featurise.py](src/featurise.py) (default inference mode) →
[src/predict.py](src/predict.py).
Produces [data/predictions/latest.parquet](data/predictions/latest.parquet) and
[data/features/history.parquet](data/features/history.parquet).

`pipeline.py` does file-shuffling state management:
moves `features.parquet` → `for_predict.parquet`, restores
`features_full_history.parquet` over `features.parquet`. Treat that flow as
load-bearing; missing snapshots silently produce empty outputs.

## Data sources
- **Carbon Intensity API** — GB wind generation as % of mix (the target).
- **Open-Meteo** — `temperature_2m`, `wind_speed_10m` for a single hardcoded
  UK location (54.0°N, −1.5°E). Both archive (training/history) and forecast
  (next 48h) endpoints are used.

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
  UTC daily. Runs `python -m src.pipeline`, commits the updated
  `data/predictions/latest.parquet` and `data/features/history.parquet`
  as "Nightly forecast update: YYYY-MM-DD", then triggers the Render
  redeploy webhook **iff** the `RENDER_DEPLOY_HOOK_URL` repo secret is
  set. Until it is, Render redeploys still happen via Render's own git
  auto-deploy (if enabled) but not via the explicit hook ping.

## Working preferences
- Commit and push in stages — one logical change per commit, push after each.
  Don't bundle unrelated changes into a single commit even when fixing many
  small items in a row.

## Gotchas worth knowing
- The legacy `wind_mw` column is gone from the source tree but still
  appears in old git blame. The current target is `wind_perc` everywhere.
- [cv_metrics.json](cv_metrics.json) used to report trillion-scale MAPE
  on folds 2–4 because the unbounded MAPE divides by near-zero
  `wind_perc` during calm. `validate.py` now uses SMAPE instead. The
  committed JSON has only the (always-correct) RMSE numbers until the
  next manual `python src/validate.py` regenerates the SMAPE block.
- The README roadmap item "error plot for Historical Analysis tab" is
  already implemented at
  [dashboard/app.py:736-799](dashboard/app.py#L736-L799).
- The Render deploy hook URL was previously committed in plaintext in
  `nightly.yml`. The workflow now reads it from a repo secret, but the
  old URL is still in git history forever — **rotate on Render** if it
  hasn't been done already.
- `is_holiday` was constant 0 for the model's whole life: `Series.isin()`
  against a lazily-populated `holidays.UnitedKingdom()` dict returns
  all-False until the dict is seeded with years. `featurise.py` now seeds
  it, but the shipped [models/model.cbm](models/model.cbm) was trained on
  the constant-0 column — **retrain to actually benefit** from the flag.
- No automated tests yet. The CI workflow only does compile + import
  checks. Item 12 of the improvement plan tracks the test backlog.

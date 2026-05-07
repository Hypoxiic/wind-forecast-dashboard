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
- [catboost_info/](catboost_info/) — training telemetry, should not be tracked

**Gitignored:** `__pycache__/`, `.venv/`, `notebooks/`. NB: `.gitignore` is
UTF-16 LE with a broken first line containing literal `\n` escapes, so the
first three patterns are effectively dead. `__pycache__/` and `.venv/` are
**not** actually ignored — fix before relying on them.

## Run locally
```bash
pip install -r requirements.txt

# Training (run once)
python src/etl_training.py
python src/featurise.py --mode training
python src/train_model.py        # GPU-only — task_type="GPU" is hardcoded
python src/validate.py

# One nightly cycle
python -m src.pipeline

# Dashboard
python dashboard/app.py          # http://127.0.0.1:8050
```

## Render deploy contract
- Web service, free tier. Build = `pip install -r requirements.txt`.
- Start = `gunicorn dashboard.app:server --bind 0.0.0.0:$PORT`
  (see [Procfile](Procfile)). No `--workers` / `--threads`, so 1 sync worker.
- [Dockerfile](Dockerfile) is **not** the deploy path. Its `ENTRYPOINT` runs
  `python src/pipeline.py`, which would deploy the wrong process. Treat as
  orphaned.
- Cold starts ~30s; the dashboard reads 5 parquet files at module import.

## Nightly GitHub Action
[.github/workflows/nightly.yml](.github/workflows/nightly.yml), 01:30 UTC:
1. Runs `python -m src.pipeline`.
2. Commits `data/predictions/latest.parquet` and
   `data/features/history.parquet` as "Nightly forecast update: YYYY-MM-DD".
3. Pings the Render deploy webhook to redeploy.
NB: the Render deploy key is currently committed in plaintext on line 53.

## Working preferences
- Commit and push in stages — one logical change per commit, push after each.
  Don't bundle unrelated changes into a single commit even when fixing many
  small items in a row.

## Gotchas worth knowing
- `wind_mw` is dead-code residue from before the target became `wind_perc`.
  Still appears as drop-targets in [predict.py:17](src/predict.py#L17) and
  exclusion lists in `train_model.py` / `validate.py`.
- [src/etl.py](src/etl.py) is dead — `pipeline.py` imports
  `etl_inference as etl`. The name collision is confusing.
- [cv_metrics.json](cv_metrics.json) reports trillion-scale MAPE for folds 2–4
  because `wind_perc` approaches zero in calm periods and MAPE divides by it.
  The mean MAPE is meaningless; trust per-fold RMSE instead.
- The README roadmap item "error plot for Historical Analysis tab" is already
  implemented at
  [dashboard/app.py:736-799](dashboard/app.py#L736-L799).
- No tests, no pre-commit hooks, no PR CI. The nightly Action is the only
  automation.

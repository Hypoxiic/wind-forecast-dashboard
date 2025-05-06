# Wind Day‑Ahead Forecast & Dashboard

Live demo → [https://wind‑forecast-dashboard.onrender.com](https://wind‑forecast-dashboard.onrender.com)  (Free Render service – cold‑start ≤ 30 s)

*A concise, fully‑reproducible mini‑project that forecasts the **day‑ahead percentage of Great‑Britain wind generation in the national mix** and serves the results through an interactive Dash app.*

> **Note:** This project uses the [Carbon Intensity API](https://carbonintensity.org.uk/) for wind generation data (percentage contribution, 0-100%) and the [Open-Meteo API](https://open-meteo.com/) for weather forecasts.

---

## 1  Business context

Short‑horizon wind output (as a percentage of the mix) drives GB's supply‑demand balance, flexible‑asset dispatch and baseload prices. Accurate 24 h forecasts ⇒ lower imbalance costs and better market bids.

---

## 2  Automated data & modelling pipeline

```text
# Training Setup (Run manually as needed)
(Carbon‑Intensity API + Open‑Meteo archive) ─▶ src/etl_training.py   ─▶ data/raw/training/
                                             (ci_wind_*.parquet + openmeteo_*.parquet)
                                                        │
                                                        ▼
                                src/featurise.py --mode training ─▶ data/features/training_features.parquet
                                                        │           (+ lags, seasonality, v³ proxy)
                                                        ▼
                                        src/train_model.py    ─▶ models/model.cbm, metrics.json
                                        src/validate.py       ─▶ cv_metrics.json

# Daily Prediction Pipeline (Automated via GitHub Actions)
(Carbon‑Intensity API + Open‑Meteo forecast) ─▶ src/pipeline.py orchestrates:
                                                  ├─▶ src/etl_inference.py  ─▶ data/raw/ (ci.parquet, openmeteo_weather.parquet)
                                                  │
                                                  ├─▶ src/featurise.py (inf. mode) ─▶ data/features/ (features.parquet, for_predict.parquet)
                                                  │
                                                  ├─▶ src/predict.py    ─▶ data/predictions/latest.parquet
                                                  │  (using models/model.cbm)
                                                  │
                                                  └─▶ Updates data/features/history.parquet


# Dashboard
Reads historical actuals, baseline data, full historical predictions, and latest forecasts.
Displays data via dashboard/app.py in two tabs:
  - Forecast & Recent: Shows recent history + upcoming 48h forecast with static model performance KPIs.
  - Historical Analysis: Allows exploring the full historical data and predictions, with dynamically calculated performance KPIs (RMSE/MAPE for model and baseline) for the selected date range.
```

### Nightly refresh

A GitHub Actions workflow (`.github/workflows/nightly.yml`) runs at **01 : 30 UTC** every night:

1. Executes `src/pipeline.py` which:
    a. Fetches latest actual wind % (inc. today) and weather (history + 48h forecast) using `src/etl_inference.py`.
    b. Generates features for the inference period using `src/featurise.py` (inference mode).
    c. Runs the prediction using the trained model (`models/model.cbm`) via `src/predict.py`.
    d. Updates the rolling `data/features/history.parquet`.
2. Commits & pushes the updated `data/predictions/latest.parquet` and `data/features/history.parquet`.
3. Render (if configured for auto-deploy) redeploys the Dash app with the latest data.

---

## 3  Why the model works

| Factor                                                  | Contribution                                                  |
| ------------------------------------------------------- | ------------------------------------------------------------- |
| **10 m wind‑speed forecast (D‑1)**                      | Turbine power curve → explains ≈ 90 % of wind output variance. |
| **Yesterday's observed output (%age)**                  | Autocorrelation + system inertia (percentage reflects mix).   |
| Engineered extras (v³ proxy, seasonality, holiday flag) | Mop‑up residual bias.                                         |

Walk‑forward CV + Optuna tuning + early‑stopping keep the CatBoost model honest.

---

## 4  Quick‑start (local)

```bash
# Install deps (CUDA optional but speeds training)
pip install -r requirements.txt

# --- Training Pipeline (Run once or when retraining) ---
# 1. Fetch full history and weather data for training
python src/etl_training.py

# 2. Build features for training
python src/featurise.py --mode training

# 3. Train and validate the model (saves model and CV metrics)
#    NOTE: train_model.py includes lengthy Optuna tuning.
python src/train_model.py
python src/validate.py

# --- Daily Prediction Pipeline (Simulates nightly run) ---
# This single script runs the full inference pipeline:
python src/pipeline.py

# --- Launch Dashboard ---
# Reads data/features/history.parquet and data/predictions/latest.parquet
python dashboard/app.py         # http://127.0.0.1:8050
```

### Deploy to Render (free tier)

1. Repo already includes `Procfile` + `gunicorn` entry.
2. Push to GitHub and create **New → Web Service** on Render pointing at this repo.
3. Build command = `pip install -r requirements.txt` (auto)
   Start command = `gunicorn dashboard.app:server --bind 0.0.0.0:$PORT`
4. Free plan → Create. In ~2 min you get the public URL.
5. The nightly GitHub Action pushes updated predictions and history – Render auto‑redeploys.

---

## 5  Directory map

```text
wind‑forecast‑dashboard/
├─ assets/                       # Custom CSS/JS
├─ data/   (git‑ignored contents, except specific committed files)
│   ├─ raw/
│   │   ├─ ci.parquet             # Daily inference raw wind %
│   │   ├─ openmeteo_weather.parquet # Daily inference raw weather
│   │   └─ training/              # Raw data from etl_training.py (git‑ignored?)
│   │       ├─ ci_wind_perc_training.parquet
│   │       └─ openmeteo_weather_training.parquet
│   ├─ features/
│   │   ├─ features.parquet       # Used by inference pipeline
│   │   ├─ training_features.parquet # Features for model training (git‑ignored?)
│   │   └─ history.parquet        # Rolling history (Actuals + Baseline Lags) - Committed
│   └─ predictions/
│       └─ latest.parquet         # Latest Predictions (Actuals + Forecast) - Committed
├─ dashboard/app.py              # Plotly‑Dash application (2 Tabs)
├─ src/
│   ├─ etl_training.py         # Fetch full historical data
│   ├─ etl_inference.py        # Fetch data for daily inference
│   ├─ featurise.py              # Feature engineering (training & inference modes)
│   ├─ train_model.py            # GPU CatBoost training + Optuna tuning
│   ├─ validate.py               # Cross-validation script
│   ├─ predict.py                # Generate predictions from features
│   └─ pipeline.py               # Orchestrates nightly inference pipeline
├─ models/                       # Saved model, study, and related outputs
│   ├─ model.cbm                 # Saved CatBoost model - Committed
│   ├─ optuna_study_gpu.pkl      # Optuna study object (git‑ignored?)
│   ├─ catboost_full.parquet     # Predictions on full training features (git‑ignored?)
│   └─ catboost_holdout_gpu_final.parquet # Holdout preds from training (git‑ignored?)
├─ .github/workflows/nightly.yml # Nightly pipeline workflow
├─ metrics.json                  # Holdout metrics from train_model.py - Committed
├─ cv_metrics.json               # Cross-validation metrics from validate.py - Committed
├─ requirements.txt
└─ README.md
```
(Note: Consider adding large files like `training_features.parquet`, `optuna_study_gpu.pkl`, `catboost_full.parquet`, `catboost_holdout_gpu_final.parquet`, and potentially the contents of `data/raw/training/` to your `.gitignore` if they are very large and not essential to commit directly). 

---

## 6  Road‑map

* [x] GPU CatBoost + walk‑forward CV (predicting percentage)
* [x] Full‑history predictions for dashboard (percentage)
* [x] Render free‑tier deployment
* [x] Nightly GitHub Action to refresh percentage predictions & history
* [x] Two-tab dashboard structure (Forecast/Recent, Historical)
* [x] Implement dynamic KPIs for Historical Analysis tab
* [ ] Implement error plot for Historical Analysis tab
* [ ] Pre‑commit lint/format hooks (ruff, black, isort)
* [ ] Cloudfront (or Fly io) in front of Render for faster cold‑start

---

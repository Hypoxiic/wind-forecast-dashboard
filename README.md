# Wind Day‑Ahead Forecast & Dashboard

A concise, reproducible mini‑project that forecasts **day‑ahead Great Britain wind generation (MW)** and serves the results through a lightweight dashboard.
Created as a portfolio piece to demonstrate data‑engineering, modelling, and product‑thinking skills (link shared in my Argus cover letter).

---

## 1. Purpose & Business Context

* **Why wind generation?** Accurate day‑ahead output forecasts feed directly into supply–demand balances, margin calculations and price formation—core themes for Argus Consulting.
* **Outcome‑oriented:** The repo shows an end‑to‑end pipeline: ingest → feature engineering → model training → dashboard.
* **Tech stack demo:** Python • pandas • CatBoost • Plotly Dash • GitHub Actions

---

## Why do the day‑ahead results look “almost perfect”?

Wind‑generation forecasting is unusually forgiving at a 24‑‑48 h horizon because the two strongest predictors are both available:

| Driver | Why it helps | Impact |
|--------|--------------|--------|
| **Day‑ahead hub‑height wind‑speed forecast** | Electrical output is a deterministic function of wind speed (turbine power curve).  Feeding the model tomorrow’s 100 m wind speed is essentially giving it tomorrow’s “cause.” | Explains 85‑95 % of MW variance by itself. |
| **Yesterday’s observed output (lags & rolling means)** | Wind generation is highly autocorrelated at half‑hour resolution; the physical system can’t change instantaneously. | Acts as a safety‑net when the met forecast is slightly off and captures inertia/curtailment effects. |

Additional features (hour‑of‑day sine/cos, day‑of‑year seasonality, holiday flag) mop up the remaining systematic bias.

### Caveats
* **Narrow back‑test** – our test slice is ≈ 4 weeks (Apr–May 2025).  No extreme storms or widespread curtailment occurred, so the relationship stayed clean.  
* **No rolling cross‑validation yet** – a single 75 / 25 split can still hide regime‑specific error.

### Next safeguards
* **TimeSeriesSplit cross‑validation** across the full year to expose seasonality and rare events.  
* **Early‑stopping** during CatBoost training to curb variance.  
* **Longer evaluation windows** (seasonal or yearly) before claiming production‑grade accuracy.

In short, the model isn’t “cheating”; it’s simply leveraging very strong predictors at a short horizon. The planned cross‑validation step will confirm whether the low RMSE generalises across different weather regimes.

---

## 2. Data Sources & Licences

| Role                       | Source                                               | Notes                              |
| -------------------------- | ---------------------------------------------------- | ---------------------------------- |
| **Target** (actual output) | National Grid ESO – *Historic Generation Mix* CSV    | Half‑hourly MW • public‑domain     |
| **Forecast driver**        | **[Open‑Meteo](https://open-meteo.com/)** hourly API | 100 m hub‑height wind speed • free |

> *Why not Met Office DataPoint?* New sign‑ups closed (service sunset Sept 2025). Open‑Meteo offers similar coverage without an API key.

---

## 3. Quick Start

```bash
# 0. Clone the repo & create a fresh environment
python -m venv .venv
source .venv/Scripts/activate      # Windows PowerShell

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run ETL, Feature Engineering & Model Training
# (This downloads ~50-100MB data and takes ~1-2 minutes)
python src/etl.py && python src/featurise.py && python src/train_model.py

# 3. Launch the Dashboard
python dashboard/app.py  # Runs at http://127.0.0.1:8050/
```

---

## 4. Directory Map

```
wind-forecast-dashboard/
├─ data/                    # Auto-generated, git-ignored
│   ├─ raw/                 # eso_wind.parquet, openmeteo_weather.parquet
│   ├─ features/            # features.parquet
│   └─ predictions/         # catboost_test.parquet
├─ dashboard/               # Plotly Dash app
│   └─ app.py
├─ notebooks/               # Exploration & development (optional)
│   ├─ 01_get_data.ipynb
│   └─ 03_model_dev.ipynb   # (Superseded by src/train_model.py)
├─ src/                     # Core ETL, feature, and training scripts
│   ├─ etl.py
│   ├─ featurise.py
│   └─ train_model.py
├─ .github/                 # GitHub Actions workflows
│   └─ workflows/
│        ├─ ci.yml
│        └─ retrain.yml     # (Phase 5)
├─ metrics.json             # Model evaluation metrics
├─ requirements.txt         # Python dependencies
└─ README.md                # ← You are here
```

---

## 5. Current Progress

**Phase 0 –** repo scaffold, GPT-4.1 prompts, README template

**Phase 1 –** `src/etl.py`

* Robust CSV parser for ESO (skips licence & footer rows) – *URL updated 2024-05-03*
* Open-Meteo 100 m wind-speed forecasts (key-less) – *Extended to fetch 2024+ historical*
* DST handling for Open-Meteo timestamps
* Logging, retry logic, parquet output to `data/raw/`
* *Status: Working*

**Phase 2 / 2.5 –** `src/featurise.py`

* Time-aligned merge (≤ 30 min tolerance) – *timezone bug fixed*
* Basic lags & calendar features
* *Added features:* 24/48h lags, 24/48h rolling stats, day-of-year cyclical
* Features saved to `data/features/`
* *Status: Working*

**Phase 3 –** `src/train_model.py`
* 75 / 25 chronological split (3162 train, 1054 test)
* Baseline MAPE 0.9232 | RMSE 3660 MW  *(Note: Previous run MAPE differed)*
* CatBoost MAPE 0.0687 | RMSE 327 MW (-91 % RMSE vs Baseline)
* `metrics.json` saved at project root
* *Status: Completed*

**Phase 3.5 (Validation) –** `src/validate.py`
* TimeSeriesSplit CV (5 folds, ~15% test size)
* Early stopping used during training.
* Mean CV MAPE: 0.2068 ± 0.2188
* Mean CV RMSE: 629.83 ± 310.45 MW
* `cv_metrics.json` saved at project root.
* Average feature importance logged.
* *Status: Completed*

---

## 6. TODO / Next Steps

* [ ] Expand feature set (rolling means, gusts, holiday flags) - *Some basic features already added*
* [ ] Run patched model development notebook (`notebooks/03_model_dev.ipynb`) - *Superseded by train_model.py*
    * [ ] Analyze feature importance - *Can add to train_model.py or a new notebook*
    * [ ] Potentially tune CatBoost parameters
    * [ ] Finalize model choice and save trained model artifact (e.g., `model.cbm`)
* [ ] Refactor model training logic into `src/model.py` script - *Partially done in train_model.py*
* [ ] Phase 4 – interactive Plotly Dash dashboard (live metrics & forecast plot)
* [ ] Single-page Dash app (`dashboard/app.py`) with KPI cards & horizon slider - *Stub created*
* [ ] `requirements.txt` + pre-commit hooks (black, ruff, isort)
* [ ] GitHub Actions `retrain.yml` – nightly ETL → model retrain → dashboard redeploy
* [ ] Update README & docs after each phase

---

*This README is a living document—updated after every major commit.*

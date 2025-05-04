# Wind Day‑Ahead Forecast & Dashboard

*A concise, fully‑reproducible mini‑project that forecasts **day‑ahead Great‑Britain wind generation (MW)** and serves the results through a lightweight Plotly‑Dash app.*  Originally built as a portfolio piece – now a polished tutorial repo.

---

## 1  Business context – why short‑horizon wind matters

| Question            | Answer                                                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Who needs this?** | Power‐trading desks, system operators, renewable asset owners.                                                                             |
| **Why day‑ahead?**  | GB’s wholesale market clears in D‑1 auctions; accurate generation forecasts feed directly into supply‑demand balances and price formation. |
| **Why ML?**         | Modern tree ensembles digest non‑linear power‑curves, lags and regime shifts better than textbook statistical models.                      |

---

## 2  Project outline

```
(raw CSV + API) ─▶  src/etl.py        ─▶  parquet        ─▶  src/featurise.py ─▶  features.parquet  ─▶  src/train_model.py ─▶  model.cbm + metrics + predictions
                                                             ▲                                                            ▼
                                                             └──────────────────────── dashboard/app.py ◀────────────────────┘
```

* **ETL** – grabs historical ESO wind output & Open‑Meteo hub‑height forecasts, stores them as parquet.
* **Feature engineering** – adds power‑curve proxies, calendar cycles, lags, rolling stats.
* **Model** – CatBoost on GPU, tuned with Optuna & 5‑fold walk‑forward CV (early‑stopping).
* **Dashboard** – KPI cards, light/dark toggle, forecast plot & error histogram.  Immediate feedback for any experiment.

---

## 3  Why do the results look *so* good?

Day‑ahead wind is a case where two public inputs carry almost all the predictive power:

1. **100 m wind‑speed forecast** – the turbine power‑curve is deterministic → ≈ 90 % of the variance.
2. **Yesterday’s output** – half‑hour wind is autocorrelated; also captures curtailment inertia.

> *With those two features a GB wind model can reach \~98 % R². The remaining work is guarding against over‑fitting and extreme regimes.*

Safeguards we added:

* **Expanding walk‑forward CV** – exposes seasonality / weather regime shifts.
* **Early‑stopping** inside each fold.
* **Hold‑out window** – last 24 h are never seen during tuning.

---

## 4  What’s new in v0.5 (2025‑05‑05)

| Area                | v0.4                                               | **v0.5**                                                              |
| ------------------- | -------------------------------------------------- | --------------------------------------------------------------------- |
| Prediction artefact | Hold‑out‑only `catboost_holdout_gpu_final.parquet` | **Full‑range** `catboost_full.parquet` saved automatically.           |
| Dashboard           | Hard‑coded path                                    | Reads whichever file exists (`*_full` ➜ preferred, else `*_holdout`). |
| README              | Outdated metrics & paths                           | **This file** – re‑written with correct numbers.                      |

---

## 5  Quick‑start

```bash
# 0  Dependencies (CUDA optional but recommended)
pip install -r requirements.txt

# 1  Full pipeline (ETL → features → model → full predictions)
python src/etl.py && \
python src/featurise.py && \
python src/train_model.py        # creates data/predictions/catboost_full.parquet

# 2  Launch dashboard (light theme by default)
python dashboard/app.py
```

You should see KPI cards like:

* **Baseline 48 h‑lag**  RMSE ≈ 4 900 MW | MAPE ≈ 1.0
* **CatBoost**          RMSE ≈  315 MW | MAPE ≈ 0.03 (↓ > 90 %)

*(Exact numbers vary with the latest data pull.)*

---

## 6  Directory map

```
wind‑forecast‑dashboard/
├─ assets/                       # Custom CSS / JS (fonts, theme fixes)
│   └─ z_override.css            # forces uniform 16 px Inter font in both themes
├─ data/                         # Auto‑generated, git‑ignored
│   ├─ raw/                      # eso_wind.parquet, openmeteo_weather.parquet
│   ├─ features/                 # features.parquet
│   └─ predictions/              # catboost_full.parquet, catboost_holdout_gpu_final.parquet
├─ dashboard/
│   └─ app.py                    # Plotly‑Dash application
├─ src/
│   ├─ etl.py                    # raw‑data ingestion
│   ├─ featurise.py              # feature engineering
│   └─ train_model.py            # GPU CatBoost + Optuna tuning + prediction export
├─ models/                       # Saved model.cbm + feature importance TSVs
├─ metrics.json                  # Hold‑out & CV metrics
├─ optuna_study_gpu.pkl          # Full Optuna study object (75 trials)
├─ .github/workflows/            # CI + (future) nightly retrain
└─ README.md
```

---

## 7  Road‑map

* [x] GPU walk‑forward CV & Optuna tuning.
* [x] Full prediction export for dashboard.
* [x] Theme‑consistent typography & KPI tool‑tips.
* [ ] Pre‑commit hooks (black, ruff, isort).
* [ ] GitHub Actions workflow to run the pipeline nightly & push updated dashboard artefacts.
* [ ] Cloud deployment (Render / Fly.io).

---

*This README is a living document – updated after every major commit.*

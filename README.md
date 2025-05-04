# Wind Day‑Ahead Forecast & Dashboard

Live demo → [https://wind‑forecast-dashboard.onrender.com](https://wind‑forecast-dashboard.onrender.com)  (Free Render service – cold‑start ≤ 30 s)

*A concise, fully‑reproducible mini‑project that forecasts **day‑ahead Great‑Britain wind generation** and serves the results through an interactive Dash app. Built as a portfolio piece for the Argus Media Graduate Analyst application.*

---

## 1  Business context

Short‑horizon wind output drives GB’s supply‑demand balance, flexible‑asset dispatch, and dayahead baseload prices. Accurate 24 h forecasts = lower imbalance costs and better market bids.

---

## 2  Pipeline

```text
(raw CSV + API) ─▶  src/etl.py        ─▶  data/raw/                  |  ESO wind + Open‑Meteo
                         │
                         ▼
              src/featurise.py        ─▶  data/features/features.parquet
                         │  (+ power‑curve proxies, lags, seasonality)
                         ▼
              src/train_model.py      ─▶  model.cbm            ─┐
                                             │                 │
                                             └▶  data/predictions/
                                                      ├─ catboost_full.parquet   # entire history
                                                      └─ catboost_holdout.parquet # last 24 h only

Dash app reads the *full* predictions file if it exists, else the hold‑out slice.
```

---

## 3  Why are the metrics so good?

| Factor                                                  | Contribution                                          |
| ------------------------------------------------------- | ----------------------------------------------------- |
| **100 m wind‑speed forecast (D‑1)**                     | Turbine power curve → explains ≈ 90 % of MW variance. |
| **Yesterday’s observed output**                         | Autocorrelation + curtailment inertia.                |
| Engineered extras (v³ proxy, seasonality, holiday flag) | Mop‑up residual bias.                                 |

5‑fold walk‑forward CV + early‑stopping keeps the CatBoost model honest.

---

## 4  Quick‑start

```bash
# Install deps (CUDA optional but speeds training)
pip install -r requirements.txt

# Run full pipeline
python src/etl.py && \
python src/featurise.py && \
python src/train_model.py        # writes catboost_full.parquet

# Launch dashboard locally
python dashboard/app.py           # http://127.0.0.1:8050
```

### Deploy to Render (free tier)

1. The repo already contains `Procfile` + `gunicorn` entry.  Push to GitHub.
2. Create **New → Web Service** on [https://dashboard.render.com/](https://dashboard.render.com/) → pick this repo.
3. Build command = `pip install -r requirements.txt` (auto)
   Start command = `gunicorn dashboard.app:server --bind 0.0.0.0:$PORT`
4. Free plan → Create. In \~2 min you get the public URL (above).

Every subsequent `git push main` auto‑redeploys.

---

## 5  Directory map

```text
wind‑forecast‑dashboard/
├─ assets/                       # custom CSS/JS (fonts, debug close btn)
│   └─ z_override.css            # forces uniform 16 px Inter font
├─ catboost_info/               # CatBoost train logs (auto‑gen, .gitignored)
├─ data/  (git‑ignored)         # all artefacts produced by pipeline
│   ├─ raw/                     # eso_wind.parquet, openmeteo_weather.parquet
│   ├─ features/                # features.parquet
│   └─ predictions/             # catboost_full.parquet + hold‑out slice
├─ dashboard/
│   └─ app.py                   # Plotly‑Dash application (exposes server)
├─ src/
│   ├─ etl.py                   # raw‑data ingestion
│   ├─ featurise.py             # feature engineering
│   ├─ train_model.py           # GPU CatBoost + Optuna tuning + full preds
│   └─ validate.py              # cross‑validation & feature importance
├─ models/                      # saved model.cbm + importance TSVs
├─ Procfile                     # Render/Heroku entry‑point
├─ requirements.txt             # python deps inc. gunicorn
├─ metrics.json                 # hold‑out & CV summary
├─ optuna_study_gpu.pkl         # 75‑trial tuning study
└─ README.md                    # ← this file
```

---

## 6  Road‑map

* [x] GPU CatBoost + walk‑forward CV
* [x] Full‑history predictions for dashboard
* [x] Render free‑tier deployment
* [ ] Nightly GitHub Action to retrain & redeploy
* [ ] Pre‑commit lint/format hooks (ruff, black, isort)
* [ ] Cloudfront in front of Render for faster cold‑start

---


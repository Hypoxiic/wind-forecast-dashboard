# Wind Day‑Ahead Forecast & Dashboard

Live demo → [https://wind‑forecast-dashboard.onrender.com](https://wind‑forecast-dashboard.onrender.com)  (Free Render service – cold‑start ≤ 30 s)

*A concise, fully‑reproducible mini‑project that forecasts **day‑ahead Great‑Britain wind generation** and serves the results through an interactive Dash app.*

---

## 1  Business context

Short‑horizon wind output drives GB's supply‑demand balance, flexible‑asset dispatch and baseload prices. Accurate 24 h forecasts ⇒ lower imbalance costs and better market bids.

---

## 2  Automated data & modelling pipeline

```text
(Carbon‑Intensity API  +  Open‑Meteo archive) ─▶  src/etl.py            ─▶  data/raw/
                                                 (ci_wind.parquet + openmeteo.parquet)
                                                            │
                                                            ▼
                                            src/featurise.py  ─▶  data/features/features.parquet
                                                            │  (+ lags, seasonality, v³ proxy)
                                                            ▼
                                            src/train_model.py ─▶  models/catboost_best.cbm   ─┐
                                                                                               │
                                                                                               ▼
                                            src/predict.py     ─▶  data/predictions/latest.parquet  ← nightly

Dash app reads `latest.parquet` every 10 min (auto‑refresh) so the chart is always current.
```

### Nightly refresh

A GitHub Actions workflow runs at **01 : 30 UTC** every night:

1. Re‑download yesterday's wind out‑turn from the **Carbon‑Intensity API**.
2. Fetch the newest weather features from **Open‑Meteo** (month cache keeps the run fast).
3. Re‑build features and re‑inference the CatBoost model.
4. Commit & push `data/predictions/latest.parquet` → Render redeploys the Dash app automatically.

---

## 3  Why the model works

| Factor                                                  | Contribution                                          |
| ------------------------------------------------------- | ----------------------------------------------------- |
| **10 m wind‑speed forecast (D‑1)**                      | Turbine power curve → explains ≈ 90 % of MW variance. |
| **Yesterday's observed output**                         | Autocorrelation + curtailment inertia.                |
| Engineered extras (v³ proxy, seasonality, holiday flag) | Mop‑up residual bias.                                 |

Walk‑forward CV + early‑stopping keep the CatBoost model honest.

---

## 4  Quick‑start (local)

```bash
# Install deps (CUDA optional but speeds training)
pip install -r requirements.txt

# Run the whole pipeline once
python src/etl.py && \
python src/featurise.py && \
python src/train_model.py && \
python src/predict.py

# Launch dashboard locally
python dashboard/app.py         # http://127.0.0.1:8050
```

### Deploy to Render (free tier)

1. Repo already includes `Procfile` + `gunicorn` entry.
2. Push to GitHub and create **New → Web Service** on Render pointing at this repo.
3. Build command = `pip install -r requirements.txt` (auto)
   Start command = `gunicorn dashboard.app:server --bind 0.0.0.0:$PORT`
4. Free plan → Create. In ~2 min you get the public URL.
5. The nightly GitHub Action pushes a new predictions file each night – Render auto‑redeploys.

---

## 5  Directory map

```text
wind‑forecast‑dashboard/
├─ assets/                       # custom CSS/JS (fonts, debug close btn)
├─ data/   (git‑ignored)
│   ├─ raw/
│   │   ├─ wind/ci_wind.parquet          # Carbon‑Intensity wind out‑turn (cached)
│   │   └─ meteo/openmeteo_YYYY‑MM.parquet  # monthly weather cache
│   ├─ features/features.parquet
│   └─ predictions/latest.parquet       # refreshes nightly
├─ dashboard/app.py              # Plotly‑Dash application
├─ src/
│   ├─ etl.py                    # data ingestion (CI + Open‑Meteo)
│   ├─ featurise.py              # feature engineering
│   ├─ train_model.py            # GPU CatBoost training + Optuna tuning
│   ├─ predict.py                # nightly inference
│   └─ pipeline.py               # one‑shot driver (called by GitHub Action)
├─ models/                       # saved CatBoost model(s)
├─ .github/workflows/nightly.yml # nightly CI/CD pipeline
├─ requirements.txt
└─ README.md
```

---

## 6  Road‑map

* [x] GPU CatBoost + walk‑forward CV
* [x] Full‑history predictions for dashboard
* [x] Render free‑tier deployment
* [x] **Nightly GitHub Action to refresh predictions**
* [ ] Pre‑commit lint/format hooks (ruff, black, isort)
* [ ] Cloudfront (or Fly io) in front of Render for faster cold‑start

---

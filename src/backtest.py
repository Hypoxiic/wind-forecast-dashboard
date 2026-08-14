"""Out-of-sample historical predictions for the dashboard.

models/catboost_full.parquet feeds the Historical tab's chart and its RMSE /
SMAPE cards. It used to be filled by asking the production model to predict its
own training data, so the tab advertised an in-sample fit (~1.3 %-points) while
the hero card showed the honest hold-out figure (~5.8). Two numbers, 4x apart,
on the same page.

This runs a proper expanding-window backtest instead: for each fold, models are
fitted only on data preceding it, so every prediction written here is one the
model could actually have made at the time. Rows before the first fold get NaN
rather than an in-sample guess.

    python -m src.backtest              # uses metrics.json best_params
    BACKTEST_FOLDS=6 python -m src.backtest
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DEVICE       = os.environ.get("CATBOOST_DEVICE", "GPU").upper()
N_FOLDS      = int(os.environ.get("BACKTEST_FOLDS", "8"))
# Fraction of the series reserved as the initial training block; the backtest
# covers everything after it.
WARMUP_FRAC  = float(os.environ.get("BACKTEST_WARMUP", "0.35"))
WITH_BAND    = os.environ.get("BACKTEST_BAND", "1") != "0"

FEAT_PATH    = Path("data/features/training_features.parquet")
METRICS_PATH = Path("metrics.json")
OUT_PATH     = Path("models/catboost_full.parquet")
CAT_FEATURES = ["is_holiday"]
QUANTILES    = {"wind_perc_pred_p10": 0.1, "wind_perc_pred_p90": 0.9}


def load_params() -> tuple[dict, list[str], float]:
    m = json.loads(METRICS_PATH.read_text())
    p = dict(m["best_params"])
    if "lr" in p:
        p["learning_rate"] = p.pop("lr")
    if "l2" in p:
        p["l2_leaf_reg"] = p.pop("l2")
    if p.get("bootstrap_type") == "Bayesian":
        p.pop("subsample", None)
    p.update(loss_function="RMSE", eval_metric="RMSE",
             task_type=DEVICE, random_state=42, verbose=0)
    if DEVICE == "GPU":
        p["devices"] = "0"
    return p, list(m["features"]), float(m.get("conformal_delta_pts", 0.0))


def main() -> None:
    params, features, delta = load_params()
    df = pd.read_parquet(FEAT_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.dropna(subset=["wind_perc"]).sort_values("datetime").reset_index(drop=True)

    n = len(df)
    start = int(n * WARMUP_FRAC)
    bounds = np.linspace(start, n, N_FOLDS + 1).astype(int)
    logging.info("Backtesting %d folds over rows %d..%d of %d (%s)",
                 N_FOLDS, start, n, n, DEVICE)

    out = pd.DataFrame({"datetime": df["datetime"]})
    for col in ["wind_perc_pred", *QUANTILES]:
        out[col] = np.nan

    X, y = df[features], df["wind_perc"]
    for i in range(N_FOLDS):
        lo, hi = bounds[i], bounds[i + 1]
        t0 = time.time()
        tr = slice(0, lo)
        fit_pool = Pool(X.iloc[tr], y.iloc[tr], cat_features=CAT_FEATURES)
        test_pool = Pool(X.iloc[lo:hi], cat_features=CAT_FEATURES)

        point = CatBoostRegressor(**params).fit(fit_pool).predict(test_pool)
        point = np.clip(point, 0, 100)
        out.iloc[lo:hi, out.columns.get_loc("wind_perc_pred")] = point

        if WITH_BAND:
            for col, alpha in QUANTILES.items():
                qp = dict(params)
                qp["loss_function"] = f"Quantile:alpha={alpha}"
                qp["eval_metric"] = f"Quantile:alpha={alpha}"
                q = CatBoostRegressor(**qp).fit(fit_pool).predict(test_pool)
                # Same crossing guard + conformal widening as predict.py.
                q = (np.minimum(q, point) - delta if alpha < 0.5
                     else np.maximum(q, point) + delta)
                out.iloc[lo:hi, out.columns.get_loc(col)] = np.clip(q, 0, 100)

        err = point - y.iloc[lo:hi].to_numpy()
        logging.info("  fold %d/%d rows %6d..%6d  train=%6d  RMSE %5.2f  (%.0fs)",
                     i + 1, N_FOLDS, lo, hi, lo,
                     float(np.sqrt(np.mean(err ** 2))), time.time() - t0)

    covered = out["wind_perc_pred"].notna()
    oos_rmse = float(np.sqrt(np.mean((out.loc[covered, "wind_perc_pred"]
                                      - y[covered]) ** 2)))
    logging.info("Out-of-sample over %d rows (%.0f%% of history): RMSE %.3f pts",
                 int(covered.sum()), 100 * covered.mean(), oos_rmse)
    out.to_parquet(OUT_PATH, index=False)
    logging.info("Wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()

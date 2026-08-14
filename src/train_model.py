"""
Wind day-ahead training script.

* Expanding-window walk-forward validation
* Optuna hyper-parameter search
* Power-curve features (v3 and clipped v3) computed in featurise.py
* Target = wind_perc (% of GB mix)
* Metrics & artefacts saved

Device selection: defaults to GPU. Set CATBOOST_DEVICE=CPU to train on CPU
(useful for contributors without CUDA, or for CI smoke tests).
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm.auto import tqdm

# ──────────────────────────
# Device selection
# ──────────────────────────
DEVICE = os.environ.get("CATBOOST_DEVICE", "GPU").upper()
if DEVICE not in ("GPU", "CPU"):
    raise ValueError(f"CATBOOST_DEVICE must be 'GPU' or 'CPU', got {DEVICE!r}")

if DEVICE == "GPU":
    try:
        from catboost.dev_utils.installation_check import check_gpu_installation  # type: ignore
        check_gpu_installation()
        logging.info("CatBoost GPU installation check passed.")
    except Exception as e:
        logging.warning(
            f"CatBoost GPU check failed or tool unavailable: {e}. "
            "Make sure your drivers / CUDA toolkit are OK, or set CATBOOST_DEVICE=CPU."
        )

# ──────────────────────────
# Config & logging
# ──────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DATA_DIR   = Path("data")
FEAT_PATH  = DATA_DIR / "features" / "training_features.parquet"
PRED_DIR   = DATA_DIR / "predictions"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = Path("metrics.json")
MODEL_PATH   = MODELS_DIR / "model.cbm"
STUDY_PATH   = MODELS_DIR / "optuna_study_gpu.pkl"

# ──────────────────────────
# Load & enrich features
# ──────────────────────────
logging.info(f"Loading features from {FEAT_PATH}…")
df = pd.read_parquet(FEAT_PATH)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
logging.info(f"Data shape: {df.shape}")

TARGET        = "wind_perc"

# Drop rows where the target variable is NaN before splitting X and y
df.dropna(subset=[TARGET], inplace=True)
logging.info(f"Data shape after dropping NaN targets: {df.shape}")

if df.empty:
    logging.error(f"No data left after dropping NaN from target column '{TARGET}'. Check feature generation.")
    # Exit or raise an error, as training cannot proceed
    raise ValueError(f"DataFrame empty after dropping NaNs from target '{TARGET}'.")

CAT_FEATURES  = ["is_holiday"]

# The published forecast reaches ~48h ahead, so a wind_perc observation is only
# usable as a feature if it predates the target by at least that much. An
# earlier version took every column indiscriminately, which pulled in
# wind_perc_lag_{30m,1h,3h,24h}: available while training, NaN in production.
# They carried 88% of that model's importance, so the quoted accuracy described
# a one-hour nowcast rather than the day-ahead forecast actually served.
# featurise.py no longer emits them; this is the belt-and-braces guard.
FORECAST_HORIZON_H = 48

def _is_admissible(col: str) -> bool:
    if not col.startswith("wind_perc_lag_"):
        return True
    suffix = col.removeprefix("wind_perc_lag_")
    if suffix.endswith("h") and suffix[:-1].isdigit():
        return int(suffix[:-1]) >= FORECAST_HORIZON_H
    return False          # "30m", "1h" and friends are all inside the horizon

_candidates = [c for c in df.columns if c not in {"datetime", TARGET}]
FEATURES    = [c for c in _candidates if _is_admissible(c)]
_rejected   = sorted(set(_candidates) - set(FEATURES))
if _rejected:
    logging.warning(
        "Excluding %d feature(s) inside the %dh forecast horizon: %s",
        len(_rejected), FORECAST_HORIZON_H, _rejected)
logging.info(f"Using {len(FEATURES)} features.")

X = df[FEATURES]
y = df[TARGET]

# ──────────────────────────
# Walk‑forward CV
# ──────────────────────────
N_SPLITS  = 5
# 30 days of hourly rows per fold. This was 48 rows (two days) — far too few
# for a stable estimate on a series this seasonal, and it made the quoted
# hold-out number swing between retrains.
TEST_SIZE = 24 * 30
# Windows carved off the end for honest reporting: the model never sees the
# hold-out, and the conformal delta is measured on data the quantile models
# were not fitted on.
HOLDOUT_HOURS = 24 * 30
CALIB_HOURS   = 24 * 60

# Chronological split, defined before tuning so the search never sees them.
# `fit` trains, `calib` sizes the conformal correction, `holdout` is touched by
# neither and is what the reported metrics describe.
n_rows     = len(X)
holdout_lo = n_rows - HOLDOUT_HOURS
calib_lo   = holdout_lo - CALIB_HOURS
if calib_lo <= TEST_SIZE * N_SPLITS:
    raise ValueError(
        f"Not enough rows ({n_rows}) for {N_SPLITS} CV folds of {TEST_SIZE} "
        f"plus a {CALIB_HOURS}h calibration and {HOLDOUT_HOURS}h hold-out.")

# Optuna tunes on everything BEFORE the calibration window. Letting the search
# score folds that overlap the hold-out would pick hyper-parameters with the
# hold-out visible, which is exactly the kind of quiet optimism this rewrite
# exists to remove.
X_tune, y_tune = X.iloc[:calib_lo], y.iloc[:calib_lo]

tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, gap=0)
logging.info(f"Using TimeSeriesSplit with {N_SPLITS} splits, test_size={TEST_SIZE}, "
             f"over the first {len(X_tune)} of {n_rows} rows.")

# ──────────────────────────
# Optuna objective
# ──────────────────────────
def objective(trial: optuna.Trial) -> float:
    # Choose bootstrap first (affects allowed params)
    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
    )

    params: dict = {
        # Core tree parameters
        "iterations":    trial.suggest_int("iterations", 500, 4000),
        "depth":         trial.suggest_int("depth", 4, 11),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2", 0.5, 20.0, log=True),

        # Row‑sampling (GPU‑compatible). NOT allowed with Bayesian bootstrap.
        "subsample":     trial.suggest_float("subsample", 0.6, 1.0),

        # Other knobs.
        # boosting_type is fixed to Plain: Ordered boosting costs ~10x the
        # wall-clock per fold on this dataset and lost every probe against
        # Plain, so searching it just burns the trial budget.
        "boosting_type":   "Plain",
        "bootstrap_type":  bootstrap_type,
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "border_count":    trial.suggest_int("border_count", 32, 255),

        "loss_function":  "RMSE",
        "eval_metric":    "RMSE",
        "task_type":      DEVICE,
        "random_state":   42,
        "verbose":        0,
    }
    if DEVICE == "GPU":
        params["devices"] = "0"

    # Remove subsample if Bayesian bootstrap (not supported)
    if bootstrap_type == "Bayesian":
        params.pop("subsample", None)
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)

    rmses = []
    fold_iter = tqdm(
        tscv.split(X_tune),
        total=N_SPLITS,
        desc=f"Trial {trial.number:03d}",
        leave=False,
    )
    for fold, (train_idx, test_idx) in enumerate(fold_iter):
        train_pool = Pool(
            X_tune.iloc[train_idx], y_tune.iloc[train_idx], cat_features=CAT_FEATURES
        )
        valid_pool = Pool(
            X_tune.iloc[test_idx], y_tune.iloc[test_idx], cat_features=CAT_FEATURES
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = CatBoostRegressor(**params)
                model.fit(
                    train_pool,
                    eval_set=valid_pool,
                    use_best_model=True,
                    early_stopping_rounds=50,
                    verbose=False,
                )
        except Exception as e:
            logging.error(f"Trial failed (fold {fold + 1}): {e}")
            return float("inf")

        pred_perc = model.predict(X_tune.iloc[test_idx])
        mse  = mean_squared_error(
            y_tune.iloc[test_idx], pred_perc
        )
        rmse = mse ** 0.5
        rmses.append(rmse)

    return float(np.mean(rmses))


# ──────────────────────────
# Hyper‑parameter tuning
# ──────────────────────────
# OPTUNA_TRIALS=0 skips the search entirely and reuses the best params stored
# in metrics.json from the previous tuning run — the right choice for a
# retrain-on-new-data where the search budget has already been spent. Set a
# positive number to retune from scratch (100 trials ≈ hours on CPU).
N_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "100"))
logging.info(f"Optuna study starts – {N_TRIALS} {DEVICE} trials with inner tqdm fold bars.")
if N_TRIALS > 0:
    # Persistent SQLite storage: each invocation ADDS N_TRIALS new trials to
    # the same study, so long tuning runs can be split into resumable
    # batches (overnight automation, interrupted shells). Set OPTUNA_STORAGE
    # to a different URL to start a separate study.
    storage = os.environ.get("OPTUNA_STORAGE", f"sqlite:///{MODELS_DIR}/optuna_study.db")
    study = optuna.create_study(
        direction="minimize",
        study_name="wind_catboost_tuning",
        storage=storage,
        load_if_exists=True,
    )
    logging.info(f"Study has {len(study.trials)} existing trials; adding {N_TRIALS}.")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params_optuna = study.best_params
    best_value = study.best_value
    logging.info(f"Optuna best CV RMSE: {best_value:.4f}")
    logging.info(f"Best params: {best_params_optuna}")

    # Save study (optional)
    try:
        import joblib
        joblib.dump(study, STUDY_PATH)
    except Exception as e:
        logging.warning(f"Could not save Optuna study: {e}")
else:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"OPTUNA_TRIALS=0 but {METRICS_PATH} does not exist — run with tuning at least once."
        )
    prev = json.loads(METRICS_PATH.read_text())
    best_params_optuna = prev["best_params"]
    best_value = prev.get("optuna_best_cv_rmse", float("nan"))
    logging.info(f"Reusing best params from {METRICS_PATH} (CV RMSE {best_value:.4f}).")
    study = None

# ──────────────────────────
# Prepare final parameters
# ──────────────────────────
final_params = best_params_optuna.copy()
if "lr" in final_params:
    final_params["learning_rate"] = final_params.pop("lr")
if "l2" in final_params:
    final_params["l2_leaf_reg"] = final_params.pop("l2")

# Remove column sampling (shouldn't exist, but be safe) and adjust subsample if needed
for key in ("colsample", "colsample_bylevel", "rsm"):
    final_params.pop(key, None)
if final_params.get("bootstrap_type") == "Bayesian":
    final_params.pop("subsample", None)  # Bayesian bootstrap can't have subsample

final_params.update(
    {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "task_type": DEVICE,
        "random_state": 42,
        "verbose": 200,
    }
)
if DEVICE == "GPU":
    final_params["devices"] = "0"

if "iterations" not in final_params:
    raise ValueError("'iterations' missing from best params – investigate Optuna output")

# ──────────────────────────
# Final training
# ──────────────────────────
X_fit,  y_fit  = X.iloc[:calib_lo],            y.iloc[:calib_lo]
X_cal,  y_cal  = X.iloc[calib_lo:holdout_lo],  y.iloc[calib_lo:holdout_lo]
X_hold, y_hold = X.iloc[holdout_lo:],          y.iloc[holdout_lo:]
logging.info("Split: fit=%d  calib=%d  holdout=%d rows", len(X_fit), len(X_cal), len(X_hold))

QUANTILES = {"p10": 0.1, "p90": 0.9}

def _fit_point(Xt, yt):
    m = CatBoostRegressor(**final_params)
    m.fit(Pool(Xt, yt, cat_features=CAT_FEATURES))
    return m

def _fit_quantiles(Xt, yt):
    out = {}
    for tag, alpha in QUANTILES.items():
        q = final_params.copy()
        q["loss_function"] = f"Quantile:alpha={alpha}"
        q["eval_metric"]   = f"Quantile:alpha={alpha}"
        logging.info(f"Training {tag} quantile model (alpha={alpha}) …")
        m = CatBoostRegressor(**q)
        m.fit(Pool(Xt, yt, cat_features=CAT_FEATURES))
        out[tag] = m
    return out

# ── Stage 1: evaluation models, fitted without calib/holdout ──────────────
logging.info(f"Training evaluation model on {len(X_fit)} samples …")
eval_model = _fit_point(X_fit, y_fit)
eval_qs    = _fit_quantiles(X_fit, y_fit)

# ── Conformal calibration ─────────────────────────────────────────────────
# Quantile regression is not calibrated by construction: the P10/P90 pair
# routinely misses its nominal 80%. Split-conformal fixes that distribution-
# free — take the empirical (1-alpha) quantile of the conformity score
# max(lo - y, y - hi) on data the models never saw, then widen both bounds by
# it. A negative delta means the raw band was too wide and gets narrowed.
NOMINAL_COVERAGE = 0.80
cal_lo = eval_qs["p10"].predict(X_cal)
cal_hi = eval_qs["p90"].predict(X_cal)
conformal_delta = float(np.quantile(
    np.maximum(cal_lo - y_cal.values, y_cal.values - cal_hi), NOMINAL_COVERAGE))
logging.info("Conformal delta on %d calibration rows: %+.3f pts",
             len(X_cal), conformal_delta)

def _band(qs, Xt, point):
    lo = np.minimum(qs["p10"].predict(Xt), point)
    hi = np.maximum(qs["p90"].predict(Xt), point)
    return (np.clip(lo - conformal_delta, 0, 100),
            np.clip(hi + conformal_delta, 0, 100))

# ── Hold-out evaluation (the honest numbers) ──────────────────────────────
pred_hold = np.clip(eval_model.predict(X_hold), 0, 100)
raw_lo = np.minimum(eval_qs["p10"].predict(X_hold), pred_hold)
raw_hi = np.maximum(eval_qs["p90"].predict(X_hold), pred_hold)
cal_lo_h, cal_hi_h = _band(eval_qs, X_hold, pred_hold)

rmse_final = float(np.sqrt(mean_squared_error(y_hold, pred_hold)))
mape_final = float(mean_absolute_percentage_error(y_hold, pred_hold))
baseline_col = "wind_perc_lag_48h"
baseline_rmse = (
    float(np.sqrt(mean_squared_error(
        y_hold[df[baseline_col].iloc[holdout_lo:].notna().values],
        df[baseline_col].iloc[holdout_lo:].dropna())))
    if baseline_col in df.columns else float("nan"))

cov_raw = float(np.mean((y_hold >= raw_lo)   & (y_hold <= raw_hi)))
cov_cal = float(np.mean((y_hold >= cal_lo_h) & (y_hold <= cal_hi_h)))
width_raw = float(np.mean(raw_hi - raw_lo))
width_cal = float(np.mean(cal_hi_h - cal_lo_h))

logging.info(f"Hold-out RMSE: {rmse_final:.4f} pts  (48h-persistence {baseline_rmse:.2f})")
logging.info(f"Band coverage raw {cov_raw:.3f} -> conformal {cov_cal:.3f} "
             f"(nominal {NOMINAL_COVERAGE}); width {width_raw:.2f} -> {width_cal:.2f} pts")

# ── Stage 2: production models, refitted on everything ────────────────────
# The delta measured above is carried over: it is an estimate of this model
# family's miscalibration, and refitting on the extra 90 days does not
# meaningfully change that while it does help the capacity trend stay current.
logging.info(f"Refitting production models on all {n_rows} samples …")
model = _fit_point(X, y)
quantile_models = _fit_quantiles(X, y)
for tag, qm in quantile_models.items():
    qm.save_model(str(MODELS_DIR / f"model_{tag}.cbm"))
    logging.info(f"Saved {tag} quantile model → {MODELS_DIR / f'model_{tag}.cbm'}")

# -------- save predictions for the *entire* feature set --------
full_pred_perc = np.clip(model.predict(X), 0, 100)
full_p10, full_p90 = _band(quantile_models, X, full_pred_perc)
pd.DataFrame({
    "datetime": df["datetime"],
    "wind_perc_pred": full_pred_perc,
    "wind_perc_pred_p10": full_p10,
    "wind_perc_pred_p90": full_p90,
}).to_parquet(MODELS_DIR / "catboost_full.parquet", index=False)

# ──────────────────────────
# Evaluation & artefacts
# ──────────────────────────
metrics = {
    "optuna_best_cv_rmse": best_value,
    "holdout_days": HOLDOUT_HOURS // 24,
    "holdout_rmse_perc": rmse_final,
    "holdout_mape_perc": mape_final,
    "holdout_baseline_rmse_perc": baseline_rmse,
    "holdout_band_coverage_p10_p90": cov_cal,
    "holdout_band_coverage_uncalibrated": cov_raw,
    "holdout_band_mean_width_pts": width_cal,
    "conformal_delta_pts": conformal_delta,
    "conformal_nominal_coverage": NOMINAL_COVERAGE,
    "n_features": len(FEATURES),
    "features": FEATURES,
    "best_params": best_params_optuna,
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)
logging.info(f"Saved metrics → {METRICS_PATH}")

# Save holdout predictions
# These come from the evaluation model (never fitted on the hold-out), so the
# file stays a fair record of out-of-sample behaviour.
holdout_df = pd.DataFrame({
    "datetime": df["datetime"].iloc[holdout_lo:],
    "wind_perc_actual": y_hold.values,
    "wind_perc_pred": pred_hold,
    "wind_perc_pred_p10": cal_lo_h,
    "wind_perc_pred_p90": cal_hi_h,
})
holdout_df.to_parquet(
    MODELS_DIR / "catboost_holdout_gpu_final.parquet", index=False
)
logging.info(f"Saved holdout predictions → {MODELS_DIR / 'catboost_holdout_gpu_final.parquet'}")

# Save model
model.save_model(str(MODEL_PATH))
logging.info(f"Saved final model → {MODEL_PATH}")

# Updated final log message
logging.info(f"{DEVICE} training script finished - model, metrics & predictions saved.")

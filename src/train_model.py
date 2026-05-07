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
FEATURES      = [c for c in df.columns if c not in {"datetime", TARGET}]
logging.info(f"Using {len(FEATURES)} features.")

X = df[FEATURES]
y = df[TARGET]

# ──────────────────────────
# Walk‑forward CV
# ──────────────────────────
N_SPLITS  = 5
TEST_SIZE = 24 * 2  # 48 half‑hours (24 h)

tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, gap=0)
logging.info(f"Using TimeSeriesSplit with {N_SPLITS} splits, test_size={TEST_SIZE}.")

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

        # Other knobs
        "boosting_type":   trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
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
        tscv.split(X),
        total=N_SPLITS,
        desc=f"Trial {trial.number:03d}",
        leave=False,
    )
    for fold, (train_idx, test_idx) in enumerate(fold_iter):
        train_pool = Pool(
            X.iloc[train_idx], y.iloc[train_idx], cat_features=CAT_FEATURES
        )
        valid_pool = Pool(
            X.iloc[test_idx], y.iloc[test_idx], cat_features=CAT_FEATURES
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

        pred_perc = model.predict(X.iloc[test_idx]) # <-- RENAMED pred_cf to pred_perc
        mse  = mean_squared_error(
            y.iloc[test_idx], pred_perc
        )
        rmse = mse ** 0.5
        rmses.append(rmse)

    return float(np.mean(rmses))


# ──────────────────────────
# Hyper‑parameter tuning
# ──────────────────────────
N_TRIALS = 100
logging.info(f"Optuna study starts – {N_TRIALS} {DEVICE} trials with inner tqdm fold bars.")
study = optuna.create_study(direction="minimize", study_name="wind_catboost_gpu_tuning")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params_optuna = study.best_params
best_value = study.best_value
logging.info(f"Optuna best CV RMSE: {best_value:.4f}")
logging.info(f"Best params: {best_params_optuna}")

# Save study (optional)
try:
    import joblib
    joblib.dump(study, STUDY_PATH)
except Exception as e:
    logging.warning(f"Could not save Optuna study: {e}")

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
holdout_size = TEST_SIZE
X_train_full, y_train_full = X.iloc[:-holdout_size], y.iloc[:-holdout_size]
X_holdout, y_holdout       = X.iloc[-holdout_size:], y.iloc[-holdout_size:]

logging.info(f"Training final {DEVICE} model on {X_train_full.shape[0]} samples …")
model = CatBoostRegressor(**final_params)
model.fit(Pool(X_train_full, y_train_full, cat_features=CAT_FEATURES))

# -------- save predictions for the *entire* feature set --------
full_pred_perc = model.predict(X)

pd.DataFrame({
    "datetime": df["datetime"],
    "wind_perc_pred": full_pred_perc,
}).to_parquet(
    MODELS_DIR / "catboost_full.parquet",
    index=False
)


# ──────────────────────────
# Evaluation & artefacts
# ──────────────────────────
pred_hold_perc = model.predict(X_holdout)
actual_hold_perc = y_holdout

mse_final  = mean_squared_error(actual_hold_perc, pred_hold_perc)
rmse_final = mse_final ** 0.5
mape_final = mean_absolute_percentage_error(actual_hold_perc, pred_hold_perc)

# Metrics dict update
metrics = {
    "optuna_best_cv_rmse": study.best_value,
    "holdout_rmse_perc": rmse_final,
    "holdout_mape_perc": mape_final,
    "best_params": best_params_optuna,
}

logging.info(f"Holdout RMSE (perc): {rmse_final:.4f}")
logging.info(f"Holdout MAPE (perc): {mape_final:.4f}")

# Save metrics
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)
logging.info(f"Saved metrics → {METRICS_PATH}")

# Save holdout predictions
holdout_df = pd.DataFrame({
    "datetime": df["datetime"].iloc[-holdout_size:],
    "wind_perc_actual": actual_hold_perc,
    "wind_perc_pred": pred_hold_perc,
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

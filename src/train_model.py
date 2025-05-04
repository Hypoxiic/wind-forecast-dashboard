"""
Wind day‑ahead – Enhanced GPU Training Script
---------------------------------------------
* Expanding‑window walk‑forward validation
* GPU‑accelerated Optuna hyper‑parameter search
* No column sampling (GPU limitation)
* Power‑curve features (v³ and clipped v³)
* Target = capacity factor → rescale back to MW
* Metrics & artefacts saved
"""

from __future__ import annotations

import json
import logging
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
# Optional: GPU installation check
# ──────────────────────────
try:
    from catboost.dev_utils.installation_check import check_gpu_installation  # type: ignore
    check_gpu_installation()
    logging.info("CatBoost GPU installation check passed.")
except Exception as e:
    logging.warning(
        f"CatBoost GPU check failed or tool unavailable: {e}. "
        "Make sure your drivers / CUDA toolkit are OK."
    )

# ──────────────────────────
# Config & logging
# ──────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DATA_DIR   = Path("data")
FEAT_PATH  = DATA_DIR / "features" / "features.parquet"
PRED_DIR   = DATA_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

METRICS_PATH = Path("metrics.json")
MODEL_PATH   = Path("model.cbm")
STUDY_PATH   = Path("optuna_study_gpu.pkl")

# ──────────────────────────
# Load & enrich features
# ──────────────────────────
logging.info(f"Loading features from {FEAT_PATH}…")
df = pd.read_parquet(FEAT_PATH)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
logging.info(f"Data shape: {df.shape}")

# Power‑curve proxies
RATED_MS = 15.0
df["wind_speed_v3"]      = df["wind_speed_ms"] ** 3
df["wind_speed_v3_clip"] = np.clip(df["wind_speed_ms"], 0, RATED_MS) ** 3

# Capacity‑factor target
capacity_mw          = df["wind_mw"].max()
df["capacity_factor"] = df["wind_mw"] / capacity_mw
logging.info(f"Calculated max capacity (approx): {capacity_mw:.1f} MW")

TARGET        = "capacity_factor"
CAT_FEATURES  = ["is_holiday"]
FEATURES      = [c for c in df.columns if c not in {"datetime", "wind_mw", TARGET}]
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

        # Fixed for GPU regression
        "loss_function":  "RMSE",
        "eval_metric":    "RMSE",
        "task_type":      "GPU",
        "devices":        "0",
        "random_state":   42,
        "verbose":        0,
    }

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

        pred_cf = model.predict(X.iloc[test_idx])
        mse  = mean_squared_error(
            y.iloc[test_idx] * capacity_mw, pred_cf * capacity_mw
        )
        rmse = mse ** 0.5
        rmses.append(rmse)

    return float(np.mean(rmses))


# ──────────────────────────
# Hyper‑parameter tuning
# ──────────────────────────
N_TRIALS = 75
logging.info(f"Optuna study starts – {N_TRIALS} GPU trials with inner tqdm fold bars.")
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
        "task_type": "GPU",
        "devices": "0",
        "random_state": 42,
        "verbose": 200,
    }
)

if "iterations" not in final_params:
    raise ValueError("'iterations' missing from best params – investigate Optuna output")

# ──────────────────────────
# Final training
# ──────────────────────────
holdout_size = TEST_SIZE
X_train_full, y_train_full = X.iloc[:-holdout_size], y.iloc[:-holdout_size]
X_holdout, y_holdout       = X.iloc[-holdout_size:], y.iloc[-holdout_size:]

logging.info(f"Training final GPU model on {X_train_full.shape[0]} samples …")
model = CatBoostRegressor(**final_params)
model.fit(Pool(X_train_full, y_train_full, cat_features=CAT_FEATURES))

# -------- new: save predictions for the *entire* feature set --------
full_pred_cf = model.predict(X)            # X == all data
full_pred_mw = full_pred_cf * capacity_mw

pd.DataFrame({
    "datetime": df["datetime"],
    "wind_mw_pred": full_pred_mw
}).to_parquet(
    PRED_DIR / "catboost_full.parquet",     # <- NEW file
    index=False
)


# ──────────────────────────
# Evaluation & artefacts
# ──────────────────────────
pred_hold_cf = model.predict(X_holdout)
pred_hold_mw = pred_hold_cf * capacity_mw
actual_hold_mw = y_holdout * capacity_mw

mse_final  = mean_squared_error(actual_hold_mw, pred_hold_mw)
rmse_final = mse_final ** 0.5
mape_final = mean_absolute_percentage_error(actual_hold_mw, pred_hold_mw)
logging.info(f"Hold‑out RMSE = {rmse_final:.2f} MW | MAPE = {mape_final:.4f}")

metrics = {
    "holdout_rmse_mw": rmse_final,
    "holdout_mape": mape_final,
    "capacity_mw_approx": capacity_mw,
    "optuna_best_cv_rmse": best_value,
    "optuna_n_trials": N_TRIALS,
    "optuna_best_params_raw": best_params_optuna,
    "final_model_params": final_params,
}
METRICS_PATH.write_text(json.dumps(metrics, indent=4))

model.save_model(str(MODEL_PATH))

pred_df = pd.DataFrame(
    {
        "datetime": df["datetime"].iloc[-holdout_size:],
        "wind_mw_actual": actual_hold_mw,
        "wind_mw_pred": pred_hold_mw,
    }
)
pred_df.to_parquet(PRED_DIR / "catboost_holdout_gpu_final.parquet", index=False)

logging.info("✓ Enhanced GPU training complete – model, metrics & predictions saved.")

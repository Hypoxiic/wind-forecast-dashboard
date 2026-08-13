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
holdout_size = TEST_SIZE
X_train_full, y_train_full = X.iloc[:-holdout_size], y.iloc[:-holdout_size]
X_holdout, y_holdout       = X.iloc[-holdout_size:], y.iloc[-holdout_size:]

logging.info(f"Training final {DEVICE} model on {X_train_full.shape[0]} samples …")
model = CatBoostRegressor(**final_params)
model.fit(Pool(X_train_full, y_train_full, cat_features=CAT_FEATURES))

# ──────────────────────────
# Quantile models (P10 / P90) for the dashboard uncertainty band.
# Same tuned hyper-params; only the loss changes. Quantile crossing is
# guarded against at prediction time (p90 = max(p50, p90) etc.).
# ──────────────────────────
QUANTILES = {"p10": 0.1, "p90": 0.9}
quantile_models: dict[str, CatBoostRegressor] = {}
for tag, alpha in QUANTILES.items():
    q_params = final_params.copy()
    q_params["loss_function"] = f"Quantile:alpha={alpha}"
    q_params["eval_metric"] = f"Quantile:alpha={alpha}"
    logging.info(f"Training {tag} quantile model (alpha={alpha}) …")
    qm = CatBoostRegressor(**q_params)
    qm.fit(Pool(X_train_full, y_train_full, cat_features=CAT_FEATURES))
    qm.save_model(str(MODELS_DIR / f"model_{tag}.cbm"))
    quantile_models[tag] = qm
    logging.info(f"Saved {tag} quantile model → {MODELS_DIR / f'model_{tag}.cbm'}")

# -------- save predictions for the *entire* feature set --------
full_pred_perc = model.predict(X)
full_pred_p10 = np.minimum(quantile_models["p10"].predict(X), full_pred_perc)
full_pred_p90 = np.maximum(quantile_models["p90"].predict(X), full_pred_perc)

pd.DataFrame({
    "datetime": df["datetime"],
    "wind_perc_pred": full_pred_perc,
    "wind_perc_pred_p10": full_pred_p10,
    "wind_perc_pred_p90": full_pred_p90,
}).to_parquet(
    MODELS_DIR / "catboost_full.parquet",
    index=False
)


# ──────────────────────────
# Evaluation & artefacts
# ──────────────────────────
pred_hold_perc = model.predict(X_holdout)
pred_hold_p10 = np.minimum(quantile_models["p10"].predict(X_holdout), pred_hold_perc)
pred_hold_p90 = np.maximum(quantile_models["p90"].predict(X_holdout), pred_hold_perc)
actual_hold_perc = y_holdout

mse_final  = mean_squared_error(actual_hold_perc, pred_hold_perc)
rmse_final = mse_final ** 0.5
mape_final = mean_absolute_percentage_error(actual_hold_perc, pred_hold_perc)

# Uncertainty-band quality on the holdout: coverage should be near 80% and
# the average width says how informative the band is.
band_coverage = float(np.mean((actual_hold_perc >= pred_hold_p10) & (actual_hold_perc <= pred_hold_p90)))
band_mean_width = float(np.mean(pred_hold_p90 - pred_hold_p10))

# Metrics dict update
metrics = {
    "optuna_best_cv_rmse": best_value,
    "holdout_rmse_perc": rmse_final,
    "holdout_mape_perc": mape_final,
    "holdout_band_coverage_p10_p90": band_coverage,
    "holdout_band_mean_width_pts": band_mean_width,
    "best_params": best_params_optuna,
}

logging.info(f"Holdout RMSE (perc): {rmse_final:.4f}")
logging.info(f"Holdout MAPE (perc): {mape_final:.4f}")
logging.info(f"Holdout P10–P90 band: coverage {band_coverage:.3f}, mean width {band_mean_width:.2f} pts")

# Save metrics
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)
logging.info(f"Saved metrics → {METRICS_PATH}")

# Save holdout predictions
holdout_df = pd.DataFrame({
    "datetime": df["datetime"].iloc[-holdout_size:],
    "wind_perc_actual": actual_hold_perc,
    "wind_perc_pred": pred_hold_perc,
    "wind_perc_pred_p10": pred_hold_p10,
    "wind_perc_pred_p90": pred_hold_p90,
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

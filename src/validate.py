# src/validate.py

import pandas as pd
import numpy as np
import json
import logging
import os
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from collections import defaultdict

DEVICE = os.environ.get("CATBOOST_DEVICE", "CPU").upper()
if DEVICE not in ("GPU", "CPU"):
    raise ValueError(f"CATBOOST_DEVICE must be 'GPU' or 'CPU', got {DEVICE!r}")


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE, bounded in [0, 2]. Robust when actuals approach zero,
    which the unbounded MAPE used previously was not — wind_perc dropping into
    single-digit percentages during calm periods produced fold-level MAPE in
    the trillions.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features" / "training_features.parquet"
CV_METRICS_PATH = Path("cv_metrics.json") # Save CV metrics in root directory

TARGET = "wind_perc"
N_SPLITS = 5
TEST_FRACTION = 0.15 # Approximate test size for TimeSeriesSplit test_size calculation
EARLY_STOPPING_ROUNDS = 50

# --- Load Data ---
logging.info(f"Loading features from {FEATURES_PATH}...")
if not FEATURES_PATH.exists():
    logging.error(f"Features file not found at {FEATURES_PATH}")
    raise FileNotFoundError(f"Features file not found at {FEATURES_PATH}")

df = pd.read_parquet(FEATURES_PATH)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True) # Ensure tz-aware datetime type
df = df.sort_values("datetime").reset_index(drop=True) # Ensure sorted for CV
logging.info(f"Loaded DataFrame shape: {df.shape}")

# Drop rows where the target variable is NaN
df.dropna(subset=[TARGET], inplace=True)
logging.info(f"DataFrame shape after dropping NaN targets: {df.shape}")

if df.empty:
    logging.error(f"No data left after dropping NaN from target column '{TARGET}'. Check training_features.parquet.")
    raise ValueError(f"DataFrame empty after dropping NaNs from target '{TARGET}'.")

# Define features (X) and target (y)
FEATURES = [col for col in df.columns if col not in ["datetime", TARGET]]
X, y = df[FEATURES], df[TARGET]

# --- Time Series Cross-Validation ---
logging.info(f"Starting TimeSeriesSplit CV with {N_SPLITS} splits...")

# Calculate test_size based on fraction and total samples
n_samples = len(df)
test_size = int(n_samples * TEST_FRACTION)
tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=test_size)

fold_metrics = defaultdict(list)
feature_importances = pd.DataFrame(index=FEATURES)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    logging.info(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    logging.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # Initialize and train CatBoost model with early stopping
    cat_kwargs = dict(
        random_state=42,
        verbose=0,
        eval_metric="RMSE",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        task_type=DEVICE,
        # Match the production model's categorical declaration
        # (train_model.py CAT_FEATURES) so cv_metrics.json measures the
        # same configuration that ships.
        cat_features=["is_holiday"],
    )
    if DEVICE == "GPU":
        cat_kwargs["devices"] = "0"
    model = CatBoostRegressor(**cat_kwargs)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100 # Show progress within fit
    )

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    smape = symmetric_mape(y_val.to_numpy(), y_pred)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5

    logging.info(f"Fold {fold+1} SMAPE: {smape:.4f}, RMSE (perc): {rmse:.4f}")
    fold_metrics["smape_perc"].append(smape)
    fold_metrics["rmse_perc"].append(rmse)

    # Store feature importance for this fold
    feature_importances[f'fold_{fold+1}'] = model.get_feature_importance()

# --- Aggregate Results ---
logging.info("--- CV Results ---")
mean_smape = float(np.mean(fold_metrics["smape_perc"]))
std_smape = float(np.std(fold_metrics["smape_perc"]))
mean_rmse = float(np.mean(fold_metrics["rmse_perc"]))
std_rmse = float(np.std(fold_metrics["rmse_perc"]))

logging.info(f"Mean SMAPE: {mean_smape:.4f} ± {std_smape:.4f}")
logging.info(f"Mean RMSE (perc): {mean_rmse:.4f} ± {std_rmse:.4f}")

cv_results = {
    "mean_smape": mean_smape,
    "std_smape": std_smape,
    "mean_rmse_perc": mean_rmse,
    "std_rmse_perc": std_rmse,
    "folds": {
        "smape": fold_metrics["smape_perc"],
        "rmse_perc": fold_metrics["rmse_perc"],
    },
}

# --- Save CV Metrics ---
logging.info(f"Saving CV metrics to {CV_METRICS_PATH}...")
with open(CV_METRICS_PATH, 'w') as f:
    json.dump(cv_results, f, indent=4)
logging.info("CV Metrics saved.")

# --- Average Feature Importance ---
logging.info("--- Average Feature Importance (Top 20) ---")
feature_importances['mean'] = feature_importances.mean(axis=1)
feature_importances = feature_importances.sort_values('mean', ascending=False)
logging.info("\n" + feature_importances['mean'].head(20).to_string())

logging.info("Validation script finished.") 
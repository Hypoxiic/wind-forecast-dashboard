# src/validate.py

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from catboost import CatBoostRegressor
from collections import defaultdict

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features" / "features.parquet"
CV_METRICS_PATH = Path("cv_metrics.json") # Save CV metrics in root directory

TARGET = "wind_mw"
N_SPLITS = 5
TEST_FRACTION = 0.15 # Approximate test size for TimeSeriesSplit test_size calculation
EARLY_STOPPING_ROUNDS = 50

# --- Load Data ---
logging.info(f"Loading features from {FEATURES_PATH}...")
if not FEATURES_PATH.exists():
    logging.error(f"Features file not found at {FEATURES_PATH}")
    raise FileNotFoundError(f"Features file not found at {FEATURES_PATH}")

df = pd.read_parquet(FEATURES_PATH)
df["datetime"] = pd.to_datetime(df["datetime"]) # Ensure datetime type
df = df.sort_values("datetime").reset_index(drop=True) # Ensure sorted for CV
logging.info(f"Loaded DataFrame shape: {df.shape}")

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
    model = CatBoostRegressor(
        random_state=42,
        verbose=0, # Keep verbose low for CV loop
        eval_metric='RMSE',
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
        # task_type="GPU" # Uncomment if GPU available
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100 # Show progress within fit
    )

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5

    logging.info(f"Fold {fold+1} MAPE: {mape:.4f}, RMSE: {rmse:.2f}")
    fold_metrics["mape"].append(mape)
    fold_metrics["rmse"].append(rmse)

    # Store feature importance for this fold
    feature_importances[f'fold_{fold+1}'] = model.get_feature_importance()

# --- Aggregate Results ---
logging.info("--- CV Results ---")
mean_mape = np.mean(fold_metrics["mape"])
std_mape = np.std(fold_metrics["mape"])
mean_rmse = np.mean(fold_metrics["rmse"])
std_rmse = np.std(fold_metrics["rmse"])

logging.info(f"Mean MAPE: {mean_mape:.4f} ± {std_mape:.4f}")
logging.info(f"Mean RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")

cv_results = {
    "mean_mape": mean_mape,
    "std_mape": std_mape,
    "mean_rmse": mean_rmse,
    "std_rmse": std_rmse,
    "folds": {
        "mape": fold_metrics["mape"],
        "rmse": fold_metrics["rmse"]
    }
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
# src/train_model.py

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from catboost import CatBoostRegressor

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "features" / "features.parquet"
METRICS_PATH = Path("metrics.json") # Save in root directory
MODEL_SAVE_PATH = Path("model.cbm") # Path to save the trained model

TARGET = "wind_mw"

# --- Load Data ---
logging.info(f"Loading features from {FEATURES_PATH}...")
if not FEATURES_PATH.exists():
    logging.error(f"Features file not found at {FEATURES_PATH}")
    raise FileNotFoundError(f"Features file not found at {FEATURES_PATH}")

df = pd.read_parquet(FEATURES_PATH)

# Ensure datetime type
df["datetime"] = pd.to_datetime(df["datetime"])
logging.info(f"Loaded DataFrame shape: {df.shape}")

# --- Diagnostic Print Statements ---
logging.info("--- Data Diagnostics ---")
logging.info(f"Earliest date: {df['datetime'].min()}")
logging.info(f"Latest date : {df['datetime'].max()}")
logging.info(f"Total rows  : {len(df)}")
logging.info("------------------------")

# --- Train/Test Split ---
logging.info("Splitting data into train/test sets...")
# # Train on 2024 Q1–Q3, test on Q4 to date. (Old method - failed due to actual start date)
# train_start = "2024-01-01"
# train_end = "2024-09-30 23:59:59" # Inclusive end of Q3
# test_start = "2024-10-01"

# --- New Split Method: Use percentage based on available data --- 
split_fraction = 0.75
split_index = int(len(df) * split_fraction)
split_date = df['datetime'].iloc[split_index]

df_train = df[df["datetime"] < split_date].copy()
df_test = df[df["datetime"] >= split_date].copy()
# --- End New Split Method ---

if df_train.empty or df_test.empty:
     logging.error("Train or test split resulted in empty DataFrame. Check date ranges and data.")
     raise ValueError("Empty train or test set after split.")

logging.info(f"Train shape: {df_train.shape}")
logging.info(f"Test shape: {df_test.shape}")

# Define features (X) and target (y)
FEATURES = [col for col in df.columns if col not in ["datetime", TARGET]]

# Check if lag feature exists for baseline
BASELINE_LAG_FEATURE = 'wind_mw_lag_48h'
if BASELINE_LAG_FEATURE not in FEATURES:
    logging.error(f"Baseline feature '{BASELINE_LAG_FEATURE}' not found in columns.")
    raise ValueError(f"Missing required baseline feature: {BASELINE_LAG_FEATURE}")

X_train, y_train = df_train[FEATURES], df_train[TARGET]
X_test, y_test = df_test[FEATURES], df_test[TARGET]

# --- Baseline Model: Persistence (ŷ(t) = y(t - 48 hours)) ---
logging.info("Calculating baseline (persistence) metrics...")
y_pred_baseline = X_test[BASELINE_LAG_FEATURE]

mape_baseline = mean_absolute_percentage_error(y_test, y_pred_baseline)
# rmse_baseline = mean_squared_error(y_test, y_pred_baseline, squared=False) # Old way
rmse_baseline = mean_squared_error(y_test, y_pred_baseline) ** 0.5 # New way

logging.info(f"Baseline MAPE: {mape_baseline:.4f}")
logging.info(f"Baseline RMSE: {rmse_baseline:.2f}")

metrics = {
    "baseline": {
        "mape": mape_baseline,
        "rmse": rmse_baseline
    }
}

# --- Model Training: CatBoostRegressor ---
logging.info("Training CatBoostRegressor model...")
# Use default parameters for now, consider adding task_type='GPU' if GPU available
model = CatBoostRegressor(
    random_state=42,
    verbose=100 # Print progress every 100 iterations
    # task_type="GPU" # Uncomment this if a compatible GPU and drivers are available
)

model.fit(X_train, y_train)

logging.info("Evaluating CatBoost model...")
y_pred_catboost = model.predict(X_test)

mape_catboost = mean_absolute_percentage_error(y_test, y_pred_catboost)
# rmse_catboost = mean_squared_error(y_test, y_pred_catboost, squared=False) # Old way
rmse_catboost = mean_squared_error(y_test, y_pred_catboost) ** 0.5 # New way

logging.info(f"CatBoost MAPE: {mape_catboost:.4f}")
logging.info(f"CatBoost RMSE: {rmse_catboost:.2f}")

metrics["catboost"] = {
    "mape": mape_catboost,
    "rmse": rmse_catboost
}

# --- Save Metrics ---
logging.info(f"Saving metrics to {METRICS_PATH}...")
with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=4)
logging.info("Metrics saved.")

# --- Save CatBoost Predictions on Test Set ---
logging.info("Saving CatBoost predictions on the test set...")
PRED_DIR = DATA_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True) # Create dir if it doesn't exist
PRED_PATH = PRED_DIR / "catboost_test.parquet"

# Create DataFrame with datetime and predictions
pred_df = pd.DataFrame({
    'datetime': df_test['datetime'], # Get datetimes from the test set DataFrame
    'wind_mw_pred': y_pred_catboost
})

pred_df.to_parquet(PRED_PATH, index=False)
logging.info(f"Test set predictions saved to {PRED_PATH}")

# --- Save Model ---
# logging.info(f"Saving trained model to {MODEL_SAVE_PATH}...")
# model.save_model(str(MODEL_SAVE_PATH)) # Uncomment to save the model
# logging.info("Model saved.")

logging.info("Script finished.") 
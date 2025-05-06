"""
src/featurise.py
----------------
Merge raw ci wind generation & Open‑Meteo wind‑speed data,
then create modelling features (lags, rolling stats, calendar cycles, holidays).

Output:  data/features/features.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from holidays import UnitedKingdom

# ──────────────────────────
# Config & logging
# ──────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

# Path definitions (will be adjusted by mode)
DATA_ROOT = Path("data")
RAW_DIR_BASE = DATA_ROOT / "raw" # Default base for raw data
FEAT_DIR_BASE = DATA_ROOT / "features"
FEAT_DIR_BASE.mkdir(parents=True, exist_ok=True)

# Global variables for paths, to be set by main()
CI_PARQUET = None
MET_PARQUET = None
OUT_PARQUET = None

# ──────────────────────────
# Load helpers
# ──────────────────────────
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw parquet files for ci & Open‑Meteo."""
    ci = pd.read_parquet(CI_PARQUET)
    met = pd.read_parquet(MET_PARQUET)
    logging.info("Loaded raw ci (%s rows) & Open‑Meteo (%s rows)",
                 len(ci), len(met))
    logging.info("ci dtypes:\\n%s", ci.dtypes)
    logging.info("met dtypes:\\n%s", met.dtypes)
    logging.info("ci head:\\n%s", ci.head().to_string())
    logging.info("met head:\\n%s", met.head().to_string())
    logging.info("ci tail:\\n%s", ci.tail().to_string())
    logging.info("met tail:\\n%s", met.tail().to_string())
    return ci, met


# ──────────────────────────
# Feature engineering
# ──────────────────────────
def engineer_features(ci: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:
    """
    • align 30‑min timestamps (nearest ±30 min)
    • add lags (30 m / 1 h / 3 h / 24 h / 48 h)
    • rolling means / stds on wind‑speed
    • calendar + cyclical encodings
    • UK holiday flag
    """
    # --- ensure timezone‑aware UTC ---
    for df_loop_var in (ci, met): # Changed df to df_loop_var to avoid clash
        if df_loop_var["datetime"].dt.tz is None:
            df_loop_var["datetime"] = pd.to_datetime(df_loop_var["datetime"], utc=True)
        else:
            df_loop_var["datetime"] = df_loop_var["datetime"].dt.tz_convert("UTC")

    logging.info("Pre-merge ci shape: %s, met shape: %s", ci.shape, met.shape)
    # --- merge nearest (tolerance ±30 min) ---
    df = pd.merge_asof(
        met.sort_values("datetime"),
        ci.sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    ).sort_values("datetime").reset_index(drop=True)
    logging.info("Post-merge df shape: %s", df.shape)
    logging.info("Post-merge df head:\\n%s", df.head().to_string())
    logging.info("Post-merge df tail:\\n%s", df.tail().to_string())
    logging.info("Post-merge NaNs:\\n%s", df.isnull().sum().to_string())

    # ─── Power curve proxies ──────────
    # Moved from train_model.py for consistency
    RATED_MS = 15.0
    if "wind_speed_10m" in df.columns:
        df["wind_speed_v3"]      = df["wind_speed_10m"] ** 3
        df["wind_speed_v3_clip"] = np.clip(df["wind_speed_10m"], 0, RATED_MS) ** 3
    else:
        # Add columns as NaN if base speed column is missing after merge (shouldn't happen ideally)
        df["wind_speed_v3"] = np.nan
        df["wind_speed_v3_clip"] = np.nan

    # ─── lags ─────────────────────────
    # df is hourly after merge_asof(met, ci, ...)
    # Original wind_perc was 30-min, but is now aligned to hourly by the merge.
    # So, shift(N) on both wind_perc and wind_speed_10m will be an N-hour shift.
    hourly_lag_config = {   # label : hours_to_shift
        "30m": 1,       # No 0.5h shift on hourly data, use 1h as proxy or consider if needed
        "1h":  1,
        "3h":  3,
        "24h": 24,
        "48h": 48,
    }

    for label, lag_hours in hourly_lag_config.items():
        df[f"wind_perc_lag_{label}"]    = df["wind_perc"].shift(lag_hours)
        df[f"wind_speed_lag_{label}"] = df["wind_speed_10m"].shift(lag_hours)

    # ─── rolling stats on wind‑speed ─
    roll_steps = {6: "3h", 48: "24h", 96: "48h"}  # windows in 30‑min steps
    for steps, label in roll_steps.items():
        # Use min_periods=1 to ensure calculation even if full window not available (common at start of series)
        df[f"wind_speed_roll_mean_{label}"] = df["wind_speed_10m"].rolling(window=steps, min_periods=1).mean()
        df[f"wind_speed_roll_std_{label}"]  = df["wind_speed_10m"].rolling(window=steps, min_periods=1).std()

    # ─── calendar & cycles ────────────
    df["hour"]      = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["dayofyear"] = df["datetime"].dt.dayofyear

    # cyclic encodings
    df["sin_hour"]       = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]       = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_dayofyear"]  = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_dayofyear"]  = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # UK public holidays
    uk_holidays = UnitedKingdom()
    df["is_holiday"] = df["datetime"].dt.date.isin(uk_holidays).astype(int)
    logging.info("df shape after all features added (before dropna): %s", df.shape)
    logging.info("df NaNs after all features added (before dropna):\\n%s", df.isnull().sum().to_string())

    # For prediction, 'wind_perc' will be NaN for future dates.
    # We only drop rows if essential *predictor* features are NaN after lags/rolls.
    # 'wind_perc' and its direct lags are targets or used for baseline, not primary predictors for future.
    
    # Identify columns that are purely derived from 'wind_perc' (excluding 'wind_perc' itself)
    # These are columns like 'wind_perc_lag_...'
    wind_perc_derived_lags = [col for col in df.columns if "wind_perc_lag" in col]

    # Columns that, if NaN, would justify dropping a row for feature generation.
    # This typically includes weather data and its derivatives, and calendar features.
    # Exclude 'wind_perc' and its direct lags as they will be NaN for future predictions.
    predictor_columns = [
        col for col in df.columns 
        if col not in ["wind_perc"] + wind_perc_derived_lags
    ]
    logging.info("Predictor columns for dropna: %s", predictor_columns)

    # Log NaN counts for predictor columns specifically
    logging.info("NaN counts for predictor_columns (before dropna):\\n%s", df[predictor_columns].isnull().sum().to_string())
    
    initial_rows = len(df)
    # Drop rows where essential predictor features are missing.
    # This will keep rows where 'wind_perc' is NaN (future) but weather data is present.
    df = df.dropna(subset=predictor_columns).reset_index(drop=True)
    final_rows = len(df)

    logging.info(
        f"Feature dataframe shape after engineering: {df.shape} (dropped {initial_rows - final_rows} rows based on predictor availability)"
    )

    return df


# ──────────────────────────
# Main
# ──────────────────────────
def main() -> None:
    global CI_PARQUET, MET_PARQUET, OUT_PARQUET # Declare global to modify

    parser = argparse.ArgumentParser(description="Featurisation script for wind generation data.")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="inference", 
        choices=["training", "inference"],
        help="Mode of operation: 'training' for full historical data, 'inference' for daily prediction data."
    )
    args = parser.parse_args()

    logging.info(f"Running featurise.py in {args.mode} mode.")

    if args.mode == "training":
        CURRENT_RAW_DIR = RAW_DIR_BASE / "training"
        CURRENT_FEAT_DIR = FEAT_DIR_BASE
        CI_PARQUET = CURRENT_RAW_DIR / "ci_wind_perc_training.parquet"
        MET_PARQUET = CURRENT_RAW_DIR / "openmeteo_weather_training.parquet"
        OUT_PARQUET = CURRENT_FEAT_DIR / "training_features.parquet"
        logging.info(f"Training mode: Using full historical raw data.")
        logging.info(f"  Input CI: {CI_PARQUET}")
        logging.info(f"  Input Met: {MET_PARQUET}")
        logging.info(f"  Output Features: {OUT_PARQUET}")
    else: # Inference mode (default)
        CURRENT_RAW_DIR = RAW_DIR_BASE # Uses the top-level raw directory
        CURRENT_FEAT_DIR = FEAT_DIR_BASE
        CI_PARQUET = CURRENT_RAW_DIR / "ci.parquet"
        MET_PARQUET = CURRENT_RAW_DIR / "openmeteo_weather.parquet"
        OUT_PARQUET = CURRENT_FEAT_DIR / "features.parquet"
        logging.info(f"Inference mode: Using daily raw data for prediction.")
        logging.info(f"  Input CI: {CI_PARQUET}")
        logging.info(f"  Input Met: {MET_PARQUET}")
        logging.info(f"  Output Features: {OUT_PARQUET}")

    # Check existence based on CURRENT_RAW_DIR
    if not CI_PARQUET.parent.exists() or not MET_PARQUET.parent.exists():
        # A bit more specific error for clarity
        if args.mode == "training" and not (RAW_DIR_BASE / "training").exists():
            logging.error(f"Input directory for training data does not exist: {RAW_DIR_BASE / 'training'}")
        elif args.mode == "inference" and not RAW_DIR_BASE.exists(): # Assuming ci.parquet and openmeteo_weather.parquet are directly in raw for inference
            logging.error(f"Input directory for inference data does not exist: {RAW_DIR_BASE}")
        else:
            logging.error(f"One or both input data parent directories do not exist: {CI_PARQUET.parent}, {MET_PARQUET.parent}")
        return

    ci, met = load_raw() # load_raw will use the global CI_PARQUET and MET_PARQUET
    if ci.empty and met.empty:
        logging.warning("Both CI and MET dataframes are empty. Skipping feature engineering.")
        # Create an empty parquet file if it does not exist to prevent downstream errors
        if not OUT_PARQUET.exists():
            pd.DataFrame().to_parquet(OUT_PARQUET, index=False)
            logging.info(f"Created empty output file as inputs were empty: {OUT_PARQUET}")
        return
    
    features = engineer_features(ci, met)
    
    if features.empty:
        logging.warning("Feature engineering resulted in an empty DataFrame. Saving empty parquet.")
    features.to_parquet(OUT_PARQUET, index=False)
    logging.info("Saved engineered features → %s (%s rows)", OUT_PARQUET, len(features))


if __name__ == "__main__":
    main()

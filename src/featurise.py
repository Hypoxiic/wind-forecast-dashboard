"""
src/featurise.py
----------------
Merge raw ESO wind generation & Open‑Meteo wind‑speed data,
then create modelling features (lags, rolling stats, calendar cycles, holidays).

Output:  data/features/features.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from holidays import UnitedKingdom

# ──────────────────────────
# Config & logging
# ──────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

RAW_DIR   = Path("data/raw")
FEAT_DIR  = Path("data/features")
FEAT_DIR.mkdir(parents=True, exist_ok=True)

ESO_PARQUET  = RAW_DIR / "eso_wind.parquet"
MET_PARQUET  = RAW_DIR / "openmeteo_weather.parquet"
OUT_PARQUET  = FEAT_DIR / "features.parquet"


# ──────────────────────────
# Load helpers
# ──────────────────────────
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw parquet files for ESO & Open‑Meteo."""
    eso = pd.read_parquet(ESO_PARQUET)
    met = pd.read_parquet(MET_PARQUET)
    logging.info("Loaded raw ESO (%s rows) & Open‑Meteo (%s rows)",
                 len(eso), len(met))
    return eso, met


# ──────────────────────────
# Feature engineering
# ──────────────────────────
def engineer_features(eso: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:
    """
    • align 30‑min timestamps (nearest ±30 min)
    • add lags (30 m / 1 h / 3 h / 24 h / 48 h)
    • rolling means / stds on wind‑speed
    • calendar + cyclical encodings
    • UK holiday flag
    """
    # --- ensure timezone‑aware UTC ---
    for df in (eso, met):
        if df["datetime"].dt.tz is None:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        else:
            df["datetime"] = df["datetime"].dt.tz_convert("UTC")

    # --- merge nearest (tolerance ±30 min) ---
    df = pd.merge_asof(
        eso.sort_values("datetime"),
        met.sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    ).sort_values("datetime").reset_index(drop=True)

    # ─── lags ─────────────────────────
    lag_steps = {   # step : label
        1  : "30m",
        2  : "1h",
        6  : "3h",
        48 : "24h",
        96 : "48h",
    }
    for steps, label in lag_steps.items():
        df[f"wind_perc_lag_{label}"]      = df["wind_perc"].shift(steps)
        df[f"wind_speed_lag_{label}"]   = df["wind_speed_10m"].shift(steps)

    # ─── rolling stats on wind‑speed ─
    roll_steps = {6: "3h", 48: "24h", 96: "48h"}  # windows in 30‑min steps
    for steps, label in roll_steps.items():
        df[f"wind_speed_roll_mean_{label}"] = df["wind_speed_10m"].rolling(steps).mean()
        df[f"wind_speed_roll_std_{label}"]  = df["wind_speed_10m"].rolling(steps).std()

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

    # drop rows containing NaNs introduced by lags/rolls
    df = df.dropna().reset_index(drop=True)
    logging.info("Feature dataframe shape after engineering: %s", df.shape)

    return df


# ──────────────────────────
# Main
# ──────────────────────────
def main() -> None:
    eso, met = load_raw()
    features = engineer_features(eso, met)
    features.to_parquet(OUT_PARQUET, index=False)
    logging.info("Saved engineered features → %s", OUT_PARQUET)


if __name__ == "__main__":
    main()

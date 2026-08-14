"""
src/featurise.py
----------------
Merge raw ci wind generation & Open‑Meteo wind‑speed data,
then create modelling features (lags, rolling stats, calendar cycles, holidays).

Output:  data/features/training_features.parquet (training mode)
         data/features/for_predict.parquet       (inference mode)
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

# The published forecast runs out to ~48h. A wind_perc observation is only
# knowable at issue time if it predates the target by at least that much, so
# every wind_perc-derived feature is shifted by this embargo. Features built
# from shorter lags are future actuals: available while training, NaN when
# serving, and the model silently learns to depend on them.
FORECAST_HORIZON_H = 48

# Rolling history of the target, used to compute the embargoed features. The
# nightly ETL only fetches ~3 days, which is nowhere near enough for the
# 365-day trend, so inference mode reads the backbone from here instead.
HISTORY_PARQUET = FEAT_DIR_BASE / "history.parquet"
SNAPSHOT_PARQUET = FEAT_DIR_BASE / "features.parquet"

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


def target_history_features(backbone: pd.DataFrame,
                            embargo_h: int = FORECAST_HORIZON_H) -> pd.DataFrame:
    """Causal features derived from the target's own history.

    `backbone` is any frame with `datetime` + `wind_perc`; it is resampled onto
    a strict hourly grid first, so windows are *time* spans regardless of
    whether the source was 30-minute (old snapshots) or hourly (current ETL).
    Row-based windows silently mean different things across those two eras.

    Every column is shifted by `embargo_h` so it only ever reflects data that
    predates the forecast issue time. This is what keeps the model honest: the
    same values are present during training and at serve time.
    """
    bb = backbone[["datetime", "wind_perc"]].copy()
    bb["datetime"] = pd.to_datetime(bb["datetime"], utc=True)
    s = (bb.dropna(subset=["datetime"])
           .set_index("datetime")["wind_perc"]
           .resample("1h").mean()
           .sort_index())

    out = pd.DataFrame(index=s.index)
    # Plain lags at or beyond the embargo. 48h is the persistence baseline the
    # dashboard also plots; the rest give the model a few days of context.
    for h in (48, 72, 96, 168):
        out[f"wind_perc_lag_{h}h"] = s.shift(h)
    # Trailing level and volatility. The 365-day mean is the installed-capacity
    # proxy: GB wind capacity grew a lot over 2018-2026 and the target is a
    # *share* of the mix, so without this the model can only absorb capacity
    # growth as a vague time trend.
    for hours, lab in ((24, "1d"), (24 * 7, "7d"), (24 * 30, "30d"), (24 * 365, "365d")):
        out[f"wp_roll_mean_{lab}"] = (
            s.rolling(hours, min_periods=max(2, hours // 4)).mean().shift(embargo_h))
    for hours, lab in ((24 * 7, "7d"), (24 * 30, "30d")):
        out[f"wp_roll_std_{lab}"] = (
            s.rolling(hours, min_periods=max(2, hours // 4)).std().shift(embargo_h))

    return out.reset_index()


def load_backbone(fresh: pd.DataFrame) -> pd.DataFrame:
    """Long wind_perc history for inference, newest values winning.

    The nightly ETL fetches ~3 days, which cannot support a 365-day trailing
    mean. history.parquet carries the full series from 2019, so union the two
    and let the fresh rows override on overlap.
    """
    frames = []
    for path in (HISTORY_PARQUET, SNAPSHOT_PARQUET):
        if path.exists():
            try:
                past = pd.read_parquet(path, columns=["datetime", "wind_perc"])
                frames.append(past)
                logging.info("Backbone: %s rows from %s", len(past), path.name)
            except (OSError, ValueError, KeyError) as e:
                logging.warning("Could not read backbone from %s: %s", path, e)
    frames.append(fresh[["datetime", "wind_perc"]])
    bb = (pd.concat(frames, ignore_index=True)
            .dropna(subset=["datetime"])
            .drop_duplicates(subset="datetime", keep="last")
            .sort_values("datetime")
            .reset_index(drop=True))
    span = (pd.to_datetime(bb["datetime"], utc=True).max()
            - pd.to_datetime(bb["datetime"], utc=True).min()).days
    logging.info("Backbone spans %s days (%s rows)", span, len(bb))
    if span < 400:
        logging.warning(
            "Backbone spans only %s days; the 365-day capacity feature will be "
            "NaN or unreliable. Check that history.parquet is present.", span)
    return bb


# ──────────────────────────
# Feature engineering
# ──────────────────────────
def engineer_features(ci: pd.DataFrame, met: pd.DataFrame,
                      backbone: pd.DataFrame | None = None) -> pd.DataFrame:
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
    # v³ captures the turbine power curve below rated speed; the clipped
    # variant captures saturation above rated. Computed at 10m (legacy) and
    # at 100m, much closer to real hub heights (archive only exposes 100m).
    RATED_MS = 15.0
    for height in ("10m", "100m"):
        col = f"wind_speed_{height}"
        if col in df.columns:
            df[f"wind_speed_v3_{height}"]      = df[col] ** 3
            df[f"wind_speed_v3_clip_{height}"] = np.clip(df[col], 0, RATED_MS) ** 3
        else:
            df[f"wind_speed_v3_{height}"] = np.nan
            df[f"wind_speed_v3_clip_{height}"] = np.nan
    # Legacy aliases kept so older snapshots/models still read familiar names.
    df["wind_speed_v3"]      = df["wind_speed_v3_10m"]
    df["wind_speed_v3_clip"] = df["wind_speed_v3_clip_10m"]

    # ─── Gust / direction / pressure ──
    if "wind_gusts_10m" in df.columns:
        # Gust factor: turbulence indicator, helps in rampy conditions.
        df["gust_factor"] = df["wind_gusts_10m"] / df["wind_speed_10m"].clip(lower=0.5)
    if "wind_direction_10m" in df.columns:
        df["sin_wind_dir"] = np.sin(np.deg2rad(df["wind_direction_10m"]))
        df["cos_wind_dir"] = np.cos(np.deg2rad(df["wind_direction_10m"]))
    if "surface_pressure" in df.columns:
        # Pressure level + 3h tendency: synoptic-scale frontal passage signal.
        df["pressure_delta_3h"] = df["surface_pressure"].diff(3)

    # ─── air density ──────────────────
    # Turbine power is proportional to rho * v^3, not v^3 alone. rho = P/(R·T)
    # from the ideal gas law; cold dense air carries materially more energy at
    # the same wind speed.
    if {"surface_pressure", "temperature_2m"} <= set(df.columns):
        df["air_density"] = (df["surface_pressure"] * 100.0) / (
            287.05 * (df["temperature_2m"] + 273.15))
        if "wind_speed_100m" in df.columns:
            df["rho_v3_100m"] = df["air_density"] * df["wind_speed_100m"] ** 3

    # ─── weather lags ─────────────────
    # df is hourly after merge_asof(met, ci, ...). These lag the *weather*,
    # which is forecast out over the whole window, so they are known at issue
    # time and are safe to use at any horizon.
    for label, lag_hours in {"1h": 1, "3h": 3, "24h": 24, "48h": 48}.items():
        df[f"wind_speed_lag_{label}"] = df["wind_speed_10m"].shift(lag_hours)
        if "wind_speed_100m" in df.columns:
            df[f"wind_speed_100m_lag_{label}"] = df["wind_speed_100m"].shift(lag_hours)

    # ─── target history (embargoed) ───
    # Deliberately NOT df["wind_perc"].shift(1/3/24): those are future actuals
    # at forecast time. They were 88% of the old model's importance and arrived
    # as NaN in production. See target_history_features().
    if backbone is None:
        backbone = df
    hist_feats = target_history_features(backbone)
    df = df.merge(hist_feats, on="datetime", how="left")

    # ─── rolling stats on wind‑speed ─
    # Window sizes are in HOURLY rows (the data is hourly after merge_asof),
    # so the actual spans are 6h/48h/96h — double what the "3h/24h/48h"
    # labels claim. The mislabeled column names are kept because the shipped
    # model.cbm was trained on them; correcting spans/names needs a retrain.
    roll_steps = {6: "3h", 48: "24h", 96: "48h"}
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

    # UK public holidays. The holidays dict populates itself lazily per year,
    # so it must be seeded with every year present in the data — Series.isin()
    # against a fresh UnitedKingdom() sees an empty dict and returns all-False.
    years = sorted(df["datetime"].dt.year.unique())
    uk_holidays = UnitedKingdom(years=years)
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
    # The embargoed target-history columns (wind_perc_lag_*h, wp_roll_*) are
    # NaN while their window warms up — 365 days for the capacity trend. They
    # must be excluded from the dropna subset or the early years are wiped and
    # inference (which has no local history) drops every row.
    warmup_columns = [c for c in df.columns
                      if c.startswith("wp_roll_") or c.startswith("wind_perc_lag_")]
    predictor_columns = [
        col for col in df.columns
        if col not in ["wind_perc"] + wind_perc_derived_lags + warmup_columns
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
def main(mode: str = "inference") -> None:
    global CI_PARQUET, MET_PARQUET, OUT_PARQUET # Declare global to modify

    logging.info(f"Running featurise.py in {mode} mode.")

    if mode == "training":
        CURRENT_RAW_DIR = RAW_DIR_BASE / "training"
        CI_PARQUET = CURRENT_RAW_DIR / "ci_wind_perc_training.parquet"
        MET_PARQUET = CURRENT_RAW_DIR / "openmeteo_weather_training.parquet"
        OUT_PARQUET = FEAT_DIR_BASE / "training_features.parquet"
        logging.info("Training mode: Using full historical raw data.")
    elif mode == "inference":
        CURRENT_RAW_DIR = RAW_DIR_BASE # Uses the top-level raw directory
        CI_PARQUET = CURRENT_RAW_DIR / "ci.parquet"
        MET_PARQUET = CURRENT_RAW_DIR / "openmeteo_weather.parquet"
        # Written straight to the prediction input. data/features/features.parquet
        # (the committed full-history snapshot) is never overwritten by the
        # nightly run, so no move/restore dance is needed in pipeline.py.
        OUT_PARQUET = FEAT_DIR_BASE / "for_predict.parquet"
        logging.info("Inference mode: Using daily raw data for prediction.")
    else:
        raise ValueError(f"Unknown featurise mode: {mode!r}")

    logging.info(f"  Input CI: {CI_PARQUET}")
    logging.info(f"  Input Met: {MET_PARQUET}")
    logging.info(f"  Output Features: {OUT_PARQUET}")

    if not CI_PARQUET.exists() or not MET_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing raw input(s): {CI_PARQUET}, {MET_PARQUET} — run the ETL step first."
        )

    ci, met = load_raw() # load_raw will use the global CI_PARQUET and MET_PARQUET
    if ci.empty or met.empty:
        raise ValueError(
            "Raw CI and/or weather data is empty; refusing to build features "
            "so stale downstream outputs are kept instead of overwritten."
        )

    # Training has the whole series locally; inference must borrow it from
    # history.parquet, since the ETL only fetched the last ~3 days.
    backbone = None if mode == "training" else load_backbone(ci)
    features = engineer_features(ci, met, backbone=backbone)
    if features.empty:
        raise ValueError(
            "Feature engineering produced no rows; aborting instead of writing an empty file."
        )
    features.to_parquet(OUT_PARQUET, index=False)
    logging.info("Saved engineered features → %s (%s rows)", OUT_PARQUET, len(features))


if __name__ == "__main__":
    # argparse lives here, NOT in main(): main() is also called in-process by
    # pipeline.py, and parsing sys.argv there reads the parent process's
    # arguments (SystemExit on anything unrecognised, and a pipeline run
    # started with "--mode training" would silently redirect featurise).
    parser = argparse.ArgumentParser(description="Featurisation script for wind generation data.")
    parser.add_argument(
        "--mode",
        type=str,
        default="inference",
        choices=["training", "inference"],
        help="Mode of operation: 'training' for full historical data, 'inference' for daily prediction data."
    )
    main(parser.parse_args().mode)

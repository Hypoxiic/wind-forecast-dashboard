# src/etl_inference.py
"""
Lean ETL for nightly inference using Carbon Intensity API:
* Yesterday's wind generation percentage (Carbon Intensity API)
* Yesterday's weather             (Open‑Meteo archive, 1 day)
* Next‑48 h weather forecast      (Open‑Meteo forecast endpoint)

NOTE: Carbon Intensity API provides wind generation as a percentage ('perc')
      of the total mix, not absolute MW. Downstream scripts (featurise, train)
      expecting 'wind_mw' will need modification.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------- User‑configurable ---------------------------------------- #
LAT, LON = 54.0, -1.5  # Example coordinates (Central UK)
HOURLY_VARS = "temperature_2m,wind_speed_10m" # Weather variables
DATA_DIR = Path("data") # Ensure this path is correct relative to execution
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Output file names expected by downstream scripts (even if content differs)
WIND_OUT_PATH = RAW_DIR / "ci.parquet"
METEO_OUT_PATH = RAW_DIR / "openmeteo_weather.parquet"

# --------------------------------------------------------------------------- #

# ---------- Setup Logging -------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- Carbon‑Intensity API Configuration ----------------------------- #
CI_API_BASE_URL = "https://api.carbonintensity.org.uk"
CI_HEADERS = {'Accept': 'application/json'}

# ---------- Open‑Meteo API Configuration ----------------------------------- #
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OM_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# ---------- Retry Mechanism ------------------------------------------------ #
@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
def _make_request(url: str, params: dict | None = None, headers: dict | None = None) -> dict:
    """Makes a GET request with retries and basic error checking."""
    try:
        response = requests.get(url, params=params, headers=headers, timeout=60)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {url}: {e}")
        raise # Re-raise exception to trigger tenacity retry

# ---------- Carbon‑Intensity Wind Percentage (Yesterday) ------------------- #
def fetch_ci_wind_perc_history(today: date, days_history: int = 3) -> pd.DataFrame:
    """
    Fetches wind generation percentage for the specified number of past days up to and including today.
    """
    # start_date remains X days ago to provide context for lags if needed by other processes
    # but the primary goal is to get data up to 'today'
    start_date_ci = today - timedelta(days=days_history) 
    end_date_ci = today # Fetch up to and including today

    from_dt_str = start_date_ci.strftime('%Y-%m-%dT00:00Z')
    # For 'to' date, CI API typically includes the whole day. So T23:30Z for 'today' is fine.
    to_dt_str = end_date_ci.strftime('%Y-%m-%dT23:30Z') 

    url = f"{CI_API_BASE_URL}/generation/{from_dt_str}/{to_dt_str}"
    logging.info(f"Fetching Carbon Intensity generation data from: {url}")

    try:
        js = _make_request(url, headers=CI_HEADERS)
    except Exception as e:
        logging.error(f"Failed to fetch or parse Carbon Intensity data: {e}")
        return pd.DataFrame(columns=["datetime", "wind_perc"]) # Return empty df on failure

    if "data" not in js or not isinstance(js["data"], list):
        logging.warning(f"Unexpected response structure from {url}: {js}")
        return pd.DataFrame(columns=["datetime", "wind_perc"])

    rows = []
    for interval_data in js["data"]:
        ts = interval_data.get("from")
        generation_mix = interval_data.get("generationmix")

        if not ts or not generation_mix:
            logging.warning(f"Skipping interval due to missing 'from' or 'generationmix': {interval_data}")
            continue

        wind_entry = next((item for item in generation_mix if item.get("fuel") == "wind"), None)

        if wind_entry and "perc" in wind_entry:
            rows.append({"datetime": ts, "wind_perc": wind_entry["perc"]})
        else:
            # Log if wind data is missing for an interval, maybe append NaN or 0?
            logging.warning(f"Wind percentage not found for interval starting {ts}. Appending NaN.")
            rows.append({"datetime": ts, "wind_perc": float('nan')}) # Or use 0 if preferred

    if not rows:
        logging.warning(f"No valid wind percentage data found for the period {start_date_ci} to {end_date_ci}.")
        return pd.DataFrame(columns=["datetime", "wind_perc"])

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# ---------- Open‑Meteo Weather Fetchers (Unchanged from original) ---------- #
def fetch_weather_history(today: date, days_history: int = 3) -> pd.DataFrame:
    """Fetches past hourly weather data from Open-Meteo."""
    start_date = today - timedelta(days=days_history)
    end_date = today - timedelta(days=1)
    url = (
        f"{OM_ARCHIVE_URL}?latitude={LAT}&longitude={LON}"
        f"&hourly={HOURLY_VARS}&start_date={start_date}&end_date={end_date}&timezone=UTC"
    )
    logging.info(f"Fetching past weather from Open-Meteo: {url}")
    try:
        js = _make_request(url)
        if "hourly" not in js or "time" not in js["hourly"]:
             logging.error(f"Unexpected Open-Meteo response structure: {js}")
             return pd.DataFrame() # Return empty df
        df = pd.DataFrame(js["hourly"])
        df = df.rename(columns={"time": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        # Fill NaNs for weather variables
        weather_cols = HOURLY_VARS.split(',')
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        return df
    except Exception as e:
        logging.error(f"Failed to fetch past weather: {e}")
        return pd.DataFrame()


def fetch_weather_forecast(today: date) -> pd.DataFrame:
    """Fetches the next 48 hours of hourly weather forecast from Open-Meteo."""
    start_date = today
    end_date = today + timedelta(days=1) # API includes start and end day
    url = (
        f"{OM_FORECAST_URL}?latitude={LAT}&longitude={LON}"
        f"&hourly={HOURLY_VARS}&start_date={start_date}&end_date={end_date}"
        "&timezone=UTC"
    )
    logging.info(f"Fetching 48h weather forecast from Open-Meteo: {url}")
    try:
        js = _make_request(url)
        if "hourly" not in js or "time" not in js["hourly"]:
             logging.error(f"Unexpected Open-Meteo response structure: {js}")
             return pd.DataFrame() # Return empty df
        df = pd.DataFrame(js["hourly"])
        df = df.rename(columns={"time": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        # Fill NaNs for weather variables
        weather_cols = HOURLY_VARS.split(',')
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        # Forecast API might give more than 48h, ensure we only take needed range
        forecast_end_time = pd.Timestamp(f"{end_date} 23:59:59", tz='UTC')
        df = df[df['datetime'] <= forecast_end_time].reset_index(drop=True)
        return df
    except Exception as e:
        logging.error(f"Failed to fetch weather forecast: {e}")
        return pd.DataFrame()

# ---------- Main Execution Logic ------------------------------------------- #
def main():
    """Main ETL function for inference."""
    start_time = time.time()
    today = date.today()
    logging.info(f"Starting inference ETL run for {today}")
    days_needed_history = 3 # Fetch yesterday + 2 prior days for 48h lag

    # --- Fetch Data ---
    logging.info(f"Fetching Carbon Intensity wind percentage ({days_needed_history} days history)...")
    wind_perc_df = fetch_ci_wind_perc_history(today, days_needed_history)
    if wind_perc_df.empty:
        logging.error("Failed to get wind percentage data. Aborting.")
        return
    logging.info(f"Fetched {len(wind_perc_df)} rows of wind percentage data.")

    logging.info(f"Fetching Open‑Meteo weather ({days_needed_history} days history)...")
    meteo_history_df = fetch_weather_history(today, days_needed_history)
    if meteo_history_df.empty:
        logging.error("Failed to get historical weather data. Aborting.")
        return
    logging.info(f"Fetched {len(meteo_history_df)} rows of historical weather.")
    logging.info(f"NaNs in meteo_history_df:\n{meteo_history_df.isnull().sum().to_string()}")

    logging.info("Fetching Open‑Meteo weather (forecast 48h)...")
    meteo_forecast_df = fetch_weather_forecast(today)
    if meteo_forecast_df.empty:
        logging.error("Failed to get forecast weather data. Aborting.")
        return
    logging.info(f"Fetched {len(meteo_forecast_df)} rows of forecast weather.")
    logging.info(f"NaNs in meteo_forecast_df:\n{meteo_forecast_df.isnull().sum().to_string()}")

    # --- Combine Weather Data ---
    meteo_df = pd.concat([meteo_history_df, meteo_forecast_df], ignore_index=True)
    meteo_df = meteo_df.drop_duplicates(subset=["datetime"], keep="last") # Keep forecast if overlap
    meteo_df = meteo_df.sort_values("datetime").reset_index(drop=True)
    logging.info(f"Combined weather data shape: {meteo_df.shape}")

    # --- Save Data ---
    logging.info(f"Saving wind percentage data to {WIND_OUT_PATH}")
    wind_perc_df.to_parquet(WIND_OUT_PATH, index=False)

    logging.info(f"Saving combined weather data to {METEO_OUT_PATH}")
    meteo_df.to_parquet(METEO_OUT_PATH, index=False)

    end_time = time.time()
    logging.info(f"Inference ETL finished successfully in {end_time - start_time:.2f} seconds.")

    # Return the dataframes if needed by a calling script (like the original pipeline)
    # Adjust the return based on whether pipeline.py uses the return values directly
    # or reads the saved files. The current pipeline.py doesn't seem to use returns.
    # return wind_perc_df, meteo_df


if __name__ == "__main__":
    main()
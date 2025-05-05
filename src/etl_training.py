# src/etl_training.py
"""
ETL script for fetching historical data for model training:
* Historical wind generation percentage (Carbon Intensity API)
* Historical weather data             (Open‑Meteo Archive API)

Saves data to data/raw/ directory in Parquet format.

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
# Define the historical period for training data
# Inclusive start date, exclusive end date (like Python slicing)
TRAIN_START_DATE = date(2019, 1, 1) # Example: Start of 2019
TRAIN_END_DATE   = date(2023, 12, 31) # Example: End of 2023

# Location for weather data
LAT, LON = 54.0, -1.5  # Example coordinates (Central UK)
HOURLY_VARS = "temperature_2m,wind_speed_10m" # Weather variables

# Data directories
DATA_DIR = Path("data") # Ensure this path is correct relative to execution
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Output file names expected by featurise.py
WIND_OUT_PATH = RAW_DIR / "eso_wind.parquet"
METEO_OUT_PATH = RAW_DIR / "openmeteo_weather.parquet"

# Fetching configuration
CI_FETCH_CHUNK_DAYS = 30 # Fetch Carbon Intensity data in chunks (e.g., 30 days)
OM_FETCH_CHUNK_YEARS = 1 # Fetch Open Meteo data in chunks (e.g., 1 year)
# --------------------------------------------------------------------------- #

# ---------- Setup Logging -------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- Carbon‑Intensity API Configuration ----------------------------- #
CI_API_BASE_URL = "https://api.carbonintensity.org.uk"
CI_HEADERS = {'Accept': 'application/json'}

# ---------- Open‑Meteo API Configuration ----------------------------------- #
OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ---------- Retry Mechanism ------------------------------------------------ #
@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=5, max=60))
def _make_request(url: str, params: dict | None = None, headers: dict | None = None) -> dict:
    """Makes a GET request with retries and basic error checking."""
    try:
        logging.debug(f"Requesting URL: {url} with params: {params}")
        response = requests.get(url, params=params, headers=headers, timeout=120) # Longer timeout for potentially large data
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {url}: {e}. Retrying...")
        raise # Re-raise exception to trigger tenacity retry

# ---------- Carbon‑Intensity Historical Wind Percentage Fetcher ------------- #
def fetch_ci_wind_perc_historical(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetches historical wind generation percentage from Carbon Intensity API
    in chunks.
    """
    all_rows = []
    current_start = start_date

    logging.info(f"Fetching Carbon Intensity wind % from {start_date} to {end_date}")

    while current_start <= end_date:
        chunk_end = min(current_start + timedelta(days=CI_FETCH_CHUNK_DAYS - 1), end_date)
        from_dt_str = current_start.strftime('%Y-%m-%dT00:00Z')
        # Fetch up to the end of the chunk_end day
        to_dt_str = chunk_end.strftime('%Y-%m-%dT23:30Z')

        url = f"{CI_API_BASE_URL}/generation/{from_dt_str}/{to_dt_str}"
        logging.info(f"  Fetching chunk: {current_start} to {chunk_end}")

        try:
            js = _make_request(url, headers=CI_HEADERS)
            time.sleep(1) # Be polite to the API

            if "data" not in js or not isinstance(js["data"], list):
                logging.warning(f"  Unexpected response structure for chunk {current_start}-{chunk_end}. Skipping.")
                current_start += timedelta(days=CI_FETCH_CHUNK_DAYS)
                continue

            chunk_rows = []
            for interval_data in js["data"]:
                ts = interval_data.get("from")
                generation_mix = interval_data.get("generationmix")

                if not ts or not generation_mix:
                    continue # Skip malformed intervals

                wind_entry = next((item for item in generation_mix if item.get("fuel") == "wind"), None)

                if wind_entry and "perc" in wind_entry:
                    chunk_rows.append({"datetime": ts, "wind_perc": wind_entry["perc"]})
                else:
                    # Log if wind data is missing for an interval, append NaN
                    logging.debug(f"  Wind percentage not found for interval {ts}. Appending NaN.")
                    chunk_rows.append({"datetime": ts, "wind_perc": float('nan')})

            all_rows.extend(chunk_rows)
            logging.info(f"  Fetched {len(chunk_rows)} records for chunk.")

        except Exception as e:
            logging.error(f"  Failed to fetch or parse chunk {current_start}-{chunk_end}: {e}. Skipping chunk.")
            # Optionally implement logic to retry the chunk later or abort

        # Move to the next chunk
        current_start += timedelta(days=CI_FETCH_CHUNK_DAYS)

    if not all_rows:
        logging.error("No Carbon Intensity data could be fetched for the specified period.")
        return pd.DataFrame(columns=["datetime", "wind_perc"])

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    # De-duplicate based on datetime, keeping the first occurrence
    df = df.drop_duplicates(subset=["datetime"], keep="first")
    df = df.sort_values("datetime").reset_index(drop=True)
    logging.info(f"Total unique Carbon Intensity records fetched: {len(df)}")
    return df


# ---------- Open‑Meteo Historical Weather Fetcher -------------------------- #
def fetch_weather_historical(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetches historical hourly weather data from Open-Meteo Archive API
    in yearly chunks.
    """
    all_meteo_dfs = []
    current_start = start_date

    logging.info(f"Fetching Open-Meteo weather from {start_date} to {end_date}")

    while current_start <= end_date:
        # Fetch data year by year, or up to end_date if less than a year remaining
        chunk_end_year = min(current_start.year + OM_FETCH_CHUNK_YEARS - 1, end_date.year)
        chunk_end = date(chunk_end_year, 12, 31)
        # Ensure the final chunk doesn't exceed the overall end_date
        chunk_end = min(chunk_end, end_date)

        chunk_start_str = current_start.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d')

        url = (
            f"{OM_ARCHIVE_URL}?latitude={LAT}&longitude={LON}"
            f"&hourly={HOURLY_VARS}&start_date={chunk_start_str}&end_date={chunk_end_str}"
            "&timezone=UTC"
        )
        logging.info(f"  Fetching weather chunk: {current_start} to {chunk_end}")

        try:
            js = _make_request(url)
            time.sleep(1) # Be polite

            if "hourly" not in js or "time" not in js["hourly"]:
                logging.warning(f"  Unexpected Open-Meteo response for chunk {current_start}-{chunk_end}. Skipping.")
                # Move to the next year/chunk start
                current_start = date(chunk_end.year + 1, 1, 1)
                continue

            df_chunk = pd.DataFrame(js["hourly"])
            df_chunk = df_chunk.rename(columns={"time": "datetime"})
            df_chunk["datetime"] = pd.to_datetime(df_chunk["datetime"], utc=True)
            all_meteo_dfs.append(df_chunk)
            logging.info(f"  Fetched {len(df_chunk)} weather records for chunk.")

        except Exception as e:
            logging.error(f"  Failed to fetch weather chunk {current_start}-{chunk_end}: {e}. Skipping chunk.")
            # Optionally implement logic to retry the chunk later or abort

        # Move to the start of the next chunk (day after the current chunk ended)
        current_start = chunk_end + timedelta(days=1)


    if not all_meteo_dfs:
        logging.error("No Open-Meteo weather data could be fetched.")
        return pd.DataFrame()

    # Concatenate all chunks
    meteo_df = pd.concat(all_meteo_dfs, ignore_index=True)
    meteo_df = meteo_df.drop_duplicates(subset=["datetime"], keep="first")
    meteo_df = meteo_df.sort_values("datetime").reset_index(drop=True)
    logging.info(f"Total unique Open-Meteo records fetched: {len(meteo_df)}")
    return meteo_df


# ---------- Main Execution Logic ------------------------------------------- #
def main():
    """Main ETL function for training data."""
    start_time = time.time()
    logging.info(f"Starting training ETL run for period: {TRAIN_START_DATE} to {TRAIN_END_DATE}")

    # --- Fetch Historical Data ---
    wind_perc_df = fetch_ci_wind_perc_historical(TRAIN_START_DATE, TRAIN_END_DATE)
    if wind_perc_df.empty:
        logging.error("Failed to fetch historical wind percentage data. Aborting.")
        return

    meteo_df = fetch_weather_historical(TRAIN_START_DATE, TRAIN_END_DATE)
    if meteo_df.empty:
        logging.error("Failed to fetch historical weather data. Aborting.")
        return

    # --- Save Data ---
    logging.info(f"Saving historical wind percentage data to {WIND_OUT_PATH}")
    wind_perc_df.to_parquet(WIND_OUT_PATH, index=False)

    logging.info(f"Saving historical weather data to {METEO_OUT_PATH}")
    meteo_df.to_parquet(METEO_OUT_PATH, index=False)

    end_time = time.time()
    logging.info(f"Training ETL finished successfully in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
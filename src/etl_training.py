#! /usr/bin/env python
# src/etl_training.py

from __future__ import annotations
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# ── USER CONFIG ─────────────────────────────────────────────────────────────
TRAIN_START_DATE = date(2017, 9, 27)
# Dynamic end: the model should keep learning from the newest history each
# retrain, not freeze at a hard-coded date. CI data is near-real-time; the
# Open-Meteo archive lags ~5 days, so the weather fetch is clamped below.
TRAIN_END_DATE   = date.today() - timedelta(days=2)
ARCHIVE_MAX_LAG_DAYS = 5

# For Carbon-Intensity wind percent
CI_API_BASE_URL   = "https://api.carbonintensity.org.uk"
CI_HEADERS        = {"Accept": "application/json"}
CI_CHUNK_DAYS     = 30

# For Open-Meteo archive
OM_ARCHIVE_URL    = "https://archive-api.open-meteo.com/v1/archive"
OM_CHUNK_YEARS    = 1
LAT, LON          = 54.0, -1.5
# Hub-height wind (100m sits much closer to turbine hub height than 10m;
# the archive API only exposes 100m, not 80m/120m), gusts, direction and
# pressure are all free from the same Open-Meteo archive.
HOURLY_VARS       = ("temperature_2m,wind_speed_10m,wind_speed_100m,"
                     "wind_gusts_10m,wind_direction_10m,surface_pressure")

# ── PATHS ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
TRAIN_RAW_DIR = DATA_DIR / "raw" / "training"
TRAIN_RAW_DIR.mkdir(parents=True, exist_ok=True)

WIND_OUT_PATH = TRAIN_RAW_DIR / "ci_wind_perc_training.parquet"
METEO_OUT_PATH= TRAIN_RAW_DIR / "openmeteo_weather_training.parquet"

# ── LOGGING SETUP ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── RETRY UTILITY ────────────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
def _make_request(url: str, params: dict | None = None, headers: dict | None = None) -> dict:
    resp = requests.get(url, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()

# ── FETCH CARBON-INTENSITY WIND PERCENT ──────────────────────────────────────
def fetch_ci_wind_perc_historical(start_date: date, end_date: date) -> pd.DataFrame:
    all_rows = []
    cur = start_date
    logging.info(f"Fetching CI wind % from {start_date} to {end_date}")
    while cur <= end_date:
        chunk_end = min(cur + timedelta(days=CI_CHUNK_DAYS - 1), end_date)
        url = f"{CI_API_BASE_URL}/generation/{cur.strftime('%Y-%m-%dT00:00Z')}/" \
              f"{chunk_end.strftime('%Y-%m-%dT23:30Z')}"
        logging.info(f"  CI chunk: {cur} → {chunk_end}")
        try:
            js = _make_request(url, headers=CI_HEADERS)
            for rec in js.get("data", []):
                ts = rec.get("from")
                mix = rec.get("generationmix", [])
                wind = next((x["perc"] for x in mix if x.get("fuel")=="wind"), float("nan"))
                all_rows.append({"datetime": ts, "wind_perc": wind})
        except Exception as e:
            logging.error(f"    Failed chunk {cur}–{chunk_end}: {e}")
        cur = chunk_end + timedelta(days=1)

    df = pd.DataFrame(all_rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    logging.info(f"Total CI wind % rows fetched: {len(df)}")
    return df

# ── FETCH OPEN-METEO ARCHIVE ─────────────────────────────────────────────────
def fetch_weather_historical(start_date: date, end_date: date) -> pd.DataFrame:
    parts = []
    cur = start_date
    logging.info(f"Fetching weather from {start_date} to {end_date}")
    while cur <= end_date:
        chunk_end = date(min(cur.year + OM_CHUNK_YEARS - 1, end_date.year), 12, 31)
        chunk_end = min(chunk_end, end_date)
        params = {
            "latitude": LAT, "longitude": LON,
            "hourly": HOURLY_VARS,
            "start_date": cur.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
            "timezone": "UTC",
        }
        logging.info(f"  Weather chunk: {cur} → {chunk_end}")
        try:
            js = _make_request(OM_ARCHIVE_URL, params=params)
            hr = js.get("hourly", {})
            df = pd.DataFrame(hr).rename(columns={"time": "datetime"})
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            parts.append(df)
        except Exception as e:
            logging.error(f"    Failed weather chunk {cur}–{chunk_end}: {e}")
        cur = chunk_end + timedelta(days=1)

    if not parts:
        return pd.DataFrame()
    met = pd.concat(parts, ignore_index=True) \
             .drop_duplicates("datetime") \
             .sort_values("datetime") \
             .reset_index(drop=True)
    logging.info(f"Total weather rows fetched: {len(met)}")
    return met

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    start = time.time()

    # Incremental CI fetch: the raw target history is append-only, so keep the
    # existing parquet and only request the tail that is missing.
    ci_start = TRAIN_START_DATE
    existing_wind = pd.DataFrame()
    if WIND_OUT_PATH.exists():
        existing_wind = pd.read_parquet(WIND_OUT_PATH)
        if not existing_wind.empty:
            last_ts = pd.to_datetime(existing_wind["datetime"], utc=True).max()
            ci_start = max(ci_start, last_ts.date() + timedelta(days=1))
            logging.info(f"Existing CI data through {last_ts}; fetching from {ci_start}")

    if ci_start <= TRAIN_END_DATE:
        wind_new = fetch_ci_wind_perc_historical(ci_start, TRAIN_END_DATE)
        wind_df = pd.concat([existing_wind, wind_new], ignore_index=True)
        wind_df = wind_df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    else:
        logging.info("CI wind % already up to date.")
        wind_df = existing_wind
    wind_df.to_parquet(WIND_OUT_PATH, index=False)
    logging.info(f"Saved CI wind % → {WIND_OUT_PATH} ({len(wind_df)} rows)")

    # Weather is always a full refetch: the hourly variable set can change
    # between versions (e.g. hub-height winds added), and partial-schema
    # merges are messier than a clean re-pull of the archive.
    met_end = min(TRAIN_END_DATE, date.today() - timedelta(days=ARCHIVE_MAX_LAG_DAYS))
    met_df   = fetch_weather_historical(TRAIN_START_DATE, met_end)
    met_df.to_parquet(METEO_OUT_PATH, index=False)
    logging.info(f"Saved weather history → {METEO_OUT_PATH}")

    logging.info(f"Training ETL finished in {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()

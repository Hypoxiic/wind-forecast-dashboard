"""
Lean ETL for nightly inference:
* Yesterday's wind MW  (Carbon‑Intensity)
* Yesterday's weather  (Open‑Meteo archive, 1 day)
* Next‑48 h forecast    (Open‑Meteo forecast endpoint)
"""

from __future__ import annotations
import time
from datetime import date, datetime, timedelta
from pathlib import Path
import requests, pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------- User‑configurable ---------------------------------------- #
LAT, LON   = 54.0, -1.5
HOURLY_VARS = "temperature_2m,wind_speed_10m"
DATA_DIR = Path("data")
# --------------------------------------------------------------------------- #

# ---------- Carbon‑Intensity wind (yesterday) ------------------------------ #
def fetch_ci_wind_yesterday(today: date) -> pd.DataFrame:
    y0 = today - timedelta(days=1)
    url = (
        "https://api.carbonintensity.org.uk/generation/"
        f"{y0:%Y-%m-%dT00:00Z}/{y0:%Y-%m-%dT23:00Z}"
    )
    js = requests.get(url, timeout=60).json()["data"]
    rows = []
    for blk in js:
        ts = blk["from"]
        wind = next(m for m in blk["generationmix"] if m["fuel"] == "wind")
        if "actual" in wind:
            rows.append({"datetime": ts, "wind_mw": wind["actual"]})
    return pd.DataFrame(rows)

# ---------- Open‑Meteo helpers --------------------------------------------- #
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST= "https://api.open-meteo.com/v1/forecast"

@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=30))
def _get_json(url: str) -> dict:
    return requests.get(url, timeout=60).json()

def fetch_weather_yesterday(today: date) -> pd.DataFrame:
    y0 = today - timedelta(days=1)
    url = (
        f"{ARCHIVE}?latitude={LAT}&longitude={LON}"
        f"&hourly={HOURLY_VARS}&start_date={y0}&end_date={y0}&timezone=UTC"
    )
    js = _get_json(url)
    return pd.DataFrame(js["hourly"])

def fetch_weather_forecast(today: date) -> pd.DataFrame:
    tomorrow = today + timedelta(days=1)
    url = (
        f"{FORECAST}?latitude={LAT}&longitude={LON}"
        f"&hourly={HOURLY_VARS}&start_date={today}&end_date={tomorrow}"
        "&timezone=UTC"
    )
    js = _get_json(url)
    return pd.DataFrame(js["hourly"])

# ---------- Main entry (called by pipeline.py) ----------------------------- #
def main():
    today = date.today()
    print("· CI wind (yesterday)")
    wind_df = fetch_ci_wind_yesterday(today)
    print(f"  rows: {len(wind_df)}")

    print("· Open‑Meteo yesterday")
    meteo_y   = fetch_weather_yesterday(today)
    print(f"  rows: {len(meteo_y)}")

    print("· Open‑Meteo forecast 48 h")
    meteo_f   = fetch_weather_forecast(today)
    print(f"  rows: {len(meteo_f)}")

    # concat weather so downstream code sees one frame
    meteo_df = pd.concat([meteo_y, meteo_f], ignore_index=True)

    return wind_df, meteo_df

if __name__ == "__main__":
    main()

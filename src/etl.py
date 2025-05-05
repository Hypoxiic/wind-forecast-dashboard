"""
ETL helpers for the nightly wind‑forecast pipeline.

* ESO CSV (historic wind generation)
* Open‑Meteo archive API, fetched month‑by‑month with retry + local cache
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --------------------------------------------------------------------------- #
# Configuration – tweak here not in the code body
# --------------------------------------------------------------------------- #

# Where to save raw + cached files
DATA_DIR   = Path("data")
ESO_CSV    = DATA_DIR / "eso" / "wind_uk.csv"         # 15‑min ESO metered wind
METEO_DIR  = DATA_DIR / "meteo"                       # monthly archive parquet files

# Open‑Meteo parameters – change lat/lon to your location
LAT, LON   = 54.0, -1.5
METEO_VARS = "temperature_2m,wind_speed_10m"

# Chunk size for archive calls – 30 keeps us well within 31‑day limit
CHUNK_DAYS = 30

# --------------------------------------------------------------------------- #
# 1.  ESO WIND CSV
# --------------------------------------------------------------------------- #

ESO_URL = (
    "https://data.nationalgrideso.com/system/energy-supply/dataset/"
    "wind-and-solar-generation/download"
)

def fetch_eso_csv() -> pd.DataFrame:
    """Download ESO wind CSV if not present / outdated."""
    ESO_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Download if file missing or older than 24 h
    if not ESO_CSV.exists() or (time.time() - ESO_CSV.stat().st_mtime) > 86_400:
        print("Downloading ESO wind CSV …")
        r = requests.get(ESO_URL, timeout=60)
        r.raise_for_status()
        ESO_CSV.write_bytes(r.content)

    df = pd.read_csv(ESO_CSV)
    df.rename(columns=str.lower, inplace=True)
    return df


# --------------------------------------------------------------------------- #
# 2.  OPEN‑METEO ARCHIVE (chunked, cached, retried)
# --------------------------------------------------------------------------- #

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def _fetch_archive_chunk(day0: date, day1: date) -> pd.DataFrame:
    """Fetch a ≤31‑day window from the archive endpoint (with retry)."""
    url = (
        f"{ARCHIVE_URL}"
        f"?latitude={LAT}&longitude={LON}"
        f"&hourly={METEO_VARS}"
        f"&start_date={day0}&end_date={day1}"
        "&timezone=UTC"
    )
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    js = r.json()

    if "hourly" not in js:
        raise ValueError(f"Open‑Meteo response missing 'hourly' key: {js}")

    return pd.DataFrame(js["hourly"])


def _month_cache_path(day: date) -> Path:
    """Return e.g. data/meteo/2025-05.parquet"""
    METEO_DIR.mkdir(parents=True, exist_ok=True)
    return METEO_DIR / f"{day:%Y-%m}.parquet"


def _fetch_or_load_month(year: int, month: int) -> pd.DataFrame:
    """Load the month from cache or fetch it (and then cache)."""
    first = date(year, month, 1)
    cache = _month_cache_path(first)

    if cache.exists():
        return pd.read_parquet(cache)

    # Determine last day of month
    nxt = (first.replace(day=28) + timedelta(days=4)).replace(day=1)  # first of next month
    last = nxt - timedelta(days=1)

    print(f"  · Fetching {first:%Y‑%m} …")  # nice progress print
    df = _fetch_archive_chunk(first, last)
    df.to_parquet(cache, index=False)
    return df


def fetch_openmeteo_archive(start: date, end: date) -> pd.DataFrame:
    """Return concatenated hourly dataframe for full [start, end] range."""
    dfs: List[pd.DataFrame] = []
    cur = start

    while cur <= end:
        dfs.append(_fetch_or_load_month(cur.year, cur.month))
        # jump to 1st of next month
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

    out = pd.concat(dfs, ignore_index=True)
    # Keep only requested interval (e.g. if start is mid‑month)
    mask = (out["time"] >= str(start)) & (out["time"] <= str(end))
    return out.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3.  MAIN ENTRY
# --------------------------------------------------------------------------- #

def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestrate downloads – called from src/pipeline.py."""
    print("Downloading ESO data …")
    eso_df = fetch_eso_csv()
    print(f"ESO rows: {len(eso_df):,}")

    # determine date span to fetch (archive endpoint goes back to 1940)
    start_date = date(2024, 1, 1)         # <- adjust earliest date you need
    end_date   = date.today()

    print(f"Downloading Open‑Meteo archive {start_date} → {end_date} …")
    meteo_df   = fetch_openmeteo_archive(start_date, end_date)
    print(f"Open‑Meteo rows: {len(meteo_df):,}")

    return eso_df, meteo_df


if __name__ == "__main__":
    main()

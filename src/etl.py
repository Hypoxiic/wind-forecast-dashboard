"""
ETL helpers for the nightly wind‑forecast pipeline.

* ESO historic metered wind generation
* Open‑Meteo archive API (month‑by‑month, cached, retry)
"""

from __future__ import annotations

import io                      # 🔹  for BytesIO
import os
import time
import zipfile                 # 🔹  to peek inside .zip responses
from datetime import date, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# --------------------------------------------------------------------------- #
# Configuration – tweak here, not in the code body
# --------------------------------------------------------------------------- #

DATA_DIR   = Path("data")
ESO_CSV    = DATA_DIR / "eso" / "wind_uk.csv"        # cached, plain CSV we save
METEO_DIR  = DATA_DIR / "meteo"

LAT, LON   = 54.0, -1.5                             # 🔹  edit to your location
METEO_VARS = "temperature_2m,wind_speed_10m"

CHUNK_DAYS = 30
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Direct download link for the ESO “wind‑and‑solar‑generation” dataset
# (this link returns a ZIP that holds a single CSV file)
ESO_URL = (
    "https://data.nationalgrideso.com/system/energy-supply/"
    "wind-and-solar-generation/download/wind_and_solar_generation.csv.zip"
)
# If National Grid rename the file, adjust the trailing filename.          🔹


# --------------------------------------------------------------------------- #
# 1.  ESO WIND CSV (robust to zip / gzip)
# --------------------------------------------------------------------------- #

def _read_csv_bytes(buf: bytes) -> pd.DataFrame:
    """
    Take raw bytes that are either:
    • plain CSV text
    • gz compressed CSV
    • a ZIP archive with exactly one CSV inside
    and return a DataFrame.
    """
    # ZIP?
    if buf[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(buf)) as z:
            csv_members = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_members:
                raise ValueError("ZIP from ESO contains no .csv file!")
            with z.open(csv_members[0]) as f:
                return pd.read_csv(f)

    # GZIP?
    if buf[:2] == b"\x1f\x8b":
        return pd.read_csv(io.BytesIO(buf), compression="gzip")

    # Plain text CSV
    return pd.read_csv(io.BytesIO(buf))


def fetch_eso_csv() -> pd.DataFrame:
    """Download and cache ESO wind data, handling ZIP / GZIP transparently."""
    ESO_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Download if cache missing or older than 24 h
    need_download = (not ESO_CSV.exists()) or (
        time.time() - ESO_CSV.stat().st_mtime > 86_400
    )

    if need_download:
        print("Downloading ESO wind CSV …")
        r = requests.get(ESO_URL, timeout=90)
        r.raise_for_status()

        df = _read_csv_bytes(r.content)
        df.to_csv(ESO_CSV, index=False)   # save as flat CSV for next runs
    else:
        df = pd.read_csv(ESO_CSV)

    df.rename(columns=str.lower, inplace=True)
    return df


# --------------------------------------------------------------------------- #
# 2.  OPEN‑METEO ARCHIVE (unchanged from previous version)
# --------------------------------------------------------------------------- #

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def _fetch_archive_chunk(day0: date, day1: date) -> pd.DataFrame:
    url = (
        f"{ARCHIVE_URL}?latitude={LAT}&longitude={LON}"
        f"&hourly={METEO_VARS}"
        f"&start_date={day0}&end_date={day1}&timezone=UTC"
    )
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    js = r.json()
    if "hourly" not in js:
        raise ValueError(f"Open‑Meteo: no 'hourly' key → {js}")
    return pd.DataFrame(js["hourly"])


def _month_cache_path(day: date) -> Path:
    METEO_DIR.mkdir(parents=True, exist_ok=True)
    return METEO_DIR / f"{day:%Y-%m}.parquet"


def _fetch_or_load_month(year: int, month: int) -> pd.DataFrame:
    first = date(year, month, 1)
    cache = _month_cache_path(first)

    if cache.exists():
        return pd.read_parquet(cache)

    # last day of month
    nxt = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    last = nxt - timedelta(days=1)

    print(f"  · Fetching {first:%Y‑%m} …")
    df = _fetch_archive_chunk(first, last)
    df.to_parquet(cache, index=False)
    return df


def fetch_openmeteo_archive(start: date, end: date) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    cur = start
    while cur <= end:
        dfs.append(_fetch_or_load_month(cur.year, cur.month))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)  # next month
    out = pd.concat(dfs, ignore_index=True)
    mask = (out["time"] >= str(start)) & (out["time"] <= str(end))
    return out.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3.  MAIN ENTRY
# --------------------------------------------------------------------------- #

def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Downloading ESO data …")
    eso_df = fetch_eso_csv()
    print(f"ESO rows: {len(eso_df):,}")

    start_date = date(2024, 1, 1)          # 🔹  pick earliest needed
    end_date   = date.today()

    print(f"Downloading Open‑Meteo archive {start_date} → {end_date} …")
    meteo_df   = fetch_openmeteo_archive(start_date, end_date)
    print(f"Open‑Meteo rows: {len(meteo_df):,}")

    return eso_df, meteo_df


if __name__ == "__main__":
    main()

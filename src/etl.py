"""
ETL for the nightly wind‑forecast pipeline.

* fetch_wind_ci : Carbon‑Intensity UK hourly wind generation (MW)
* fetch_openmeteo_archive : Open‑Meteo archive, month‑by‑month, cached
"""

from __future__ import annotations

import os, time, io, zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --------------------------------------------------------------------------- #
# Directories
# --------------------------------------------------------------------------- #

DATA_DIR   = Path("data")
WIND_PATH  = DATA_DIR / "wind" / "ci_wind.parquet"   # cache for wind out‑turn
METEO_DIR  = DATA_DIR / "meteo"                      # Open‑Meteo monthly cache
METEO_DIR.mkdir(parents=True, exist_ok=True)
WIND_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1.  Carbon‑Intensity UK wind out‑turn
# --------------------------------------------------------------------------- #

CI_API = "https://api.carbonintensity.org.uk/generation/{start}/{end}"

def _parse_ci_block(block: dict) -> dict[str, float] | None:
    """Return {'datetime': iso, 'wind_mw': value} or None."""
    ts = block["from"]

    # find total MW if provided
    total_mw = next(
        (mix.get("actual") for mix in block["generationmix"]
         if mix["fuel"] in ("all", "total") and "actual" in mix),
        None,
    )

    wind = next(m for m in block["generationmix"] if m["fuel"] == "wind")

    if "actual" in wind:
        return {"datetime": ts, "wind_mw": wind["actual"]}

    if total_mw is not None and "perc" in wind:
        return {"datetime": ts, "wind_mw": total_mw * wind["perc"] / 100.0}

    # nothing useful this hour
    return None


def fetch_wind_ci(start: date, end: date) -> pd.DataFrame:
    """Hourly wind MW between [start, end] inclusive, cached on disk."""
    # --- load existing cache -------------------------------------------------
    if WIND_PATH.exists():
        cache = pd.read_parquet(WIND_PATH)
    else:
        cache = pd.DataFrame(columns=["datetime", "wind_mw"])

    have_dates = set(cache["datetime"])

    rows: list[dict] = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=30), end)
        url = CI_API.format(
            start=f"{cur:%Y-%m-%dT00:00Z}",
            end=f"{nxt:%Y-%m-%dT23:00Z}",
        )

        js = requests.get(url, timeout=60).json()["data"]
        for blk in js:
            record = _parse_ci_block(blk)
            if record and record["datetime"] not in have_dates:
                rows.append(record)

        cur = nxt + timedelta(days=1)
        time.sleep(1)          # polite pause

    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_parquet(WIND_PATH, index=False)

    # keep only requested window
    mask = (cache["datetime"] >= f"{start}T00:00Z") & (cache["datetime"] <= f"{end}T23:59Z")
    return cache.loc[mask].reset_index(drop=True)

# --------------------------------------------------------------------------- #
# 2.  Open‑Meteo archive (same as previous version)
# --------------------------------------------------------------------------- #

LAT, LON   = 54.0, -1.5            # change to your site
METEO_VARS = "temperature_2m,wind_speed_10m"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
)
def _fetch_archive_chunk(day0: date, day1: date) -> pd.DataFrame:
    url = (
        f"{ARCHIVE_URL}?latitude={LAT}&longitude={LON}"
        f"&hourly={METEO_VARS}&start_date={day0}&end_date={day1}&timezone=UTC"
    )
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    js = r.json()
    if "hourly" not in js:
        raise ValueError(f"Open‑Meteo response lacks 'hourly': {js}")
    return pd.DataFrame(js["hourly"])

def _month_cache_path(day: date) -> Path:
    return METEO_DIR / f"{day:%Y-%m}.parquet"

def _fetch_or_load_month(year: int, month: int) -> pd.DataFrame:
    first = date(year, month, 1)
    cache = _month_cache_path(first)
    if cache.exists():
        return pd.read_parquet(cache)

    # calc last day of month
    nxt = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    last = nxt - timedelta(days=1)

    print(f"  · Open‑Meteo {first:%Y‑%m}")
    part = _fetch_archive_chunk(first, last)
    part.to_parquet(cache, index=False)
    return part

def fetch_openmeteo_archive(start: date, end: date) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    cur = start
    while cur <= end:
        dfs.append(_fetch_or_load_month(cur.year, cur.month))
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    all_df = pd.concat(dfs, ignore_index=True)
    mask = (all_df["time"] >= str(start)) & (all_df["time"] <= str(end))
    return all_df.loc[mask].reset_index(drop=True)

# --------------------------------------------------------------------------- #
# 3.  Main entry
# --------------------------------------------------------------------------- #

def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    #start_date = date(2024, 1, 1)   # earliest you want
    end_date   = date.today()
    start_date = end_date - timedelta(days=60)   # 🔸 was date(2024, 1, 1)

    print("Fetching Carbon‑Intensity wind …")
    wind_df  = fetch_wind_ci(start_date, end_date)
    print(f"Wind rows: {len(wind_df):,}")

    print(f"Fetching Open‑Meteo {start_date} → {end_date} …")
    meteo_df = fetch_openmeteo_archive(start_date, end_date)
    print(f"Open‑Meteo rows: {len(meteo_df):,}")

    return wind_df, meteo_df

if __name__ == "__main__":
    main()

"""One-off repair for gaps in data/features/history.parquet.

The nightly job only fetches ~3 days, so any stretch where it failed leaves a
permanent hole in the rolling history — and those holes now matter, because
the model's strongest features are trailing windows over the target. This
script refetches wind_perc straight from the Carbon Intensity API for a date
range and merges it in, keeping existing rows for timestamps it cannot supply.

    python -m src.backfill_history --days 420
    python -m src.backfill_history --start 2026-05-01 --end 2026-08-14

Safe to re-run: the merge is keyed on datetime and only fills what is missing
unless --overwrite is passed.
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

CI_API_BASE_URL = "https://api.carbonintensity.org.uk"
CI_HEADERS = {"Accept": "application/json"}
HISTORY_PATH = Path("data/features/history.parquet")
CHUNK_DAYS = 10          # the API caps a single generation query at ~14 days


def fetch_range(start: date, end: date) -> pd.DataFrame:
    """wind_perc for [start, end], fetched in chunks the API will accept."""
    rows: list[dict] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), end)
        url = (f"{CI_API_BASE_URL}/generation/"
               f"{cursor:%Y-%m-%d}T00:00Z/{chunk_end:%Y-%m-%d}T23:30Z")
        try:
            r = requests.get(url, headers=CI_HEADERS, timeout=45)
            r.raise_for_status()
            data = r.json().get("data", [])
        except (requests.RequestException, ValueError) as e:
            logging.warning("chunk %s..%s failed: %s", cursor, chunk_end, e)
            cursor = chunk_end + timedelta(days=1)
            continue

        for interval in data:
            ts = interval.get("from")
            mix = interval.get("generationmix")
            if not ts or not mix:
                continue
            wind = next((m for m in mix if m.get("fuel") == "wind"), None)
            if wind and "perc" in wind:
                rows.append({"datetime": ts, "wind_perc": wind["perc"]})

        logging.info("  %s..%s -> %d intervals (running total %d)",
                     cursor, chunk_end, len(data), len(rows))
        cursor = chunk_end + timedelta(days=1)
        time.sleep(0.3)          # be polite to a free public API

    if not rows:
        return pd.DataFrame(columns=["datetime", "wind_perc"])
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return (df.dropna(subset=["wind_perc"])
              .drop_duplicates("datetime")
              .sort_values("datetime")
              .reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=420,
                    help="how far back from today to repair (default 420)")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="let fetched values replace existing ones, not just fill gaps")
    args = ap.parse_args()

    today = datetime.utcnow().date()
    end = date.fromisoformat(args.end) if args.end else today
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=args.days)
    logging.info("Backfilling %s .. %s", start, end)

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"{HISTORY_PATH} not found; run the pipeline once first.")
    hist = pd.read_parquet(HISTORY_PATH)
    hist["datetime"] = pd.to_datetime(hist["datetime"], utc=True)
    before_nonnull = int(hist["wind_perc"].notna().sum())

    fetched = fetch_range(start, end)
    if fetched.empty:
        logging.error("Nothing fetched; leaving %s untouched.", HISTORY_PATH)
        return
    logging.info("Fetched %d intervals from the API.", len(fetched))

    # Align to the grid history.parquet already uses, then fill.
    merged = hist.merge(fetched, on="datetime", how="outer", suffixes=("", "_new"))
    if args.overwrite:
        merged["wind_perc"] = merged["wind_perc_new"].combine_first(merged["wind_perc"])
    else:
        merged["wind_perc"] = merged["wind_perc"].combine_first(merged["wind_perc_new"])
    merged = (merged.drop(columns=["wind_perc_new"])
                    .sort_values("datetime")
                    .drop_duplicates("datetime", keep="last")
                    .reset_index(drop=True))

    after_nonnull = int(merged["wind_perc"].notna().sum())
    logging.info("wind_perc non-null: %d -> %d (+%d); rows %d -> %d",
                 before_nonnull, after_nonnull, after_nonnull - before_nonnull,
                 len(hist), len(merged))

    # The dashboard reads this column straight off the file.
    merged["wind_perc_lag_48h"] = (
        merged.set_index("datetime")["wind_perc"]
              .resample("1h").mean().shift(48)
              .reindex(merged["datetime"]).to_numpy())

    merged.to_parquet(HISTORY_PATH, index=False)
    logging.info("Wrote %s", HISTORY_PATH)


if __name__ == "__main__":
    main()

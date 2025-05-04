"""
src/etl.py
Phase 1 – Data acquisition for GB wind day‑ahead forecast project
---------------------------------------------------------------
• Pull half‑hourly wind generation (historic) from National Grid ESO
• Pull day‑ahead wind‑speed forecast from Open‑Meteo (100 m hub height)
• Save both as parquet under data/raw/

Run:  python src/etl.py
"""

from __future__ import annotations
import io, time, logging, datetime as dt
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# National Grid ESO CSV (single file with all years)
NG_ESO_URL = (
    "https://api.neso.energy/dataset/88313ae5-94e4-4ddc-a790-593554d8c6b9/"
    "resource/f93d1835-75bc-43e5-84ad-12472b180a98/download/df%5Ffuel%5Fckan.csv"
)

# Open‑Meteo hourly forecast (100 m wind speed)
LAT, LON = 54.0, -1.0  # Dogger Bank area
OPEN_METEO_BASE = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=wind_speed_100m"
    "&timezone=Europe/London"
)

# --------------------------------------------------
def fetch_eso_wind(retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """Download and clean the ESO generation CSV (half‑hourly MW)."""
    for a in range(1, retries + 1):
        try:
            logging.info("Downloading ESO wind CSV …")
            csv_bytes = requests.get(NG_ESO_URL, timeout=60).content

            rows = csv_bytes.decode("utf‑8", errors="ignore").splitlines()
            header_idx = next(i for i, r in enumerate(rows)
                              if r.lower().startswith("datetime"))
            clean_csv = "\n".join(rows[header_idx:])

            df = (
                pd.read_csv(io.StringIO(clean_csv),
                            engine="python", on_bad_lines="skip")
                  .rename(str.lower, axis=1)
                  .loc[:, ["datetime", "wind"]]
                  .rename(columns={"wind": "wind_mw"})
            )
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            logging.info("ESO rows: %s", len(df))
            return df
        except Exception as e:
            logging.warning("ESO fetch failed (%s/%s): %s", a, retries, e)
            if a < retries:
                time.sleep(delay)
            else:
                raise

def fetch_openmeteo(start: str, end: str,
                    retries: int = 3, delay: int = 5) -> pd.DataFrame:
    """Download Open‑Meteo historical forecast and upsample to 30 min."""
    url = f"{OPEN_METEO_BASE}&start_date={start}&end_date={end}"
    for a in range(1, retries + 1):
        try:
            logging.info("Downloading Open‑Meteo JSON %s → %s …", start, end)
            js = requests.get(url, timeout=90).json()
            times = js["hourly"]["time"]
            speed = js["hourly"]["wind_speed_100m"]

            df = pd.DataFrame({
                "datetime": (
                    pd.to_datetime(times)
                        .tz_localize("Europe/London", nonexistent="shift_forward",
                            ambiguous="NaT")
                            .tz_convert("UTC")                     # no .dt here
                ),
                "wind_speed_ms": speed,
            }).dropna()

            # Hourly → 30 min
            # make datetime the index
            df = df.set_index("datetime")

            # 1️⃣ drop any duplicate timestamps (keep the first copy)
            df = df[~df.index.duplicated(keep="first")]

            # 2️⃣ upsample from 1 h to 30 min and forward‑fill
            df = (
                df.resample("30min")      # “30T” is deprecated; use “30min”
                    .ffill()
                    .reset_index()
                )
            logging.info("Open‑Meteo rows: %s", len(df))
            return df
        except Exception as e:
            logging.warning("Open‑Meteo fetch failed (%s/%s): %s", a, retries, e)
            if a < retries:
                time.sleep(delay)
            else:
                raise

# --------------------------------------------------
def main() -> None:
    eso_df = fetch_eso_wind()

    start_date = "2024-01-01"
    end_date = dt.datetime.utcnow().strftime("%Y‑%m‑%d")
    meteo_df = fetch_openmeteo(start_date, end_date)

    (RAW_DIR / "eso_wind.parquet").write_bytes(eso_df.to_parquet(index=False))
    (RAW_DIR / "openmeteo_weather.parquet").write_bytes(
        meteo_df.to_parquet(index=False)
    )
    logging.info("Saved parquet files to %s", RAW_DIR.resolve())

if __name__ == "__main__":
    main()

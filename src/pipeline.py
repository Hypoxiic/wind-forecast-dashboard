#! /usr/bin/env python
# src/pipeline.py

from pathlib import Path
import pandas as pd

from src import etl_inference as etl, featurise, predict

def main():
    # ─── Step 1: pull raw data for inference ────────────────────────────────
    etl.main()        # writes raw/eso_wind.parquet & raw/openmeteo_weather.parquet

    # ─── Step 2: build ONLY the new inference features ──────────────────────
    # by default featurise.main() writes data/features/features.parquet
    featurise.main()

    # ─── Step 3: split off those new rows for prediction, then append to history ─
    BASE       = Path(__file__).resolve().parents[1]
    feats_pth  = BASE / "data" / "features" / "features.parquet"
    predict_pth= BASE / "data" / "features" / "for_predict.parquet"
    history_pth= BASE / "data" / "features" / "history.parquet"

    # 3a) load new features
    new_feats = pd.read_parquet(feats_pth)

    # 3b) write them out as 'for_predict.parquet' for the model to read
    new_feats.to_parquet(predict_pth, index=False)

    # 3c) build/append full rolling history
    if history_pth.exists():
        hist = pd.read_parquet(history_pth)
    else:
        # first run: if you have a snapshot of full history (e.g. features_full_history.parquet),
        # point here; otherwise start history with what's just been generated
        orig = BASE / "data" / "features" / "features_full_history.parquet"
        hist = pd.read_parquet(orig) if orig.exists() else new_feats.copy()

    # append & dedupe on datetime
    hist = pd.concat([hist, new_feats], ignore_index=True)
    hist = (
        hist
        .drop_duplicates(subset="datetime", keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # save updated history
    hist.to_parquet(history_pth, index=False)

    # ─── Step 4: run prediction on just the new features ─────────────────────
    # predict.main() should load "data/features/for_predict.parquet" (see note below)
    predict.main()


if __name__ == "__main__":
    main()

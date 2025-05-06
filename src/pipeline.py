#! /usr/bin/env python
# src/pipeline.py

import shutil
from pathlib import Path
import pandas as pd
import sys, os
# ensure `src/` is on the import path so we can do `import etl_inference`, etc.
sys.path.insert(0, os.path.dirname(__file__))
import etl_inference as etl
import featurise
import predict

def main():
    # ─── Define paths & safeguard static full history ────────────────────────
    BASE           = Path(__file__).resolve().parents[1]
    feats_dir      = BASE / "data" / "features"
    orig_feats     = feats_dir / "features.parquet"
    full_hist      = feats_dir / "features_full_history.parquet"
    predict_feats  = feats_dir / "for_predict.parquet"
    history_pth    = feats_dir / "history.parquet"

    # If we haven’t yet preserved the original full-history, do so now
    if not full_hist.exists() and orig_feats.exists():
        print(f"Creating static full-history snapshot: {full_hist.name}")
        shutil.copy(orig_feats, full_hist)

    # ─── Step 1: pull raw data for inference ────────────────────────────────
    etl.main()        # writes data/raw/ci.parquet & data/raw/openmeteo_weather.parquet

    # ─── Step 2: build ONLY the new inference features ──────────────────────
    # featurise.main() will overwrite data/features/features.parquet with just the newest rows
    featurise.main()

    # ─── Step 3: split off inference features & restore full history ────────
    if orig_feats.exists():
        print(f"Moving new features from {orig_feats.name} to {predict_feats.name}")
        shutil.move(orig_feats, predict_feats)
    else:
        print(f"Error: expected {orig_feats.name} but it was not found.")
        # create an empty file to avoid downstream errors
        predict_feats.touch()

    if full_hist.exists():
        print(f"Restoring full history from {full_hist.name} to {orig_feats.name}")
        shutil.copy(full_hist, orig_feats)
    else:
        print(f"Warning: static snapshot {full_hist.name} not found; {orig_feats.name} may be incomplete.")

    # Load the newly generated features for prediction
    if predict_feats.exists() and predict_feats.stat().st_size > 0:
        new_feats = pd.read_parquet(predict_feats)
        print(f"Loaded {len(new_feats)} new feature rows from {predict_feats.name}")
    else:
        print(f"Warning: {predict_feats.name} is missing or empty; no new features to append.")
        new_feats = pd.DataFrame(columns=[])

    # ─── Step 4: build/append rolling history ────────────────────────────────
    # Prefer reinitializing from the static snapshot if history is stale
    if history_pth.exists():
        hist = pd.read_parquet(history_pth)
        if full_hist.exists():
            full = pd.read_parquet(full_hist)
            if len(hist) < len(full):
                print(f"Detected stale history ({len(hist)} rows) < full snapshot ({len(full)} rows). Reinitialising.")
                hist = full
    elif full_hist.exists():
        print(f"Initialising rolling history from static snapshot {full_hist.name}")
        hist = pd.read_parquet(full_hist)
    else:
        print("No history file found; starting history with new features only.")
        hist = new_feats.copy()

    # Append new features if any
    if not new_feats.empty:
        hist = pd.concat([hist, new_feats], ignore_index=True)
        hist = (
            hist
            .drop_duplicates(subset="datetime", keep="last")
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        print(f"Appended new features. History now has {len(hist)} rows.")
    else:
        print("Skipping history append step (no new features).")

    # Save updated rolling history
    if not hist.empty:
        print(f"Saving updated rolling history to {history_pth.name}")
        hist.to_parquet(history_pth, index=False)
    else:
        print(f"Skipping saving empty history to {history_pth.name}")

    # ─── Step 5: run prediction on just the new features ────────────────────
    print("Starting prediction step...")
    predict.main()   # should read data/features/for_predict.parquet and write data/predictions/latest.parquet
    print("Prediction step finished.")

if __name__ == "__main__":
    main()

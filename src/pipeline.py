#! /usr/bin/env python
# src/pipeline.py

import shutil
from pathlib import Path
import pandas as pd

from src import etl_inference as etl, featurise, predict

def main():
    # ─── Define paths and safeguard full history ────────────────────────────
    BASE = Path(__file__).resolve().parents[1]
    feats_dir = BASE / "data" / "features"
    orig_feats = feats_dir / "features.parquet"
    full_hist = feats_dir / "features_full_history.parquet"
    predict_feats = feats_dir / "for_predict.parquet"
    history_pth = feats_dir / "history.parquet"

    # safeguard the static history
    if not full_hist.exists() and orig_feats.exists():
        print(f"Creating {full_hist.name} from current {orig_feats.name}")
        shutil.copy(orig_feats, full_hist)

    # ─── Step 1: pull raw data for inference ────────────────────────────────
    etl.main()        # writes raw/eso_wind.parquet & raw/openmeteo_weather.parquet

    # ─── Step 2: build ONLY the new inference features ──────────────────────
    # featurise.main() overwrites data/features/features.parquet with ONLY new rows
    featurise.main()

    # ─── Step 3: Separate inference features, restore history, update rolling ─
    # keep the new rows for prediction
    if orig_feats.exists(): # Ensure the file was created by featurise
        print(f"Moving new features from {orig_feats.name} to {predict_feats.name}")
        shutil.move(orig_feats, predict_feats)
    else:
        print(f"Error: {orig_feats.name} not found after featurise step.")
        # Handle error: Maybe exit or create an empty predict_feats?
        predict_feats.touch() # Create empty file to avoid downstream error

    # restore the full history file to the main features.parquet path
    if full_hist.exists():
        print(f"Restoring full history from {full_hist.name} to {orig_feats.name}")
        shutil.copy(full_hist, orig_feats)
    else:
        print(f"Warning: {full_hist.name} not found. {orig_feats.name} may be incomplete.")

    # now load new_feats from predict_feats (the ones just generated)
    if predict_feats.exists() and predict_feats.stat().st_size > 0:
        new_feats = pd.read_parquet(predict_feats)
        print(f"Loaded {len(new_feats)} new feature rows from {predict_feats.name}")
    else:
        print(f"Warning: {predict_feats.name} is missing or empty. No new features to append.")
        new_feats = pd.DataFrame() # Ensure new_feats is an empty DataFrame

    # build/append full rolling history
    if history_pth.exists():
        print(f"Loading existing rolling history from {history_pth.name}")
        hist = pd.read_parquet(history_pth)
    elif full_hist.exists():
        print(f"Initialising rolling history from static {full_hist.name}")
        hist = pd.read_parquet(full_hist)
    else:
        print(f"Warning: No history file found. Initialising with new features only.")
        hist = new_feats.copy() if not new_feats.empty else pd.DataFrame()

    # append & dedupe on datetime
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
        print("Skipping history append step as no new features were loaded.")

    # save updated history
    if not hist.empty:
        print(f"Saving updated rolling history to {history_pth.name}")
        hist.to_parquet(history_pth, index=False)
    else:
        print(f"Skipping save of empty history to {history_pth.name}")

    # ─── Step 4: run prediction on just the new features ─────────────────────
    # predict.main() loads "data/features/for_predict.parquet"
    print("Starting prediction step...")
    predict.main()
    print("Prediction step finished.")

if __name__ == "__main__":
    main()

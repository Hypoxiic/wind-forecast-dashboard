#! /usr/bin/env python
# src/pipeline.py

from pathlib import Path
import pandas as pd
import sys, os
# ensure `src/` is on the import path so we can do `import etl_inference`, etc.
sys.path.insert(0, os.path.dirname(__file__))
import etl_inference as etl
import featurise
import predict

def main():
    # ─── Define paths ─────────────────────────────────────────────────────────
    BASE           = Path(__file__).resolve().parents[1]
    feats_dir      = BASE / "data" / "features"
    orig_feats     = feats_dir / "features.parquet"
    full_hist      = feats_dir / "features_full_history.parquet"
    predict_feats  = feats_dir / "for_predict.parquet"
    history_pth    = feats_dir / "history.parquet"

    # ─── Step 1: pull raw data for inference ────────────────────────────────
    # Raises on API failure so the nightly job fails visibly instead of
    # silently republishing predictions built from stale raw parquets.
    etl.main()        # writes data/raw/ci.parquet & data/raw/openmeteo_weather.parquet

    # ─── Step 2: build inference features straight into for_predict.parquet ─
    # featurise no longer clobbers features.parquet, so the old
    # move/restore-from-snapshot dance (and its crash window) is gone.
    featurise.main(mode="inference")

    if not predict_feats.exists() or predict_feats.stat().st_size == 0:
        raise FileNotFoundError(f"{predict_feats} was not produced; aborting nightly run.")
    new_feats = pd.read_parquet(predict_feats)
    if new_feats.empty:
        raise ValueError(f"{predict_feats} contains no rows; aborting nightly run.")
    print(f"Loaded {len(new_feats)} new feature rows from {predict_feats.name}")

    # ─── Step 3: rolling history = union of snapshot, prior history, new rows ─
    # Idempotent merge keyed on datetime; later frames win on overlap, so a
    # regenerated snapshot can never wipe rows accumulated by earlier runs
    # (the old len(hist) < len(full) heuristic could).
    frames = []
    if full_hist.exists():
        frames.append(pd.read_parquet(full_hist))
    elif orig_feats.exists():
        frames.append(pd.read_parquet(orig_feats))
    if history_pth.exists():
        frames.append(pd.read_parquet(history_pth))
    frames.append(new_feats)

    hist = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="datetime", keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    print(f"Saving rolling history ({len(hist)} rows) to {history_pth.name}")
    hist.to_parquet(history_pth, index=False)

    # ─── Step 4: run prediction on just the new features ────────────────────
    print("Starting prediction step...")
    predict.main()   # reads data/features/for_predict.parquet, writes data/predictions/latest.parquet
    print("Prediction step finished.")

if __name__ == "__main__":
    main()

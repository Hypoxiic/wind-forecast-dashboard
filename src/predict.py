# src/predict.py
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor

FEAT   = Path("data/features/features.parquet")
MODEL  = Path("models/model.cbm")
OUTDIR = Path("data/predictions"); OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    feats = pd.read_parquet(FEAT)
    model = CatBoostRegressor().load_model(str(MODEL))
    preds = feats[["datetime"]].copy()
    preds["wind_perc_pred"] = model.predict(feats.drop(columns=["datetime", "wind_mw", "wind_perc"], errors='ignore'))
    preds.to_parquet(OUTDIR / "latest.parquet", index=False)

if __name__ == "__main__":
    main()

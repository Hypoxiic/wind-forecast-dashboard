# src/predict.py
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor

FEAT   = Path("data/features/features.parquet")
MODEL  = Path("models/catboost_best.cbm")        # rename from catboost_info to models/
OUTDIR = Path("data/predictions"); OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    feats = pd.read_parquet(FEAT)
    model = CatBoostRegressor().load_model(str(MODEL))
    preds = feats[["datetime"]].copy()
    preds["wind_mw_pred"] = model.predict(feats)
    preds.to_parquet(OUTDIR / "latest.parquet", index=False)

if __name__ == "__main__":
    main()

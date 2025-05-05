# src/predict.py
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor
from catboost import Pool

FEAT   = Path("data/features/for_predict.parquet")
MODEL  = Path("models/model.cbm")
OUTDIR = Path("data/predictions"); OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    feats = pd.read_parquet(FEAT)
    model = CatBoostRegressor().load_model(str(MODEL))
    preds = feats[["datetime"]].copy()
    
    # Prepare features for prediction
    X = feats.drop(columns=["datetime", "wind_mw", "wind_perc"], errors='ignore')
    pool = Pool(X, cat_features=[]) # Explicitly define no categorical features

    preds["wind_perc_pred"] = model.predict(pool)
    preds.to_parquet(OUTDIR / "latest.parquet", index=False)

if __name__ == "__main__":
    main()

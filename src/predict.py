# src/predict.py
from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor
from catboost import Pool

FEAT   = Path("data/features/for_predict.parquet")
MODEL  = Path("models/model.cbm")
OUTDIR = Path("data/predictions"); OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    if not FEAT.exists() or FEAT.stat().st_size == 0:
        raise FileNotFoundError(f"{FEAT} is missing or empty; run the featurise step first.")
    feats = pd.read_parquet(FEAT)
    if feats.empty or "datetime" not in feats.columns:
        raise ValueError(f"{FEAT} contains no usable feature rows.")

    model = CatBoostRegressor().load_model(str(MODEL))
    preds = feats[["datetime"]].copy()

    # Prepare features for prediction
    X = feats.drop(columns=["datetime", "wind_perc"], errors='ignore')
    # Declare the same categorical features as training (train_model.py
    # CAT_FEATURES) instead of relying on CatBoost's implicit int coercion.
    cat_features = [c for c in ["is_holiday"] if c in X.columns]
    pool = Pool(X, cat_features=cat_features)

    preds["wind_perc_pred"] = model.predict(pool)
    preds.to_parquet(OUTDIR / "latest.parquet", index=False)

if __name__ == "__main__":
    main()

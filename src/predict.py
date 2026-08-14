# src/predict.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost import Pool

FEAT   = Path("data/features/for_predict.parquet")
MODEL  = Path("models/model.cbm")
MODEL_P10 = Path("models/model_p10.cbm")
MODEL_P90 = Path("models/model_p90.cbm")
METRICS = Path("metrics.json")
OUTDIR = Path("data/predictions"); OUTDIR.mkdir(parents=True, exist_ok=True)


def _conformal_delta() -> float:
    """Band widening measured on a held-out calibration window at training
    time. Absent (older artefacts) means no correction."""
    try:
        return float(json.loads(METRICS.read_text())["conformal_delta_pts"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0

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

    # Uncertainty band from the quantile companions. Optional: an older
    # deployment without model_p10/p90.cbm still gets the point forecast.
    if MODEL_P10.exists() and MODEL_P90.exists():
        p10 = CatBoostRegressor().load_model(str(MODEL_P10)).predict(pool)
        p90 = CatBoostRegressor().load_model(str(MODEL_P90)).predict(pool)
        # Split-conformal correction measured at training time. Quantile
        # regression is not calibrated by construction, so without this the
        # nominal-80% band lands wherever it lands. A negative delta narrows
        # an over-wide band; missing metrics.json degrades to the raw band.
        delta = _conformal_delta()
        # Guard against quantile crossing and the 0–100 share bounds.
        lo = np.minimum(p10, preds["wind_perc_pred"]) - delta
        hi = np.maximum(p90, preds["wind_perc_pred"]) + delta
        preds["wind_perc_pred_p10"] = np.clip(np.minimum(lo, preds["wind_perc_pred"]), 0, 100)
        preds["wind_perc_pred_p90"] = np.clip(np.maximum(hi, preds["wind_perc_pred"]), 0, 100)

    preds.to_parquet(OUTDIR / "latest.parquet", index=False)

if __name__ == "__main__":
    main()

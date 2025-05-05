#! /usr/bin/env python
# src/pipeline.py
from pathlib import Path
from src import etl_inference as etl, featurise, predict   # predict is new – see next bullet

if __name__ == "__main__":
    etl.main()          # pull data needed for INFERENCE
    featurise.main()    # build lags / calendars / etc. (uses output of etl.main)
    predict.main()      # load CatBoost model → write data/predictions/latest.parquet

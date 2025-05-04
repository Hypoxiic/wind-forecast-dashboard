#! /usr/bin/env python
# src/pipeline.py
from pathlib import Path
from src import etl, featurise, predict   # predict is new – see next bullet

if __name__ == "__main__":
    etl.main()          # pull ESO + Open‑Meteo
    featurise.main()    # build lags / calendars / etc.
    predict.main()      # load CatBoost model → write data/predictions/latest.parquet

"""Unit tests for featurise.engineer_features.

Runs against the user's project Python (catboost env); no external data needed —
all inputs are small synthetic frames.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import featurise  # noqa: E402


def _synthetic(days: int = 35) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly weather + 30-min CI wind %, fully overlapping."""
    # December window: the 48h lag warm-up drops the first 2 days, and the
    # Christmas / Boxing Day holidays land well inside the kept range.
    hours = pd.date_range("2023-12-01", periods=days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    met = pd.DataFrame({
        "datetime": hours,
        "temperature_2m": 10 + rng.normal(0, 2, len(hours)),
        "wind_speed_10m": np.abs(rng.normal(15, 6, len(hours))),
        "wind_speed_100m": np.abs(rng.normal(25, 9, len(hours))),
        "wind_gusts_10m": np.abs(rng.normal(30, 8, len(hours))),
        "wind_direction_10m": rng.uniform(0, 360, len(hours)),
        "surface_pressure": 1013 + rng.normal(0, 5, len(hours)),
    })
    ci = pd.DataFrame({
        "datetime": hours,  # already hourly for simplicity
        "wind_perc": np.clip(rng.normal(30, 10, len(hours)), 0, 100),
    })
    return ci, met


@pytest.fixture()
def features() -> pd.DataFrame:
    ci, met = _synthetic()
    return featurise.engineer_features(ci, met)


def test_lag_alignment(features: pd.DataFrame):
    """wind_perc_lag_24h at row t must equal wind_perc at t-24 (hourly data)."""
    df = features.reset_index(drop=True)
    i = 100
    assert df.loc[i, "wind_perc_lag_24h"] == pytest.approx(df.loc[i - 24, "wind_perc"])
    assert df.loc[i, "wind_perc_lag_48h"] == pytest.approx(df.loc[i - 48, "wind_perc"])
    assert df.loc[i, "wind_speed_lag_24h"] == pytest.approx(df.loc[i - 24, "wind_speed_10m"])


def test_holiday_flag_seeded(features: pd.DataFrame):
    """Christmas Day must be flagged — the holidays dict is seeded (a lazily
    populated UnitedKingdom() dict returns all-False, the old silent bug)."""
    xmas = features[features["datetime"].dt.date == pd.Timestamp("2023-12-25").date()]
    assert not xmas.empty
    assert (xmas["is_holiday"] == 1).all()
    # A random non-holiday must not be flagged.
    plain = features[features["datetime"].dt.date == pd.Timestamp("2023-12-12").date()]
    assert (plain["is_holiday"] == 0).all()


def test_power_curve_proxies(features: pd.DataFrame):
    df = features
    i = 50
    assert df.loc[i, "wind_speed_v3_10m"] == pytest.approx(df.loc[i, "wind_speed_10m"] ** 3)
    assert df.loc[i, "wind_speed_v3_100m"] == pytest.approx(df.loc[i, "wind_speed_100m"] ** 3)
    # Clipped variant saturates at rated speed (15 m/s → 3375)
    assert df["wind_speed_v3_clip_10m"].max() <= 15.0 ** 3 + 1e-9
    # Legacy aliases mirror the 10m columns
    assert df.loc[i, "wind_speed_v3"] == pytest.approx(df.loc[i, "wind_speed_v3_10m"])


def test_new_weather_features(features: pd.DataFrame):
    df = features
    assert "gust_factor" in df.columns and df["gust_factor"].notna().all()
    assert "pressure_delta_3h" in df.columns
    i = 50
    expected_dir = np.sin(np.deg2rad(df.loc[i, "wind_direction_10m"]))
    assert df.loc[i, "sin_wind_dir"] == pytest.approx(expected_dir)
    assert df.loc[i, "pressure_delta_3h"] == pytest.approx(
        df.loc[i, "surface_pressure"] - df.loc[i - 3, "surface_pressure"])


def test_no_future_leakage_in_rolls(features: pd.DataFrame):
    """Rolling means must only use rows up to and including the current one."""
    df = features.reset_index(drop=True)
    i = 100
    window = df.loc[i - 47:i, "wind_speed_10m"]  # 48 hourly rows, mislabeled "24h"
    assert df.loc[i, "wind_speed_roll_mean_24h"] == pytest.approx(window.mean())

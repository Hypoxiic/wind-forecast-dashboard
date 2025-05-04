"""
GB Wind Forecast Dashboard – May 2025
------------------------------------
• Light/dark Bootstrap theme toggle (Minty / Cyborg)
• KPI cards (Baseline vs CatBoost) with colour‑coded deltas
• Forecast & error‑distribution tabs
• Robust date / series filtering
• Unified typography across themes
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, exceptions
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO, load_figure_template
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------
ROOT           = Path(__file__).resolve().parents[1]
FEATURES_PATH  = ROOT / "data" / "features" / "features.parquet"
PRED_PATH = ROOT / "data" / "predictions" / "catboost_full.parquet"
METRICS_PATH   = ROOT / "metrics.json"

THEME_LIGHT    = "minty"   # template name
THEME_DARK     = "cyborg"
CSS_LIGHT      = dbc.themes.MINTY
CSS_DARK       = dbc.themes.CYBORG

# Hide the Dash debug‑inspector dropdown
os.environ["DASH_DEBUG_UI"] = "false"

# Force global typography consistency (16 px / Inter)
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
(ASSETS / "z_override.css").write_text(
    """
    html, body {font-size:16px !important; font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;}
    """,
    encoding="utf-8",
)

# -----------------------------------------------------------------------------
# Utilities for safe loading
# -----------------------------------------------------------------------------

def safe_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def load_parquet(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path, columns=cols)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df
    return pd.DataFrame(columns=cols)

# -----------------------------------------------------------------------------
# Metrics – handle legacy & new schema
# -----------------------------------------------------------------------------

raw_metrics = safe_json(METRICS_PATH)

# CatBoost (always present once training completed)
catboost_rmse = raw_metrics.get("holdout_rmse_mw", np.nan)
catboost_mape = raw_metrics.get("holdout_mape", np.nan)

# --- Baseline: compute from data if not stored --------------------------------
features_full = load_parquet(FEATURES_PATH, ["datetime", "wind_mw", "wind_mw_lag_48h"])

if raw_metrics.get("baseline_rmse_mw") is not None:
    baseline_rmse = raw_metrics.get("baseline_rmse_mw", np.nan)
    baseline_mape = raw_metrics.get("baseline_mape", np.nan)
else:
    # Compute on‑the‑fly for full feature set (quick)
    valid = features_full.dropna(subset=["wind_mw", "wind_mw_lag_48h"])
    if valid.empty:
        baseline_rmse = baseline_mape = np.nan
    else:
        mse_b = mean_squared_error(valid["wind_mw"], valid["wind_mw_lag_48h"])
        baseline_rmse = mse_b ** 0.5
        baseline_mape = mean_absolute_percentage_error(valid["wind_mw"], valid["wind_mw_lag_48h"])

# -----------------------------------------------------------------------------
# Prediction data (CatBoost hold‑out predictions)
# -----------------------------------------------------------------------------

preds_full = load_parquet(PRED_PATH, ["datetime", "wind_mw_pred"])

# Align to same test window (75 % train / 25 % test as in training)
if not features_full.empty:
    cut = int(len(features_full) * 0.75)
    test_feats = features_full.iloc[cut:].copy()
    min_d, max_d = (
        test_feats["datetime"].dt.date.min(),
        test_feats["datetime"].dt.date.max(),
    )
else:
    test_feats = features_full.copy()
    today = pd.Timestamp.utcnow().date()
    min_d = max_d = today

# -----------------------------------------------------------------------------
# KPI helper (colour + icon)
# -----------------------------------------------------------------------------

def kpi_colour_icon(value, base, lower_better=True):
    if np.isnan(value) or np.isnan(base):
        return "secondary", ""
    better = value < base if lower_better else value > base
    worse = value > base if lower_better else value < base
    if better:
        return "success", "↓" if lower_better else "↑"
    if worse:
        return "danger", "↑" if lower_better else "↓"
    return "warning", "="

# -----------------------------------------------------------------------------
# Dash app
# -----------------------------------------------------------------------------

load_figure_template(THEME_LIGHT)
app = Dash(__name__, external_stylesheets=[CSS_LIGHT])
app.title = "GB Wind Forecast Dashboard"

# ───── KPI cards ─────────────────────────────────────────────────────────────

rmse_col, rmse_icon = kpi_colour_icon(catboost_rmse, baseline_rmse)
mape_col, mape_icon = kpi_colour_icon(catboost_mape, baseline_mape)


def card(title: str, value: str, unit: str = "", colour="light", tooltip: str | None = None):
    cid = str(uuid.uuid4())
    body = html.H4(f"{value}{unit}", className="card-title")
    comp = dbc.Card(
        [dbc.CardHeader(title), dbc.CardBody(body)],
        id=cid,
        color=colour,
        inverse=(colour not in {"light", "secondary"}),
        className="shadow-sm",
        style={"height": "100px"},
    )
    if tooltip:
        return dbc.Col([comp, dbc.Tooltip(tooltip, target=cid, placement="top")])
    return dbc.Col(comp)

base_rmse = card("Baseline RMSE", f"{baseline_rmse:,.0f}", " MW")
base_mape = card("Baseline MAPE", f"{baseline_mape:.3f}")
cb_rmse = card(
    "CatBoost RMSE",
    f"{catboost_rmse:,.0f} MW {rmse_icon}",
    colour=rmse_col,
    tooltip=(
        f"Δ {(baseline_rmse - catboost_rmse) / baseline_rmse:+.0%}"
        if not np.isnan(baseline_rmse) and not np.isnan(catboost_rmse)
        else None
    ),
)
cb_mape = card(
    "CatBoost MAPE",
    f"{catboost_mape:.3f} {mape_icon}",
    colour=mape_col,
    tooltip=(
        f"Δ {(baseline_mape - catboost_mape) / baseline_mape:+.1%}"
        if not np.isnan(baseline_mape) and not np.isnan(catboost_mape)
        else None
    ),
)

kpis = dbc.Row([base_rmse, base_mape, cb_rmse, cb_mape], className="g-4 mb-3")

# ───── Controls ──────────────────────────────────────────────────────────────

series_dd = dcc.Dropdown(
    id="series",
    multi=True,
    value=["wind_mw", "wind_mw_pred"],
    options=[
        {"label": "Actual", "value": "wind_mw"},
        {"label": "Baseline (48 h lag)", "value": "wind_mw_lag_48h"},
        {"label": "CatBoost Prediction", "value": "wind_mw_pred"},
    ],
)

date_picker = dcc.DatePickerRange(
    id="date",
    min_date_allowed=min_d,
    max_date_allowed=max_d,
    start_date=min_d,
    end_date=max_d,
    display_format="DD/MM/YYYY",
    initial_visible_month=max_d,
)

controls = dbc.Card(
    dbc.CardBody(
        dbc.Row(
            [
                dbc.Col(html.Label("Series:"), width="auto"),
                dbc.Col(series_dd, lg=5),
                dbc.Col(html.Label("Date range:"), width="auto", className="pt-2"),
                dbc.Col(date_picker, lg=4),
            ],
            align="center",
        )
    ),
    className="mb-3 shadow-sm",
)

# ───── Layout ────────────────────────────────────────────────────────────────

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.H2("GB Wind Day‑Ahead Forecast", className="display-6"), width=10),
                dbc.Col(
                    ThemeSwitchAIO(
                        aio_id="theme",
                        themes=[CSS_LIGHT, CSS_DARK],
                        switch_props={"style": {"marginTop": "12px"}},
                    ),
                    width=2,
                ),
            ]
        ),
        kpis,
        controls,
        dbc.Tabs(
            [
                dbc.Tab(label="Forecast Plot", tab_id="forecast"),
                dbc.Tab(label="Error Distribution", tab_id="errors"),
            ],
            id="tabs",
            active_tab="forecast",
            className="mb-3",
        ),
        html.Div(id="tab-content"),
    ],
    fluid=False,
    style={"maxWidth": "1200px", "paddingTop": "18px"},
)

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


@app.callback(
    Output("tab-content", "children"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"),
    Input("tabs", "active_tab"),
    Input("series", "value"),
    Input("date", "start_date"),
    Input("date", "end_date"),
)
def render_tab(toggle, tab, series_sel, start_d, end_d):
    theme = THEME_LIGHT if toggle else THEME_DARK
    load_figure_template(theme)

    # date slice
    start_ts = pd.to_datetime(start_d).tz_localize("UTC")
    end_ts = pd.to_datetime(end_d).tz_localize("UTC")

    slice_feats = test_feats[(test_feats["datetime"] >= start_ts) & (test_feats["datetime"] <= end_ts)]
    if slice_feats.empty:
        return dbc.Alert("No data in selected range.", color="warning")

    df = slice_feats.merge(preds_full, on="datetime", how="left")

    # ----------------------------- Forecast tab -----------------------------
    if tab == "forecast":
        fig = go.Figure()
        if "wind_mw" in series_sel:
            fig.add_scatter(x=df["datetime"], y=df["wind_mw"], mode="lines", name="Actual")
        if "wind_mw_lag_48h" in series_sel and "wind_mw_lag_48h" in df:
            fig.add_scatter(
                x=df["datetime"],
                y=df["wind_mw_lag_48h"],
                mode="lines",
                name="Baseline (48 h lag)",
                line=dict(dash="dot"),
            )
        if "wind_mw_pred" in series_sel and "wind_mw_pred" in df:
            fig.add_scatter(
                x=df["datetime"],
                y=df["wind_mw_pred"],
                mode="lines",
                name="CatBoost Prediction",
                line=dict(color="orange"),
            )

        fig.update_layout(
            template=theme,
            title=f"{start_ts.date()} → {end_ts.date()}",
            legend=dict(
                orientation="h", y=1.05, x=0.5, xanchor="center", yanchor="bottom"
            ),
            xaxis_title="Date",
            yaxis_title="Wind generation (MW)",
        )
        return dcc.Graph(figure=fig)

    # -------------------------- Error distribution --------------------------
    if tab == "errors":
        if {"wind_mw", "wind_mw_pred"}.issubset(df.columns):
            resid = (df["wind_mw_pred"] - df["wind_mw"]).dropna()
        else:
            resid = pd.Series(dtype=float)

        if resid.empty:
            return dbc.Alert("No overlapping actual/prediction data.", color="warning")

        fig = px.histogram(
            resid,
            nbins=50,
            template=theme,
            title=f"Prediction error distribution ({start_ts.date()} – {end_ts.date()})",
            labels={"value": "Predicted − Actual (MW)"},
        )
        fig.update_layout(bargap=0.05, yaxis_title="Frequency", coloraxis_showscale=False)
        return dcc.Graph(figure=fig)

    return dbc.Alert("Unknown tab.", color="danger")


# -----------------------------------------------------------------------------
# Run server
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

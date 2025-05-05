"""
GB Wind Forecast Dashboard – May 2025
------------------------------------
• Light/dark Bootstrap theme toggle (Minty / Cyborg)
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
LATEST_PRED_PATH = ROOT / "data" / "predictions" / "latest.parquet"
METRICS_PATH   = ROOT / "metrics.json"

THEME_LIGHT    = "minty"   # template name
THEME_DARK     = "cyborg"
CSS_LIGHT      = dbc.themes.MINTY
CSS_DARK       = dbc.themes.CYBORG

# Hide the Dash debug‑inspector dropdown
os.environ["DASH_DEBUG_UI"] = "false"

# Force global typography consistency (16 px / Inter)
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
# Metrics – handle legacy & new schema
# -----------------------------------------------------------------------------

raw_metrics = safe_json(METRICS_PATH)

# CatBoost metrics (using new percentage keys)
catboost_rmse = raw_metrics.get("holdout_rmse_perc", np.nan)
catboost_mape = raw_metrics.get("holdout_mape_perc", np.nan)

# --- Baseline: compute from data if not stored --------------------------------
# Load features needed for baseline and actuals plot
features_df = load_parquet(FEATURES_PATH, ["datetime", "wind_perc", "wind_perc_lag_48h"])

if raw_metrics.get("baseline_rmse_perc") is not None: # Check for new baseline keys if added later
    baseline_rmse = raw_metrics.get("baseline_rmse_perc", np.nan)
    baseline_mape = raw_metrics.get("baseline_mape_perc", np.nan)
elif not features_df.empty:
    # Compute on‑the‑fly using percentage columns
    valid = features_df.dropna(subset=["wind_perc", "wind_perc_lag_48h"])
    if valid.empty:
        baseline_rmse = baseline_mape = np.nan
    else:
        mse_b = mean_squared_error(valid["wind_perc"], valid["wind_perc_lag_48h"])
        baseline_rmse = mse_b ** 0.5
        mape_b = mean_absolute_percentage_error(valid["wind_perc"], valid["wind_perc_lag_48h"])
        baseline_mape = mape_b # MAPE already a percentage
else:
    baseline_rmse = baseline_mape = np.nan

# -----------------------------------------------------------------------------
# Prediction data (Load latest predictions and merge with features)
# -----------------------------------------------------------------------------

latest_preds_df = load_parquet(LATEST_PRED_PATH, ["datetime", "wind_perc_pred"])

# Merge features and latest predictions
if not features_df.empty and not latest_preds_df.empty:
    # Use outer merge to keep all times, fill gaps if any (though should align)
    plot_data = pd.merge(features_df, latest_preds_df, on="datetime", how="outer")
    plot_data = plot_data.sort_values("datetime").reset_index(drop=True)
    # Define date range based on the available latest predictions
    min_d = plot_data["datetime"].dt.date.min()
    max_d = plot_data["datetime"].dt.date.max()
elif not features_df.empty:
    # Only features available
    plot_data = features_df.copy()
    min_d = plot_data["datetime"].dt.date.min()
    max_d = plot_data["datetime"].dt.date.max()
elif not latest_preds_df.empty:
    # Only predictions available
    plot_data = latest_preds_df.copy()
    min_d = plot_data["datetime"].dt.date.min()
    max_d = plot_data["datetime"].dt.date.max()
else:
    # No data available
    plot_data = pd.DataFrame(columns=["datetime", "wind_perc", "wind_perc_lag_48h", "wind_perc_pred"])
    today = pd.Timestamp.utcnow().date()
    min_d = max_d = today

# -----------------------------------------------------------------------------
# KPI helper (colour + icon)
# -----------------------------------------------------------------------------

# ───── Intro / explanation card ─────────────────────────────────────────────
intro = dbc.Card(
    dbc.CardBody(
        [
            html.P(
                "This dashboard forecasts the day‑ahead GB wind generation percentage with a "
                "GPU‑tuned CatBoost model. Carbon Intensity wind percentage is merged with "
                "Open‑Meteo hub‑height forecasts; power‑curve proxies, lags "
                "and seasonal features feed the model. Optuna tunes hyper‑"
                "parameters via five expanding walk‑forward CV splits.",
                className="mb-2",
            ),
            html.P(
                f"Hold‑out (last 24 h) error: RMSE ≈ {catboost_rmse:.2f}% • MAPE ≈ {catboost_mape*100:.1f}% "
                f"(≈ {(catboost_rmse - baseline_rmse) / baseline_rmse:+.0%} vs 48 h‑lag baseline RMSE).",
                className="fst-italic small mb-0",
            ),
        ]
    ),
    className="mb-4 shadow-sm border-0",
)


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
# Dash app
# -----------------------------------------------------------------------------

load_figure_template(THEME_LIGHT)
app = Dash(__name__, external_stylesheets=[CSS_LIGHT])
app.title = "GB Wind Forecast Dashboard"
server = app.server
# ───── KPI cards ─────────────────────────────────────────────────────────────

rmse_col, rmse_icon = kpi_colour_icon(catboost_rmse, baseline_rmse)
mape_col, mape_icon = kpi_colour_icon(catboost_mape, baseline_mape)


def card(title: str, value: str, unit: str = "%", colour="light", tooltip: str | None = None):
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

base_rmse = card("Baseline RMSE", f"{baseline_rmse:.2f}")
base_mape = card("Baseline MAPE", f"{baseline_mape*100:.1f}")
cb_rmse = card(
    "CatBoost RMSE",
    f"{catboost_rmse:.2f}% {rmse_icon}",
    colour=rmse_col,
    tooltip=(
        f"Δ {(baseline_rmse - catboost_rmse) / baseline_rmse:+.0%}"
        if not np.isnan(baseline_rmse) and not np.isnan(catboost_rmse) and baseline_rmse != 0
        else None
    ),
)
cb_mape = card(
    "CatBoost MAPE",
    f"{baseline_mape*100:.1f}% {mape_icon}",
    colour=mape_col,
    tooltip=(
        f"Δ {(baseline_mape - catboost_mape) / baseline_mape:+.1%}"
        if not np.isnan(baseline_mape) and not np.isnan(catboost_mape) and baseline_mape != 0
        else None
    ),
)

kpis = dbc.Row([base_rmse, base_mape, cb_rmse, cb_mape], className="g-4 mb-3")

# ───── Controls ──────────────────────────────────────────────────────────────

series_dd = dcc.Dropdown(
    id="series",
    multi=True,
    value=["wind_perc", "wind_perc_pred"],
    options=[
        {"label": "Actual (%)", "value": "wind_perc"},
        {"label": "Baseline (% lag 48h)", "value": "wind_perc_lag_48h"},
        {"label": "CatBoost Prediction (%)", "value": "wind_perc_pred"},
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
                dbc.Col(html.Label("Date range:"), width="auto", className="pt-2"),
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
                dbc.Col(html.H2("GB Wind Day-Ahead Forecast", className="display-6"), width=10),
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
        intro,
        kpis,
        controls,
        dbc.Tabs(
            [
                dbc.Tab(label="Forecast Plot", tab_id="forecast"),
                dbc.Tab(label="Error Distribution", tab_id="errors"),
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
    template = THEME_DARK if toggle else THEME_LIGHT

    if not series_sel:
        return html.Div("Please select at least one series.")

    try:
        # Filter data based on date picker
        mask = (
            (plot_data["datetime"].dt.date >= pd.to_datetime(start_d).date()) &
            (plot_data["datetime"].dt.date <= pd.to_datetime(end_d).date())
        )
        df_filtered = plot_data[mask]

        if df_filtered.empty:
             return html.Div("No data available for the selected date range.")

    except Exception as e:
        print(f"Error filtering data: {e}")
        return html.Div("Error processing data for the selected date range.")


    if tab == "forecast":
        fig = px.line(
            df_filtered,
            x="datetime",
            y=series_sel,
            template=template,
            labels={"value": "Wind Generation (% of Mix)", "variable": "Series"},
            title="GB Wind Generation Forecast vs Actual",
        )
        # Enhance layout
        fig.update_layout(
            yaxis_title="Wind Generation (% of Mix)",
            legend_title="Series",
            hovermode="x unified",
            margin=dict(t=50, b=30, l=30, r=30),
        )
        return dcc.Graph(figure=fig)

    elif tab == "errors":
        # Calculate errors only if prediction and actual are selected and available
        required_cols = ["wind_perc", "wind_perc_pred"]
        if not all(s in series_sel for s in required_cols) or not all(c in df_filtered.columns for c in required_cols):
            return html.Div("Please select both 'Actual (%)' and 'CatBoost Prediction (%)' to view errors.")

        df_err = df_filtered.dropna(subset=required_cols)
        if df_err.empty:
            return html.Div("No overlapping actual and prediction data for error calculation.")

        df_err["error_perc"] = df_err["wind_perc_pred"] - df_err["wind_perc"]

        fig = px.histogram(
            df_err,
            x="error_perc",
            template=template,
            marginal="box",
            title="Prediction Error Distribution (Prediction % - Actual %)",
            labels={"error_perc": "Error (Percentage Points)"},
        )
        # Enhance layout
        fig.update_layout(
            xaxis_title="Error (Percentage Points)",
            yaxis_title="Frequency",
            margin=dict(t=50, b=30, l=30, r=30),
        )
        return dcc.Graph(figure=fig)

    return html.P("This shouldn't happen...")


# -----------------------------------------------------------------------------
# Run server
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

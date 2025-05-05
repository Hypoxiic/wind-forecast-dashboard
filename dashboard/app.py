"""
dashboard/app.py

GB Wind Day-Ahead Forecast Dashboard
------------------------------------
• Light/dark theme toggle (simple_white / plotly_dark)
• KPI cards (Baseline vs CatBoost) with colour-coded deltas
• Forecast & error-distribution tabs
• Robust date / series filtering
• Includes tomorrow’s predictions via outer-merge
"""

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO, load_figure_template
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ─── Paths & Constants ───────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parents[1]
FEATURES_PATH    = ROOT / "data" / "features" / "features.parquet"
HISTORY_PATH     = ROOT / "data" / "features" / "history.parquet"
LATEST_PRED_PATH = ROOT / "data" / "predictions" / "latest.parquet"
METRICS_PATH     = ROOT / "metrics.json"

COLOR_MAP = {
    "wind_perc":        "#2ca02c",  # green for actual
    "wind_perc_lag_48h":"#1f77b4",  # blue for baseline
    "wind_perc_pred":   "#ff7f0e",  # orange for prediction
}

THEME_LIGHT = "simple_white"
THEME_DARK  = "plotly_dark"
CSS_LIGHT   = dbc.themes.MINTY
CSS_DARK    = dbc.themes.CYBORG

os.environ["DASH_DEBUG_UI"] = "false"

# ─── Utility Functions ───────────────────────────────────────────────────────
def safe_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def load_parquet(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path, columns=cols)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df
    return pd.DataFrame(columns=cols)

# ─── Load Metrics ────────────────────────────────────────────────────────────
raw_metrics = safe_json(METRICS_PATH)
cat_rmse    = raw_metrics.get("holdout_rmse_perc", np.nan)
cat_mape    = raw_metrics.get("holdout_mape_perc", np.nan)

# ─── Load & Compute Baseline ─────────────────────────────────────────────────
# Read rolling-history of features
hist_df = load_parquet(HISTORY_PATH, ["datetime", "wind_perc", "wind_perc_lag_48h"])
# If first run and history missing, fall back to original snapshot
if hist_df.empty and FEATURES_PATH.exists():
    hist_df = load_parquet(FEATURES_PATH, ["datetime", "wind_perc", "wind_perc_lag_48h"])

if not hist_df.empty:
    cutoff = hist_df.datetime.max() - pd.Timedelta(days=1)
    sub    = hist_df[hist_df.datetime >= cutoff].dropna(subset=["wind_perc","wind_perc_lag_48h"])
    if not sub.empty:
        mse_val        = mean_squared_error(sub.wind_perc, sub.wind_perc_lag_48h)
        baseline_rmse  = np.sqrt(mse_val)
        baseline_mape  = mean_absolute_percentage_error(sub.wind_perc, sub.wind_perc_lag_48h)
    else:
        baseline_rmse = baseline_mape = np.nan
else:
    baseline_rmse = baseline_mape = np.nan

# ─── Load Latest Predictions ─────────────────────────────────────────────────
preds_df = load_parquet(LATEST_PRED_PATH, ["datetime", "wind_perc_pred"])

# ─── Merge History + Tomorrow’s Forecast ────────────────────────────────────
plot_data = pd.merge(
    hist_df,
    preds_df,
    on="datetime",
    how="outer",
    sort=True,
)
if "wind_perc_pred" not in plot_data:
    plot_data["wind_perc_pred"] = np.nan

# Recalculate slider bounds
min_d = plot_data.datetime.dt.date.min()
max_d = plot_data.datetime.dt.date.max()

# ─── KPI Card Helpers ────────────────────────────────────────────────────────
def delta_colour(val, base, lower_better=True):
    if np.isnan(val) or np.isnan(base):
        return "secondary", ""
    better = (val < base) if lower_better else (val > base)
    worse  = (val > base) if lower_better else (val < base)
    if better: return "success", "↓" if lower_better else "↑"
    if worse:  return "danger",  "↑" if lower_better else "↓"
    return "warning", "="

def make_card(title, value, unit, colour, tooltip=None):
    cid = str(uuid.uuid4())
    body = dbc.CardBody(html.H4(f"{value}{unit}", className="card-title"))
    card = dbc.Card([dbc.CardHeader(title), body],
                    id=cid, color=colour,
                    inverse=(colour not in ["light","secondary"]),
                    className="shadow-sm", style={"height":"100px"})
    if tooltip:
        return dbc.Col([card, dbc.Tooltip(tooltip, target=cid, placement="top")])
    return dbc.Col(card)

rmse_col, rmse_ic = delta_colour(cat_rmse, baseline_rmse)
mape_col, mape_ic = delta_colour(cat_mape, baseline_mape)

card_baseline_rmse = make_card("Baseline RMSE",
    f"{baseline_rmse:.2f}", "%", "light")
card_baseline_mape = make_card("Baseline MAPE",
    f"{baseline_mape*100:.1f}", "%", "light")
card_cat_rmse = make_card("CatBoost RMSE",
    f"{cat_rmse:.2f}", f"% {rmse_ic}", rmse_col,
    tooltip=f"Δ {(baseline_rmse - cat_rmse)/baseline_rmse:+.0%}"
)
card_cat_mape = make_card("CatBoost MAPE",
    f"{cat_mape*100:.1f}", f"% {mape_ic}", mape_col,
    tooltip=f"Δ {(baseline_mape - cat_mape)/baseline_mape:+.1%}"
)

# ─── Build Dash App ─────────────────────────────────────────────────────────
load_figure_template(THEME_LIGHT)
app    = Dash(__name__, external_stylesheets=[CSS_LIGHT])
server = app.server
app.title = "GB Wind Day-Ahead Forecast"

# Controls
series_dd = dcc.Dropdown(
    id="series", multi=True,
    value=["wind_perc","wind_perc_pred"],
    options=[
        {"label":"Actual (%)",           "value":"wind_perc"},
        {"label":"Baseline (lag 48h %)", "value":"wind_perc_lag_48h"},
        {"label":"Prediction (%)",       "value":"wind_perc_pred"},
    ],
)
date_picker = dcc.DatePickerRange(
    id="date", min_date_allowed=min_d, max_date_allowed=max_d,
    start_date=min_d,   end_date=max_d,
    display_format="DD/MM/YYYY"
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("GB Wind Day-Ahead Forecast", className="display-6"), width=10),
        dbc.Col(ThemeSwitchAIO(aio_id="theme",
                               themes=[CSS_LIGHT, CSS_DARK],
                               switch_props={"style":{"marginTop":"12px"}}),
                width=2),
    ], align="center"),
    dbc.Row([card_baseline_rmse, card_baseline_mape, card_cat_rmse, card_cat_mape],
            className="g-4 mb-4"),
    dbc.Card(dbc.CardBody([
        html.P(
          "This dashboard forecasts the day-ahead GB wind generation percentage with a "
          "GPU-tuned CatBoost model. Carbon Intensity wind percentage is merged with "
          "Open-Meteo hub-height forecasts; power-curve proxies, lags and seasonal "
          "features feed the model. Optuna tunes hyper-parameters via five expanding "
          "walk-forward CV splits.",
          className="mb-2"
        ),
        html.P(
          f"Hold-out (last 24 h) error: RMSE ≈ {cat_rmse:.2f}% • "
          f"MAPE ≈ {cat_mape*100:.1f}% "
          f"(≈ {(cat_rmse-baseline_rmse)/baseline_rmse:+.0%} vs 48 h-lag).",
          className="fst-italic small mb-0"
        )
    ]), className="mb-4 border-0 shadow-sm"),
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col(html.Label("Series:"), width="auto"),
            dbc.Col(series_dd, width=5),
            dbc.Col(html.Label("Date range:"), width="auto", className="pt-2"),
            dbc.Col(date_picker, width=4),
        ], align="center")
    ]), className="mb-4 shadow-sm"),
    dbc.Tabs([
        dbc.Tab(label="Forecast Plot",   tab_id="forecast"),
        dbc.Tab(label="Error Distribution", tab_id="errors"),
    ], id="tabs", active_tab="forecast", className="mb-3"),
    html.Div(id="tab-content"),
], style={"maxWidth":"1200px","paddingTop":"18px"})

# ─── Callbacks ──────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-content","children"),
    Input(ThemeSwitchAIO.ids.switch("theme"),"value"),
    Input("tabs","active_tab"),
    Input("series","value"),
    Input("date","start_date"),
    Input("date","end_date"),
)
def render_tab(is_dark, tab, series_sel, start_d, end_d):
    template = THEME_DARK if is_dark else THEME_LIGHT
    light_bg = not is_dark

    # filter by date
    df = plot_data[
        (plot_data.datetime.dt.date >= pd.to_datetime(start_d).date()) &
        (plot_data.datetime.dt.date <= pd.to_datetime(end_d).date())
    ]
    if df.empty or not series_sel:
        return html.Div("No data available for that selection.")

    if tab == "forecast":
        fig = px.line(
            df, x="datetime", y=series_sel,
            template=template,
            color_discrete_map=COLOR_MAP,
            labels={"value":"Wind Gen. (% of mix)", "variable":"Series"}
        )
        fig.update_layout(margin={"t":50,"b":30,"l":30,"r":30})
        if light_bg:
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        return dcc.Graph(figure=fig)

    # Errors tab
    if "wind_perc" in series_sel and "wind_perc_pred" in series_sel:
        err = df.dropna(subset=["wind_perc","wind_perc_pred"])
        err["error"] = err.wind_perc_pred - err.wind_perc
        fig = px.histogram(
            err, x="error", template=template,
            marginal="box",
            color_discrete_sequence=[COLOR_MAP["wind_perc_pred"]],
            labels={"error":"Error (percentage points)"}
        )
        fig.update_layout(margin={"t":50,"b":30,"l":30,"r":30})
        if light_bg:
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        return dcc.Graph(figure=fig)

    return html.Div("Please select both Actual & Prediction to view errors.")

# ─── Run Server ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)

"""
dashboard/app.py

GB Wind Day-Ahead Forecast Dashboard
------------------------------------
• Light/dark theme toggle (simple_white / plotly_dark)
• KPI cards (Baseline vs CatBoost) with colour-coded deltas
• Forecast & error-distribution tabs
• Robust date / series filtering
• Includes tomorrow's predictions via outer-merge
"""

import json
import os
import uuid
from pathlib import Path
import logging
from datetime import date

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

# Calculate Baseline RMSE and MAPE based on the last 24h of available actuals in hist_df
actuals_df = hist_df.dropna(subset=["wind_perc"])
if not actuals_df.empty:
    latest_actual_date = actuals_df.datetime.max()
    # Ensure cutoff doesn't go before the start of hist_df if hist_df is very short
    cutoff_start = actuals_df.datetime.min()
    cutoff_end = latest_actual_date
    cutoff_calc_start = latest_actual_date - pd.Timedelta(days=1)
    
    # Define the period for baseline calculation (last 24h of actuals)
    sub = hist_df[
        (hist_df.datetime >= cutoff_calc_start) & 
        (hist_df.datetime <= cutoff_end)
    ].dropna(subset=["wind_perc", "wind_perc_lag_48h"])
    
    if not sub.empty and len(sub) > 1: # Need at least 2 points for error metrics
        mse_val        = mean_squared_error(sub.wind_perc, sub.wind_perc_lag_48h)
        baseline_rmse  = np.sqrt(mse_val)
        baseline_mape  = mean_absolute_percentage_error(sub.wind_perc, sub.wind_perc_lag_48h)
    else:
        logging.warning("Not enough data points to calculate baseline metrics after filtering.")
        baseline_rmse = baseline_mape = np.nan
else:
    logging.warning("No actual data (wind_perc) found in history_df to calculate baseline metrics.")
    baseline_rmse = baseline_mape = np.nan

# ─── Load Latest Predictions ─────────────────────────────────────────────────
preds_df = load_parquet(LATEST_PRED_PATH, ["datetime", "wind_perc_pred"])

# ─── Merge History + Tomorrow's Forecast ────────────────────────────────────
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

# Controls - Define them once, and place them in the layout
series_dd = dcc.Dropdown(
    id="series", multi=True,
    value=["wind_perc","wind_perc_pred"],
    options=[
        {"label":"Actual (%)",           "value":"wind_perc"},
        {"label":"Baseline (lag 48h %)", "value":"wind_perc_lag_48h"},
        {"label":"Prediction (%)",       "value":"wind_perc_pred"},
    ],
)

date_picker_global = dcc.DatePickerRange(
    id="date-picker-global", 
    min_date_allowed=min_d, 
    max_date_allowed=max_d,
    start_date=min_d,   
    end_date=max_d,
    display_format="DD/MM/YYYY",
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("GB Wind Day-Ahead Forecast", className="display-6"), width=10),
        dbc.Col(ThemeSwitchAIO(aio_id="theme",
                               themes=[CSS_LIGHT, CSS_DARK],
                               switch_props={"style":{"marginTop":"12px"}}),
                width=2),
    ], align="center"),
    
    html.Div(id="kpi-cards-row"), 

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
          f"Hold-out (last 24 h) error from training: RMSE ≈ {cat_rmse:.2f}% • "
          f"MAPE ≈ {cat_mape*100:.1f}%.",
          className="fst-italic small mb-0"
        )
    ]), className="mb-4 border-0 shadow-sm"),

    # Add controls card to the main layout
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col(html.Label("Series:"), width="auto"),
            dbc.Col(series_dd, width=5, className="me-3"),
            dbc.Col(html.Label("Date range:"), width="auto", className="pt-2"),
            dbc.Col(date_picker_global, width=4),
        ], align="center")
    ]), className="mb-4 shadow-sm"),
    
    dbc.Tabs([
        dbc.Tab(label="Forecast & Recent", tab_id="forecast_recent", className="fw-bold"),
        dbc.Tab(label="Historical Analysis", tab_id="historical", className="fw-bold"),
    ], id="tabs", active_tab="forecast_recent", className="mb-3"),
    
    html.Div(id="tab-content"),
    
], fluid=True, className="dbc dbc-row-selectable", style={"maxWidth":"1400px","paddingTop":"18px"})

# ─── Callbacks ──────────────────────────────────────────────────────────────
@app.callback(
    [Output("tab-content","children"),
     Output("kpi-cards-row", "children")],
    [Input(ThemeSwitchAIO.ids.switch("theme"),"value"),
     Input("tabs","active_tab"),
     Input("series","value"), 
     Input("date-picker-global", "start_date"), # Use the global date picker ID
     Input("date-picker-global", "end_date")]   # Use the global date picker ID
)
def render_content(is_dark, active_tab, series_sel, start_d_global, end_d_global):
    template = THEME_DARK if is_dark else THEME_LIGHT
    light_bg = not is_dark
    kpi_cards_content = []
    tab_specific_content = []

    if active_tab == "forecast_recent":
        # --- KPI Cards for Forecast & Recent Tab ---
        card_baseline_rmse_fc = make_card("Baseline RMSE (Recent Actuals)", f"{baseline_rmse:.2f}" if not np.isnan(baseline_rmse) else "N/A", "%", "light")
        card_baseline_mape_fc = make_card("Baseline MAPE (Recent Actuals)", f"{baseline_mape*100:.1f}" if not np.isnan(baseline_mape) else "N/A", "%", "light")
        cat_rmse_col, cat_rmse_ic = delta_colour(cat_rmse, baseline_rmse)
        cat_mape_col, cat_mape_ic = delta_colour(cat_mape, baseline_mape)
        card_cat_rmse_fc = make_card("Model RMSE (Holdout)", f"{cat_rmse:.2f}", f"% {cat_rmse_ic}", cat_rmse_col, tooltip=f"Model holdout vs recent baseline Δ {(cat_rmse-baseline_rmse)/baseline_rmse:+.0%}" if not (np.isnan(cat_rmse) or np.isnan(baseline_rmse)) else "Model holdout from training")
        card_cat_mape_fc = make_card("Model MAPE (Holdout)", f"{cat_mape*100:.1f}", f"% {cat_mape_ic}", cat_mape_col, tooltip=f"Model holdout vs recent baseline Δ {(cat_mape-baseline_mape)/baseline_mape:+.1%}" if not (np.isnan(cat_mape) or np.isnan(baseline_mape)) else "Model holdout from training")
        kpi_cards_content = [dbc.Row([card_baseline_rmse_fc, card_baseline_mape_fc, card_cat_rmse_fc, card_cat_mape_fc], className="g-4 mb-4")]

        # --- Content for Forecast & Recent Tab ---
        # Define date range for this tab: e.g., last 3 days of history + 2 days of forecast
        current_date = date.today() # Use imported date
        forecast_plot_start_date = current_date - pd.Timedelta(days=3) # Show last 3 days of history
        forecast_plot_end_date = current_date + pd.Timedelta(days=2)   # Show 2 days of forecast

        df_forecast_tab = plot_data[
            (plot_data.datetime.dt.date >= forecast_plot_start_date) &
            (plot_data.datetime.dt.date <= forecast_plot_end_date)
        ]
        
        # Controls are now global, no need to redefine them here for this tab explicitly
        # We might hide/show the global date_picker or adjust its range for this tab later if needed.

        if df_forecast_tab.empty or not series_sel:
            fig_forecast_content = html.Div("No data available for the forecast window.")
        else:
            fig_fc = px.line(
                df_forecast_tab, x="datetime", y=series_sel,
                template=template, color_discrete_map=COLOR_MAP,
                labels={"value":"Wind Gen. (% of mix)", "variable":"Series"}
            )
            fig_fc.update_layout(margin={"t":30,"b":30,"l":30,"r":30}, title_text="Forecast & Recent Data (Last 3 Days History + 2 Days Forecast)", title_x=0.5)
            if light_bg: fig_fc.update_layout(paper_bgcolor="white", plot_bgcolor="white")
            fig_forecast_content = dcc.Graph(figure=fig_fc)
        
        tab_specific_content = [fig_forecast_content] # Just the graph for this tab now

    elif active_tab == "historical":
        # --- KPI Cards for Historical Analysis Tab (Placeholders) ---
        kpi_cards_content = [dbc.Row([
            dbc.Col(dbc.Alert("KPIs for selected historical range (dynamic) - Coming soon!", color="info"))
        ], className="g-4 mb-4")]
        
        # --- Content for Historical Analysis Tab ---
        # Controls are global. The global date_picker (date-picker-global) drives this tab.
        df_hist_tab = plot_data[
            (plot_data.datetime.dt.date >= pd.to_datetime(start_d_global).date()) &
            (plot_data.datetime.dt.date <= pd.to_datetime(end_d_global).date())
        ]

        if df_hist_tab.empty or not series_sel:
            fig_historical_content = html.Div("No data available for that selection.")
        else:
            fig_hist = px.line(
                df_hist_tab, x="datetime", y=series_sel,
                template=template, color_discrete_map=COLOR_MAP,
                labels={"value":"Wind Gen. (% of mix)", "variable":"Series"}
            )
            fig_hist.update_layout(margin={"t":30,"b":30,"l":30,"r":30}, title_text="Historical Data Analysis", title_x=0.5)
            if light_bg: fig_hist.update_layout(paper_bgcolor="white", plot_bgcolor="white")
            fig_historical_content = dcc.Graph(figure=fig_hist)

        error_plot_placeholder = dbc.Alert("Error distribution plot for selected range - Coming soon!", color="light", className="mt-3")
        tab_specific_content = [fig_historical_content, error_plot_placeholder] # Graph and placeholder

    return tab_specific_content, kpi_cards_content

# ─── Run Server ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)

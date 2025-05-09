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
from dash import Dash, Input, Output, dcc, html, callback_context, no_update
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO, load_figure_template
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# --- Setup dedicated file logger for dashboard debugging ---
# Get the root logger
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s") # This might conflict if set globally elsewhere, or can be primary

# Create a specific logger for this dashboard module
dash_logger = logging.getLogger("dashboard_app")
dash_logger.setLevel(logging.INFO)
# Create a file handler
log_file_path = Path(__file__).parent / "dashboard_debug.log"
try:
    # Attempt to remove old log file to start fresh each run, if desired
    if log_file_path.exists():
        os.remove(log_file_path)
except OSError as e:
    print(f"Warning: Could not remove old log file {log_file_path}: {e}")

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
# Create a logging format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
# Add the handlers to the logger
if not dash_logger.handlers: # Avoid adding multiple handlers on hot reloads
    dash_logger.addHandler(file_handler)
dash_logger.info("Dashboard logger initialized. Logging to: dashboard_debug.log")
# --- End of logger setup ---

# ─── Paths & Constants ───────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parents[1]
FEATURES_PATH    = ROOT / "data" / "features" / "features.parquet"
HISTORY_PATH     = ROOT / "data" / "features" / "history.parquet"
LATEST_PRED_PATH = ROOT / "data" / "predictions" / "latest.parquet"
FULL_HIST_PREDS_PATH = ROOT / "models" / "catboost_full.parquet"
METRICS_PATH     = ROOT / "metrics.json"

COLOR_MAP = {
    "wind_perc":        "#1E88E5",  # blue for actual (previously green)
    "wind_perc_lag_48h":"#1f77b4",  # blue for baseline
    "wind_perc_pred":   "#FF8F00",  # amber for prediction (previously orange)
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

# ─── Load & Compute Baseline & Prepare Plot Data ───────────────────────────
# Read rolling-history of features (actuals and baseline lags)
hist_df = load_parquet(HISTORY_PATH, ["datetime", "wind_perc", "wind_perc_lag_48h"])
# If first run and history missing, fall back to original snapshot from features.parquet (less likely now)
if hist_df.empty and FEATURES_PATH.exists():
    logging.warning("history.parquet is empty, attempting to fallback to features.parquet for hist_df")
    hist_df = load_parquet(FEATURES_PATH, ["datetime", "wind_perc", "wind_perc_lag_48h"])

# Load predictions
latest_preds_df = load_parquet(LATEST_PRED_PATH, ["datetime", "wind_perc_pred"])
full_hist_preds_df = load_parquet(FULL_HIST_PREDS_PATH, ["datetime", "wind_perc_pred"])

# Consolidate predictions
if not full_hist_preds_df.empty:
    all_preds_df = full_hist_preds_df.copy()
    if not latest_preds_df.empty:
        # Ensure latest_preds_df takes precedence by removing its datetime range from all_preds_df first
        all_preds_df = pd.concat([
            all_preds_df[~all_preds_df['datetime'].isin(latest_preds_df['datetime'])],
            latest_preds_df
        ]).sort_values("datetime").reset_index(drop=True)
elif not latest_preds_df.empty:
    all_preds_df = latest_preds_df.copy()
else:
    all_preds_df = pd.DataFrame(columns=["datetime", "wind_perc_pred"])
    all_preds_df['datetime'] = pd.to_datetime(all_preds_df['datetime'])

# Merge with hist_df (actuals and baseline)
if not hist_df.empty:
    plot_data = pd.merge(
        hist_df,
        all_preds_df,
        on="datetime",
        how="outer",
        sort=True,
    )
else: # Fallback if hist_df is critically empty
    logging.warning("history.parquet was empty, plot_data will be based on predictions only.")
    plot_data = all_preds_df.copy()
    # Ensure essential columns for plotting/metrics if hist_df was missing
    if "wind_perc" not in plot_data: plot_data["wind_perc"] = np.nan
    if "wind_perc_lag_48h" not in plot_data: plot_data["wind_perc_lag_48h"] = np.nan

# Final checks for essential columns in plot_data
if "datetime" not in plot_data:
    plot_data["datetime"] = pd.to_datetime([]) # Create empty datetime series if totally missing
if "wind_perc_pred" not in plot_data:
    plot_data["wind_perc_pred"] = np.nan
if "wind_perc" not in plot_data:
    plot_data["wind_perc"] = np.nan
if "wind_perc_lag_48h" not in plot_data:
    plot_data["wind_perc_lag_48h"] = np.nan

# Calculate Baseline RMSE and MAPE (using the now comprehensive plot_data for actuals source)
# but hist_df is still the more reliable source for this specific calculation if available
# Let's use the existing baseline logic that depends on hist_df for stability
# (The baseline calculation was already refined to use actuals_df from hist_df)
actuals_for_baseline_calc = hist_df.dropna(subset=["wind_perc"])
if not actuals_for_baseline_calc.empty:
    latest_actual_date = actuals_for_baseline_calc.datetime.max()
    cutoff_calc_start = latest_actual_date - pd.Timedelta(days=1)
    sub = hist_df[
        (hist_df.datetime >= cutoff_calc_start) & 
        (hist_df.datetime <= latest_actual_date)
    ].dropna(subset=["wind_perc", "wind_perc_lag_48h"])
    if not sub.empty and len(sub) > 1:
        mse_val        = mean_squared_error(sub.wind_perc, sub.wind_perc_lag_48h)
        baseline_rmse  = np.sqrt(mse_val)
        baseline_mape  = mean_absolute_percentage_error(sub.wind_perc, sub.wind_perc_lag_48h)
    else:
        baseline_rmse = baseline_mape = np.nan
else:
    baseline_rmse = baseline_mape = np.nan

# Recalculate slider bounds from the final plot_data
if not plot_data.empty and not plot_data['datetime'].dropna().empty:
    min_d = plot_data['datetime'].dropna().dt.date.min()
    max_d = plot_data['datetime'].dropna().dt.date.max()
else:
    min_d = date.today() - pd.Timedelta(days=30)
    max_d = date.today() + pd.Timedelta(days=2)

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
    
    # Custom styles for the card
    card_style = {"height":"100px", "padding": "12px", "border-radius": "8px", "box-shadow": "0 2px 4px rgba(0,0,0,0.1)"}
    
    # Add green border for all cards except metric cards (baseline and model)
    if not (title.startswith("Baseline") or title.startswith("Model")):
        card_style["border"] = "2px solid #2E7D32"
    else:
        card_style["border"] = "none"  # Remove border for metric cards
    
    # Revert back to standard colors for cards, but keep the borders consistent
    inverse = (colour not in ["light", "secondary"])
    
    card = dbc.Card([dbc.CardHeader(title), body],
                    id=cid, color=colour,
                    inverse=inverse,
                    className="shadow-sm", style=card_style)
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
    style={"width": "300px"}
)

date_picker_global = dcc.DatePickerRange(
    id="date-picker-global", 
    min_date_allowed=min_d, 
    max_date_allowed=max_d,
    start_date=min_d,   
    end_date=max_d,
    display_format="DD/MM/YYYY",
)

# Add quick date filter buttons
def create_date_preset_buttons():
    today = pd.Timestamp.today().date()
    yesterday = today - pd.Timedelta(days=1)
    last_7_days = today - pd.Timedelta(days=7)
    last_30_days = today - pd.Timedelta(days=30)
    
    return html.Div([
        dbc.Button("Last 24h", id="btn-last-24h", color="primary", size="sm", className="me-2"),
        dbc.Button("Last 7d", id="btn-last-7d", color="secondary", size="sm", className="me-2"),
        dbc.Button("Last 30d", id="btn-last-30d", color="secondary", size="sm", className="me-2"),
        dbc.Button("All Time", id="btn-all-time", color="secondary", size="sm"),
    ], className="mb-3")

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("GB Wind Day-Ahead Forecast", className="display-6"), width=8),
        dbc.Col([
            dbc.Button("Export Data", id="btn-export-main", color="secondary", size="sm", className="me-2"),
            ThemeSwitchAIO(aio_id="theme",
                          themes=[CSS_LIGHT, CSS_DARK],
                          switch_props={"style":{"marginTop":"12px"}})
        ], width=4, className="text-end"),
    ], align="center", className="mb-3"),
    
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
    ]), className="mb-4 border-0 shadow-sm", style={"border": "2px solid #2E7D32", "padding": "12px", "border-radius": "8px", "box-shadow": "0 2px 4px rgba(0,0,0,0.1)"}),

    # Add date preset buttons
    create_date_preset_buttons(),
    
    # Combine tabs and series selector in one row
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="Forecast & Recent", tab_id="forecast_recent", className="fw-bold"),
                dbc.Tab(label="Historical Analysis", tab_id="historical", className="fw-bold"),
            ], id="tabs", active_tab="forecast_recent", className="mb-0"),
        ], width=8),
        dbc.Col([
            html.Label("Series:", className="me-2", style={"display": "inline-block"}),
            html.Div(series_dd, style={"display": "inline-block", "vertical-align": "middle"}),
        ], width=4, className="text-end"),
    ], className="align-items-center mb-3"),
    
    # Move date range selector to a more intuitive design
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Label("Date Range:", className="pt-2"), width="auto"),
                dbc.Col(date_picker_global, width=4),
                dbc.Col(html.Div([
                    html.Span("Last updated: ", className="text-muted"),
                    html.Span(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
                ]), width=4, className="text-end")
            ], align="center")
        ])
    ], className="mb-4", style={"border": "2px solid #2E7D32", "padding": "12px", "border-radius": "8px", "box-shadow": "0 2px 4px rgba(0,0,0,0.1)"}),
    
    html.Div(id="tab-content"),
    
], fluid=True, className="dbc dbc-row-selectable", style={"maxWidth":"1400px", "paddingTop":"18px"})

# ─── Callbacks ──────────────────────────────────────────────────────────────
@app.callback(
    [Output("tab-content","children"),
     Output("kpi-cards-row", "children")],
    [Input(ThemeSwitchAIO.ids.switch("theme"),"value"),
     Input("tabs","active_tab"),
     Input("series","value"), 
     Input("date-picker-global", "start_date"),
     Input("date-picker-global", "end_date")]
)
def render_content(is_dark, active_tab, series_sel, start_d_global, end_d_global):
    dash_logger.info(f"--- render_content CALLED: active_tab={active_tab}, series_sel={series_sel}, start_d={start_d_global}, end_d={end_d_global} ---")
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
            dash_logger.info("Forecast tab: No data for forecast window or no series selected.")
            fig_forecast_content = html.Div("No data available for the forecast window.")
        else:
            fig_fc = px.line(
                df_forecast_tab, x="datetime", y=series_sel,
                template=template, color_discrete_map=COLOR_MAP,
                labels={"value":"Wind Gen. (% of mix)", "variable":"Series"}
            )
            
            # Chart enhancements
            for trace in fig_fc.data:
                # Increase line thickness to 2.5px
                trace.line.width = 2.5
                # Add subtle point markers every 6 hours
                trace.mode = 'lines+markers'
                trace.marker.size = 8
                trace.marker.opacity = 0.8
                # Make markers more visible
                trace.marker.line = dict(width=1, color='white')
                trace.marker.symbol = "circle"
                
                # Remove area fill - use lines only as requested
                trace.fill = None
                
                # If this is the actual wind data (blue line), make it more prominent
                if trace.name == "wind_perc":
                    trace.line.width = 3.0
                    trace.marker.size = 9
                    trace.zorder = 10  # Try to keep it on top
            
            # Add light gray vertical lines at midnight transitions
            for day in pd.date_range(start=df_forecast_tab.datetime.min().floor('D'), 
                                    end=df_forecast_tab.datetime.max().ceil('D'), 
                                    freq='D'):
                fig_fc.add_vline(x=day, line_width=1, line_dash="dot", 
                                line_color="rgba(150, 150, 150, 0.3)")
            
            # Layout enhancements
            fig_fc.update_layout(
                margin={"t":30,"b":30,"l":30,"r":30}, 
                title_text="Forecast & Recent Data (Last 3 Days History + 2 Days Forecast)", 
                title_x=0.5
            )
            
            # Add dark mode gradient overlay if in dark mode
            if not light_bg:
                fig_fc.update_layout(
                    paper_bgcolor="rgb(13,18,30)",
                    plot_bgcolor="rgb(13,18,30)",
                    # Improve grid visibility for better readability
                    xaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.1)",
                        zerolinecolor="rgba(255, 255, 255, 0.1)"
                    ),
                    yaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.1)",
                        zerolinecolor="rgba(255, 255, 255, 0.1)"
                    )
                )
            else:
                fig_fc.update_layout(
                    paper_bgcolor="white", 
                    plot_bgcolor="white",
                    # Improve grid visibility for better readability
                    xaxis=dict(
                        gridcolor="rgba(0, 0, 0, 0.1)",
                        zerolinecolor="rgba(0, 0, 0, 0.1)"
                    ),
                    yaxis=dict(
                        gridcolor="rgba(0, 0, 0, 0.1)",
                        zerolinecolor="rgba(0, 0, 0, 0.1)"
                    )
                )
                
            dash_logger.info(f"Forecast tab: Plotting {len(df_forecast_tab)} rows.")
            
            # Create error panel showing difference between prediction and actual
            error_df = df_forecast_tab.copy()
            error_df["prediction_error"] = error_df["wind_perc_pred"] - error_df["wind_perc"]
            error_df = error_df.dropna(subset=["prediction_error"])
            
            if not error_df.empty:
                # Get the min and max dates from the main chart for proper alignment
                x_min = df_forecast_tab.datetime.min()
                x_max = df_forecast_tab.datetime.max()
                
                fig_error = px.line(
                    error_df, x="datetime", y="prediction_error",
                    template=template,
                    labels={"prediction_error": "Prediction Error (%)", "datetime": ""}
                )
                
                # Style the error chart
                fig_error.update_traces(
                    line_color="#1E88E5", 
                    line_width=2,
                    # Use the same color family as main chart with very subtle fill
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(30, 136, 229, 0.05)'
                )
                
                # Explicitly set the same x-axis range
                fig_error.update_layout(
                    height=150, 
                    margin={"t":10,"b":30,"l":30,"r":30},
                    showlegend=False,
                    # Ensure identical time range as main chart
                    xaxis=dict(
                        range=[x_min, x_max],
                        gridcolor="rgba(255, 255, 255, 0.1)" if not light_bg else "rgba(0, 0, 0, 0.1)",
                        # Match the main chart's tick format
                        tickformat="%H:%M\n%b %d",
                        # Reduce number of ticks to match main chart
                        nticks=10
                    ),
                    yaxis=dict(
                        title="Error (%)",
                        gridcolor="rgba(255, 255, 255, 0.1)" if not light_bg else "rgba(0, 0, 0, 0.1)"
                    )
                )
                
                # Add a horizontal zero line
                fig_error.add_hline(
                    y=0, 
                    line_color="rgba(255, 255, 255, 0.4)" if not light_bg else "rgba(0, 0, 0, 0.4)", 
                    line_width=1
                )
                
                # Format the error chart to better align with main chart
                fig_fc.update_layout(
                    xaxis=dict(
                        tickformat="%H:%M\n%b %d",
                        nticks=10
                    )
                )
                
                error_chart = dcc.Graph(figure=fig_error)
            else:
                error_chart = html.Div("No error data available", className="text-center py-3")
            
            # Create a small table showing numeric values at 6-hour intervals
            interval_df = df_forecast_tab[df_forecast_tab.datetime.dt.hour % 6 == 0].copy()
            
            if not interval_df.empty:
                interval_df = interval_df.sort_values("datetime", ascending=False).head(8)
                
                # Format the data for display
                interval_df["datetime"] = interval_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
                interval_df["wind_perc"] = interval_df["wind_perc"].round(1)
                interval_df["wind_perc_pred"] = interval_df["wind_perc_pred"].round(1)
                
                # Create table columns based on available data
                columns = [{"name": "Timestamp", "id": "datetime"}]
                data = []
                
                for idx, row in interval_df.iterrows():
                    data_row = {"datetime": row["datetime"]}
                    
                    if not np.isnan(row["wind_perc"]):
                        data_row["actual"] = f"{row['wind_perc']}%"
                        if "actual" not in [c["id"] for c in columns]:
                            columns.append({"name": "Actual", "id": "actual"})
                    
                    if not np.isnan(row["wind_perc_pred"]):
                        data_row["predicted"] = f"{row['wind_perc_pred']}%"
                        if "predicted" not in [c["id"] for c in columns]:
                            columns.append({"name": "Predicted", "id": "predicted"})
                    
                    data.append(data_row)
                
                values_table = dbc.Table.from_dataframe(
                    pd.DataFrame(data),
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                    className="mt-3"
                )
            else:
                values_table = html.Div("No interval data available", className="text-center py-2")
            
            # Determine wind trend
            if not df_forecast_tab.empty and "wind_perc" in df_forecast_tab.columns:
                latest_values = df_forecast_tab.dropna(subset=["wind_perc"]).tail(12)
                if len(latest_values) >= 2:
                    first_half = latest_values.head(len(latest_values) // 2)["wind_perc"].mean()
                    second_half = latest_values.tail(len(latest_values) // 2)["wind_perc"].mean()
                    trend_direction = "↑" if second_half > first_half else "↓"
                    trend_color = "success" if second_half > first_half else "danger"
                    trend_indicator = dbc.Alert(
                        f"Current Wind Trend: {trend_direction} {round(abs(second_half - first_half), 1)}%",
                        color=trend_color,
                        className="mt-3 py-2",
                        style={"width": "fit-content"}
                    )
                else:
                    trend_indicator = None
            else:
                trend_indicator = None
            
            fig_forecast_content = html.Div([
                # Add last updated timestamp
                html.Div([
                    html.Span("Last updated: ", className="text-muted"),
                    html.Span(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
                ], className="mb-3 text-end"),
                
                # Main chart with export button
                html.Div([
                    dbc.Button(
                        "Export Data", 
                        color="secondary", 
                        size="sm", 
                        className="mb-2 float-end",
                        id="btn-export-data"
                    ),
                    dcc.Graph(figure=fig_fc),
                ]),
                
                # Error chart
                error_chart,
                
                # Wind trend indicator alongside interval data table
                dbc.Row([
                    dbc.Col(trend_indicator, width="auto") if trend_indicator else None,
                    dbc.Col(values_table, width=12 if not trend_indicator else None),
                ], className="mt-3"),
                
                # Legend explaining the downward arrows in metrics
                dbc.Alert(
                    [
                        html.H6("Metric Indicators", className="alert-heading mb-2"),
                        html.P("↓ indicates improvement (lower error is better)", className="mb-0 small")
                    ],
                    color="light",
                    className="mt-3"
                )
            ])
        
        tab_specific_content = [fig_forecast_content] # Just the graph for this tab now

    elif active_tab == "historical":
        # Filter data for the selected range
        df_hist_tab = plot_data[
            (plot_data.datetime.dt.date >= pd.to_datetime(start_d_global).date()) &
            (plot_data.datetime.dt.date <= pd.to_datetime(end_d_global).date())
        ]

        # --- Calculate Dynamic KPIs for Historical Analysis Tab ---
        dyn_cat_rmse, dyn_cat_mape = np.nan, np.nan
        dyn_baseline_rmse, dyn_baseline_mape = np.nan, np.nan

        # CatBoost model dynamic metrics
        df_eval_model = df_hist_tab.dropna(subset=["wind_perc", "wind_perc_pred"])
        if len(df_eval_model) > 1:
            dyn_cat_rmse = np.sqrt(mean_squared_error(df_eval_model.wind_perc, df_eval_model.wind_perc_pred))
            dyn_cat_mape = mean_absolute_percentage_error(df_eval_model.wind_perc, df_eval_model.wind_perc_pred)

        # Baseline dynamic metrics
        df_eval_baseline = df_hist_tab.dropna(subset=["wind_perc", "wind_perc_lag_48h"])
        if len(df_eval_baseline) > 1:
            dyn_baseline_rmse = np.sqrt(mean_squared_error(df_eval_baseline.wind_perc, df_eval_baseline.wind_perc_lag_48h))
            dyn_baseline_mape = mean_absolute_percentage_error(df_eval_baseline.wind_perc, df_eval_baseline.wind_perc_lag_48h)
        
        # --- KPI Cards for Historical Analysis Tab (Dynamic) ---
        dyn_cat_rmse_col, dyn_cat_rmse_ic = delta_colour(dyn_cat_rmse, dyn_baseline_rmse)
        dyn_cat_mape_col, dyn_cat_mape_ic = delta_colour(dyn_cat_mape, dyn_baseline_mape)

        card_dyn_baseline_rmse = make_card("Baseline RMSE (Selected Range)", f"{dyn_baseline_rmse:.2f}" if not np.isnan(dyn_baseline_rmse) else "N/A", "%", "light")
        card_dyn_baseline_mape = make_card("Baseline MAPE (Selected Range)", f"{dyn_baseline_mape*100:.1f}" if not np.isnan(dyn_baseline_mape) else "N/A", "%", "light")
        card_dyn_cat_rmse = make_card("Model RMSE (Selected Range)", f"{dyn_cat_rmse:.2f}" if not np.isnan(dyn_cat_rmse) else "N/A", f"% {dyn_cat_rmse_ic}", dyn_cat_rmse_col, tooltip=f"Model vs Baseline Δ {(dyn_cat_rmse-dyn_baseline_rmse)/dyn_baseline_rmse:+.0%}" if not (np.isnan(dyn_cat_rmse) or np.isnan(dyn_baseline_rmse)) else "N/A")
        card_dyn_cat_mape = make_card("Model MAPE (Selected Range)", f"{dyn_cat_mape*100:.1f}" if not np.isnan(dyn_cat_mape) else "N/A", f"% {dyn_cat_mape_ic}", dyn_cat_mape_col, tooltip=f"Model vs Baseline Δ {(dyn_cat_mape-dyn_baseline_mape)/dyn_baseline_mape:+.1%}" if not (np.isnan(dyn_cat_mape) or np.isnan(dyn_baseline_mape)) else "N/A")
        kpi_cards_content = [dbc.Row([card_dyn_baseline_rmse, card_dyn_baseline_mape, card_dyn_cat_rmse, card_dyn_cat_mape], className="g-4 mb-4")]
        
        # --- Content for Historical Analysis Tab (Plot) ---
        # (Keep diagnostic logging as is for now, can be removed later)
        if not df_hist_tab.empty:
            dash_logger.info(f"Historical tab: df_hist_tab from {start_d_global} to {end_d_global} (example slice):\n" \
                         f"{df_hist_tab[(df_hist_tab.datetime >= pd.Timestamp('2019-01-01', tz='UTC')) & (df_hist_tab.datetime <= pd.Timestamp('2019-01-03', tz='UTC'))][['datetime', 'wind_perc', 'wind_perc_pred']].to_string()}")
            if series_sel:
                 dash_logger.info(f"NaN counts in selected df_hist_tab for series {series_sel}:\n{df_hist_tab[series_sel if isinstance(series_sel, list) else [series_sel]].isnull().sum().to_string()}")
            else:
                dash_logger.info("Historical tab: No series selected.")
        else:
            dash_logger.info("Historical tab: df_hist_tab is empty for the selected range.")

        if df_hist_tab.empty or not series_sel:
            fig_historical_content = html.Div("No data available for that selection.")
        else:
            fig_hist = px.line(
                df_hist_tab, x="datetime", y=series_sel,
                template=template, color_discrete_map=COLOR_MAP,
                labels={"value":"Wind Gen. (% of mix)", "variable":"Series"}
            )
            
            # Chart enhancements
            for trace in fig_hist.data:
                # Increase line thickness to 2.5px
                trace.line.width = 2.5
                # Add subtle point markers every 6 hours
                trace.mode = 'lines+markers'
                trace.marker.size = 8
                trace.marker.opacity = 0.8
                # Make markers more visible
                trace.marker.line = dict(width=1, color='white')
                trace.marker.symbol = "circle"
                
                # Remove area fill - use lines only as requested
                trace.fill = None
                
                # If this is the actual wind data (blue line), make it more prominent
                if trace.name == "wind_perc":
                    trace.line.width = 3.0
                    trace.marker.size = 9
                    trace.zorder = 10  # Try to keep it on top
            
            # Add light gray vertical lines at midnight transitions
            for day in pd.date_range(start=df_hist_tab.datetime.min().floor('D'), 
                                    end=df_hist_tab.datetime.max().ceil('D'), 
                                    freq='D'):
                fig_hist.add_vline(x=day, line_width=1, line_dash="dot", 
                                line_color="rgba(150, 150, 150, 0.3)")
            
            # Layout enhancements
            fig_hist.update_layout(
                margin={"t":30,"b":30,"l":30,"r":30}, 
                title_text="Historical Data Analysis", 
                title_x=0.5
            )
            
            # Add dark mode gradient overlay if in dark mode
            if not light_bg:
                fig_hist.update_layout(
                    paper_bgcolor="rgb(13,18,30)",
                    plot_bgcolor="rgb(13,18,30)",
                    # Improve grid visibility for better readability
                    xaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.1)",
                        zerolinecolor="rgba(255, 255, 255, 0.1)"
                    ),
                    yaxis=dict(
                        gridcolor="rgba(255, 255, 255, 0.1)",
                        zerolinecolor="rgba(255, 255, 255, 0.1)"
                    )
                )
            else:
                fig_hist.update_layout(
                    paper_bgcolor="white", 
                    plot_bgcolor="white",
                    # Improve grid visibility for better readability
                    xaxis=dict(
                        gridcolor="rgba(0, 0, 0, 0.1)",
                        zerolinecolor="rgba(0, 0, 0, 0.1)"
                    ),
                    yaxis=dict(
                        gridcolor="rgba(0, 0, 0, 0.1)",
                        zerolinecolor="rgba(0, 0, 0, 0.1)"
                    )
                )
                
            # Create error panel showing difference between prediction and actual
            error_df = df_hist_tab.copy()
            error_df["prediction_error"] = error_df["wind_perc_pred"] - error_df["wind_perc"]
            error_df = error_df.dropna(subset=["prediction_error"])
            
            if not error_df.empty:
                # Get the min and max dates from the main chart for proper alignment
                x_min = df_hist_tab.datetime.min()
                x_max = df_hist_tab.datetime.max()
                
                fig_error = px.line(
                    error_df, x="datetime", y="prediction_error",
                    template=template,
                    labels={"prediction_error": "Prediction Error (%)", "datetime": ""}
                )
                
                # Style the error chart
                fig_error.update_traces(
                    line_color="#1E88E5", 
                    line_width=2,
                    # Use the same color family as main chart with very subtle fill
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(30, 136, 229, 0.05)'
                )
                
                # Explicitly set the same x-axis range
                fig_error.update_layout(
                    height=150, 
                    margin={"t":10,"b":30,"l":30,"r":30},
                    showlegend=False,
                    # Ensure identical time range as main chart
                    xaxis=dict(
                        range=[x_min, x_max],
                        gridcolor="rgba(255, 255, 255, 0.1)" if not light_bg else "rgba(0, 0, 0, 0.1)",
                        # Match the main chart's tick format
                        tickformat="%H:%M\n%b %d",
                        # Reduce number of ticks to match main chart
                        nticks=10
                    ),
                    yaxis=dict(
                        title="Error (%)",
                        gridcolor="rgba(255, 255, 255, 0.1)" if not light_bg else "rgba(0, 0, 0, 0.1)"
                    )
                )
                
                # Add a horizontal zero line
                fig_error.add_hline(
                    y=0, 
                    line_color="rgba(255, 255, 255, 0.4)" if not light_bg else "rgba(0, 0, 0, 0.4)", 
                    line_width=1
                )
                
                # Format the error chart to better align with main chart
                fig_hist.update_layout(
                    xaxis=dict(
                        tickformat="%H:%M\n%b %d",
                        nticks=10
                    )
                )
                
                error_content = dcc.Graph(figure=fig_error)
            else:
                error_content = dbc.Alert("No error data available for the selected range", color="light")
            
            # Create a small table showing numeric values at 6-hour intervals (sample only)
            sample_interval = max(1, len(df_hist_tab) // 8)  # Get up to 8 sample points
            interval_df = df_hist_tab.iloc[::sample_interval].head(8).copy()
            
            if not interval_df.empty:
                # Format the data for display
                interval_df["datetime"] = interval_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
                interval_df["wind_perc"] = interval_df["wind_perc"].round(1)
                interval_df["wind_perc_pred"] = interval_df["wind_perc_pred"].round(1)
                
                values_table = dbc.Table.from_dataframe(
                    interval_df[["datetime", "wind_perc", "wind_perc_pred"]].rename(
                        columns={"datetime": "Timestamp", "wind_perc": "Actual (%)", "wind_perc_pred": "Predicted (%)"}
                    ),
                    striped=True, 
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size="sm",
                    className="mt-3",
                )
            else:
                values_table = html.Div("No interval data available", className="text-center py-2")
            
            # Main page content
            fig_historical_content = html.Div([
                # Add last updated timestamp and export button
                html.Div([
                    html.Span("Last updated: ", className="text-muted"),
                    html.Span(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")),
                    dbc.Button(
                        "Export Data", 
                        color="secondary", 
                        size="sm", 
                        className="float-end",
                        id="btn-export-hist-data"
                    ),
                ], className="mb-3 d-flex justify-content-between"),
                
                # Main chart
                dcc.Graph(figure=fig_hist),
                
                # Error content (either chart or message)
                error_content,
                
                # Sample data table
                html.Div([
                    html.H6("Sample Data Points", className="mt-3"),
                    values_table
                ])
            ])

        # Replace the placeholder error plot with our new error panel
        tab_specific_content = [fig_historical_content]

    return tab_specific_content, kpi_cards_content

# Add callbacks for date preset buttons
@app.callback(
    [Output("date-picker-global", "start_date"),
     Output("date-picker-global", "end_date")],
    [Input("btn-last-24h", "n_clicks"),
     Input("btn-last-7d", "n_clicks"),
     Input("btn-last-30d", "n_clicks"),
     Input("btn-all-time", "n_clicks")],
    prevent_initial_call=True
)
def update_date_range(last_24h, last_7d, last_30d, all_time):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    today = pd.Timestamp.today().date()
    
    if button_id == "btn-last-24h":
        return today - pd.Timedelta(days=1), today
    elif button_id == "btn-last-7d":
        return today - pd.Timedelta(days=7), today
    elif button_id == "btn-last-30d":
        return today - pd.Timedelta(days=30), today
    elif button_id == "btn-all-time":
        return min_d, max_d
    
    return no_update, no_update

# ─── Run Server ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)

# Dash 3.0.4 bug: debug graph "top‑down" / "left‑right" labels swapped
import json
import pandas as pd
import numpy as np
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, State, ctx, exceptions
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO, load_figure_template

# --- Constants & Theme Definitions ---
PROJECT_ROOT = Path(__file__).parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "features.parquet"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "catboost_test.parquet"
METRICS_PATH = PROJECT_ROOT / "metrics.json"

# Themes for ThemeSwitchAIO
template_theme1 = "minty" # Light theme
template_theme2 = "cyborg" # Dark theme
url_theme1 = dbc.themes.MINTY
url_theme2 = dbc.themes.CYBORG

# --- Load Data Outside Callbacks (Read-only) ---
try:
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    baseline = metrics.get("baseline", {})
    catboost = metrics.get("catboost", {})
except FileNotFoundError:
    baseline = {"rmse": np.nan, "mape": np.nan}
    catboost = {"rmse": np.nan, "mape": np.nan}

try:
    features_df = pd.read_parquet(FEATURES_PATH)
    features_df["datetime"] = pd.to_datetime(features_df["datetime"])
    # Determine split date for test set identification
    split_fraction = 0.75
    split_index = int(len(features_df) * split_fraction)
    split_date = features_df['datetime'].iloc[split_index]
    min_date = features_df['datetime'].min().date()
    max_date = features_df['datetime'].max().date()

    # Separate test features now
    test_features_df = features_df[features_df["datetime"] >= split_date].copy()

except FileNotFoundError:
    test_features_df = pd.DataFrame() # Handle case where features are missing
    min_date = pd.Timestamp('today').date() - pd.Timedelta(days=30) # Default range if no data
    max_date = pd.Timestamp('today').date()


try:
    preds_df = pd.read_parquet(PREDICTIONS_PATH)
    preds_df["datetime"] = pd.to_datetime(preds_df["datetime"])
    # Merge predictions onto test features
    plot_data_df = pd.merge(test_features_df, preds_df, on="datetime", how="left")
except FileNotFoundError:
    plot_data_df = test_features_df.copy()
    plot_data_df['wind_mw_pred'] = np.nan # Ensure column exists


# --- Helper Functions for Styling ---
def get_kpi_style(metric_value, baseline_value, lower_is_better=True):
    style = {} # Basic style handled by dbc.Card
    if pd.isna(metric_value) or pd.isna(baseline_value):
         return "secondary", "" # Return color string for dbc.Card

    if lower_is_better:
        if metric_value < baseline_value:
            color = "success"
            icon = "↓"
        elif metric_value > baseline_value:
            color = "danger"
            icon = "↑"
        else:
             color = "warning"
             icon = "="
    else: # Higher is better
         if metric_value > baseline_value:
            color = "success"
            icon = "↑"
         elif metric_value < baseline_value:
            color = "danger"
            icon = "↓"
         else:
             color = "warning"
             icon = "="
    return color, icon


# --- App Initialization ---
# Load default light template for figures
load_figure_template(template_theme1)
app = Dash(__name__, external_stylesheets=[url_theme1]) # Start with light theme URL
app.title = "GB Wind Forecast Dashboard"

# --- Prepare KPI Card Styles ---
rmse_color_cb, rmse_icon_cb = get_kpi_style(catboost.get('rmse',np.nan), baseline.get('rmse',np.nan))
mape_color_cb, mape_icon_cb = get_kpi_style(catboost.get('mape',np.nan), baseline.get('mape',np.nan))
kpi_card_style = {"height": "120px"} # Uniform height

# --- Calculate Relative Improvements --- 
rmse_baseline_val = baseline.get('rmse', np.nan)
rmse_catboost_val = catboost.get('rmse', np.nan)
mape_baseline_val = baseline.get('mape', np.nan)
mape_catboost_val = catboost.get('mape', np.nan)

rmse_delta_text = "N/A"
if not pd.isna(rmse_baseline_val) and rmse_baseline_val != 0 and not pd.isna(rmse_catboost_val):
    rmse_improvement = (rmse_baseline_val - rmse_catboost_val) / rmse_baseline_val
    rmse_delta_text = f"Δ RMSE {rmse_improvement:+.0%} vs Baseline"

mape_delta_text = "N/A"
if not pd.isna(mape_baseline_val) and mape_baseline_val != 0 and not pd.isna(mape_catboost_val):
     mape_improvement = (mape_baseline_val - mape_catboost_val) / mape_baseline_val
     mape_delta_text = f"Δ MAPE {mape_improvement:+.1%} vs Baseline"


# --- Reusable Components ---
header = html.Div([ # Use Div instead of H1 for easier styling control
    html.H2("GB Wind Day-Ahead Forecast", className="display-6"), # Smaller header
    html.Hr()
], style={'textAlign': 'left'})

kpi_cards = dbc.Row( # Use Row with gutter class for spacing
    [
        dbc.Col(dbc.Card([
            dbc.CardHeader("Baseline RMSE"),
            # Add comma formatting to RMSE
            dbc.CardBody(html.H4(f"{baseline.get('rmse', 0):,.0f} MW", className="card-title"))
        ], className="shadow-sm", style=kpi_card_style), lg=3, md=6), # Responsive widths
         dbc.Col(dbc.Card([
            dbc.CardHeader("Baseline MAPE"),
             # Ensure 3 decimal places for MAPE
            dbc.CardBody(html.H4(f"{baseline.get('mape', 0):.3f}", className="card-title"))
        ], className="shadow-sm", style=kpi_card_style), lg=3, md=6),
        dbc.Col([ # Wrap Card in Div to attach Tooltip
            dbc.Card([
                dbc.CardHeader("CatBoost RMSE"),
                dbc.CardBody(html.H4(f"{catboost.get('rmse', 0):,.0f} MW {rmse_icon_cb}", className="card-title")) # Comma format
            ], id="catboost-rmse-card", color=rmse_color_cb, inverse=True, className="shadow", style=kpi_card_style),
            dbc.Tooltip(rmse_delta_text, target="catboost-rmse-card", placement="top")
        ], lg=3, md=6),
        dbc.Col([ # Wrap Card in Div to attach Tooltip
             dbc.Card([
                dbc.CardHeader("CatBoost MAPE"),
                dbc.CardBody(html.H4(f"{catboost.get('mape', 0):.3f} {mape_icon_cb}", className="card-title")) # Ensure 3 decimals
            ], id="catboost-mape-card", color=mape_color_cb, inverse=True, className="shadow", style=kpi_card_style),
             dbc.Tooltip(mape_delta_text, target="catboost-mape-card", placement="top")
        ], lg=3, md=6),
    ],
    className="g-4 mb-4", # Gutter spacing, margin bottom
)

controls = dbc.Card( # Group controls in a Card
    dbc.CardBody([
        dbc.Row([
            dbc.Col(html.Label("Select Series:"), width="auto"),
            dbc.Col(
                dcc.Dropdown(
                    id='series-selector',
                    options=[
                        {'label': 'Actual', 'value': 'wind_mw'},
                        {'label': 'Baseline (48h Lag)', 'value': 'wind_mw_lag_48h'},
                        {'label': 'CatBoost Prediction', 'value': 'wind_mw_pred'}
                    ],
                    value=['wind_mw', 'wind_mw_pred'], # Default selection
                    multi=True
                ), lg=6, md=8 # Control width
            ),
             # Date Range Slider (using test data range)
            dbc.Col(html.Label("Date Range:"), width="auto", className="pt-2"),
             dbc.Col(
                dcc.DatePickerRange(
                    id='date-picker-range',
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    initial_visible_month=max_date,
                    start_date=min_date, # Default to full range
                    end_date=max_date,
                    className="dbc" # Apply dbc styling
                ), lg=4, md=12 # Control width
            )
        ], align="center")
    ]), className="mb-4 shadow-sm"
)

tabs = dbc.Tabs(
    [
        dbc.Tab(label="Forecast Plot", tab_id="tab-forecast"),
        dbc.Tab(label="Error Distribution", tab_id="tab-errors"),
    ],
    id="tabs",
    active_tab="tab-forecast",
    className="mb-3",
)

# --- App Layout ---
app.layout = dbc.Container(
    [
        dbc.Row([
            dbc.Col(header, width=10),
            dbc.Col(ThemeSwitchAIO(aio_id="theme", themes=[url_theme1, url_theme2], switch_props={"style": {"marginTop": "15px"}}), width=2, align="center")
        ]),
        kpi_cards,
        controls,
        tabs,
        html.Div(id="tab-content"),
        # Hidden store for graph template persistence (optional but good practice)
        dcc.Store(id="graph-template-store", data=template_theme1) 
    ],
    fluid=False, # Fixed width
    className="dbc", # Apply dbc styling to container children
    style={"maxWidth": "1200px", "paddingTop": "20px"}
)


# --- Callbacks ---

# Callback to switch themes and graph templates
@app.callback(
    Output("tab-content", "children"), # Regenerate content on theme switch too
    Output("graph-template-store", "data"), # Store template name
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"),
    Input("tabs", "active_tab"),
    Input("series-selector", "value"),
    Input("date-picker-range", "start_date"),
    Input("date-picker-range", "end_date"),
)
def update_layout_and_tab_content(toggle, active_tab, selected_series, start_date, end_date):
    template = template_theme1 if toggle else template_theme2
    load_figure_template(template) # Update plotly template

    # ---- convert picker strings to Timestamps ----
    start_ts = pd.to_datetime(start_date)
    end_ts   = pd.to_datetime(end_date)
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise exceptions.PreventUpdate

    date_filtered_data = plot_data_df[
        (plot_data_df['datetime'] >= start_ts) &
        (plot_data_df['datetime'] <= end_ts)
    ]

    # Render active tab content
    if active_tab == "tab-forecast":
        # --- Create Forecast Plot ---
        if not selected_series or date_filtered_data.empty:
             fig_forecast = go.Figure().update_layout(
                 title="Please select series and ensure date range has data",
                 template=template # Use selected template
             )
        else:
            fig_forecast = go.Figure()
            title_parts = []
            # Add traces based on selected_series
            if 'wind_mw' in selected_series:
                fig_forecast.add_trace(go.Scatter(x=date_filtered_data["datetime"], y=date_filtered_data["wind_mw"],
                                         mode='lines', name='Actual'))
                title_parts.append("Actual")
            if 'wind_mw_lag_48h' in selected_series:
                 fig_forecast.add_trace(go.Scatter(x=date_filtered_data["datetime"], y=date_filtered_data["wind_mw_lag_48h"],
                                          mode='lines', name='Baseline (48h Lag)', line=dict(dash='dot')))
                 title_parts.append("Baseline")
            if 'wind_mw_pred' in selected_series and 'wind_mw_pred' in date_filtered_data.columns:
                fig_forecast.add_trace(go.Scatter(x=date_filtered_data["datetime"], y=date_filtered_data["wind_mw_pred"],
                                         mode='lines', name='CatBoost Prediction', line=dict(color='orange')))
                title_parts.append("CatBoost")

            fig_forecast.update_layout(
                title=" vs. ".join(title_parts) + f" ({start_date} to {end_date})",
                xaxis_title="Date",
                yaxis_title="Wind Generation (MW)",
                template=template, # Use selected template
                legend_title_text="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5) # Center legend above plot
            )
        tab_content = dcc.Graph(figure=fig_forecast)

    elif active_tab == "tab-errors":
        # --- Create Error Distribution Plot ---
        # Calculate residuals only on filtered data
        residuals = date_filtered_data['wind_mw_pred'] - date_filtered_data['wind_mw']
        residuals = residuals.dropna() # Drop NaNs if predictions or actuals are missing

        if residuals.empty:
            fig_errors = go.Figure().update_layout(title="No data available for error distribution", template=template)
        else:
            hist_color = "#2ecc71"
            fig_errors = px.histogram(
                residuals, nbins=50, title="Distribution of CatBoost Prediction Errors (MW)", template=template,
                color_discrete_sequence=[hist_color]
            )
            fig_errors.update_layout(
                xaxis_title="Prediction Error (Predicted - Actual) MW",
                yaxis_title="Frequency",
                bargap=0.1
            )

        tab_content = dcc.Graph(figure=fig_errors)
    else:
        tab_content = html.Div("Invalid tab selected")

    return tab_content, template # Return standardized variable


# --- Main Execution ---
if __name__ == "__main__":
    app.run(debug=True) 
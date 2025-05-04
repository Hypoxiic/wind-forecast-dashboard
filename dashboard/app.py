import json, pandas as pd
from pathlib import Path
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import numpy as np

# --- Load metrics ---
metrics_path = Path(__file__).parents[1] / "metrics.json"
project_root = Path(__file__).parents[1]
features_path = project_root / "data" / "features" / "features.parquet"
predictions_path = project_root / "data" / "predictions" / "catboost_test.parquet"

try:
    with open(metrics_path) as f:
        metrics = json.load(f)
    baseline = metrics["baseline"]
    catboost = metrics["catboost"]
except FileNotFoundError:
    # Provide default values if metrics file not found yet
    baseline = {"rmse": float('nan'), "mape": float('nan')}
    catboost = {"rmse": float('nan'), "mape": float('nan')}

# --- Helper Functions for Styling ---
def get_kpi_style(metric_value, baseline_value, lower_is_better=True):
    style = {"padding": "10px", "borderRadius": "5px", "color": "white"}
    if pd.isna(metric_value) or pd.isna(baseline_value):
         style["backgroundColor"] = "grey"
         return style, ""

    if lower_is_better:
        if metric_value < baseline_value:
            style["backgroundColor"] = "green"
            icon = "↓"
        elif metric_value > baseline_value:
            style["backgroundColor"] = "red"
            icon = "↑"
        else:
             style["backgroundColor"] = "orange"
             icon = "="
    else: # Higher is better (not used here, but for completeness)
         if metric_value > baseline_value:
            style["backgroundColor"] = "green"
            icon = "↑"
         elif metric_value < baseline_value:
            style["backgroundColor"] = "red"
            icon = "↓"
         else:
             style["backgroundColor"] = "orange"
             icon = "="
    return style, icon

# --- App ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "GB Wind Forecast Dashboard"

rmse_style_cb, rmse_icon_cb = get_kpi_style(catboost['rmse'], baseline['rmse'], lower_is_better=True)
mape_style_cb, mape_icon_cb = get_kpi_style(catboost['mape'], baseline['mape'], lower_is_better=True)

app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H1("GB Wind Day-Ahead Forecast"), width=12)),
        # --- KPIs ---
        dbc.Row(
            [
                dbc.Col( # Baseline RMSE
                    dbc.Card([
                        dbc.CardHeader("Baseline RMSE"),
                        dbc.CardBody(html.H4(f"{baseline['rmse']:.0f} MW", className="card-title"))
                    ]), width=3
                ),
                 dbc.Col( # Baseline MAPE
                    dbc.Card([
                        dbc.CardHeader("Baseline MAPE"),
                        dbc.CardBody(html.H4(f"{baseline['mape']:.3f}", className="card-title"))
                    ]), width=3
                ),
                dbc.Col( # CatBoost RMSE
                    dbc.Card([
                        dbc.CardHeader("CatBoost RMSE"),
                        dbc.CardBody(html.H4(f"{catboost['rmse']:.0f} MW {rmse_icon_cb}", className="card-title"))
                    ], color=rmse_style_cb["backgroundColor"], inverse=True), width=3 # Use Card color
                ),
                dbc.Col( # CatBoost MAPE
                     dbc.Card([
                        dbc.CardHeader("CatBoost MAPE"),
                        dbc.CardBody(html.H4(f"{catboost['mape']:.3f} {mape_icon_cb}", className="card-title"))
                    ], color=mape_style_cb["backgroundColor"], inverse=True), width=3 # Use Card color
                ),
            ],
            className="mb-4", # Margin bottom
        ),
        html.Hr(),
        # --- Controls ---
        dbc.Row(
            [
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
                    ),
                ),
                # TODO: Add Date Range Slider?
            ],
            align="center",
            className="mb-3",
        ),
        # --- Tabs for Plots ---
        dbc.Tabs(
            [
                dbc.Tab(label="Forecast Plot", tab_id="tab-forecast"),
                dbc.Tab(label="Error Distribution", tab_id="tab-errors"),
            ],
            id="tabs",
            active_tab="tab-forecast",
            className="mb-3",
        ),
        html.Div(id="tab-content") # Content will be rendered by callback
    ],
    fluid=False, # Use fixed-width container
    style={"maxWidth": "1200px"} # Set max width
)

# --- Callback to Render Tab Content ---
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("series-selector", "value") # Pass series selector value
)
def render_tab_content(active_tab, selected_series):
    # --- Load Data --- (Load needed data within the callback)
    try:
        feat = pd.read_parquet(features_path)
        feat["datetime"] = pd.to_datetime(feat["datetime"])
        # Determine test set split consistently
        split_fraction = 0.75
        split_index = int(len(feat) * split_fraction)
        split_date = feat['datetime'].iloc[split_index]
        test_features = feat[feat["datetime"] >= split_date].copy()
    except FileNotFoundError:
        return dbc.Alert("Error: Features file not found.", color="danger")

    try:
        preds = pd.read_parquet(predictions_path)
        preds["datetime"] = pd.to_datetime(preds["datetime"])
        plot_data = pd.merge(test_features, preds, on="datetime", how="left")
    except FileNotFoundError:
        plot_data = test_features.copy() # Use features if preds not found
        plot_data['wind_mw_pred'] = np.nan # Add empty pred column


    if active_tab == "tab-forecast":
        # --- Create Forecast Plot ---
        if not selected_series:
            fig_forecast = go.Figure().update_layout(title="Please select series to display")
        else:
            fig_forecast = go.Figure()
            title_parts = []
            if 'wind_mw' in selected_series:
                fig_forecast.add_trace(go.Scatter(x=plot_data["datetime"], y=plot_data["wind_mw"],
                                         mode='lines', name='Actual'))
                title_parts.append("Actual")
            if 'wind_mw_lag_48h' in selected_series:
                 fig_forecast.add_trace(go.Scatter(x=plot_data["datetime"], y=plot_data["wind_mw_lag_48h"],
                                          mode='lines', name='Baseline (48h Lag)', line=dict(dash='dot')))
                 title_parts.append("Baseline")
            if 'wind_mw_pred' in selected_series and 'wind_mw_pred' in plot_data.columns:
                fig_forecast.add_trace(go.Scatter(x=plot_data["datetime"], y=plot_data["wind_mw_pred"],
                                         mode='lines', name='CatBoost Prediction', line=dict(color='orange')))
                title_parts.append("CatBoost")

            fig_forecast.update_layout(
                title=" vs. ".join(title_parts) + " (Test Set)",
                xaxis_title="Date",
                yaxis_title="Wind Generation (MW)",
                template="plotly_white",
                legend_title_text=""
            )
        return dcc.Graph(figure=fig_forecast)

    elif active_tab == "tab-errors":
        # --- Create Error Distribution Plot ---
        if 'wind_mw_pred' not in plot_data.columns or plot_data['wind_mw_pred'].isna().all():
             return dbc.Alert("CatBoost predictions not available to calculate errors.", color="warning")

        residuals = plot_data["wind_mw"] - plot_data["wind_mw_pred"]
        fig_errors = px.histogram(
            residuals,
            nbins=50,
            title="Distribution of CatBoost Prediction Errors (Actual - Predicted)",
            template="plotly_white"
        )
        fig_errors.update_layout(xaxis_title="Error (MW)", yaxis_title="Frequency")
        return dcc.Graph(figure=fig_errors)

    return html.P("This shouldn't be displayed") # Fallback

if __name__ == "__main__":
    app.run(debug=True) 
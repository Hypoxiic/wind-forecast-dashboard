import json, pandas as pd
from pathlib import Path
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

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
app = Dash(__name__)
app.title = "GB Wind Forecast Dashboard"

rmse_style_cb, rmse_icon_cb = get_kpi_style(catboost['rmse'], baseline['rmse'], lower_is_better=True)
mape_style_cb, mape_icon_cb = get_kpi_style(catboost['mape'], baseline['mape'], lower_is_better=True)

app.layout = html.Div(
    style={"fontFamily": "Arial", "maxWidth": "1000px", "margin": "0 auto"},
    children=[
        html.H1("GB Wind Day-Ahead Forecast"),
        # --- KPIs ---
        html.Div(
            style={"display": "flex", "gap": "1rem", "justifyContent": "space-around"},
            children=[
                 # Baseline KPIs (neutral style)
                html.Div([
                    html.H3("Baseline RMSE"),
                    html.H2(f"{baseline['rmse']:.0f} MW")
                ], style={"textAlign": "center"}),
                 html.Div([
                    html.H3("Baseline MAPE"),
                    html.H2(f"{baseline['mape']:.3f}")
                ], style={"textAlign": "center"}),
                 # CatBoost KPIs (styled)
                 html.Div([
                    html.H3("CatBoost RMSE"),
                    html.H2(f"{catboost['rmse']:.0f} MW {rmse_icon_cb}")
                 ], style={**rmse_style_cb, "textAlign": "center"} ),
                html.Div([
                    html.H3("CatBoost MAPE"),
                    html.H2(f"{catboost['mape']:.3f} {mape_icon_cb}")
                ], style={**mape_style_cb, "textAlign": "center"}),
            ],
        ),
        html.Hr(),
        # --- Plot Controls ---
        html.Div([
            html.Label("Select Series:", style={"marginRight": "10px"}),
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
        ], style={"marginBottom": "20px"}),
        # --- Plot ---
        dcc.Graph(id="forecast-plot")
    ]
)

# --- Callback to load data and update plot ---
@app.callback(
    Output('forecast-plot', 'figure'), # Correct Output target
    Input('series-selector', 'value')  # Input from dropdown
)
def update_plot(selected_series):
    if not selected_series: # Handle empty selection
        return go.Figure().update_layout(title="Please select series to display")

    # --- Load Data ---
    try:
        feat = pd.read_parquet(features_path)
        feat["datetime"] = pd.to_datetime(feat["datetime"]) # Ensure datetime
    except FileNotFoundError:
        return go.Figure().update_layout(title="Error: Features file not found.")

    try:
        preds = pd.read_parquet(predictions_path)
        preds["datetime"] = pd.to_datetime(preds["datetime"]) # Ensure datetime
    except FileNotFoundError:
         # Allow plot without predictions if file missing
        preds = pd.DataFrame({'datetime': [], 'wind_mw_pred': []})
        preds["datetime"] = pd.to_datetime(preds["datetime"])

    # --- Prepare Data for Plotting ---
    # Determine test set split consistently
    split_fraction = 0.75
    split_index = int(len(feat) * split_fraction)
    split_date = feat['datetime'].iloc[split_index]
    test_features = feat[feat["datetime"] >= split_date].copy()

    # Merge predictions onto the test feature set
    plot_data = pd.merge(test_features, preds, on="datetime", how="left")

    # --- Create Plot ---
    fig = go.Figure()
    title_parts = []

    if 'wind_mw' in selected_series:
        fig.add_trace(go.Scatter(x=plot_data["datetime"], y=plot_data["wind_mw"],
                                 mode='lines', name='Actual'))
        title_parts.append("Actual")

    if 'wind_mw_lag_48h' in selected_series:
         fig.add_trace(go.Scatter(x=plot_data["datetime"], y=plot_data["wind_mw_lag_48h"],
                                  mode='lines', name='Baseline (48h Lag)', line=dict(dash='dot')))
         title_parts.append("Baseline")

    if 'wind_mw_pred' in selected_series and 'wind_mw_pred' in plot_data.columns:
        fig.add_trace(go.Scatter(x=plot_data["datetime"], y=plot_data["wind_mw_pred"],
                                 mode='lines', name='CatBoost Prediction', line=dict(color='orange')))
        title_parts.append("CatBoost")

    fig.update_layout(
        title=" vs. ".join(title_parts) + " (Test Set)",
        xaxis_title="Date",
        yaxis_title="Wind Generation (MW)",
        template="plotly_white",
        legend_title_text=""
    )

    return fig

if __name__ == "__main__":
    app.run(debug=True) 
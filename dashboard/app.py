"""
dashboard/app.py
Clean, end‑to‑end version – May 2025
------------------------------------
• Light/dark theme toggle (Minty / Cyborg)
• KPI cards with tool‑tips
• Forecast + Error‑distribution tabs
• Robust date / series filtering
"""

import os, json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, exceptions
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO, load_figure_template
import uuid   
# -----------------------------------------------------------------------------
#  Paths & constants
# -----------------------------------------------------------------------------
ROOT            = Path(__file__).resolve().parents[1]
FEATURES_PATH   = ROOT / "data" / "features" / "features.parquet"
PRED_PATH       = ROOT / "data" / "predictions" / "catboost_test.parquet"
METRICS_PATH    = ROOT / "metrics.json"

THEME_LIGHT     = "minty"   # template name
THEME_DARK      = "cyborg"
CSS_LIGHT       = dbc.themes.MINTY
CSS_DARK        = dbc.themes.CYBORG

# Hide the Dash debug‑inspector dropdown
os.environ["DASH_DEBUG_UI"] = "false"

# -----------------------------------------------------------------------------
#  Load static artefacts (metrics)
# -----------------------------------------------------------------------------
def safe_load_json(path: Path, default: dict) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default

metrics   = safe_load_json(METRICS_PATH, {})
baseline  = metrics.get("baseline",  {"rmse": np.nan, "mape": np.nan})
catboost  = metrics.get("catboost",  {"rmse": np.nan, "mape": np.nan})

# -----------------------------------------------------------------------------
#  Load raw data
# -----------------------------------------------------------------------------
def load_parquet(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path, columns=cols)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df
    return pd.DataFrame(columns=["datetime", *cols[1:]])

features_full = load_parquet(FEATURES_PATH,
                             ["datetime", "wind_mw", "wind_mw_lag_48h"])
preds_full    = load_parquet(PRED_PATH,
                             ["datetime", "wind_mw_pred"])

# restrict to test‑set fraction (as in training script)
if not features_full.empty:
    cut = int(len(features_full) * 0.75)
    test_feats = features_full.iloc[cut:].copy()
    min_d, max_d = test_feats["datetime"].dt.date.min(), test_feats["datetime"].dt.date.max()
else:
    test_feats = features_full.copy()
    today  = pd.Timestamp.utcnow().date()
    min_d, max_d = today, today

# -----------------------------------------------------------------------------
#  Helper – KPI card colour / icon
# -----------------------------------------------------------------------------
def kpi_colour_icon(value, base, lower_better=True):
    if np.isnan(value) or np.isnan(base):
        return "secondary", ""
    better = value < base if lower_better else value > base
    worse  = value > base if lower_better else value < base
    if better:
        return "success", "↓" if lower_better else "↑"
    if worse:
        return "danger",  "↑" if lower_better else "↓"
    return "warning", "="

# -----------------------------------------------------------------------------
#  Dash app
# -----------------------------------------------------------------------------
load_figure_template(THEME_LIGHT)
app = Dash(__name__, external_stylesheets=[CSS_LIGHT])
app.title = "GB Wind Forecast Dashboard"

# -- KPI cards ---------------------------------------------------------------
rmse_col, rmse_icon = kpi_colour_icon(catboost["rmse"], baseline["rmse"])
mape_col, mape_icon = kpi_colour_icon(catboost["mape"], baseline["mape"])

def card(title, value, unit="", colour="light", tooltip=None):
    card_id = str(uuid.uuid4())      # <-- give every card a unique id
    body    = html.H4(f"{value}{unit}", className="card-title")

    comp    = dbc.Card(
        [dbc.CardHeader(title), dbc.CardBody(body)],
        id=card_id,                 # <-- set the id here
        color=colour,
        inverse=(colour not in ["light", "secondary"]),
        className="shadow-sm",
        style={"height": "100px"}
    )

    if tooltip:
        return dbc.Col([comp,
                        dbc.Tooltip(tooltip, target=card_id, placement="top")])
    return dbc.Col(comp)

base_rmse = card("Baseline RMSE",  f"{baseline['rmse']:,.0f} MW")
base_mape = card("Baseline MAPE",  f"{baseline['mape']:.3f}")
cb_rmse   = card("CatBoost RMSE",  f"{catboost['rmse']:,.0f} MW {rmse_icon}",
                 colour=rmse_col,
                 tooltip=f"Δ {(baseline['rmse']-catboost['rmse'])/baseline['rmse']:+.0%}")
cb_mape   = card("CatBoost MAPE",  f"{catboost['mape']:.3f} {mape_icon}",
                 colour=mape_col,
                 tooltip=f"Δ {(baseline['mape']-catboost['mape'])/baseline['mape']:+.1%}")

kpis = dbc.Row([base_rmse, base_mape, cb_rmse, cb_mape], className="g-4 mb-3")

# -- Controls ----------------------------------------------------------------
series_dd = dcc.Dropdown(
    id="series", multi=True, value=["wind_mw", "wind_mw_pred"],
    options=[
        {"label": "Actual",             "value": "wind_mw"},
        {"label": "Baseline (48 h lag)","value": "wind_mw_lag_48h"},
        {"label": "CatBoost Prediction","value": "wind_mw_pred"},
    ])

date_picker = dcc.DatePickerRange(
    id="date",   min_date_allowed=min_d, max_date_allowed=max_d,
    start_date=min_d, end_date=max_d, display_format="DD/MM/YYYY", initial_visible_month=max_d)

controls = dbc.Card(dbc.CardBody(
    dbc.Row([
        dbc.Col(html.Label("Series:"), width="auto"),
        dbc.Col(series_dd, lg=5),
        dbc.Col(html.Label("Date range:"), width="auto", className="pt-2"),
        dbc.Col(date_picker, lg=4)
    ], align="center")), className="mb-3 shadow-sm")

# -- Layout ------------------------------------------------------------------
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("GB Wind Day‑Ahead Forecast", className="display-6"), width=10),
        dbc.Col(ThemeSwitchAIO(aio_id="theme", themes=[CSS_LIGHT, CSS_DARK],
                               switch_props={"style": {"marginTop": "12px"}}), width=2)
    ]),
    kpis, controls,
    dbc.Tabs([
        dbc.Tab(label="Forecast Plot",       tab_id="forecast"),
        dbc.Tab(label="Error Distribution",  tab_id="errors")
    ], id="tabs", active_tab="forecast", className="mb-3"),
    html.Div(id="tab-content")
], fluid=False, style={"maxWidth": "1200px", "paddingTop": "18px"})

# -----------------------------------------------------------------------------
#  Callback
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
    # theme
    theme   = THEME_LIGHT if toggle else THEME_DARK
    load_figure_template(theme)

    # ----- slice & merge data ------------------------------------------------
    s = pd.to_datetime(start_d).tz_localize("UTC")
    e = pd.to_datetime(end_d  ).tz_localize("UTC")

    feats = test_feats[(test_feats["datetime"] >= s) & (test_feats["datetime"] <= e)]
    if feats.empty:
        return dbc.Alert("No data in selected range.", color="warning")

    df = feats.merge(preds_full, on="datetime", how="left")

    # ------------------------------------------------------------------------
    if tab == "forecast":
        fig = go.Figure()
        if "wind_mw" in series_sel:
            fig.add_scatter(x=df["datetime"], y=df["wind_mw"],
                            mode="lines", name="Actual")
        if "wind_mw_lag_48h" in series_sel and "wind_mw_lag_48h" in df:
            fig.add_scatter(x=df["datetime"], y=df["wind_mw_lag_48h"],
                            mode="lines", name="Baseline (48 h lag)",
                            line=dict(dash="dot"))
        if "wind_mw_pred" in series_sel and "wind_mw_pred" in df:
            fig.add_scatter(x=df["datetime"], y=df["wind_mw_pred"],
                            mode="lines", name="CatBoost Prediction",
                            line=dict(color="orange"))

        fig.update_layout(template=theme,
                          title=f"{s.date()} → {e.date()}",
                          legend=dict(orientation="h", y=1.05, x=0.5,
                                      xanchor="center", yanchor="bottom"),
                          xaxis_title="Date", yaxis_title="Wind generation (MW)")
        return dcc.Graph(figure=fig)

    # ----- error distribution ----------------------------------------------
    if tab == "errors":
        if {"wind_mw", "wind_mw_pred"}.issubset(df.columns):
            resid = (df["wind_mw_pred"] - df["wind_mw"]).dropna()
        else:
            resid = pd.Series(dtype=float)

        if resid.empty:
            return dbc.Alert("No overlapping actual/prediction data.", color="warning")

        fig = px.histogram(resid, nbins=50, template=theme,
                           title=f"Prediction error distribution ({s.date()} – {e.date()})",
                           labels={"value": "Predicted − Actual (MW)"})
        fig.update_layout(bargap=0.05, yaxis_title="Frequency",
                          coloraxis_showscale=False)
        return dcc.Graph(figure=fig)

    return dbc.Alert("Unknown tab.", color="danger")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # force 16 px base font in both themes
    from pathlib import Path
    css = ROOT / "assets" / "z_override.css"
    css.parent.mkdir(exist_ok=True)
    css.write_text("html,body{font-size:16px !important;}")

    app.run(debug=True)

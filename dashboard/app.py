"""
GB Wind Day-Ahead Forecast Dashboard
------------------------------------
• Light/dark theme toggle (simple_white / plotly_dark)
• KPI cards (Baseline vs CatBoost) with colour-coded deltas
• Forecast & error-distribution tabs
• Robust date / series filtering
• Includes tomorrow’s predictions via outer merge
"""

from __future__ import annotations
import json, os, uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeSwitchAIO, load_figure_template
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ─── Paths & Constants ───────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parents[1]
FEATURES_PATH   = ROOT / "data" / "features" / "features.parquet"
LATEST_PRED_PATH= ROOT / "data" / "predictions" / "latest.parquet"
METRICS_PATH    = ROOT / "metrics.json"

COLOR_MAP = {
    "wind_perc":       "#2ca02c",  # green
    "wind_perc_lag_48h":"#1f77b4", # blue
    "wind_perc_pred":  "#ff7f0e",  # orange
}

THEME_LIGHT     = "simple_white"
THEME_DARK      = "plotly_dark"
CSS_LIGHT       = dbc.themes.MINTY
CSS_DARK        = dbc.themes.CYBORG

os.environ["DASH_DEBUG_UI"] = "false"

# ─── Helpers ─────────────────────────────────────────────────────────────────
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
raw_metrics    = safe_json(METRICS_PATH)
cat_rmse       = raw_metrics.get("holdout_rmse_perc", np.nan)
cat_mape       = raw_metrics.get("holdout_mape_perc", np.nan)

# ─── Load Historical Features ────────────────────────────────────────────────
# **This file must contain your full history** or you’ll only ever get a few rows.
hist = load_parquet(FEATURES_PATH, ["datetime", "wind_perc", "wind_perc_lag_48h"])

if not hist.empty:
    cutoff   = hist.datetime.max() - pd.Timedelta(days=1)
    sub      = hist[hist.datetime >= cutoff].dropna(subset=["wind_perc","wind_perc_lag_48h"])
    if not sub.empty:
        # MSE → RMSE
        mse_val = mean_squared_error(sub.wind_perc, sub.wind_perc_lag_48h)
        baseline_rmse = np.sqrt(mse_val)
        baseline_mape = mean_absolute_percentage_error(sub.wind_perc, sub.wind_perc_lag_48h)
    else:
        baseline_rmse = baseline_mape = np.nan
else:
    baseline_rmse = baseline_mape = np.nan


# ─── Load Latest Predictions ─────────────────────────────────────────────────
preds = load_parquet(LATEST_PRED_PATH, ["datetime","wind_perc_pred"])
# if empty or missing, preds will simply be an empty DF

# ─── Merge History + Tomorrow’s Forecast ────────────────────────────────────
plot_data = pd.merge(
    hist,
    preds,
    on="datetime",
    how="outer",
    sort=True
)
# ensure column exists
if "wind_perc_pred" not in plot_data:
    plot_data["wind_perc_pred"] = np.nan

# recalc date bounds
min_d = plot_data.datetime.dt.date.min()
max_d = plot_data.datetime.dt.date.max()

# ─── KPI Card Helper ─────────────────────────────────────────────────────────
def card(title, value, unit, colour, tooltip=None):
    cid = str(uuid.uuid4())
    card_body = dbc.CardBody([html.H4(f"{value}{unit}", className="card-title")])
    c = dbc.Card([dbc.CardHeader(title), card_body],
                 id=cid, color=colour,
                 inverse=(colour not in ["light","secondary"]),
                 className="shadow-sm", style={"height":"100px"})
    if tooltip:
        return dbc.Col([c, dbc.Tooltip(tooltip, target=cid)])
    return dbc.Col(c)

def delta_colour(val, base, lower_better=True):
    if np.isnan(val) or np.isnan(base): return "secondary",""
    better = (val<base) if lower_better else (val>base)
    worse  = (val>base) if lower_better else (val<base)
    if better: return "success","↓" if lower_better else "↑"
    if worse:  return "danger","↑" if lower_better else "↓"
    return "warning","="

rmse_col, rmse_ic = delta_colour(cat_rmse, baseline_rmse)
mape_col, mape_ic = delta_colour(cat_mape, baseline_mape)

base_rmse = card("Baseline RMSE", f"{baseline_rmse:.2f}", "%", "light")
base_mape = card("Baseline MAPE", f"{baseline_mape*100:.1f}", "%", "light")
cb_rmse   = card(
    "CatBoost RMSE",
    f"{cat_rmse:.2f}",         "% "+rmse_ic,
    rmse_col,
    tooltip=f"Δ {(baseline_rmse-cat_rmse)/baseline_rmse:+.0%}"
)
cb_mape   = card(
    "CatBoost MAPE",
    f"{cat_mape*100:.1f}",     "% "+mape_ic,
    mape_col,
    tooltip=f"Δ {(baseline_mape-cat_mape)/baseline_mape:+.1%}"
)

# ─── Build App ───────────────────────────────────────────────────────────────
load_figure_template(THEME_LIGHT)
app = Dash(__name__, external_stylesheets=[CSS_LIGHT])
server = app.server
app.title = "GB Wind Day-Ahead Forecast"

controls = dbc.Row([
    dbc.Col(base_rmse), dbc.Col(base_mape),
    dbc.Col(cb_rmse),   dbc.Col(cb_mape)
], className="g-4 mb-3")

series_dd = dcc.Dropdown(
    id="series", multi=True,
    value=["wind_perc","wind_perc_pred"],
    options=[
      {"label":"Actual (%)",             "value":"wind_perc"},
      {"label":"Baseline (lag 48h %)",   "value":"wind_perc_lag_48h"},
      {"label":"CatBoost Pred. (%)",     "value":"wind_perc_pred"},
    ]
)
date_picker = dcc.DatePickerRange(
    id="date", min_date_allowed=min_d, max_date_allowed=max_d,
    start_date=min_d, end_date=max_d, display_format="DD/MM/YYYY"
)

app.layout = dbc.Container([
    dbc.Row([
      dbc.Col(html.H2("GB Wind Day-Ahead Forecast", className="display-6"), width=10),
      dbc.Col(ThemeSwitchAIO(aio_id="theme",
                             themes=[CSS_LIGHT, CSS_DARK],
                             switch_props={"style":{"marginTop":"12px"}}),
              width=2),
    ]),
    html.Hr(),
    controls,
    dbc.Card(dbc.CardBody([
        html.P(
          "This dashboard forecasts … five expanding walk-forward CV splits.",
          className="mb-2"
        ),
        html.P(
          f"Hold-out (last 24 h) error: RMSE ≈ {cat_rmse:.2f}% • "
          f"MAPE ≈ {cat_mape*100:.1f}% "
          f"(≈ {(cat_rmse-baseline_rmse)/baseline_rmse:+.0%} vs 48h-lag)",
          className="fst-italic small mb-0"
        )
    ]), className="mb-4"),
    dbc.Card(dbc.CardBody([
      dbc.Row([dbc.Col(html.Label("Series:")), dbc.Col(series_dd, width=6),
               dbc.Col(html.Label("Date range:")), dbc.Col(date_picker, width=4)],
              align="center")
    ]), className="mb-4"),
    dbc.Tabs(
      [dbc.Tab(label="Forecast Plot",   tab_id="forecast"),
       dbc.Tab(label="Error Distribution", tab_id="errors")],
      id="tabs", active_tab="forecast", className="mb-3"
    ),
    html.Div(id="tab-content")
], style={"maxWidth":"1200px","padding":"1rem"})

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

    # filter
    df = plot_data.copy()
    df = df[(df.datetime.dt.date >= pd.to_datetime(start_d).date()) &
            (df.datetime.dt.date <= pd.to_datetime(end_d).date())]
    if df.empty or not series_sel:
        return html.Div("No data for that selection.")

    if tab=="forecast":
        fig = px.line(
          df, x="datetime", y=series_sel,
          template=template, color_discrete_map=COLOR_MAP,
          labels={"variable":"Series","value":"Wind Gen. (% of mix)"}
        )
        fig.update_layout(margin={"t":50,"b":30,"l":30,"r":30})
        if light_bg:
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        return dcc.Graph(figure=fig)

    # errors tab
    if "wind_perc" in series_sel and "wind_perc_pred" in series_sel:
        err_df = df.dropna(subset=["wind_perc","wind_perc_pred"])
        err_df["error"] = err_df.wind_perc_pred - err_df.wind_perc
        fig = px.histogram(
          err_df, x="error", template=template,
          marginal="box", color_discrete_sequence=[COLOR_MAP["wind_perc_pred"]],
          labels={"error":"Error (pct points)"}
        )
        fig.update_layout(margin={"t":50,"b":30,"l":30,"r":30})
        if light_bg:
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        return dcc.Graph(figure=fig)

    return html.Div("Select both Actual & Prediction to see errors.")

if __name__=="__main__":
    app.run(debug=True)

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
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, State, dcc, html, callback_context, no_update
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash_bootstrap_templates import ThemeSwitchAIO

# Mantine 7 needs React 18's useId hook; Dash 2.x defaults to React 16,
# which fails silently ("r.useId is not a function") and dmc components
# never mount. Must be set BEFORE the Dash app is created.
from dash import _dash_renderer
_dash_renderer._set_react_version("18.2.0")
from sklearn.metrics import mean_squared_error

# --- Dashboard logger ---
# Logs to stderr by default (captured by gunicorn/Render). Set
# DASHBOARD_LOG_FILE to also write a local debug log. File-handler setup must
# never crash worker boot (read-only filesystem, permissions), and no file is
# deleted at import time (multiple gunicorn workers race on that).
dash_logger = logging.getLogger("dashboard_app")
dash_logger.setLevel(logging.INFO)
if not dash_logger.handlers: # Avoid adding multiple handlers on hot reloads
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    dash_logger.addHandler(stream_handler)
    log_file_path = os.getenv("DASHBOARD_LOG_FILE")
    if log_file_path:
        try:
            file_handler = logging.FileHandler(log_file_path, delay=True)
            file_handler.setFormatter(formatter)
            dash_logger.addHandler(file_handler)
        except OSError as e:
            dash_logger.warning("Could not open log file %s: %s", log_file_path, e)
dash_logger.info("Dashboard logger initialized.")

# ─── Paths & Constants ───────────────────────────────────────────────────────
ROOT             = Path(__file__).resolve().parents[1]
FEATURES_PATH    = ROOT / "data" / "features" / "features.parquet"
HISTORY_PATH     = ROOT / "data" / "features" / "history.parquet"
LATEST_PRED_PATH = ROOT / "data" / "predictions" / "latest.parquet"
FULL_HIST_PREDS_PATH = ROOT / "models" / "catboost_full.parquet"
METRICS_PATH     = ROOT / "metrics.json"

# ─── Design tokens ───────────────────────────────────────────────────────────
# Mirrors assets/z_override.css. Series hues are slots 1–3 of the validated
# dataviz reference palette (CVD-checked in both modes with the skill's
# validator); chrome is the matching warm-neutral surface/ink set. Color
# follows the entity: Actual is always blue, the persistence baseline always
# aqua, the forecast always amber — in every theme and any series subset.
SERIES_NAMES = {
    "wind_perc":         "Actual",
    "wind_perc_lag_48h": "48h persistence",
    "wind_perc_pred":    "Forecast",
}
SERIES_COLORS = {
    "light": {"wind_perc": "#2a78d6", "wind_perc_lag_48h": "#1baf7a", "wind_perc_pred": "#eda100"},
    "dark":  {"wind_perc": "#3987e5", "wind_perc_lag_48h": "#199e70", "wind_perc_pred": "#c98500"},
}
DIVERGING = {  # error panel: warm = over-forecast, cool = under-forecast
    "light": {"over": "#e34948", "under": "#2a78d6"},
    "dark":  {"over": "#e66767", "under": "#3987e5"},
}
CHROME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", baseline="#c3c2b7"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", baseline="#383835"),
}
FONT_STACK = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

def _make_template(mode: str) -> go.layout.Template:
    """One plotly template per theme: recessive hairline grid, quiet ink,
    left-aligned title, horizontal legend, unified hover."""
    c = CHROME[mode]
    return go.layout.Template(layout=go.Layout(
        font=dict(family=FONT_STACK, size=12.5, color=c["ink2"]),
        title=dict(font=dict(family=FONT_STACK, size=15, color=c["ink"]),
                   x=0, xanchor="left"),
        paper_bgcolor=c["surface"],
        plot_bgcolor=c["surface"],
        colorway=[SERIES_COLORS[mode][k] for k in SERIES_NAMES],
        xaxis=dict(showgrid=False, linecolor=c["baseline"], linewidth=1,
                   ticks="outside", tickcolor=c["baseline"], ticklen=4,
                   zeroline=False, automargin=True),
        yaxis=dict(gridcolor=c["grid"], gridwidth=1, zeroline=False,
                   showline=False, ticks="", automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12)),
        hoverlabel=dict(bgcolor=c["surface"], bordercolor=c["grid"],
                        font=dict(family=FONT_STACK, size=12, color=c["ink"])),
        hovermode="x unified",
        margin=dict(t=56, b=40, l=48, r=16),
    ))

pio.templates["wf_light"] = _make_template("light")
pio.templates["wf_dark"]  = _make_template("dark")
THEME_LIGHT = "wf_light"
THEME_DARK  = "wf_dark"
CSS_LIGHT   = dbc.themes.MINTY
CSS_DARK    = dbc.themes.CYBORG
FONT_CSS    = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"

# Wind-glyph SVG used as both favicon (with backdrop) and header logo (bare).
_FAVICON_SVG = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Crect width='32' height='32' rx='7' fill='%232a78d6'/%3E"
    "%3Cpath d='M6 12h13a3.2 3.2 0 1 0-3.2-3.2M6 17h17a3.4 3.4 0 1 1-3.4 3.4M6 22h9a2.8 2.8 0 1 1-2.8 2.8'"
    " fill='none' stroke='white' stroke-width='2.4' stroke-linecap='round'/%3E%3C/svg%3E"
)
_WIND_GLYPH = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
    "%3Cpath d='M4 12h14a3.2 3.2 0 1 0-3.2-3.2M4 18h20a3.4 3.4 0 1 1-3.4 3.4M4 24h10a2.8 2.8 0 1 1-2.8 2.8'"
    " fill='none' stroke='white' stroke-width='2.6' stroke-linecap='round'/%3E%3C/svg%3E"
)

# ─── Utility Functions ───────────────────────────────────────────────────────
def safe_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def load_parquet(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists():
        # Read everything, then shape to `cols`: files written by older
        # pipeline versions may lack newer columns (e.g. the P10/P90 band),
        # and read_parquet(columns=...) raises on a missing column.
        df = pd.read_parquet(path)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols]
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df
    return pd.DataFrame(columns=cols)

def fmt_metric(val, fmt: str = "{:.2f}") -> str:
    """Format a metric for display, showing N/A instead of literal 'nan'."""
    return fmt.format(val) if not np.isnan(val) else "N/A"

def smape(y_true, y_pred) -> float:
    """Symmetric MAPE as a 0–2 fraction. Plain MAPE divides by wind_perc,
    which is near-zero on calm hours and explodes to trillions of %; SMAPE is
    bounded and is what validate.py already uses for the CV metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not mask.any():
        return np.nan
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"

def style_series_figure(fig, df, series_sel, mode_key, *, end_labels=False,
                        shade_forecast=False, area_actual=False, dense=False):
    """Apply the design-system mark specs to a px.line figure in place:
    2px lines (1.6px on dense multi-year data), no per-point markers, an
    end-dot with a surface ring + direct value label on short windows, a
    soft wash under the actual, and an amber-tinted forecast region."""
    c = CHROME[mode_key]
    colors = SERIES_COLORS[mode_key]
    for tr in fig.data:
        col = tr.name  # px names traces by source column
        tr.name = SERIES_NAMES.get(col, col)
        tr.mode = "lines"
        tr.line.width = 1.6 if dense else 2
        # Spline smoothing is SVG-only: px.line silently upgrades large frames
        # to Scattergl (WebGL), where shape="spline" is invalid and raises.
        if not dense and getattr(tr, "type", "scatter") == "scatter":
            tr.line.shape = "spline"
            tr.line.smoothing = 0.55
        tr.hovertemplate = "%{y:.1f}%<extra>" + tr.name + "</extra>"
        if area_actual and col == "wind_perc":
            tr.fill = "tozeroy"
            tr.fillcolor = _rgba(colors[col], 0.08)

    if end_labels:
        for col in series_sel:
            if col not in df.columns:
                continue
            s = df.dropna(subset=[col])
            if s.empty:
                continue
            x0 = s["datetime"].iloc[-1]
            y0 = float(s[col].iloc[-1])
            fig.add_trace(go.Scatter(
                x=[x0], y=[y0], mode="markers",
                marker=dict(size=9, color=colors.get(col),
                            line=dict(width=2, color=c["surface"])),
                showlegend=False, hoverinfo="skip"))
            fig.add_annotation(
                x=x0, y=y0, text=f"<b>{y0:.0f}%</b>",
                xanchor="left", yanchor="middle", xshift=9, showarrow=False,
                font=dict(size=12, color=c["ink"], family=FONT_STACK))

    if shade_forecast:
        actuals = df.dropna(subset=["wind_perc"])
        if not actuals.empty:
            last_actual = actuals["datetime"].max()
            if (df["datetime"] > last_actual).any():
                fig.add_vrect(x0=last_actual, x1=df["datetime"].max(),
                              fillcolor=colors["wind_perc_pred"],
                              opacity=0.06, line_width=0)
                fig.add_vline(x=last_actual, line_width=1, line_dash="dot",
                              line_color=c["muted"])
                fig.add_annotation(
                    x=last_actual, y=1, yref="paper", yanchor="top", yshift=-4,
                    xanchor="left", xshift=6, text="forecast →", showarrow=False,
                    font=dict(size=11, color=c["muted"], family=FONT_STACK))

    fig.update_yaxes(ticksuffix="%", rangemode="tozero", title=None)
    fig.update_xaxes(title=None)

def add_uncertainty_band(fig, df, mode_key):
    """P10–P90 uncertainty band behind the forecast line, drawn from the
    quantile-model columns. Traces are appended, then moved to the back so
    the series lines and end labels stay on top."""
    if not {"wind_perc_pred_p10", "wind_perc_pred_p90"} <= set(df.columns):
        return
    band = df.dropna(subset=["wind_perc_pred_p10", "wind_perc_pred_p90"])
    if band.empty:
        return
    col = SERIES_COLORS[mode_key]["wind_perc_pred"]
    t_lo = go.Scatter(
        x=band["datetime"], y=band["wind_perc_pred_p10"], mode="lines",
        line=dict(width=0), hoverinfo="skip", showlegend=False, name="_p10")
    t_hi = go.Scatter(
        x=band["datetime"], y=band["wind_perc_pred_p90"], mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor=_rgba(col, 0.15),
        hovertemplate="%{y:.1f}%<extra>P10–P90 range</extra>",
        name="Forecast range (P10–P90)", showlegend=True)
    n_before = len(fig.data)
    fig.add_traces([t_lo, t_hi])
    data = list(fig.data)
    fig.data = tuple(data[n_before:] + data[:n_before])

def build_error_figure(error_df, x_min, x_max, mode_key, template):
    """Diverging error panel — warm fill above zero (over-forecast), cool
    below (under-forecast), hairline zero baseline. One logical series, so
    no legend box; the title carries the reading."""
    c = CHROME[mode_key]
    d = DIVERGING[mode_key]
    err = error_df.dropna(subset=["prediction_error"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=err["datetime"], y=err["prediction_error"].clip(lower=0),
        mode="lines", line=dict(width=1.5, color=d["over"]),
        fill="tozeroy", fillcolor=_rgba(d["over"], 0.14),
        hovertemplate="%{y:.1f} pts<extra>over-forecast</extra>", name="over"))
    fig.add_trace(go.Scatter(
        x=err["datetime"], y=err["prediction_error"].clip(upper=0),
        mode="lines", line=dict(width=1.5, color=d["under"]),
        fill="tozeroy", fillcolor=_rgba(d["under"], 0.14),
        hovertemplate="%{y:.1f} pts<extra>under-forecast</extra>", name="under"))
    fig.update_layout(
        template=template, height=170, showlegend=False, hovermode="x",
        title=dict(text="Forecast error (%-points · above zero = over-forecast)",
                   font=dict(size=13)),
        margin=dict(t=36, b=28, l=48, r=16),
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(zeroline=True, zerolinecolor=c["baseline"], zerolinewidth=1,
                   title=None),
    )
    return fig

def latest_actual_ts():
    """Timestamp of the most recent actual observation, or None. Defined
    before hist_df loads; resolves hist_df from module globals at call time."""
    try:
        actuals = hist_df.dropna(subset=["wind_perc"])
        if not actuals.empty:
            return actuals["datetime"].max()
    except Exception:
        pass
    return None

def data_last_updated() -> str:
    """Freshness stamp = latest actual observation, not the file mtime. A file
    can look freshly written (a redeploy) while its data is weeks stale; the
    latest observation is the honest 'data current as of' signal."""
    ts = latest_actual_ts()
    if ts is not None and pd.notna(ts):
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    mtimes = [p.stat().st_mtime for p in (LATEST_PRED_PATH, HISTORY_PATH) if p.exists()]
    if not mtimes:
        return "unknown"
    return pd.Timestamp(max(mtimes), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")

def staleness_banner():
    """A warning strip when the latest actual observation is well in the past —
    surfaces a stalled nightly pipeline instead of silently showing old data."""
    ts = latest_actual_ts()
    if ts is None or pd.isna(ts):
        return None
    age_days = (pd.Timestamp.now(tz="UTC") - ts).days
    if age_days < 2:
        return None
    return dbc.Alert(
        f"⚠ Data may be stale — the latest actual observation is {age_days} days old "
        f"({ts.strftime('%Y-%m-%d %H:%M UTC')}). The nightly update may not have run.",
        color="warning", className="mb-3 py-2 small", dismissable=True,
    )

def freshness_chip():
    """Header pill: latest observation + a pulse dot (green fresh / amber old)."""
    ts = latest_actual_ts()
    if ts is None or pd.isna(ts):
        return html.Span([html.Span(className="wf-livedot old"),
                          "No data loaded"], className="wf-chip")
    age_h = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600
    dot = "ok" if age_h <= 36 else "old"
    return html.Span([
        html.Span(className=f"wf-livedot {dot}"),
        html.Span("Data through "),
        html.Span(ts.strftime("%d %b %Y %H:%M UTC"), className="wf-chip-strong"),
    ], className="wf-chip")

def forecast_hero():
    """The one number the view leads with: mean forecast wind share over the
    day-ahead horizon, with its peak and a delta vs the last day of actuals."""
    preds = plot_data.dropna(subset=["wind_perc_pred"])
    if preds.empty:
        return None
    la = latest_actual_ts()
    future = preds[preds["datetime"] > la] if (la is not None and pd.notna(la)) else preds.iloc[0:0]
    window = future.head(48) if not future.empty else preds.tail(24)
    label = "Day-ahead forecast" if not future.empty else "Latest forecast window"
    mean_pred = float(window["wind_perc_pred"].mean())
    if np.isnan(mean_pred):
        return None
    peak = window.loc[window["wind_perc_pred"].idxmax()]

    delta = None
    acts = plot_data.dropna(subset=["wind_perc"]).tail(24)
    if not acts.empty:
        d = mean_pred - float(acts["wind_perc"].mean())
        cls = "good" if d >= 0.5 else ("bad" if d <= -0.5 else "flat")
        arrow = "▲" if d >= 0.5 else ("▼" if d <= -0.5 else "•")
        delta = html.Div(f"{arrow} {d:+.1f} pts vs last 24h of actuals",
                         className=f"wf-delta wf-delta-{cls}")

    return dbc.Col(dbc.Card(dbc.CardBody([
        html.Div(label, className="wf-metric-label"),
        html.Div([html.Span(f"{mean_pred:.0f}", className="wf-hero-value"),
                  html.Span("% of GB mix", className="wf-hero-unit")]),
        delta,
        html.Div(f"Peak {peak['wind_perc_pred']:.0f}% around {peak['datetime']:%a %H:%M} UTC",
                 className="wf-hero-sub"),
    ]), className="wf-metric-card wf-hero h-100"), md=4)

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

# Load predictions (P10/P90 band columns are NaN-filled when an older
# artefact without quantiles is deployed)
PRED_COLS = ["datetime", "wind_perc_pred", "wind_perc_pred_p10", "wind_perc_pred_p90"]
latest_preds_df = load_parquet(LATEST_PRED_PATH, PRED_COLS)
full_hist_preds_df = load_parquet(FULL_HIST_PREDS_PATH, PRED_COLS)

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
    all_preds_df = pd.DataFrame(columns=PRED_COLS)
    # utc=True: must stay tz-aware or the outer merge with tz-aware hist_df
    # raises at import time and gunicorn workers boot-loop.
    all_preds_df['datetime'] = pd.to_datetime(all_preds_df['datetime'], utc=True)

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
for _band_col in ("wind_perc_pred_p10", "wind_perc_pred_p90"):
    if _band_col not in plot_data:
        plot_data[_band_col] = np.nan

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
        baseline_mape  = smape(sub.wind_perc, sub.wind_perc_lag_48h)
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

def make_card(title, value, unit, colour, tooltip=None, delta=None, md=None):
    """Metric tile: muted uppercase label, large proportional value, optional
    delta chip. `colour` (success/danger from delta_colour) tints only the
    delta text — tiles stay quiet so the data reads first."""
    cid = str(uuid.uuid4())
    delta_cls = {"success": "good", "danger": "bad"}.get(colour, "flat")
    children = [
        html.Div(title, className="wf-metric-label"),
        html.Div([html.Span(value, className="wf-metric-value"),
                  html.Span(unit, className="wf-metric-unit")]),
    ]
    if delta:
        children.append(html.Div(delta, className=f"wf-delta wf-delta-{delta_cls}"))
    card = dbc.Card(dbc.CardBody(children), id=cid, className="wf-metric-card h-100")
    body = [card, dbc.Tooltip(tooltip, target=cid, placement="top")] if tooltip else card
    return dbc.Col(body, md=md)

# ─── Build Dash App ─────────────────────────────────────────────────────────
# assets_folder: the stylesheet lives in the repo-root assets/, but Dash
# defaults to dashboard/assets/ (next to this file), so the custom CSS was
# never actually served. Point it at the real folder. The Inter stylesheet is
# a separate link, untouched by ThemeSwitchAIO's theme swapping.
app    = Dash(__name__, external_stylesheets=[CSS_LIGHT, FONT_CSS, dmc.styles.DATES],
              suppress_callback_exceptions=True,
              assets_folder=str(ROOT / "assets"))
server = app.server
app.title = "GB Wind · Day-Ahead Forecast"
app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        <link rel="icon" href="{_FAVICON_SVG}">
        {{%css%}}
    </head>
    <body>
        {{%app_entry%}}
        <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
    </body>
</html>"""

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

# Two single-date Mantine masked inputs (DateInput): real <input> fields that
# accept typed DD/MM/YYYY committed on Enter/blur with immediate visual
# feedback. NOTE: Mantine 7's calendar pickers (DatePickerInput) render as
# buttons and do NOT support typing — that's why these are DateInput.
date_picker_start = dmc.DateInput(
    id="date-start",
    value=min_d.isoformat(),
    minDate=min_d,
    maxDate=max_d,
    valueFormat="DD/MM/YYYY",
    size="sm", w=150, clearable=False,
)
date_picker_end = dmc.DateInput(
    id="date-end",
    value=max_d.isoformat(),
    minDate=min_d,
    maxDate=max_d,
    valueFormat="DD/MM/YYYY",
    size="sm", w=150, clearable=False,
)

# Quick date filters rendered as one segmented control
def create_date_preset_buttons():
    return html.Div(html.Div([
        dbc.Button("Last 24h", id="btn-last-24h", size="sm"),
        dbc.Button("Last 7d",  id="btn-last-7d",  size="sm"),
        dbc.Button("Last 30d", id="btn-last-30d", size="sm"),
        dbc.Button("All time", id="btn-all-time", size="sm"),
    ], className="wf-segmented"), className="mb-3")

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.Img(src=_WIND_GLYPH, className="wf-logo"),
            html.Div([
                html.H2("GB Wind · Day-Ahead Forecast", className="wf-title mb-0"),
                html.Div("How much of Britain's electricity the wind will supply tomorrow — "
                         "CatBoost vs a 48h-persistence baseline",
                         className="wf-subtitle mt-1"),
            ]),
        ], className="wf-brand"), width=8),
        dbc.Col([
            dbc.Button("Export CSV", id="btn-export-main", size="sm", className="wf-btn me-3"),
            ThemeSwitchAIO(aio_id="theme",
                          themes=[CSS_LIGHT, CSS_DARK],
                          switch_props={"style":{"marginTop":"6px"}})
        ], width=4, className="text-end d-flex align-items-center justify-content-end"),
    ], align="center", className="mb-3 mt-3"),

    html.Div(freshness_chip(), className="mb-3"),
    html.Div(id="theme-anchor", style={"display": "none"}),

    staleness_banner(),

    html.Div(id="kpi-cards-row"),

    dbc.Card(dbc.CardBody(html.Details([
        html.Summary("About this forecast"),
        html.P(
          "A GPU-tuned CatBoost model forecasts the day-ahead share of GB electricity "
          "generation coming from wind. Carbon Intensity wind percentage is merged with "
          "Open-Meteo weather; power-curve proxies, lags and seasonal features feed the "
          "model, with Optuna tuning over five expanding walk-forward CV splits.",
          className="mb-2 mt-2"
        ),
        html.P(
          f"Hold-out (last 48 h) error from training: RMSE ≈ {fmt_metric(cat_rmse)} %-points • "
          f"MAPE ≈ {fmt_metric(cat_mape * 100, '{:.1f}')}%. Errors are points on the 0–100 "
          "share scale, not relative error.",
          className="fst-italic small mb-0"
        ),
    ], className="wf-about")), className="mb-4 wf-panel"),

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

    # The date presets + range picker only drive the Historical Analysis
    # chart. A callback hides this whole block on the Forecast tab, where the
    # chart is a fixed "recent + forecast horizon" window — otherwise the
    # controls change but the chart doesn't, which looks broken.
    html.Div(id="date-controls", children=[
        create_date_preset_buttons(),
        dbc.Card(dbc.CardBody(
            # MantineProvider is required around any dmc component;
            # forceColorScheme follows the theme switch (see clientside cb).
            dmc.MantineProvider(
                dbc.Row([
                    dbc.Col(html.Label("Date Range:", className="mb-0 d-flex align-items-center",
                                       style={"height": "30px"}), width="auto"),
                    dbc.Col(date_picker_start, width="auto"),
                    dbc.Col(html.Span("–", className="text-secondary"), width="auto"),
                    dbc.Col(date_picker_end, width="auto"),
                    dbc.Col(html.Span("Type a date + Enter (or use the preset buttons above)",
                                      className="text-secondary small"),
                            className="text-end"),
                ], align="center", className="g-2"),
                id="mantine-provider", forceColorScheme="light",
            )
        ), className="mb-4 wf-panel"),
    ]),

    dcc.Loading(html.Div(id="tab-content"), type="circle", color="#2a78d6",
                parent_className="wf-loading", delay_show=150),

    dcc.Download(id="download-data"),

], fluid=True, className="dbc dbc-row-selectable", style={"maxWidth":"1400px", "paddingTop":"18px"})

# ─── Callbacks ──────────────────────────────────────────────────────────────
@app.callback(
    [Output("tab-content","children"),
     Output("kpi-cards-row", "children")],
    [Input(ThemeSwitchAIO.ids.switch("theme"),"value"),
     Input("tabs","active_tab"),
     Input("series","value"),
     Input("date-start", "value"),
     Input("date-end", "value")]
)
def render_content(theme_switch_on, active_tab, series_sel, start_d_global, end_d_global):
    # Mantine single-date values are ISO strings; either may be None while
    # the user is mid-edit — treat missing ends as no constraint.
    dash_logger.info(f"--- render_content CALLED: active_tab={active_tab}, series_sel={series_sel}, start_d={start_d_global}, end_d={end_d_global} ---")
    # ThemeSwitchAIO semantics: switch ON (True) selects themes[0], which is
    # the LIGHT stylesheet here — treating True as dark inverts every figure
    # template against the page CSS.
    light_bg = bool(theme_switch_on)
    mode_key = "light" if light_bg else "dark"
    template = THEME_LIGHT if light_bg else THEME_DARK
    colors = SERIES_COLORS[mode_key]
    kpi_cards_content = []
    tab_specific_content = []

    if active_tab == "forecast_recent":
        # --- KPI row: hero forecast tile + four compact metric tiles ---
        cat_rmse_col, cat_rmse_ic = delta_colour(cat_rmse, baseline_rmse)
        cat_mape_col, cat_mape_ic = delta_colour(cat_mape, baseline_mape)
        tiles = [
            forecast_hero(),
            make_card("Baseline RMSE · last 24h", fmt_metric(baseline_rmse), " pts", "light", md=2,
                      tooltip="48h-persistence baseline error over the most recent 24h of actuals"),
            make_card("Baseline SMAPE · last 24h", fmt_metric(baseline_mape * 100, "{:.1f}"), "%", "light", md=2,
                      tooltip="Symmetric MAPE of the 48h-persistence baseline over the most recent 24h of actuals"),
            make_card("Model RMSE · holdout", fmt_metric(cat_rmse), " pts", cat_rmse_col, md=2,
                      delta=f"{cat_rmse_ic} vs baseline" if cat_rmse_ic else None,
                      tooltip="CatBoost error on the final 48h training holdout — points on the 0–100 share scale"),
            make_card("Model MAPE · holdout", fmt_metric(cat_mape * 100, "{:.1f}"), "%", cat_mape_col, md=2,
                      delta=f"{cat_mape_ic} vs baseline" if cat_mape_ic else None,
                      tooltip="CatBoost MAPE on the final 48h training holdout"),
        ]
        kpi_cards_content = [dbc.Row([t for t in tiles if t is not None], className="g-3 mb-4")]

        # --- Window anchored to the newest data point, not date.today():
        # when the nightly job lags, today's window would be empty and the
        # tab would go blank. ---
        if not plot_data.empty and not plot_data["datetime"].dropna().empty:
            anchor_date = plot_data["datetime"].dropna().dt.date.max()
        else:
            anchor_date = date.today()
        forecast_plot_start_date = anchor_date - pd.Timedelta(days=5)
        forecast_plot_end_date = anchor_date

        df_forecast_tab = plot_data[
            (plot_data.datetime.dt.date >= forecast_plot_start_date) &
            (plot_data.datetime.dt.date <= forecast_plot_end_date)
        ]

        if df_forecast_tab.empty or not series_sel:
            dash_logger.info("Forecast tab: no data for the window or no series selected.")
            fig_forecast_content = html.Div("No data available for the forecast window.")
        else:
            fig_fc = px.line(
                df_forecast_tab, x="datetime", y=series_sel,
                template=template, color_discrete_map=colors,
                labels={"variable": "", "datetime": ""},
            )
            style_series_figure(fig_fc, df_forecast_tab, series_sel, mode_key,
                                end_labels=True, shade_forecast=True, area_actual=True)
            # Band only where the forecast is selected — it decorates the
            # prediction line, not the actuals.
            if "wind_perc_pred" in series_sel:
                add_uncertainty_band(fig_fc, df_forecast_tab, mode_key)
            fig_fc.update_layout(title_text="Recent actuals & day-ahead forecast",
                                 margin=dict(t=64))
            dash_logger.info(f"Forecast tab: plotting {len(df_forecast_tab)} rows.")

            # Error panel — only where actuals overlap the forecast
            error_df = df_forecast_tab.copy()
            error_df["prediction_error"] = error_df["wind_perc_pred"] - error_df["wind_perc"]
            error_df = error_df.dropna(subset=["prediction_error"])
            if not error_df.empty:
                error_chart = dcc.Graph(
                    figure=build_error_figure(error_df, df_forecast_tab.datetime.min(),
                                              df_forecast_tab.datetime.max(), mode_key, template),
                    config={"displayModeBar": False})
            else:
                error_chart = html.Div("No overlapping actuals to score in this window yet.",
                                       className="text-muted small py-2")

            # Numeric table at 6-hour marks
            interval_df = df_forecast_tab[df_forecast_tab.datetime.dt.hour % 6 == 0].copy()
            if not interval_df.empty:
                interval_df = interval_df.sort_values("datetime", ascending=False).head(8)
                interval_df["datetime"] = interval_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
                interval_df["wind_perc"] = interval_df["wind_perc"].round(1)
                interval_df["wind_perc_pred"] = interval_df["wind_perc_pred"].round(1)

                columns = [{"name": "Timestamp", "id": "datetime"}]
                data = []
                for _, row in interval_df.iterrows():
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
                    pd.DataFrame(data), striped=True, hover=True,
                    responsive=True, size="sm", className="mt-3")
            else:
                values_table = html.Div("No interval data available", className="text-center py-2")

            # Wind trend chip over the latest actuals
            trend_indicator = None
            latest_values = df_forecast_tab.dropna(subset=["wind_perc"]).tail(12)
            if len(latest_values) >= 2:
                first_half = latest_values.head(len(latest_values) // 2)["wind_perc"].mean()
                second_half = latest_values.tail(len(latest_values) // 2)["wind_perc"].mean()
                rising = second_half > first_half
                trend_indicator = html.Span([
                    "▲ " if rising else "▼ ",
                    f"wind {'rising' if rising else 'easing'} · {abs(second_half - first_half):.1f} pts",
                ], className=f"wf-trend {'up' if rising else 'down'} mt-3")

            fig_forecast_content = html.Div([
                dcc.Graph(figure=fig_fc, config={"displaylogo": False}),
                error_chart,
                dbc.Row([
                    dbc.Col(trend_indicator, width="auto") if trend_indicator else None,
                    dbc.Col(values_table, width=12 if not trend_indicator else None),
                ], className="mt-2 align-items-start"),
            ])

        tab_specific_content = [fig_forecast_content]

    elif active_tab == "historical":
        # Fall back to the full range while the picker is mid-selection.
        eff_start = pd.to_datetime(start_d_global).date() if start_d_global else min_d
        eff_end   = pd.to_datetime(end_d_global).date() if end_d_global else max_d
        df_hist_tab = plot_data[
            (plot_data.datetime.dt.date >= eff_start) &
            (plot_data.datetime.dt.date <= eff_end)
        ]

        # Dynamic KPIs. SMAPE, not MAPE: over a wide range MAPE divides by
        # near-zero calm-hour wind_perc and renders quadrillion-percent junk.
        dyn_cat_rmse, dyn_cat_mape = np.nan, np.nan
        dyn_baseline_rmse, dyn_baseline_mape = np.nan, np.nan
        df_eval_model = df_hist_tab.dropna(subset=["wind_perc", "wind_perc_pred"])
        if len(df_eval_model) > 1:
            dyn_cat_rmse = np.sqrt(mean_squared_error(df_eval_model.wind_perc, df_eval_model.wind_perc_pred))
            dyn_cat_mape = smape(df_eval_model.wind_perc, df_eval_model.wind_perc_pred)
        df_eval_baseline = df_hist_tab.dropna(subset=["wind_perc", "wind_perc_lag_48h"])
        if len(df_eval_baseline) > 1:
            dyn_baseline_rmse = np.sqrt(mean_squared_error(df_eval_baseline.wind_perc, df_eval_baseline.wind_perc_lag_48h))
            dyn_baseline_mape = smape(df_eval_baseline.wind_perc, df_eval_baseline.wind_perc_lag_48h)

        dyn_cat_rmse_col, dyn_cat_rmse_ic = delta_colour(dyn_cat_rmse, dyn_baseline_rmse)
        dyn_cat_mape_col, dyn_cat_mape_ic = delta_colour(dyn_cat_mape, dyn_baseline_mape)
        kpi_cards_content = [dbc.Row([
            make_card("Baseline RMSE · selected range", fmt_metric(dyn_baseline_rmse), " pts", "light", md=3,
                      tooltip="48h-persistence baseline over the selected dates"),
            make_card("Baseline SMAPE · selected range", fmt_metric(dyn_baseline_mape * 100, "{:.1f}"), "%", "light", md=3),
            make_card("Model RMSE · selected range", fmt_metric(dyn_cat_rmse), " pts", dyn_cat_rmse_col, md=3,
                      delta=f"{dyn_cat_rmse_ic} vs baseline" if dyn_cat_rmse_ic else None,
                      tooltip="Includes in-sample fit over the training era — treat early-history skill as optimistic"),
            make_card("Model SMAPE · selected range", fmt_metric(dyn_cat_mape * 100, "{:.1f}"), "%", dyn_cat_mape_col, md=3,
                      delta=f"{dyn_cat_mape_ic} vs baseline" if dyn_cat_mape_ic else None),
        ], className="g-3 mb-4")]

        dash_logger.info(
            "Historical tab: %s rows selected from %s to %s (series=%s)",
            len(df_hist_tab), start_d_global, end_d_global, series_sel,
        )

        if df_hist_tab.empty or not series_sel:
            fig_historical_content = html.Div("No data available for that selection.")
        else:
            span_days = max((df_hist_tab.datetime.max() - df_hist_tab.datetime.min()).days, 0)
            fig_hist = px.line(
                df_hist_tab, x="datetime", y=series_sel,
                template=template, color_discrete_map=colors,
                labels={"variable": "", "datetime": ""},
            )
            style_series_figure(fig_hist, df_hist_tab, series_sel, mode_key,
                                end_labels=span_days <= 45, dense=span_days > 45)
            if "wind_perc_pred" in series_sel:
                add_uncertainty_band(fig_hist, df_hist_tab, mode_key)
            fig_hist.update_layout(title_text="Wind share of GB generation — history & model",
                                   margin=dict(t=64))

            error_df = df_hist_tab.copy()
            error_df["prediction_error"] = error_df["wind_perc_pred"] - error_df["wind_perc"]
            error_df = error_df.dropna(subset=["prediction_error"])
            if not error_df.empty:
                error_content = dcc.Graph(
                    figure=build_error_figure(error_df, df_hist_tab.datetime.min(),
                                              df_hist_tab.datetime.max(), mode_key, template),
                    config={"displayModeBar": False})

                # ── Error analytics: distribution + worst days ──
                hist_fig = px.histogram(
                    error_df, x="prediction_error", nbins=60, template=template,
                    color_discrete_sequence=[SERIES_COLORS[mode_key]["wind_perc"]])
                hist_fig.update_layout(
                    title=dict(text="Error distribution (%-points)", font=dict(size=13)),
                    height=190, showlegend=False, bargap=0.05,
                    margin=dict(t=36, b=28, l=48, r=16),
                    xaxis=dict(title=None), yaxis=dict(title=None))
                hist_fig.add_vline(x=0, line_width=1, line_dash="dot",
                                   line_color=CHROME[mode_key]["muted"])
                mean_err = float(error_df["prediction_error"].mean())
                hist_fig.add_vline(x=mean_err, line_width=1,
                                   line_color=SERIES_COLORS[mode_key]["wind_perc_pred"])
                hist_fig.add_annotation(
                    x=mean_err, y=1, yref="paper", yanchor="top", xanchor="left",
                    xshift=6, yshift=-2, text=f"mean {mean_err:+.1f} pts",
                    showarrow=False, font=dict(size=11, color=CHROME[mode_key]["ink2"]))
                error_hist_content = dcc.Graph(figure=hist_fig, config={"displayModeBar": False})

                daily = (error_df.assign(date=error_df["datetime"].dt.date)
                         .groupby("date")["prediction_error"]
                         .agg(mean_abs_err=lambda s: s.abs().mean(),
                              mean_err="mean", n="size")
                         .reset_index()
                         .sort_values("mean_abs_err", ascending=False)
                         .head(8))
                daily["mean_abs_err"] = daily["mean_abs_err"].round(1)
                daily["mean_err"] = daily["mean_err"].round(1)
                worst_days_content = html.Div([
                    html.H6("Worst-forecast days (selected range)", className="mt-3"),
                    dbc.Table.from_dataframe(
                        daily[["date", "mean_abs_err", "mean_err", "n"]].rename(columns={
                            "date": "Date", "mean_abs_err": "Mean |error| (pts)",
                            "mean_err": "Mean error (pts)", "n": "Hours"}),
                        striped=True, hover=True, responsive=True, size="sm"),
                ])
                error_analytics = dbc.Row([
                    dbc.Col(error_hist_content, md=6),
                    dbc.Col(worst_days_content, md=6),
                ], className="g-3 mt-1")
            else:
                error_content = html.Div("No overlapping actuals in the selected range.",
                                         className="text-muted small py-2")
                error_analytics = None

            sample_interval = max(1, len(df_hist_tab) // 8)
            interval_df = df_hist_tab.iloc[::sample_interval].head(8).copy()
            if not interval_df.empty:
                interval_df["datetime"] = interval_df["datetime"].dt.strftime("%Y-%m-%d %H:%M")
                interval_df["wind_perc"] = interval_df["wind_perc"].round(1)
                interval_df["wind_perc_pred"] = interval_df["wind_perc_pred"].round(1)
                values_table = dbc.Table.from_dataframe(
                    interval_df[["datetime", "wind_perc", "wind_perc_pred"]].rename(
                        columns={"datetime": "Timestamp", "wind_perc": "Actual (%)",
                                 "wind_perc_pred": "Predicted (%)"}),
                    striped=True, hover=True, responsive=True, size="sm", className="mt-3")
            else:
                values_table = html.Div("No interval data available", className="text-center py-2")

            fig_historical_content = html.Div([
                dcc.Graph(figure=fig_hist, config={"displaylogo": False}),
                error_content,
                error_analytics if error_analytics else html.Div(),
                html.Div([html.H6("Sample data points", className="mt-3"), values_table]),
            ])

        tab_specific_content = [fig_historical_content]

    return tab_specific_content, kpi_cards_content


# Show the date controls only on the Historical Analysis tab. The Forecast
# tab shows a fixed recent+forecast window, so the presets/picker there would
# change value but never move the chart — which reads as "the buttons are
# broken". Hiding them removes that trap.
@app.callback(
    Output("date-controls", "style"),
    Input("tabs", "active_tab"),
)
def toggle_date_controls(active_tab):
    return {} if active_tab == "historical" else {"display": "none"}

# Add callbacks for date preset buttons
@app.callback(
    [Output("date-start", "value"),
     Output("date-end", "value")],
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
    # Clamp presets to the available data range so the pickers never receive
    # dates outside their min/max allowed bounds. Mantine wants ISO strings.
    end = min(pd.Timestamp.today().date(), max_d)

    if button_id == "btn-last-24h":
        start = max(end - pd.Timedelta(days=1), min_d)
    elif button_id == "btn-last-7d":
        start = max(end - pd.Timedelta(days=7), min_d)
    elif button_id == "btn-last-30d":
        start = max(end - pd.Timedelta(days=30), min_d)
    elif button_id == "btn-all-time":
        start, end = min_d, max_d
    else:
        return no_update, no_update

    return start.isoformat(), end.isoformat()

# Export the currently selected date range as CSV. One global button
# (always in the layout) drives this, so no suppressed-callback juggling.
@app.callback(
    Output("download-data", "data"),
    Input("btn-export-main", "n_clicks"),
    [State("date-start", "value"),
     State("date-end", "value")],
    prevent_initial_call=True,
)
def export_data(n_main, start_d, end_d):
    if not n_main:
        return no_update
    df = plot_data
    if start_d and end_d:
        df = df[
            (df.datetime.dt.date >= pd.to_datetime(start_d).date())
            & (df.datetime.dt.date <= pd.to_datetime(end_d).date())
        ]
    return dcc.send_data_frame(df.to_csv, "wind_forecast_data.csv", index=False)

# Stamp the active theme on <html> so the CSS design tokens follow the switch
# (ThemeSwitchAIO ON = themes[0] = the light stylesheet). The same signal sets
# the Mantine provider's color scheme so the date picker matches the page.
app.clientside_callback(
    """function(on){
        var scheme = on ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', scheme);
        return ['', scheme];
    }""",
    [Output("theme-anchor", "children"),
     Output("mantine-provider", "forceColorScheme")],
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"),
)

# ─── Run Server ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Debug mode (Werkzeug debugger + hot reload) is opt-in via DASH_DEBUG=1.
    # It must never be the default: the Werkzeug console is remote code
    # execution if the port is ever reachable from another machine.
    app.run(debug=os.getenv("DASH_DEBUG", "").lower() in ("1", "true", "yes"))

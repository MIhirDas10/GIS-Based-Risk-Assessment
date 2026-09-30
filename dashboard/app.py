from __future__ import annotations

import os
from typing import Any

import folium
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_S = 15

RISK_ORDER = ["Low", "Moderate", "High", "Critical"]
RISK_COLORS = {
    "Low": "#488C6F",       # Premium soft sage green
    "Moderate": "#D9A752",  # Muted gold / ochre
    "High": "#D97341",      # Warm terracotta
    "Critical": "#C94A4A",  # Crimson red
}
RISK_EMOJI = {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Critical": "🔴"}

st.set_page_config(
    page_title="DengueRisk BD",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── ELEGANT EDITORIAL DESIGN SYSTEM (LIGHT LINEN & BRONZE STYLE) ───────────
CSS = """
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* Main App Reset */
.stApp {
    background-color: #F7F6F3 !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(240, 237, 230, 0.6) 0, transparent 50%),
        radial-gradient(at 100% 0%, rgba(217, 167, 82, 0.05) 0, transparent 50%) !important;
    color: #2D3139 !important;
}

*, *::before, *::after {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

[data-testid="stHeader"] {
    background: rgba(247, 246, 243, 0.9) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #E6E4DD;
}

.block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1560px !important;
}

/* Base Headings */
h1, h2, h3, h4, .editorial-title {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #1E2229 !important;
    font-weight: 700 !important;
}

/* Elegant Editorial Header */
.header-container {
    border-bottom: 2px double #D4D1C5;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 1rem;
}

.title-area {
    display: flex;
    flex-direction: column;
}

.editorial-tag {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #8C8269 !important;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.editorial-title {
    font-size: 2.3rem;
    line-height: 1.1;
    margin: 0;
    letter-spacing: -0.01em;
}

.editorial-subtitle {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-style: italic;
    font-size: 1.05rem;
    color: #5C6270 !important;
    margin-top: 0.35rem;
}

/* Clean Professional Pills */
.status-panel {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.editorial-pill {
    background: #FFFFFF;
    border: 1px solid #E0DDD5;
    border-radius: 4px;
    padding: 0.35rem 0.75rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: #4D5360 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    transition: all 0.2s ease;
}

.editorial-pill:hover {
    border-color: #C5A880;
    background: #FAF9F6;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
}
.status-dot.active {
    background-color: #488C6F;
    box-shadow: 0 0 0 2px rgba(72, 140, 111, 0.2);
}
.status-dot.inactive {
    background-color: #C94A4A;
    box-shadow: 0 0 0 2px rgba(201, 74, 74, 0.2);
}

/* Luxurious Fine-Art Cards */
.editorial-card {
    background: #FFFFFF;
    border: 1px solid #ECEAE4;
    border-radius: 0; /* Sophisticated sharp edges */
    padding: 1.25rem 1.5rem;
    box-shadow: 0 8px 24px -6px rgba(27,38,59,0.04), 0 2px 4px rgba(27,38,59,0.02);
    position: relative;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.editorial-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: #C5A880; /* Subtle brass/gold bar on top */
    opacity: 0.85;
}

.editorial-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px -8px rgba(27,38,59,0.08), 0 4px 10px rgba(27,38,59,0.03);
    border-color: #C5A880;
}

.card-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8C8269 !important;
    margin-bottom: 0.5rem;
}

.card-value {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 1.7rem;
    font-weight: 700;
    color: #1E2229 !important;
    line-height: 1.1;
}

.card-value.mono {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem;
    font-weight: 500;
    color: #2C3539 !important;
}

.card-note {
    font-size: 0.72rem;
    color: #6E7582 !important;
    margin-top: 0.45rem;
    border-top: 1px dashed #ECEAE4;
    padding-top: 0.4rem;
    font-style: italic;
}

/* Custom Badges */
.risk-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 1px solid transparent;
}

/* Beautiful Custom Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 2px solid #E6E4DD;
    padding-bottom: 0;
}

.stTabs [data-baseweb="tab"] {
    border: 1px solid #E6E4DD;
    border-bottom: none;
    border-radius: 0 !important;
    background: #FAF9F6;
    padding: 0.65rem 1.5rem;
    color: #5C6270 !important;
    font-weight: 600;
    font-size: 0.82rem;
    transition: all 0.25s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: #FFFFFF;
    color: #1E2229 !important;
    border-color: #C5A880;
}

.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #1E2229 !important;
    border-color: #C5A880 #C5A880 transparent #C5A880 !important;
    border-width: 2px 1px 1px 1px !important;
    font-weight: 700 !important;
    box-shadow: 0 -4px 0 #C5A880;
}

/* Section Header Style */
.section-headline {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 1.2rem;
    font-weight: 700;
    color: #2D3139 !important;
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.section-headline::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #D4D1C5, transparent);
}

/* Form elements */
[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid #D4D1C5 !important;
    border-radius: 0px !important;
    color: #1E2229 !important;
}

[data-testid="stDataFrame"] {
    border-radius: 0px !important;
    border: 1px solid #ECEAE4 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.01) !important;
}

[data-testid="stRadio"] label {
    color: #4D5360 !important;
    font-weight: 500 !important;
}
</style>
"""


def inject_css() -> None:
    import re
    # Strip CSS comments to prevent markdown parser confusion
    clean = re.sub(r"/\*.*?\*/", "", CSS, flags=re.DOTALL)
    # Flatten newlines and multiple spaces into a single space
    clean = clean.replace("\n", " ")
    clean = re.sub(r"\s+", " ", clean)
    st.markdown(clean, unsafe_allow_html=True)


# ── DESIGN HELPER COMPONENTS ───────────────────────────────────────────────

def editorial_kpi(label: str, value: str, note: str = "", is_mono: bool = False) -> None:
    value_class = "card-value mono" if is_mono else "card-value"
    html = (
        f'<div class="editorial-card">'
        f'<div class="card-label">{label}</div>'
        f'<div class="{value_class}">{value}</div>'
        f'<div class="card-note">{note}</div>'
        f'</div>'
    ).replace("\n", "").replace("  ", "")
    st.markdown(html, unsafe_allow_html=True)


def status_pill(label: str, ok: bool) -> str:
    dot_class = "active" if ok else "inactive"
    value = "Active" if ok else "Offline"
    return (
        f'<span class="editorial-pill">'
        f'<span class="status-dot {dot_class}"></span>'
        f'{label}: {value}'
        f'</span>'
    ).replace("\n", "").replace("  ", "")


# ── DATA FETCHING & PIPELINE ───────────────────────────────────────────────

@st.cache_data(ttl=45, show_spinner=False)
def api_get(path: str) -> dict[str, Any] | list[dict[str, Any]]:
    response = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def safe_api_get(path: str) -> tuple[Any | None, str | None]:
    try:
        return api_get(path), None
    except requests.RequestException as exc:
        return None, str(exc)


def properties_frame(geojson: dict[str, Any]) -> pd.DataFrame:
    rows = [
        feature.get("properties", {})
        for feature in geojson.get("features", [])
        if feature.get("properties")
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "risk_tier" in df:
        df["risk_tier"] = pd.Categorical(df["risk_tier"], categories=RISK_ORDER, ordered=True)
    return df


def format_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def latest_week_label(geojson: dict[str, Any]) -> str:
    df = properties_frame(geojson)
    if df.empty or "year" not in df or "week" not in df:
        return "-"
    row = df.sort_values(["year", "week"]).iloc[-1]
    return f"{int(row['year'])}-W{int(row['week']):02d}"


def top_risk_table(df: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if df.empty:
        return df
    columns = ["district_name", "risk_tier", "predicted_cases", "risk_score"]
    optional = ["confidence_lower", "confidence_upper"]
    available = [col for col in columns + optional if col in df.columns]
    out = df.sort_values("risk_score", ascending=False).head(limit)[available].copy()
    rename = {
        "district_name": "District",
        "risk_tier": "Risk",
        "predicted_cases": "Predicted",
        "risk_score": "Score",
        "confidence_lower": "CI low",
        "confidence_upper": "CI high",
    }
    return out.rename(columns=rename)


# ── PREMIUM EDITORIAL CHARTS (WEYL / WHITE PAPER STYLE) ───────────────────

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#4D5360", family="Inter"),
)
GRID_COLOR = "#ECEAE4"


def donut_risk(df: pd.DataFrame) -> go.Figure:
    counts = df["risk_tier"].value_counts().reindex(RISK_ORDER, fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=[RISK_COLORS[t] for t in counts.index], line=dict(color="#FFFFFF", width=2)),
        hole=0.68, textinfo="none", hovertemplate="%{label}: %{value} districts<extra></extra>",
        direction="clockwise", sort=False,
    ))
    total = int(counts.sum())
    fig.update_layout(
        **CHART_LAYOUT, height=210, showlegend=False, margin=dict(l=15, r=15, t=15, b=15),
        annotations=[
            dict(
                text=f"<span style='font-family:Playfair Display, serif;font-weight:700;font-size:24px;color:#1E2229'>{total}</span><br><span style='font-size:9px;color:#8C8269;text-transform:uppercase;letter-spacing:0.05em'>Districts</span>",
                showarrow=False,
            ),
        ]
    )
    return fig


def bar_risk(df: pd.DataFrame) -> go.Figure:
    counts = df["risk_tier"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
    counts.columns = ["tier", "n"]
    fig = go.Figure(go.Bar(
        x=counts["tier"], y=counts["n"],
        marker=dict(color=[RISK_COLORS[t] for t in counts["tier"]], line_width=0),
        text=counts["n"], textposition="outside", textfont=dict(color="#4D5360", size=10, family="Inter"),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=190,
        showlegend=False,
        margin=dict(l=15, r=15, t=15, b=15),
        yaxis=dict(gridcolor=GRID_COLOR, title=None, tickfont=dict(size=9)),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9)),
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def history_figure(history: list[dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(history)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df.empty:
        fig.update_layout(**CHART_LAYOUT, height=280, margin=dict(l=15, r=15, t=15, b=15))
        return fig
    
    df["label"] = df["year"].astype(str) + "-W" + df["week"].astype(int).astype(str).str.zfill(2)
    
    # Elegant area curve for cases
    fig.add_trace(go.Scatter(
        x=df["label"], y=df["dengue_cases"], mode="lines", name="Cases",
        line=dict(color="#C94A4A", width=2, shape="spline"),
        fill="tozeroy", fillcolor="rgba(201, 74, 74, 0.05)",
    ), secondary_y=False)
    
    # Classically designed bars for rainfall
    if "rainfall_mm" in df:
        fig.add_trace(go.Bar(
            x=df["label"], y=df["rainfall_mm"], name="Rainfall",
            marker=dict(color="rgba(72, 140, 111, 0.15)", line=dict(color="rgba(72, 140, 111, 0.3)", width=0.5)),
        ), secondary_y=True)
        
    fig.update_layout(
        **CHART_LAYOUT, height=280, hovermode="x unified",
        margin=dict(l=15, r=15, t=15, b=15),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color="#5C6270", size=10)),
        showlegend=True
    )
    fig.update_yaxes(title_text="Weekly Cases", secondary_y=False, gridcolor=GRID_COLOR, title_font=dict(size=10), tickfont=dict(size=9))
    fig.update_yaxes(title_text="Rainfall (mm)", secondary_y=True, showgrid=False, title_font=dict(size=10), tickfont=dict(size=9))
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9))
    return fig


def model_charts(metrics: dict[str, Any]) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Predictive Error Metrics", "Model Performance Scores"),
        horizontal_spacing=0.15
    )
    
    fig.add_trace(go.Bar(
        x=["RMSE", "MAE"], y=[metrics.get("rmse", 0), metrics.get("mae", 0)],
        marker=dict(color=["#8C8269", "#C5A880"], line_width=0),
        text=[format_number(metrics.get("rmse"), 0), format_number(metrics.get("mae"), 0)],
        textposition="outside", textfont=dict(color="#2D3139", size=10),
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=["R² Metric", "Outbreak F1"], y=[metrics.get("r2", 0), metrics.get("outbreak_f1", 0)],
        marker=dict(color=["#A3B899", "#C94A4A"], line_width=0),
        text=[format_number(metrics.get("r2"), 3), format_number(metrics.get("outbreak_f1"), 3)],
        textposition="outside", textfont=dict(color="#2D3139", size=10),
    ), row=1, col=2)
    
    fig.update_layout(
        **CHART_LAYOUT, height=270, margin=dict(l=10, r=10, t=35, b=10), showlegend=False,
    )
    fig.update_yaxes(gridcolor=GRID_COLOR, tickfont=dict(size=9))
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9))
    fig.update_yaxes(range=[0, 1.15], row=1, col=2)
    fig.update_annotations(font=dict(color="#8C8269", size=10, family="Playfair Display"))
    return fig


# ── SENSITIVE SCIENTIFIC RISK MAP ──────────────────────────────────────────

def build_risk_map(geojson: dict[str, Any]) -> folium.Map:
    # Classic, clean light-styled geographical base layer
    fmap = folium.Map(
        location=[23.685, 90.3563], zoom_start=7,
        tiles="CartoDB positron", control_scale=True, prefer_canvas=True
    )

    def style(f: dict) -> dict:
        tier = f.get("properties", {}).get("risk_tier", "Moderate")
        c = f.get("properties", {}).get("risk_color") or RISK_COLORS.get(tier, "#D9A752")
        return {
            "fillColor": c,
            "color": "#FFFFFF",
            "weight": 1.0,
            "fillOpacity": 0.72
        }

    def highlight(_: dict) -> dict:
        return {
            "weight": 2.2,
            "color": "#1E2229",
            "fillOpacity": 0.85
        }

    folium.GeoJson(
        geojson, name="risk", style_function=style, highlight_function=highlight,
        tooltip=folium.GeoJsonTooltip(
            fields=["district_name", "risk_tier", "predicted_cases", "risk_score"],
            aliases=["District", "Risk Category", "Predicted Cases", "Infection Score"],
            localize=True, sticky=False,
        ),
    ).add_to(fmap)
    return fmap


# ── LAYOUT SECTIONS ────────────────────────────────────────────────────────

def render_header(health: dict[str, Any] | None) -> None:
    pills_html = ""
    if health:
        pills = [
            status_pill("Postgres DB", bool(health.get("db_ok"))),
            status_pill("Redis Cache", bool(health.get("redis_ok"))),
            status_pill("MLflow Registry", bool(health.get("mlflow_ok"))),
            f'<span class="editorial-pill">'
            f'  <span class="status-dot active" style="background:#C5A880"></span>'
            f'  Data Freshness: {health.get("data_freshness") or "—"}'
            f'</span>'
        ]
        pills_html = "".join(pills)
    else:
        pills_html = '<span class="editorial-pill"><span class="status-dot inactive"></span>Surveillance API: Down</span>'
        
    hdr_html = (
        f'<div class="header-container">'
        f'<div class="title-area">'
        f'<div class="editorial-tag">National Disease Intelligence Report</div>'
        f'<div class="editorial-title">Dengue Epidemic Prediction Portal</div>'
        f'<div class="editorial-subtitle">Gis-Based Predictive Modeling and Risk Assessment for Bangladesh</div>'
        f'</div>'
        f'<div class="status-panel">{pills_html}</div>'
        f'</div>'
    ).replace("\n", "").replace("  ", "")
    st.markdown(hdr_html, unsafe_allow_html=True)


def render_kpi_row(health, metrics, geojson):
    df = properties_frame(geojson or {})
    total = df["predicted_cases"].sum() if "predicted_cases" in df else None
    high_plus = int(df["risk_tier"].isin(["High", "Critical"]).sum()) if "risk_tier" in df else 0
    crit = int((df["risk_tier"] == "Critical").sum()) if "risk_tier" in df else 0
    week = latest_week_label(geojson or {})

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        editorial_kpi("Surveillance API Node", (health or {}).get("status", "—").upper(), API_URL)
    with c2:
        editorial_kpi("Forecasting Window", week, "Current epidemiological week", is_mono=True)
    with c3:
        editorial_kpi("Aggregated Predictions", format_number(total, 0), f"Calculated across {len(df)} districts")
    with c4:
        editorial_kpi("Elevated Risk Vectors", str(high_plus), f"{crit} classified as critical alert")
    with c5:
        editorial_kpi("Historical Model RMSE", format_number((metrics or {}).get("rmse"), 1), "Production Registry v" + str((metrics or {}).get("version", "?")))


def render_map_tab(current_geojson):
    # High-end layout selections
    mode = st.radio("Temporal Horizon Offset", ["Current Week", "+1 Week Forecast", "+2 Weeks Forecast", "+3 Weeks Forecast", "+4 Weeks Forecast"],
                    horizontal=True, label_visibility="collapsed")
    weeks_map = {"Current Week": 0, "+1 Week Forecast": 1, "+2 Weeks Forecast": 2, "+3 Weeks Forecast": 3, "+4 Weeks Forecast": 4}
    weeks = weeks_map[mode]
    geojson = current_geojson
    if weeks > 0:
        geojson, err = safe_api_get(f"/risk/forecast/{weeks}")
        if err:
            st.warning(f"Predictive vector unavailable: {err}")
            return

    df = properties_frame(geojson or {})
    if df.empty:
        st.info("No spatial data vectors returned.")
        return

    left, right = st.columns([1.8, 1], gap="large")
    with left:
        st_folium(build_risk_map(geojson), height=580, use_container_width=True, returned_objects=[])
    with right:
        st.markdown('<div class="section-headline">Statistical Distribution</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(donut_risk(df), use_container_width=True)
        with c2:
            st.plotly_chart(bar_risk(df), use_container_width=True)

        st.markdown('<div class="section-headline">Priority Surveillance Alerts</div>', unsafe_allow_html=True)
        st.dataframe(
            top_risk_table(df), use_container_width=True, hide_index=True,
            column_config={
                "Predicted": st.column_config.NumberColumn(format="%.1f"),
                "Score": st.column_config.NumberColumn(format="%.1f"),
                "CI low": st.column_config.NumberColumn(format="%.1f"),
                "CI high": st.column_config.NumberColumn(format="%.1f"),
            },
        )


def render_district_tab(districts):
    if not districts:
        st.warning("Spatial registry is currently blank.")
        return
    lookup = {f"{d['district_name']} ({d['division_name']})": d for d in districts}
    default = next((k for k in lookup if k.startswith("Dhaka ")), sorted(lookup)[0])
    
    # Spacious search layout
    sel = st.selectbox("Geographical Query Selector", sorted(lookup), index=sorted(lookup).index(default))
    detail, err = safe_api_get(f"/district/{lookup[sel]['district_id']}")
    if err:
        st.warning(f"District database offline: {err}")
        return

    dist = detail.get("district", {})
    pred = detail.get("latest_prediction") or {}
    hist = detail.get("history") or []

    tier = pred.get("risk_tier", "-")
    tier_color = RISK_COLORS.get(tier, "#64748b")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        editorial_kpi("Queried Territory", dist.get("district_name", "-"), f"Division: {dist.get('division_name', '-')}")
    with c2:
        badge_html = f'<span class="risk-badge" style="background-color:{tier_color}12; color:{tier_color}; border-color:{tier_color}33;">{RISK_EMOJI.get(tier, "")} {tier}</span>'
        dist_html = (
            f'<div class="editorial-card">'
            f'<div class="card-label">Assessed Risk Tier</div>'
            f'<div>{badge_html}</div>'
            f'<div class="card-note">Week: {pred.get("year", "-")}-W{pred.get("week", "-")}</div>'
            f'</div>'
        ).replace("\n", "").replace("  ", "")
        st.markdown(dist_html, unsafe_allow_html=True)
    with c3:
        editorial_kpi("Inference Case Rate", format_number(pred.get("predicted_cases"), 1), "Mean weekly prediction value")
    with c4:
        editorial_kpi("Total Territory Area", f'{format_number(dist.get("area_km2"), 0)} km²', "Calculated boundary size")

    ch_col, tbl_col = st.columns([1.6, 1], gap="large")
    with ch_col:
        st.markdown('<div class="section-headline">Epidemiological Baseline Timeline</div>', unsafe_allow_html=True)
        st.plotly_chart(history_figure(hist), use_container_width=True)
    with tbl_col:
        st.markdown('<div class="section-headline">Registered Weekly Metrics</div>', unsafe_allow_html=True)
        hdf = pd.DataFrame(hist)
        if not hdf.empty:
            hdf = hdf.rename(columns={"year": "Year", "week": "Wk", "dengue_cases": "Cases",
                                       "temp_mean_c": "Temp °C", "rainfall_mm": "Rain mm"})
        st.dataframe(hdf, use_container_width=True, hide_index=True, height=280)


def render_model_tab(health, metrics):
    if not metrics:
        st.warning("Evaluation metrics repository unavailable.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        editorial_kpi("Algorithm Paradigm", f'v{metrics.get("version", "?")} XGBoost', f"Registry: {metrics.get('stage', '-')}")
    with c2:
        editorial_kpi("Root Mean Squared Error", format_number(metrics.get("rmse"), 1), "Historical weekly deviation", is_mono=True)
    with c3:
        editorial_kpi("Mean Absolute Error", format_number(metrics.get("mae"), 1), "Historical median absolute deviation", is_mono=True)
    with c4:
        f1 = metrics.get("outbreak_f1")
        color = "#488C6F" if f1 and f1 > 0.6 else "#D9A752"
        f1_html = (
            f'<div class="editorial-card">'
            f'<div class="card-label">Outbreak Classification (F1)</div>'
            f'<div class="card-value mono" style="color:{color} !important;">{format_number(f1, 3)}</div>'
            f'<div class="card-note">Evaluated at ≥20 weekly cases threshold</div>'
            f'</div>'
        ).replace("\n", "").replace("  ", "")
        st.markdown(f1_html, unsafe_allow_html=True)

    st.markdown('<div class="section-headline">Model Evaluation Summary</div>', unsafe_allow_html=True)
    st.plotly_chart(model_charts(metrics), use_container_width=True)

    if health:
        st.caption(
            f"Active Model Variant: {health.get('model_version') or '—'} · "
            f"Ground Truth Registry: {health.get('data_freshness') or '—'}"
        )


# ── MAIN SURVEILLANCE RUNNER ───────────────────────────────────────────────

def main() -> None:
    inject_css()

    health, health_err = safe_api_get("/health")
    metrics, _ = safe_api_get("/metrics/model")
    districts, _ = safe_api_get("/districts")
    current_geojson, map_err = safe_api_get("/risk/current")

    render_header(health)

    if health_err:
        st.error(f"Surveillance API node unreachable at `{API_URL}` — {health_err}")
        st.stop()

    render_kpi_row(health, metrics, current_geojson)

    if map_err:
        st.warning(f"Spatial geography vectors missing: {map_err}")

    tab_map, tab_dist, tab_model = st.tabs(["🗺  Risk Assessment GIS", "📊  Territorial Deep-Dive", "🤖  Predictive Engine Profile"])
    with tab_map:
        render_map_tab(current_geojson)
    with tab_dist:
        render_district_tab(districts or [])
    with tab_model:
        render_model_tab(health, metrics)


if __name__ == "__main__":
    main()

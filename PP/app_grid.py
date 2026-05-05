"""
app_grid.py — Algeria Green Hydrogen LCOH Grid Map
Run: streamlit run app_grid.py
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import math

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Algeria H₂ LCOH Map",
    page_icon="🇩🇿",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
:root {
    --bg:#070d1a; 
    --bg2:#0d1829; 
    --bg3:#111f35;
    --border:#1e3050; 
    --amber:#f5a623; 
    --amber2:#ffd07a;
    --teal:#00d4b8; 
    --red:#ff4d6d;
    --text:#c8d8f0;
    --muted:#5a7a9a; 
    --white:#eaf2ff;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'Syne',sans-serif;}
[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
h1,h2,h3{font-family:'Syne',sans-serif!important;color:var(--white)!important;}
[data-testid="metric-container"]{background:var(--bg3)!important;border:1px solid var(--border)!important;border-radius:8px!important;padding:12px 16px!important;}
[data-testid="metric-container"] label{color:var(--muted)!important;font-size:11px!important;letter-spacing:.08em;text-transform:uppercase;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--amber)!important;font-family:'Space Mono',monospace!important;font-size:1.5rem!important;}
.stButton>button{background:linear-gradient(135deg,var(--amber),#e08800)!important;color:#070d1a!important;font-family:'Space Mono',monospace!important;font-weight:700!important;border:none!important;border-radius:6px!important;padding:10px 24px!important;width:100%;}
[data-testid="stSelectbox"]>div>div{background:var(--bg3)!important;border:1px solid var(--border)!important;color:var(--white)!important;border-radius:6px!important;}
[data-testid="stTabs"] button{font-family:'Space Mono',monospace!important;font-size:12px!important;color:var(--muted)!important;letter-spacing:.06em;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--amber)!important;border-bottom:2px solid var(--amber)!important;}
.info-box{background:var(--bg3);border-left:3px solid var(--amber);border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;font-size:13px;}
.warn-box{background:#1a0d0d;border-left:3px solid var(--red);border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;font-size:13px;color:#ffaaaa;}
.section-title{font-family:'Space Mono',monospace;font-size:11px;letter-spacing:.15em;color:var(--muted);text-transform:uppercase;border-bottom:1px solid var(--border);padding-bottom:6px;margin:20px 0 12px 0;}
.stat-card{background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:16px;text-align:center;}
.stat-card .val{font-family:'Space Mono',monospace;font-size:1.4rem;color:var(--amber);}
.stat-card .lbl{font-size:11px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-top:4px;}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; [data-testid="collapsedControl"] {
    background: #f5a623 !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    top: 10px !important;
    color: #070d1a !important;
}

[data-testid="collapsedControl"]:hover {
    background: #ffd07a !important;
    transform: scale(1.1);
    transition: all 0.2s ease;
}
.block-container{padding-top:1.5rem!important;}
</style>
""", unsafe_allow_html=True)

PLOT_BG = dict(
    paper_bgcolor="#0d1829", plot_bgcolor="#070d1a",
    font=dict(family="Space Mono, monospace", color="#c8d8f0", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
)

DATA_DIR   = __import__("pathlib").Path(__file__).parent / "data"
GRID_PATH  = DATA_DIR / "lcoh_grid.csv"
STATS_PATH = DATA_DIR / "grid_cf_stats.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_cell_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cell_id" not in df.columns:
        df["cell_id"] = df["lat"].round(4).astype(str) + "_" + df["lon"].round(4).astype(str)
    df["cell_id"] = df["cell_id"].astype(str)
    return df


def detect_resolution(df: pd.DataFrame) -> float:
    lons = np.sort(df["lon"].unique())
    return float(np.round(np.median(np.diff(lons)), 4)) if len(lons) > 1 else 1.0


def make_complete_grid(df: pd.DataFrame, resolution: float) -> pd.DataFrame:
    """All (lat,lon) points in the bounding box — NaN for cells not in df."""
    half = resolution / 2.0
    lats = np.round(np.arange(
        np.floor(df["lat"].min() / resolution) * resolution,
        np.ceil (df["lat"].max() / resolution) * resolution + half,
        resolution), 6)
    lons = np.round(np.arange(
        np.floor(df["lon"].min() / resolution) * resolution,
        np.ceil (df["lon"].max() / resolution) * resolution + half,
        resolution), 6)

    full = pd.DataFrame(
        [(round(float(la), 4), round(float(lo), 4)) for la in lats for lo in lons],
        columns=["lat", "lon"],
    )
    full["cell_id"] = full["lat"].astype(str) + "_" + full["lon"].astype(str)

    df2 = df.copy()
    df2["lat"] = df2["lat"].round(4)
    df2["lon"] = df2["lon"].round(4)
    merged = full.merge(df2, on=["lat", "lon"], how="left", suffixes=("", "_d"))
    if "cell_id_d" in merged.columns:
        merged.drop(columns=["cell_id_d"], inplace=True)
    return merged


@st.cache_data
def build_geojson(lats, lons, cell_ids, resolution: float) -> dict:
    """One exact-size GeoJSON polygon per cell — neighbours share edges."""
    half = resolution / 2.0
    features = []
    for lat, lon, cid in zip(lats, lons, cell_ids):
        features.append({
            "type": "Feature",
            "id": str(cid),
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon - half, lat - half],
                    [lon + half, lat - half],
                    [lon + half, lat + half],
                    [lon - half, lat + half],
                    [lon - half, lat - half],
                ]],
            },
        })
    return {"type": "FeatureCollection", "features": features}


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_grid() -> pd.DataFrame:
    if not GRID_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(GRID_PATH)
    df = df[df["lcoh_usd_per_kg"].notna() & (df["lcoh_usd_per_kg"] < 50)]
    return ensure_cell_id(df)


@st.cache_data
def load_stats() -> pd.DataFrame:
    if not STATS_PATH.exists():
        return pd.DataFrame()
    return ensure_cell_id(pd.read_csv(STATS_PATH, index_col=0))


# ── Financial rescaling ───────────────────────────────────────────────────────

def rescale_lcoh(base_df, new_discount, new_lifetime, new_solar_capex,
                 new_elec_capex, new_elec_eff, new_grid_price, new_dzd_usd,
                 base_discount=0.08, base_lifetime=25, base_solar_capex=600.0,
                 base_elec_capex=700.0, base_elec_eff=55.0,
                 base_grid_price=9.0,  base_dzd_usd=134.5) -> pd.DataFrame:
    df = base_df.copy()

    def crf(r, n):
        return (1/n) if r == 0 else r*(1+r)**n / ((1+r)**n - 1)

    crf_r   = crf(new_discount, new_lifetime) / crf(base_discount, base_lifetime)
    sol_r   = new_solar_capex / base_solar_capex
    elec_r  = (new_elec_capex * (new_elec_eff / base_elec_eff)) / base_elec_capex
    grid_r  = (new_grid_price / new_dzd_usd) / (base_grid_price / base_dzd_usd)

    df["cost_solar_scaled"]  = df["cost_solar"]       * crf_r * sol_r
    df["cost_wind_scaled"]   = df["cost_wind"]         * crf_r
    df["cost_elec_scaled"]   = df["cost_electrolyzer"] * crf_r * elec_r
    df["cost_h2t_scaled"]    = df["cost_h2_storage"]   * crf_r
    df["cost_bat_scaled"]    = df["cost_battery"]      * crf_r
    df["cost_grid_scaled"]   = df["cost_grid"]         * grid_r

    df["total_scaled"]    = (df["cost_solar_scaled"] + df["cost_wind_scaled"] +
                             df["cost_elec_scaled"]  + df["cost_h2t_scaled"]  +
                             df["cost_bat_scaled"]   + df["cost_grid_scaled"])
    h2_kg = df["h2_produced_kg"].replace(0, np.nan)
    df["lcoh_scaled"]     = df["total_scaled"] / h2_kg
    df["lcoh_dzd_scaled"] = df["lcoh_scaled"] * new_dzd_usd
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🇩🇿 H₂ LCOH Grid Map")
    st.markdown("<div style='color:#5a7a9a;font-size:12px;margin-bottom:16px'>Algeria — Research Tool</div>",
                unsafe_allow_html=True)

    grid_df  = load_grid()
    stats_df = load_stats()
    has_data = len(grid_df) > 0

    if has_data:
        st.markdown(f"<div class='info-box'>✅ Grid loaded — <b>{len(grid_df)} cells</b></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("""<div class='warn-box'>⚠️ No grid found.<br>
            Run: <code>python atlite_grid.py</code><br>
            Then: <code>python pypsa_grid.py</code></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🗺️ Map Display</div>", unsafe_allow_html=True)
    map_metric = st.selectbox("Color cells by", [
        "LCOH (USD/kg H₂)", "LCOH (DZD/kg H₂)", "Solar Full Load Hours",
        "Wind CF Mean", "Renewable Share (%)",
        "Electrolyzer Utilization (%)", "Solar Capacity (kW)",
    ])
    colorscale = st.selectbox("Color scale",
        ["RdYlGn_r", "Viridis", "Plasma", "Turbo", "RdBu_r", "YlOrRd"], index=0)

    # ── Opacity slider so user can tune how much map shows through ────────────
    cell_opacity = st.slider("Cell opacity (lower = see map through)", 0.2, 1.0, 0.55, 0.05)

    st.markdown("<div class='section-title'>💰 Financials</div>", unsafe_allow_html=True)
    discount_rate    = st.slider("Discount Rate / WACC (%)", 3, 20, 8) / 100
    project_lifetime = st.slider("Project Lifetime (yr)", 10, 30, 25)
    grid_price_dzd   = st.slider("Grid Price (DZD/kWh)", 3.0, 25.0, 9.0, 0.5)
    dzd_to_usd       = st.number_input("DZD / USD", value=134.5, step=1.0)

    st.markdown("<div class='section-title'>⚙️ Technology Costs</div>", unsafe_allow_html=True)
    solar_capex     = st.slider("Solar CAPEX (USD/kW)",             200,  1500, 600,  25)
    elec_capex      = st.slider("Electrolyzer CAPEX (USD/kW)",      200,  2000, 700,  50)
    elec_efficiency = st.slider("Electrolyzer Efficiency (kWh/kg)", 40.0, 80.0, 55.0, 1.0)

    st.markdown("---")
    apply_btn = st.button("🔄 APPLY TO MAP", use_container_width=True)

    st.markdown("<div class='section-title'>📥 Export</div>", unsafe_allow_html=True)
    if has_data:
        st.download_button("⬇️ Download CSV", data=grid_df.to_csv(index=False),
                           file_name="algeria_lcoh_grid.csv", mime="text/csv",
                           use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🇩🇿 Algeria — Green Hydrogen LCOH Grid Map")
st.markdown("<p style='color:#5a7a9a;margin-top:-12px;font-size:14px'>"
            "Spatial Levelized Cost of Hydrogen · ERA5 Weather · PyPSA Optimization</p>",
            unsafe_allow_html=True)

if not has_data:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class='info-box'><b>Step 1 — CDS API Key</b><br><br>
        Register at <a href='https://cds.climate.copernicus.eu' target='_blank'>cds.climate.copernicus.eu</a>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='info-box'><b>Step 2 — Download ERA5</b><br><br>
        <code>python atlite_grid.py</code><br>~2–4 GB, once only.</div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='info-box'><b>Step 3 — Compute LCOH</b><br><br>
        <code>python pypsa_grid.py</code><br>~9 min, once only.</div>""", unsafe_allow_html=True)
    st.info("Refresh after running both scripts.")
    st.stop()


# ── Rescale ───────────────────────────────────────────────────────────────────
if "scaled_df" not in st.session_state or apply_btn:
    with st.spinner("Rescaling LCOH..."):
        st.session_state.scaled_df = rescale_lcoh(
            grid_df, discount_rate, project_lifetime, solar_capex,
            elec_capex, elec_efficiency, grid_price_dzd, dzd_to_usd,
        )

scaled_df = st.session_state.scaled_df
valid = scaled_df[scaled_df["lcoh_scaled"].notna() & (scaled_df["lcoh_scaled"] < 50)]

# ── KPIs ──────────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🏆 Min LCOH",   f"${valid['lcoh_scaled'].min():.2f}/kg")
m2.metric("📊 Mean LCOH",  f"${valid['lcoh_scaled'].mean():.2f}/kg")
m3.metric("📈 Max LCOH",   f"${valid['lcoh_scaled'].max():.2f}/kg")
m4.metric("🔢 Grid Cells", f"{len(valid)}")
m5.metric("♻️ Max Renew.", f"{valid['pct_renewable'].max():.0f}%")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["LCOH MAP", "SOLAR RESOURCE", "STATISTICS", "⚡ SITE SUITABILITY", "🗺️ INFRASTRUCTURE"])

# ══════════════════════════════════════════════════════════════════════════════
#  GRID MAP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def make_grid_map(df_data: pd.DataFrame, col_name: str, col_label: str,
                  cs: str, title: str, opacity: float = 0.55,
                  height: int = 640) -> go.Figure:

    resolution = detect_resolution(df_data)

    full_grid = make_complete_grid(df_data, resolution)
    geo_bg = build_geojson(
        tuple(full_grid["lat"].round(4)),
        tuple(full_grid["lon"].round(4)),
        tuple(full_grid["cell_id"]),
        resolution,
    )

    df_plot = df_data[df_data[col_name].notna()].copy()
    df_plot["cell_id"] = df_plot["cell_id"].astype(str)
    geo_data = build_geojson(
        tuple(df_plot["lat"].round(4)),
        tuple(df_plot["lon"].round(4)),
        tuple(df_plot["cell_id"]),
        resolution,
    )

    flh   = df_plot.get("solar_full_load_hours", pd.Series(0, index=df_plot.index))
    renew = df_plot.get("pct_renewable",         pd.Series(0, index=df_plot.index))
    hover = (
        "<b>Cell " + df_plot["cell_id"] + "</b><br>"
        + "📍 " + df_plot["lat"].round(2).astype(str) + "°N, "
        + df_plot["lon"].round(2).astype(str) + "°E<br>"
        + col_label + ": <b>" + df_plot[col_name].round(3).astype(str) + "</b><br>"
        + "Solar FLH: " + flh.round(0).astype(str) + " h/yr<br>"
        + "Renew: " + renew.round(1).astype(str) + "%"
    )

    fig = go.Figure()

    fig.add_trace(go.Choroplethmapbox(
        geojson=geo_bg,
        locations=full_grid["cell_id"],
        z=[0] * len(full_grid),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        showscale=False,
        marker_opacity=1.0,
        marker_line_width=0.8,
        marker_line_color="#2a4060",
        hoverinfo="skip",
        name="",
    ))

    fig.add_trace(go.Choroplethmapbox(
        geojson=geo_data,
        locations=df_plot["cell_id"],
        z=df_plot[col_name],
        colorscale=cs,
        zmin=float(df_plot[col_name].min()),
        zmax=float(df_plot[col_name].max()),
        marker_opacity=opacity,
        marker_line_width=0.8,
        marker_line_color="#2a4060",
        colorbar=dict(
            title=dict(text=col_label, font=dict(color="#c8d8f0", size=11)),
            tickfont=dict(color="#c8d8f0"),
            bgcolor="#0d1829", bordercolor="#1e3050", thickness=14,
        ),
        text=hover,
        hoverinfo="text",
        name=col_label,
    ))

    fig.update_layout(
        **PLOT_BG,
        mapbox=dict(style="carto-darkmatter", center=dict(lat=28.0, lon=2.5), zoom=4.2),
        height=height,
        showlegend=False,
        title=dict(
            text=title,
            font=dict(color="#f5a623", size=14, family="Space Mono"),
            x=0.01,
        ),
    )
    return fig
# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LCOH MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    METRIC_MAP = {
        "LCOH (USD/kg H₂)":            ("lcoh_scaled",                  "LCOH USD/kg"),
        "LCOH (DZD/kg H₂)":            ("lcoh_dzd_scaled",              "LCOH DZD/kg"),
        "Solar Full Load Hours":        ("solar_full_load_hours",        "FLH/yr"),
        "Wind CF Mean":                 ("wind_cf_mean",                 "Wind CF"),
        "Renewable Share (%)":          ("pct_renewable",                "Renew. %"),
        "Electrolyzer Utilization (%)": ("electrolyzer_utilization_pct", "Elec. util %"),
        "Solar Capacity (kW)":          ("solar_capacity_kw",            "Solar kW"),
    }
    col_name, col_label = METRIC_MAP[map_metric]
    if col_name not in scaled_df.columns:
        col_name, col_label = "lcoh_usd_per_kg", "LCOH USD/kg"

    fig = make_grid_map(scaled_df, col_name, col_label, colorscale,
                        f"Algeria — {map_metric}", opacity=cell_opacity, height=640)
    st.plotly_chart(fig, use_container_width=True)

    best5 = (
        valid.nsmallest(5, "lcoh_scaled")
        [["lat","lon","lcoh_scaled","lcoh_dzd_scaled","solar_full_load_hours","pct_renewable"]]
        .round({"lcoh_scaled":3,"lcoh_dzd_scaled":0,"solar_full_load_hours":0,"pct_renewable":1})
    )
    best5.columns = ["Lat","Lon","LCOH USD/kg","LCOH DZD/kg","Solar FLH","Renew. %"]
    st.markdown("#### 🏆 Top 5 Lowest-LCOH Cells")
    st.dataframe(best5, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SOLAR RESOURCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if len(stats_df) > 0:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### ☀️ Solar Full Load Hours")
            st.plotly_chart(
                make_grid_map(stats_df, "solar_full_load_hours", "FLH/yr",
                              "YlOrRd", "Solar PV Full Load Hours (ERA5)",
                              opacity=cell_opacity, height=420),
                use_container_width=True)
        with c_right:
            st.markdown("#### 💨 Wind Capacity Factor")
            st.plotly_chart(
                make_grid_map(stats_df, "wind_cf_mean", "Wind CF",
                              "Blues", "Wind CF — Vestas V112 3MW (ERA5)",
                              opacity=cell_opacity, height=420),
                use_container_width=True)
    else:
        st.info("Run atlite_grid.py first to generate resource maps.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### LCOH Distribution")
    fig_hist = go.Figure(go.Histogram(
        x=valid["lcoh_scaled"], nbinsx=30,
        marker=dict(color="#f5a623", line=dict(color="#070d1a", width=0.5)),
        hovertemplate="LCOH: $%{x:.2f}/kg<br>Cells: %{y}<extra></extra>",
    ))
    fig_hist.add_vline(x=valid["lcoh_scaled"].mean(), line_dash="dot", line_color="#00d4b8",
                       annotation_text=f"Mean: ${valid['lcoh_scaled'].mean():.2f}",
                       annotation_font_color="#00d4b8")
    fig_hist.update_layout(**PLOT_BG,
        xaxis=dict(title="LCOH (USD/kg H₂)", gridcolor="#1e3050"),
        yaxis=dict(title="Grid Cells", gridcolor="#1e3050"), height=320,
        title=dict(text="LCOH Distribution Across Algeria", font=dict(color="#f5a623")))
    st.plotly_chart(fig_hist, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### LCOH vs Latitude")
        fig_lat = go.Figure(go.Scatter(
            x=valid["lat"], y=valid["lcoh_scaled"], mode="markers",
            marker=dict(color=valid["lcoh_scaled"], colorscale="RdYlGn_r",
                        size=8, line=dict(color="#070d1a", width=0.5), showscale=False),
            hovertemplate="Lat: %{x:.1f}°N<br>LCOH: $%{y:.3f}/kg<extra></extra>"))
        fig_lat.update_layout(**PLOT_BG, height=300,
            xaxis=dict(title="Latitude (°N)", gridcolor="#1e3050"),
            yaxis=dict(title="LCOH (USD/kg)", gridcolor="#1e3050"))
        st.plotly_chart(fig_lat, use_container_width=True)

    with c2:
        st.markdown("#### LCOH vs Solar FLH")
        fig_flh = go.Figure(go.Scatter(
            x=valid["solar_full_load_hours"], y=valid["lcoh_scaled"], mode="markers",
            marker=dict(color=valid["lat"], colorscale="Viridis", size=8,
                        line=dict(color="#070d1a", width=0.5),
                        colorbar=dict(title="Lat", thickness=10,
                                      tickfont=dict(color="#c8d8f0"))),
            hovertemplate="FLH: %{x:.0f} h/yr<br>LCOH: $%{y:.3f}/kg<extra></extra>"))
        fig_flh.update_layout(**PLOT_BG, height=300,
            xaxis=dict(title="Solar Full Load Hours (h/yr)", gridcolor="#1e3050"),
            yaxis=dict(title="LCOH (USD/kg)", gridcolor="#1e3050"))
        st.plotly_chart(fig_flh, use_container_width=True)

    st.markdown("#### Full Grid Results")
    display_cols = ["lat","lon","lcoh_scaled","lcoh_dzd_scaled",
                    "solar_full_load_hours","wind_cf_mean",
                    "pct_renewable","solar_capacity_kw","electrolyzer_capacity_kw"]
    display_cols = [c for c in display_cols if c in valid.columns]
    rename = {"lcoh_scaled":"LCOH USD/kg","lcoh_dzd_scaled":"LCOH DZD/kg",
              "solar_full_load_hours":"Solar FLH","wind_cf_mean":"Wind CF",
              "pct_renewable":"Renew. %","solar_capacity_kw":"Solar kW",
              "electrolyzer_capacity_kw":"Elec. kW"}
    tbl = valid[display_cols].rename(columns=rename).sort_values("LCOH USD/kg")
    st.dataframe(tbl.round(3), use_container_width=True, hide_index=True)
# ============================================================
# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ELECTROLYZER SITE SUITABILITY
# ══════════════════════════════════════════════════════════════════════════════
import math

SOLAR_STATIONS = [
    {"name": "Adrar Solar Plant",      "lat": 27.9077, "lon": -0.3174, "mw": 233,  "type": "Solar"},
    {"name": "In Salah Solar",         "lat": 27.1830, "lon":  2.5040, "mw": 99,   "type": "Solar"},
    {"name": "Tindouf Solar",          "lat": 27.7520, "lon": -8.1540, "mw": 9,    "type": "Solar"},
    {"name": "Timimoune Solar",        "lat": 29.2639, "lon":  0.2306, "mw": 9,    "type": "Solar"},
    {"name": "Zaouiet Kounta Solar",   "lat": 27.9500, "lon": -0.1833, "mw": 6,    "type": "Solar"},
    {"name": "Reggane Solar",          "lat": 26.7167, "lon":  0.1667, "mw": 5,    "type": "Solar"},
    {"name": "Aoulef Solar",           "lat": 26.9667, "lon":  1.0833, "mw": 5,    "type": "Solar"},
    {"name": "Tsabit Solar",           "lat": 28.3833, "lon": -0.0667, "mw": 3,    "type": "Solar"},
    {"name": "Oued El Kebrit Solar",   "lat": 35.9194, "lon":  7.8711, "mw": 50,   "type": "Solar"},
    {"name": "Hassi R'Mel ISCC",       "lat": 33.1247, "lon":  3.3519, "mw": 150,  "type": "Solar"},
    {"name": "Saida Solar PV Park",    "lat": 34.8300, "lon":  0.1500, "mw": 29,   "type": "Solar"},
    {"name": "Ghardaia Solar",         "lat": 32.4833, "lon":  3.6667, "mw": 1,    "type": "Solar"},
]

WIND_STATIONS = [
    {"name": "Kabertene Wind Farm",    "lat": 28.4624, "lon": -0.0576, "mw": 10.2, "type": "Wind"},
    {"name": "Adrar Wind Farm",        "lat": 27.8700, "lon": -0.2900, "mw": 10,   "type": "Wind"},
]

ALL_STATIONS = SOLAR_STATIONS + WIND_STATIONS

CITIES = [
    {"name": "Algiers",        "lat": 36.7372, "lon":  3.0865},
    {"name": "Oran",           "lat": 35.6969, "lon": -0.6331},
    {"name": "Constantine",    "lat": 36.3650, "lon":  6.6147},
    {"name": "Annaba",         "lat": 36.9000, "lon":  7.7667},
    {"name": "Setif",          "lat": 36.1898, "lon":  5.4108},
    {"name": "Batna",          "lat": 35.5500, "lon":  6.1667},
    {"name": "Skikda",         "lat": 36.8761, "lon":  6.9069},
    {"name": "Ghardaia",       "lat": 32.4833, "lon":  3.6667},
    {"name": "Ouargla",        "lat": 31.9500, "lon":  5.3167},
    {"name": "Hassi Messaoud", "lat": 31.7000, "lon":  6.0500},
    {"name": "Tlemcen",        "lat": 34.8800, "lon": -1.3200},
    {"name": "Bechar",         "lat": 31.6167, "lon": -2.2167},
    {"name": "Tamanrasset",    "lat": 22.7850, "lon":  5.5228},
    {"name": "Adrar",          "lat": 27.8741, "lon": -0.2914},
    {"name": "In Salah",       "lat": 27.1956, "lon":  2.4703},
]

GRID_LINES = [
    [{"lat": 36.74, "lon": 3.09}, {"lat": 35.20, "lon": 3.40},
     {"lat": 33.80, "lon": 3.55}, {"lat": 32.48, "lon": 3.67}, {"lat": 31.95, "lon": 5.32}],
    [{"lat": 36.74, "lon": 3.09}, {"lat": 36.50, "lon": 4.50},
     {"lat": 36.37, "lon": 6.61}, {"lat": 36.90, "lon": 7.77}],
    [{"lat": 36.74, "lon": 3.09}, {"lat": 36.20, "lon": 1.80}, {"lat": 35.70, "lon": -0.63}],
    [{"lat": 35.70, "lon": -0.63}, {"lat": 34.10, "lon": -1.50}, {"lat": 31.62, "lon": -2.22}],
    [{"lat": 36.19, "lon": 5.41}, {"lat": 36.37, "lon": 6.61}],
    [{"lat": 31.70, "lon": 6.05}, {"lat": 31.95, "lon": 5.32}],
    [{"lat": 36.74, "lon": 3.09}, {"lat": 35.55, "lon": 6.17}, {"lat": 35.50, "lon": 6.17}],
    [{"lat": 33.12, "lon": 3.35}, {"lat": 32.48, "lon": 3.67}],
]


def _hav(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _min_to_lines(lat, lon, lines):
    best = float("inf")
    for line in lines:
        for pt in line:
            best = min(best, _hav(lat, lon, pt["lat"], pt["lon"]))
    return best


def _min_to_pts(lat, lon, pts):
    if not pts:
        return 9999.0
    return min(_hav(lat, lon, p["lat"], p["lon"]) for p in pts)


with tab4:
    st.markdown("## ⚡ Electrolyzer Site Suitability")
    st.markdown(
        '<p style="color:#5a7a9a;font-size:13px;margin-top:-10px">'
        "Composite scoring: LCOH · Grid proximity · Renewable stations · H₂ demand centres"
        "</p>", unsafe_allow_html=True
    )

    if not has_data:
        st.warning("No LCOH grid found. Run atlite_grid.py then pypsa_grid.py first.")
        st.stop()

    st.sidebar.markdown('<div class="section-title">Site Scoring Weights</div>', unsafe_allow_html=True)
    w_lcoh  = st.sidebar.slider("↓ LCOH (lower=better)",    0, 10, 4, key="s_lcoh")
    w_grid  = st.sidebar.slider("↓ Grid line distance",     0, 10, 3, key="s_grid")
    w_renew = st.sidebar.slider("↓ RE station distance",    0, 10, 2, key="s_renew")
    w_city  = st.sidebar.slider("↓ City/demand distance",   0, 10, 1, key="s_city")
    top_n   = st.sidebar.slider("Top N sites",              3, 20, 8, key="s_topn")
    show_gridlines = st.sidebar.checkbox("Show grid lines",  value=True, key="cb_gl")
    show_stations  = st.sidebar.checkbox("Show RE stations", value=True, key="cb_st")
    show_cities    = st.sidebar.checkbox("Show cities",      value=True, key="cb_ct")

    df_s = scaled_df[scaled_df["lcoh_scaled"].notna() & (scaled_df["lcoh_scaled"] < 50)].copy()

    df_s["dist_grid_km"]  = df_s.apply(lambda r: _min_to_lines(r["lat"], r["lon"], GRID_LINES), axis=1)
    df_s["dist_renew_km"] = df_s.apply(lambda r: _min_to_pts(r["lat"], r["lon"], ALL_STATIONS), axis=1)
    df_s["dist_city_km"]  = df_s.apply(lambda r: _min_to_pts(r["lat"], r["lon"], CITIES), axis=1)

    def _norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    df_s["n_lcoh"]  = _norm(df_s["lcoh_scaled"])
    df_s["n_grid"]  = _norm(df_s["dist_grid_km"])
    df_s["n_renew"] = _norm(df_s["dist_renew_km"])
    df_s["n_city"]  = _norm(df_s["dist_city_km"])

    tw = max(w_lcoh + w_grid + w_renew + w_city, 1)
    df_s["score"] = (
        w_lcoh  * (1 - df_s["n_lcoh"])  +
        w_grid  * (1 - df_s["n_grid"])  +
        w_renew * (1 - df_s["n_renew"]) +
        w_city  * (1 - df_s["n_city"])
    ) / tw * 100

    df_s = df_s.sort_values("score", ascending=False)
    top_sites = df_s.head(top_n)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best LCOH",     f"{top_sites.iloc[0]['lcoh_scaled']:.2f} $/kg")
    k2.metric("Best Score",    f"{top_sites.iloc[0]['score']:.1f}/100")
    k3.metric("Avg Grid Dist", f"{top_sites['dist_grid_km'].mean():.0f} km")
    k4.metric("Avg City Dist", f"{top_sites['dist_city_km'].mean():.0f} km")
    st.markdown("---")

    fig4 = go.Figure()
    resolution = detect_resolution(df_s)

    geo_bg = build_geojson(
        tuple(df_s["lat"].round(4)),
        tuple(df_s["lon"].round(4)),
        tuple(df_s["cell_id"]),
        resolution,
    )

    fig4.add_trace(go.Choroplethmapbox(
        geojson=geo_bg,
        locations=df_s["cell_id"].tolist(),
        z=df_s["lcoh_scaled"].tolist(),
        colorscale="RdYlGn_r",
        marker=dict(  # ← opacity must go inside marker=dict()
            opacity=0.30,
            line_width=0.5,
            line_color="#1e3050",
        ))),

    if show_gridlines:
        for line in GRID_LINES:
            fig4.add_trace(go.Scattermapbox(
                lat=[p["lat"] for p in line], lon=[p["lon"] for p in line],
                mode="lines", line=dict(width=2, color="#00d4b8"),
                hoverinfo="skip", showlegend=False,
            ))
        fig4.add_trace(go.Scattermapbox(
            lat=[None], lon=[None], mode="lines",
            line=dict(width=2, color="#00d4b8"), name="Grid 220kV+",
        ))

    if show_stations:
        for s in ALL_STATIONS:
            col_dot = "#ffe066" if s["type"] == "Solar" else "#7ecfff"
            sym     = "circle"  if s["type"] == "Solar" else "square"
            fig4.add_trace(go.Scattermapbox(
                lat=[s["lat"]], lon=[s["lon"]], mode="markers",
                marker=dict(size=10, color=col_dot, symbol=sym),
                text=f"<b>{s['name']}</b><br>{s['mw']} MW {s['type']}",
                hoverinfo="text", showlegend=False,
            ))
        fig4.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="markers",
            marker=dict(size=10, color="#ffe066"), name="Solar Station"))
        fig4.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="markers",
            marker=dict(size=10, color="#7ecfff", symbol="square"), name="Wind Station"))

    if show_cities:
        fig4.add_trace(go.Scattermapbox(
            lat=[c["lat"] for c in CITIES], lon=[c["lon"] for c in CITIES],
            mode="markers+text",
            marker=dict(size=6, color="#ff4d6d"),
            text=[c["name"] for c in CITIES],
            textposition="top right",
            textfont=dict(size=9, color="#c8d8f0"),
            hoverinfo="text", name="City / Industry",
        ))

    fig4.add_trace(go.Scattermapbox(
        lat=top_sites["lat"].tolist(),
        lon=top_sites["lon"].tolist(),
        mode="markers",
        marker=dict(size=18, color="#f5a623", symbol="star"),
        text=[
            f"<b>Rank #{i+1}</b><br>"
            f"Score: {row['score']:.1f}/100<br>"
            f"LCOH: {row['lcoh_scaled']:.2f} $/kg<br>"
            f"Grid: {row['dist_grid_km']:.0f} km<br>"
            f"Station: {row['dist_renew_km']:.0f} km<br>"
            f"City: {row['dist_city_km']:.0f} km<br>"
            f"Solar FLH: {row.get('solar_full_load_hours', 0):.0f} h/yr<br>"
            f"Renew: {row.get('pct_renewable', 0):.0f}%"
            for i, (_, row) in enumerate(top_sites.iterrows())
        ],
        hoverinfo="text",
        name=f"Top {top_n} Electrolyzer Sites",
    ))

    fig4.update_layout(
        **PLOT_BG,
        mapbox=dict(style="carto-darkmatter", center=dict(lat=28.0, lon=2.5), zoom=4.2),
        height=680, showlegend=True,
        legend=dict(bgcolor="#0d1829", bordercolor="#1e3050", borderwidth=1,
                    font=dict(color="#c8d8f0", size=11), x=0.75, y=0.98),
        title=dict(text="Electrolyzer Site Suitability — Algeria",
                   font=dict(color="#f5a623", size=14, family="Space Mono"), x=0.01),
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 🏆 Top Candidate Sites")
    tbl_cols = ["lat", "lon", "score", "lcoh_scaled", "dist_grid_km",
                "dist_renew_km", "dist_city_km", "solar_full_load_hours", "pct_renewable"]
    tbl_cols = [c for c in tbl_cols if c in top_sites.columns]
    tbl = top_sites[tbl_cols].copy().reset_index(drop=True)
    tbl.index += 1
    tbl.columns = [
        {"lat": "Lat", "lon": "Lon", "score": "Score /100", "lcoh_scaled": "LCOH $/kg",
         "dist_grid_km": "Grid dist km", "dist_renew_km": "Station dist km",
         "dist_city_km": "City dist km", "solar_full_load_hours": "Solar FLH h/yr",
         "pct_renewable": "Renew %"}.get(c, c)
        for c in tbl_cols
    ]
    st.dataframe(tbl.round(2), use_container_width=True)

    st.markdown("### Score Breakdown — Top 10")
    top10 = df_s.head(10).reset_index(drop=True)
    labels = [f"#{i+1} ({row['lat']:.0f}N,{row['lon']:.0f}E)"
              for i, (_, row) in enumerate(top10.iterrows())]
    fig_bar = go.Figure()
    for cname, vals, bar_col in [
        ("LCOH",    w_lcoh  * (1 - _norm(top10["lcoh_scaled"]))   / tw * 100, "#f5a623"),
        ("Grid",    w_grid  * (1 - _norm(top10["dist_grid_km"]))  / tw * 100, "#00d4b8"),
        ("Station", w_renew * (1 - _norm(top10["dist_renew_km"])) / tw * 100, "#ffe066"),
        ("City",    w_city  * (1 - _norm(top10["dist_city_km"]))  / tw * 100, "#7ecfff"),
    ]:
        fig_bar.add_trace(go.Bar(
            name=cname, x=labels, y=vals.tolist(),
            marker_color=bar_col,
            hovertemplate=f"{cname}: %{{y:.1f}}<extra></extra>",
        ))
    fig_bar.update_layout(
        **PLOT_BG, barmode="stack", height=320,
        xaxis=dict(title="Rank", tickfont=dict(size=9, color="#c8d8f0"), gridcolor="#1e3050"),
        yaxis=dict(title="Score contribution", gridcolor="#1e3050"),
        legend=dict(bgcolor="#0d1829", font=dict(color="#c8d8f0", size=11)),
        title=dict(text="Score Breakdown by Criterion",
                   font=dict(color="#f5a623", size=13, family="Space Mono"), x=0.01),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.download_button(
        "📥 Download Site Rankings CSV",
        data=df_s[["lat", "lon", "score", "lcoh_scaled",
                   "dist_grid_km", "dist_renew_km", "dist_city_km"]].to_csv(index=False),
        file_name="algeria_electrolyzer_sites.csv",
        mime="text/csv",
        use_container_width=True,
    )

import json


# ── Real-data loaders (with hardcoded fallbacks) ──────────────────────────────

@st.cache_data
def load_power_lines():
    """
    Load HV power lines from Overpass GeoJSON export.
    Expected format: GeoJSON FeatureCollection of LineString features.
    Falls back to the hardcoded GRID_LINES from tab4 if file missing.
    """
    path = DATA_DIR / "grid_lines.geojson"
    if not path.exists():
        # fallback: convert existing GRID_LINES to same format
        return [
            {"coords": [(p["lon"], p["lat"]) for p in line], "voltage": "220kV", "source": "estimated"}
            for line in GRID_LINES
        ]
    with open(path) as f:
        gj = json.load(f)
    lines = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        props = feat.get("properties", {})
        if geom["type"] == "LineString":
            coords = geom["coordinates"]
        elif geom["type"] == "MultiLineString":
            coords = [pt for seg in geom["coordinates"] for pt in seg]
        else:
            continue
        lines.append({
            "coords": coords,
            "voltage": props.get("voltage", "unknown"),
            "name": props.get("name", ""),
            "source": "osm",
        })
    return lines


@st.cache_data
def load_gas_pipelines():
    """
    Load gas pipelines from Global Energy Monitor GeoJSON.
    Falls back to empty list if file missing.
    """
    path = DATA_DIR / "gas_pipelines.geojson"
    if not path.exists():
        return []
    with open(path) as f:
        gj = json.load(f)
    pipes = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        props = feat.get("properties", {})
        if geom["type"] == "LineString":
            coords = geom["coordinates"]
        elif geom["type"] == "MultiLineString":
            coords = [pt for seg in geom["coordinates"] for pt in seg]
        else:
            continue
        pipes.append({
            "coords": coords,
            "name": props.get("Pipeline Name", props.get("name", "")),
            "status": props.get("Status", ""),
            "source": "gem",
        })
    return pipes


@st.cache_data
def load_ports():
    """
    Load ports from World Port Index CSV.
    Filter: COUNTRY == 'AL' or 'DZ' depending on WPI version.
    Falls back to hardcoded major ports if file missing.
    """
    path = DATA_DIR / "ports.csv"
    if not path.exists():
        return [
            {"name": "Arzew", "lat": 35.734, "lon": -0.311, "type": "LNG/Export"},
            {"name": "Skikda", "lat": 36.876, "lon": 6.907, "type": "LNG/Export"},
            {"name": "Algiers", "lat": 36.774, "lon": 3.056, "type": "General"},
            {"name": "Annaba", "lat": 36.906, "lon": 7.750, "type": "General"},
            {"name": "Bejaia", "lat": 36.752, "lon": 5.090, "type": "General"},
            {"name": "Mostaganem", "lat": 35.931, "lon": 0.085, "type": "General"},
            {"name": "Oran", "lat": 35.718, "lon": -0.644, "type": "General"},
        ]
    df_p = pd.read_csv(path)
    # WPI column names vary by version — handle both
    lat_col = next((c for c in df_p.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df_p.columns if "lon" in c.lower()), None)
    name_col = next((c for c in df_p.columns if "name" in c.lower() or "port" in c.lower()), None)
    if not (lat_col and lon_col and name_col):
        return []
    ports = []
    for _, row in df_p.iterrows():
        try:
            ports.append({
                "name": str(row[name_col]),
                "lat": float(row[lat_col]),
                "lon": float(row[lon_col]),
                "type": str(row.get("Harbor Type", row.get("type", "Port"))),
            })
        except (ValueError, KeyError):
            continue
    return ports


@st.cache_data
def load_power_plants():
    """
    Load solar + wind plants from Global Energy Monitor CSV.
    Falls back to the hardcoded ALL_STATIONS from tab4.
    """
    path = DATA_DIR / "power_plants.csv"
    if not path.exists():
        return ALL_STATIONS  # reuse tab4 fallback
    df_pp = pd.read_csv(path)
    plants = []
    for _, row in df_pp.iterrows():
        try:
            lat = float(row.get("Latitude", row.get("lat", float("nan"))))
            lon = float(row.get("Longitude", row.get("lon", float("nan"))))
            if pd.isna(lat) or pd.isna(lon):
                continue
            fuel = str(row.get("Fuel", row.get("type", "Solar"))).lower()
            ptype = "Wind" if "wind" in fuel else "Solar"
            plants.append({
                "name": str(row.get("Project Name", row.get("name", "Plant"))),
                "lat": lat,
                "lon": lon,
                "mw": float(row.get("Capacity (MW)", row.get("mw", 0)) or 0),
                "status": str(row.get("Status", "operating")),
                "type": ptype,
            })
        except (ValueError, KeyError):
            continue
    return plants


# ── Tab rendering ─────────────────────────────────────────────────────────────
import json


# ── Real-data loaders (with hardcoded fallbacks) ──────────────────────────────

@st.cache_data
def load_power_lines():
    """
    Load HV power lines from Overpass GeoJSON export.
    Expected format: GeoJSON FeatureCollection of LineString features.
    Falls back to the hardcoded GRID_LINES from tab4 if file missing.
    """
    path = DATA_DIR / "grid_lines.geojson"
    if not path.exists():
        # fallback: convert existing GRID_LINES to same format
        return [
            {"coords": [(p["lon"], p["lat"]) for p in line], "voltage": "220kV", "source": "estimated"}
            for line in GRID_LINES
        ]
    with open(path) as f:
        gj = json.load(f)
    lines = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        props = feat.get("properties", {})
        if geom["type"] == "LineString":
            coords = geom["coordinates"]
        elif geom["type"] == "MultiLineString":
            coords = [pt for seg in geom["coordinates"] for pt in seg]
        else:
            continue
        lines.append({
            "coords": coords,
            "voltage": props.get("voltage", "unknown"),
            "name": props.get("name", ""),
            "source": "osm",
        })
    return lines


@st.cache_data
def load_gas_pipelines():
    """
    Load gas pipelines from Global Energy Monitor GeoJSON.
    Falls back to empty list if file missing.
    """
    path = DATA_DIR / "gas_pipelines.geojson"
    if not path.exists():
        return []
    with open(path) as f:
        gj = json.load(f)
    pipes = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        props = feat.get("properties", {})
        if geom["type"] == "LineString":
            coords = geom["coordinates"]
        elif geom["type"] == "MultiLineString":
            coords = [pt for seg in geom["coordinates"] for pt in seg]
        else:
            continue
        pipes.append({
            "coords": coords,
            "name": props.get("Pipeline Name", props.get("name", "")),
            "status": props.get("Status", ""),
            "source": "gem",
        })
    return pipes


@st.cache_data
def load_ports():
    """
    Load ports from World Port Index CSV.
    Filter: COUNTRY == 'AL' or 'DZ' depending on WPI version.
    Falls back to hardcoded major ports if file missing.
    """
    path = DATA_DIR / "ports.csv"
    if not path.exists():
        return [
            {"name": "Arzew", "lat": 35.734, "lon": -0.311, "type": "LNG/Export"},
            {"name": "Skikda", "lat": 36.876, "lon": 6.907, "type": "LNG/Export"},
            {"name": "Algiers", "lat": 36.774, "lon": 3.056, "type": "General"},
            {"name": "Annaba", "lat": 36.906, "lon": 7.750, "type": "General"},
            {"name": "Bejaia", "lat": 36.752, "lon": 5.090, "type": "General"},
            {"name": "Mostaganem", "lat": 35.931, "lon": 0.085, "type": "General"},
            {"name": "Oran", "lat": 35.718, "lon": -0.644, "type": "General"},
        ]
    df_p = pd.read_csv(path)
    # WPI column names vary by version — handle both
    lat_col = next((c for c in df_p.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df_p.columns if "lon" in c.lower()), None)
    name_col = next((c for c in df_p.columns if "name" in c.lower() or "port" in c.lower()), None)
    if not (lat_col and lon_col and name_col):
        return []
    ports = []
    for _, row in df_p.iterrows():
        try:
            ports.append({
                "name": str(row[name_col]),
                "lat": float(row[lat_col]),
                "lon": float(row[lon_col]),
                "type": str(row.get("Harbor Type", row.get("type", "Port"))),
            })
        except (ValueError, KeyError):
            continue
    return ports


@st.cache_data
def load_power_plants():
    """
    Load solar + wind plants from Global Energy Monitor CSV.
    Falls back to the hardcoded ALL_STATIONS from tab4.
    """
    path = DATA_DIR / "power_plants.csv"
    if not path.exists():
        return ALL_STATIONS  # reuse tab4 fallback
    df_pp = pd.read_csv(path)
    plants = []
    for _, row in df_pp.iterrows():
        try:
            lat = float(row.get("Latitude", row.get("lat", float("nan"))))
            lon = float(row.get("Longitude", row.get("lon", float("nan"))))
            if pd.isna(lat) or pd.isna(lon):
                continue
            fuel = str(row.get("Fuel", row.get("type", "Solar"))).lower()
            ptype = "Wind" if "wind" in fuel else "Solar"
            plants.append({
                "name": str(row.get("Project Name", row.get("name", "Plant"))),
                "lat": lat,
                "lon": lon,
                "mw": float(row.get("Capacity (MW)", row.get("mw", 0)) or 0),
                "status": str(row.get("Status", "operating")),
                "type": ptype,
            })
        except (ValueError, KeyError):
            continue
    return plants


# ── Tab rendering ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — ALGERIA INFRASTRUCTURE MAP
#  Paste this block into app_grid.py
#
#  STEP 1: change your tab line from:
#    tab1, tab2, tab3, tab4 = st.tabs([...])
#  to:
#    tab1, tab2, tab3, tab4, tab5 = st.tabs([..., "🗺️ INFRASTRUCTURE"])
#
#  STEP 2: paste this entire block at the bottom of app_grid.py
#
#  STEP 3: put your downloaded data files in data/ :
#    data/grid_lines.geojson     ← from Overpass (power lines)
#    data/gas_pipelines.geojson  ← from Global Energy Monitor
#    data/ports.csv              ← from World Port Index (filter COUNTRY="AL")
#    data/power_plants.csv       ← from Global Energy Monitor solar+wind tracker
#
#  All loaders gracefully fall back to your existing hardcoded data
#  if the files are not yet present — so the tab works immediately.
# ══════════════════════════════════════════════════════════════════════════════

import json

# ── Real-data loaders (with hardcoded fallbacks) ──────────────────────────────

@st.cache_data
def load_power_lines():
    """
    Load HV power lines from Overpass GeoJSON export.
    Expected format: GeoJSON FeatureCollection of LineString features.
    Falls back to the hardcoded GRID_LINES from tab4 if file missing.
    """
    path = DATA_DIR / "grid_lines.geojson"
    if not path.exists():
        # fallback: convert existing GRID_LINES to same format
        return [
            {"coords": [(p["lon"], p["lat"]) for p in line], "voltage": "220kV", "source": "estimated"}
            for line in GRID_LINES
        ]
    with open(path) as f:
        gj = json.load(f)
    lines = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        props = feat.get("properties", {})
        if geom["type"] == "LineString":
            coords = geom["coordinates"]
        elif geom["type"] == "MultiLineString":
            coords = [pt for seg in geom["coordinates"] for pt in seg]
        else:
            continue
        lines.append({
            "coords":  coords,
            "voltage": props.get("voltage", "unknown"),
            "name":    props.get("name", ""),
            "source":  "osm",
        })
    return lines


@st.cache_data
def load_gas_pipelines():
    """
    Load gas pipelines from Global Energy Monitor GeoJSON.
    Falls back to empty list if file missing.
    """
    path = DATA_DIR / "gas_pipelines.geojson"
    if not path.exists():
        return []
    with open(path) as f:
        gj = json.load(f)
    pipes = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        props = feat.get("properties", {})
        if geom["type"] == "LineString":
            coords = geom["coordinates"]
        elif geom["type"] == "MultiLineString":
            coords = [pt for seg in geom["coordinates"] for pt in seg]
        else:
            continue
        pipes.append({
            "coords": coords,
            "name":   props.get("Pipeline Name", props.get("name", "")),
            "status": props.get("Status", ""),
            "source": "gem",
        })
    return pipes


@st.cache_data
def load_ports():
    """
    Load ports from World Port Index CSV.
    Filter: COUNTRY == 'AL' or 'DZ' depending on WPI version.
    Falls back to hardcoded major ports if file missing.
    """
    path = DATA_DIR / "ports.csv"
    if not path.exists():
        return [
            {"name": "Arzew",       "lat": 35.734, "lon": -0.311, "type": "LNG/Export"},
            {"name": "Skikda",      "lat": 36.876, "lon":  6.907, "type": "LNG/Export"},
            {"name": "Algiers",     "lat": 36.774, "lon":  3.056, "type": "General"},
            {"name": "Annaba",      "lat": 36.906, "lon":  7.750, "type": "General"},
            {"name": "Bejaia",      "lat": 36.752, "lon":  5.090, "type": "General"},
            {"name": "Mostaganem",  "lat": 35.931, "lon":  0.085, "type": "General"},
            {"name": "Oran",        "lat": 35.718, "lon": -0.644, "type": "General"},
        ]
    df_p = pd.read_csv(path)
    # WPI column names vary by version — handle both
    lat_col = next((c for c in df_p.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df_p.columns if "lon" in c.lower()), None)
    name_col = next((c for c in df_p.columns if "name" in c.lower() or "port" in c.lower()), None)
    if not (lat_col and lon_col and name_col):
        return []
    ports = []
    for _, row in df_p.iterrows():
        try:
            ports.append({
                "name": str(row[name_col]),
                "lat":  float(row[lat_col]),
                "lon":  float(row[lon_col]),
                "type": str(row.get("Harbor Type", row.get("type", "Port"))),
            })
        except (ValueError, KeyError):
            continue
    return ports


@st.cache_data
def load_power_plants():
    """
    Load solar + wind plants from Global Energy Monitor CSV.
    Falls back to the hardcoded ALL_STATIONS from tab4.
    """
    path = DATA_DIR / "power_plants.csv"
    if not path.exists():
        return ALL_STATIONS  # reuse tab4 fallback
    df_pp = pd.read_csv(path)
    plants = []
    for _, row in df_pp.iterrows():
        try:
            lat = float(row.get("Latitude", row.get("lat", float("nan"))))
            lon = float(row.get("Longitude", row.get("lon", float("nan"))))
            if pd.isna(lat) or pd.isna(lon):
                continue
            fuel = str(row.get("Fuel", row.get("type", "Solar"))).lower()
            ptype = "Wind" if "wind" in fuel else "Solar"
            plants.append({
                "name":   str(row.get("Project Name", row.get("name", "Plant"))),
                "lat":    lat,
                "lon":    lon,
                "mw":     float(row.get("Capacity (MW)", row.get("mw", 0)) or 0),
                "status": str(row.get("Status", "operating")),
                "type":   ptype,
            })
        except (ValueError, KeyError):
            continue
    return plants


# ── Tab rendering ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — ALGERIA INFRASTRUCTURE MAP
#  Paste this block into app_grid.py
#
#  STEP 1: change your tab line from:
#    tab1, tab2, tab3, tab4 = st.tabs([...])
#  to:
#    tab1, tab2, tab3, tab4, tab5 = st.tabs([..., "🗺️ INFRASTRUCTURE"])
#
#  STEP 2: paste this entire block at the bottom of app_grid.py
#
#  STEP 3: put your downloaded data files in data/ :
#    data/grid_lines.geojson     ← from Overpass (power lines)
#    data/gas_pipelines.geojson  ← from Global Energy Monitor
#    data/ports.csv              ← from World Port Index (filter COUNTRY="AL")
#    data/power_plants.csv       ← from Global Energy Monitor solar+wind tracker
#
#  All loaders gracefully fall back to your existing hardcoded data
#  if the files are not yet present — so the tab works immediately.
# ══════════════════════════════════════════════════════════════════════════════

import json

# ── Real-data loaders (with hardcoded fallbacks) ──────────────────────────────

@st.cache_data
def load_power_lines():
    """
    Load HV power lines from Overpass GeoJSON export.
    Expected format: GeoJSON FeatureCollection of LineString features.
    Falls back to the hardcoded GRID_LINES from tab4 if file missing.
    """
    path = DATA_DIR / "grid_lines.geojson"
    if not path.exists():
        # fallback: convert existing GRID_LINES to same format
        return [
            {"coords": [(p["lon"], p["lat"]) for p in line], "voltage": "220kV", "source": "estimated"}
            for line in GRID_LINES
        ]
    with open(path) as f:
        gj = json.load(f)
    lines = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        props = feat.get("properties", {})
        if geom["type"] == "LineString":
            coords = geom["coordinates"]
        elif geom["type"] == "MultiLineString":
            coords = [pt for seg in geom["coordinates"] for pt in seg]
        else:
            continue
        lines.append({
            "coords":  coords,
            "voltage": props.get("voltage", "unknown"),
            "name":    props.get("name", ""),
            "source":  "osm",
        })
    return lines


@st.cache_data
def load_gas_pipelines():
    """
    Load gas pipelines from Global Energy Monitor GeoJSON.
    Falls back to empty list if file missing.
    """
    path = DATA_DIR / "gas_pipelines.geojson"
    if not path.exists():
        return []
    with open(path) as f:
        gj = json.load(f)
    pipes = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        props = feat.get("properties", {})
        if geom["type"] == "LineString":
            coords = geom["coordinates"]
        elif geom["type"] == "MultiLineString":
            coords = [pt for seg in geom["coordinates"] for pt in seg]
        else:
            continue
        pipes.append({
            "coords": coords,
            "name":   props.get("Pipeline Name", props.get("name", "")),
            "status": props.get("Status", ""),
            "source": "gem",
        })
    return pipes


@st.cache_data
def load_ports():
    """
    Load ports from World Port Index CSV.
    Filter: COUNTRY == 'AL' or 'DZ' depending on WPI version.
    Falls back to hardcoded major ports if file missing.
    """
    path = DATA_DIR / "ports.csv"
    if not path.exists():
        return [
            {"name": "Arzew",       "lat": 35.734, "lon": -0.311, "type": "LNG/Export"},
            {"name": "Skikda",      "lat": 36.876, "lon":  6.907, "type": "LNG/Export"},
            {"name": "Algiers",     "lat": 36.774, "lon":  3.056, "type": "General"},
            {"name": "Annaba",      "lat": 36.906, "lon":  7.750, "type": "General"},
            {"name": "Bejaia",      "lat": 36.752, "lon":  5.090, "type": "General"},
            {"name": "Mostaganem",  "lat": 35.931, "lon":  0.085, "type": "General"},
            {"name": "Oran",        "lat": 35.718, "lon": -0.644, "type": "General"},
        ]
    df_p = pd.read_csv(path)
    # WPI column names vary by version — handle both
    lat_col = next((c for c in df_p.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df_p.columns if "lon" in c.lower()), None)
    name_col = next((c for c in df_p.columns if "name" in c.lower() or "port" in c.lower()), None)
    if not (lat_col and lon_col and name_col):
        return []
    ports = []
    for _, row in df_p.iterrows():
        try:
            ports.append({
                "name": str(row[name_col]),
                "lat":  float(row[lat_col]),
                "lon":  float(row[lon_col]),
                "type": str(row.get("Harbor Type", row.get("type", "Port"))),
            })
        except (ValueError, KeyError):
            continue
    return ports


@st.cache_data
def load_power_plants():
    """
    Load solar + wind plants from Global Energy Monitor CSV.
    Falls back to the hardcoded ALL_STATIONS from tab4.
    """
    path = DATA_DIR / "power_plants.csv"
    if not path.exists():
        return ALL_STATIONS  # reuse tab4 fallback
    df_pp = pd.read_csv(path)
    plants = []
    for _, row in df_pp.iterrows():
        try:
            lat = float(row.get("Latitude", row.get("lat", float("nan"))))
            lon = float(row.get("Longitude", row.get("lon", float("nan"))))
            if pd.isna(lat) or pd.isna(lon):
                continue
            fuel = str(row.get("Fuel", row.get("type", "Solar"))).lower()
            ptype = "Wind" if "wind" in fuel else "Solar"
            plants.append({
                "name":   str(row.get("Project Name", row.get("name", "Plant"))),
                "lat":    lat,
                "lon":    lon,
                "mw":     float(row.get("Capacity (MW)", row.get("mw", 0)) or 0),
                "status": str(row.get("Status", "operating")),
                "type":   ptype,
            })
        except (ValueError, KeyError):
            continue
    return plants


# ── Tab rendering ─────────────────────────────────────────────────────────────

with tab5:
    st.markdown("## 🗺️ Algeria Energy Infrastructure")
    st.markdown(
        '<p style="color:#5a7a9a;font-size:13px;margin-top:-10px">'
        "Real infrastructure data — HV power grid · Gas pipelines · Export ports · RE plants"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Layer toggles (sidebar, only shown in this tab via session state trick) ──
    st.sidebar.markdown('<div class="section-title">🗺️ Infrastructure Layers</div>', unsafe_allow_html=True)
    show_hv_lines   = st.sidebar.checkbox("⚡ HV Power lines",    value=True,  key="inf_hv")
    show_gas_pipes  = st.sidebar.checkbox("🔶 Gas pipelines",     value=True,  key="inf_gas")
    show_ports_inf  = st.sidebar.checkbox("⚓ Export ports",      value=True,  key="inf_ports")
    show_solar_inf  = st.sidebar.checkbox("☀️ Solar plants",      value=True,  key="inf_solar")
    show_wind_inf   = st.sidebar.checkbox("💨 Wind farms",        value=True,  key="inf_wind")
    show_lcoh_bg    = st.sidebar.checkbox("🌡️ LCOH background",   value=True,  key="inf_lcoh")

    # ── Load all data ──────────────────────────────────────────────────────────
    power_lines  = load_power_lines()
    gas_pipes    = load_gas_pipelines()
    ports        = load_ports()
    plants       = load_power_plants()

    solar_plants = [p for p in plants if p["type"] == "Solar"]
    wind_plants  = [p for p in plants if p["type"] == "Wind"]

    # ── Data source badges ─────────────────────────────────────────────────────
    def _src_badge(label, is_real, fallback_note=""):
        color = "#00d4b8" if is_real else "#f5a623"
        icon  = "✅" if is_real else "⚠️"
        note  = "" if is_real else f" ({fallback_note})"
        return f'<span style="background:{color}22;border:1px solid {color};border-radius:4px;padding:2px 8px;font-size:11px;color:{color};font-family:monospace">{icon} {label}{note}</span>'

    hv_real    = (DATA_DIR / "grid_lines.geojson").exists()
    gas_real   = (DATA_DIR / "gas_pipelines.geojson").exists()
    ports_real = (DATA_DIR / "ports.csv").exists()
    plants_real= (DATA_DIR / "power_plants.csv").exists()

    badge_html = " &nbsp; ".join([
        _src_badge("HV Lines",    hv_real,    "estimated"),
        _src_badge("Gas pipes",   gas_real,   "none loaded"),
        _src_badge("Ports",       ports_real, "hardcoded"),
        _src_badge("RE Plants",   plants_real,"hardcoded"),
    ])
    st.markdown(f'<div style="margin-bottom:16px">{badge_html}</div>', unsafe_allow_html=True)

    if not (hv_real or gas_real or ports_real or plants_real):
        st.markdown("""<div class='info-box'>
        <b>To load real data, add these files to your <code>data/</code> folder:</b><br><br>
        &bull; <code>grid_lines.geojson</code> — from <a href="https://overpass-turbo.eu" target="_blank">overpass-turbo.eu</a> (query: power=line, country=Algeria)<br>
        &bull; <code>gas_pipelines.geojson</code> — from <a href="https://globalenergymonitor.org" target="_blank">globalenergymonitor.org</a> → Gas Infrastructure Tracker<br>
        &bull; <code>ports.csv</code> — from <a href="https://msi.nga.mil/Publications/WPI" target="_blank">msi.nga.mil/Publications/WPI</a> (filter COUNTRY=AL)<br>
        &bull; <code>power_plants.csv</code> — from <a href="https://globalenergymonitor.org" target="_blank">globalenergymonitor.org</a> → Solar + Wind Trackers<br><br>
        The map below shows estimated/hardcoded data until then.
        </div>""", unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────────────
    i1, i2, i3, i4, i5 = st.columns(5)
    i1.metric("HV Line segments", len(power_lines),
              delta="real OSM" if hv_real else "estimated", delta_color="normal")
    i2.metric("Gas pipelines",    len(gas_pipes),
              delta="real GEM" if gas_real else "none", delta_color="off")
    i3.metric("Export ports",     len(ports),
              delta="real WPI" if ports_real else "hardcoded", delta_color="normal")
    i4.metric("Solar plants",     len(solar_plants),
              delta="real GEM" if plants_real else "hardcoded", delta_color="normal")
    i5.metric("Wind farms",       len(wind_plants),
              delta="real GEM" if plants_real else "hardcoded", delta_color="normal")
    st.markdown("---")

    # ── Build map ──────────────────────────────────────────────────────────────
    fig_inf = go.Figure()

    # Layer 0: LCOH choropleth background (optional)
    if show_lcoh_bg and has_data:
        df_bg = scaled_df[scaled_df["lcoh_scaled"].notna() & (scaled_df["lcoh_scaled"] < 50)].copy()
        resolution_bg = detect_resolution(df_bg)
        geo_bg = build_geojson(
            tuple(df_bg["lat"].round(4)),
            tuple(df_bg["lon"].round(4)),
            tuple(df_bg["cell_id"]),
            resolution_bg,
        )
        fig_inf.add_trace(go.Choroplethmapbox(
            geojson=geo_bg,
            locations=df_bg["cell_id"].tolist(),
            z=df_bg["lcoh_scaled"].tolist(),
            colorscale="RdYlGn_r",
            zmin=df_bg["lcoh_scaled"].quantile(0.05),
            zmax=df_bg["lcoh_scaled"].quantile(0.95),
            marker=dict(opacity=0.18, line_width=0, line_color="rgba(0,0,0,0)"),
            colorbar=dict(
                title=dict(text="LCOH $/kg", font=dict(color="#c8d8f0", size=11)),
                thickness=10, len=0.35, x=0.01,
                tickfont=dict(color="#c8d8f0", size=9),
                bgcolor="#0d1829", bordercolor="#1e3050", borderwidth=1,
            ),
            showscale=True,
            name="LCOH background",
        ))

    # Layer 1: HV power lines
    if show_hv_lines:
        for seg in power_lines:
            lons = [c[0] for c in seg["coords"]]
            lats = [c[1] for c in seg["coords"]]
            v    = seg.get("voltage", "")
            col  = "#00d4b8" if "400" in str(v) else "#00a896"
            fig_inf.add_trace(go.Scattermapbox(
                lat=lats, lon=lons,
                mode="lines",
                line=dict(width=1.5, color=col),
                hovertext=seg.get("name", f"HV line {v}"),
                hoverinfo="text",
                showlegend=False,
            ))
        # single legend entry
        fig_inf.add_trace(go.Scattermapbox(
            lat=[None], lon=[None], mode="lines",
            line=dict(width=2, color="#00d4b8"),
            name="⚡ HV Power lines",
        ))

    # Layer 2: Gas pipelines
    if show_gas_pipes and gas_pipes:
        for pipe in gas_pipes:
            lons = [c[0] for c in pipe["coords"]]
            lats = [c[1] for c in pipe["coords"]]
            fig_inf.add_trace(go.Scattermapbox(
                lat=lats, lon=lons,
                mode="lines",
                line=dict(width=2.5, color="#ff9f43"),
                hovertext=pipe.get("name", "Gas pipeline"),
                hoverinfo="text",
                showlegend=False,
            ))
        fig_inf.add_trace(go.Scattermapbox(
            lat=[None], lon=[None], mode="lines",
            line=dict(width=2.5, color="#ff9f43"),
            name="🔶 Gas pipelines",
        ))

    # Layer 3: Export ports
    if show_ports_inf and ports:
        lng_ports = [p for p in ports if "lng" in p.get("type","").lower() or "export" in p.get("type","").lower()]
        gen_ports = [p for p in ports if p not in lng_ports]

        if lng_ports:
            fig_inf.add_trace(go.Scattermapbox(
                lat=[p["lat"] for p in lng_ports],
                lon=[p["lon"] for p in lng_ports],
                mode="markers",
                marker=dict(size=16, color="#ff4d6d", symbol="harbor"),
                text=[
                    f"<b>⚓ {p['name']}</b><br>Type: {p.get('type','LNG/Export')}"
                    for p in lng_ports
                ],
                hoverinfo="text",
                name="⚓ LNG / Export ports",
            ))
        if gen_ports:
            fig_inf.add_trace(go.Scattermapbox(
                lat=[p["lat"] for p in gen_ports],
                lon=[p["lon"] for p in gen_ports],
                mode="markers",
                marker=dict(size=11, color="#ff8fab", symbol="harbor"),
                text=[
                    f"<b>⚓ {p['name']}</b><br>Type: {p.get('type','Port')}"
                    for p in gen_ports
                ],
                hoverinfo="text",
                name="⚓ General ports",
            ))

    # Layer 4: Solar plants (sized by MW)
    if show_solar_inf and solar_plants:
        mw_vals = [max(p["mw"], 1) for p in solar_plants]
        mw_max  = max(mw_vals)
        sizes   = [8 + 22 * (mw / mw_max) for mw in mw_vals]
        fig_inf.add_trace(go.Scattermapbox(
            lat=[p["lat"] for p in solar_plants],
            lon=[p["lon"] for p in solar_plants],
            mode="markers",
            marker=dict(size=sizes, color="#ffe066", opacity=0.85),
            text=[
                f"<b>☀️ {p['name']}</b><br>"
                f"Capacity: {p['mw']:.0f} MW<br>"
                f"Status: {p.get('status','operating')}"
                for p in solar_plants
            ],
            hoverinfo="text",
            name="☀️ Solar plants",
        ))

    # Layer 5: Wind farms (sized by MW)
    if show_wind_inf and wind_plants:
        mw_vals = [max(p["mw"], 1) for p in wind_plants]
        mw_max  = max(mw_vals)
        sizes   = [8 + 22 * (mw / mw_max) for mw in mw_vals]
        fig_inf.add_trace(go.Scattermapbox(
            lat=[p["lat"] for p in wind_plants],
            lon=[p["lon"] for p in wind_plants],
            mode="markers",
            marker=dict(size=sizes, color="#7ecfff", symbol="square", opacity=0.85),
            text=[
                f"<b>💨 {p['name']}</b><br>"
                f"Capacity: {p['mw']:.0f} MW<br>"
                f"Status: {p.get('status','operating')}"
                for p in wind_plants
            ],
            hoverinfo="text",
            name="💨 Wind farms",
        ))

    fig_inf.update_layout(
        **PLOT_BG,
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=28.5, lon=2.5),
            zoom=4.2,
        ),
        height=700,
        showlegend=True,
        legend=dict(
            bgcolor="#0d1829",
            bordercolor="#1e3050",
            borderwidth=1,
            font=dict(color="#c8d8f0", size=11),
            x=0.01, y=0.99,
            itemsizing="constant",
        ),
        title=dict(
            text="Algeria Energy Infrastructure — Real Data Overlay",
            font=dict(color="#f5a623", size=14, family="Space Mono"),
            x=0.01,
        ),
    )
    st.plotly_chart(fig_inf, use_container_width=True)

    # ── Summary tables ─────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### ⚓ Export Ports")
        if ports:
            df_ports = pd.DataFrame(ports)[["name", "lat", "lon", "type"]]
            df_ports.columns = ["Port", "Lat", "Lon", "Type"]
            st.dataframe(df_ports.round(3), use_container_width=True, hide_index=True)
        else:
            st.info("No port data loaded.")

    with col_b:
        st.markdown("#### ☀️💨 RE Plants by capacity")
        if plants:
            df_plants = pd.DataFrame(plants)
            _pcols = [c for c in ["name", "lat", "lon", "mw", "type", "status"] if c in df_plants.columns]
            df_plants = df_plants[_pcols]
            _rename = {"name":"Plant","lat":"Lat","lon":"Lon","mw":"MW","type":"Type","status":"Status"}
            df_plants.columns = [_rename.get(c, c) for c in df_plants.columns]
            df_plants = df_plants.sort_values("MW", ascending=False)
            st.dataframe(df_plants.round(2), use_container_width=True, hide_index=True)
        else:
            st.info("No plant data loaded.")

    # ── Download buttons ───────────────────────────────────────────────────────
    dl1, dl2 = st.columns(2)
    with dl1:
        if ports:
            st.download_button(
                "📥 Download Ports CSV",
                data=pd.DataFrame(ports).to_csv(index=False),
                file_name="algeria_ports.csv", mime="text/csv",
                use_container_width=True,
            )
    with dl2:
        if plants:
            st.download_button(
                "📥 Download RE Plants CSV",
                data=pd.DataFrame(plants).to_csv(index=False),
                file_name="algeria_re_plants.csv", mime="text/csv",
                use_container_width=True,
            )
"""
app_grid.py — Algeria Green Hydrogen LCOH Grid Map
Run: streamlit run app_grid.py
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

tab1, tab2, tab3 = st.tabs(["🗺️ LCOH MAP", "☀️ SOLAR RESOURCE", "📊 STATISTICS"])


# ══════════════════════════════════════════════════════════════════════════════
#  GRID MAP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def make_grid_map(df_data: pd.DataFrame, col_name: str, col_label: str,
                  cs: str, title: str, opacity: float = 0.55,
                  height: int = 640) -> go.Figure:
    """
    Two-layer seamless grid map:

      Layer 1 — ALL cells in bounding box, transparent with just a border line.
                 Shows the Algeria map underneath (terrain, borders, city names).

      Layer 2 — Only cells WITH data, semi-transparent colored fill.
                 The map is still visible through the cells.
    """
    resolution = detect_resolution(df_data)

    # Complete bounding box grid (for layer 1 outlines)
    full_grid = make_complete_grid(df_data, resolution)
    geo_bg = build_geojson(
        tuple(full_grid["lat"].round(4)),
        tuple(full_grid["lon"].round(4)),
        tuple(full_grid["cell_id"]),
        resolution,
    )

    # Only cells with real values (for layer 2 colored fill)
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

    # ── Layer 1: transparent grid outline (shows map underneath) ─────────────
    fig.add_trace(go.Choroplethmapbox(
        geojson=geo_bg,
        locations=full_grid["cell_id"],
        z=[0] * len(full_grid),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],  # fully transparent fill
        showscale=False,
        marker_opacity=1.0,           # opacity applies to fill; fill is transparent anyway
        marker_line_width=0.8,        # visible grid border lines
        marker_line_color="#2a4060",  # subtle blue-grey border — draws the grid
        hoverinfo="skip",
        name="",
    ))

    # ── Layer 2: semi-transparent colored data cells ──────────────────────────
    fig.add_trace(go.Choroplethmapbox(
        geojson=geo_data,
        locations=df_plot["cell_id"],
        z=df_plot[col_name],
        colorscale=cs,
        zmin=float(df_plot[col_name].min()),
        zmax=float(df_plot[col_name].max()),
        marker_opacity=opacity,       # user-controlled — map visible through cells
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
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=28.0, lon=2.5),
            zoom=4.2,
        ),
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

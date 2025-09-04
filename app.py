# -------------------------------------------------------------
# GreenGrid – Smart Microgrid Monitoring (Mock Dashboard)
# Run: streamlit run greengrid_app.py
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============ Page Setup ============
st.set_page_config(
    page_title="GreenGrid – Microgrid Monitoring",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======= Styles =======
DARK_BG = "#0E1117"
TEXT = "#E5E7EB"
MUTED = "#9CA3AF"
ACCENT = "#22C55E"
ACCENT_RED = "#EF4444"
ACCENT_YELLOW = "#F59E0B"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {DARK_BG};
        color: {TEXT};
    }}
    .main .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    .metric-card {{
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 16px;
        padding: 16px 18px;
        color: {TEXT};
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }}
    .metric-label {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 4px; }}
    .metric-value {{ font-weight: 800; font-size: 1.6rem; }}
    .good {{ color: {ACCENT}; }}
    .warn {{ color: {ACCENT_YELLOW}; }}
    .bad  {{ color: {ACCENT_RED}; }}
    .subtle {{ color: {MUTED}; font-size: 0.85rem; }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("## ⚡ GreenGrid — Smart IoT Microgrid Monitoring (Demo)")
st.caption("Mock data dashboard showcasing IoT + Cloud + Mobile + Analytics for rural microgrids.")

# ============ Mock Data ============
np.random.seed(7)

villages = [
    ("Barapitha, Odisha", 20.229, 85.760),
    ("Lakshmipura, Rajasthan", 26.912, 75.787),
    ("Dharnai, Bihar", 25.096, 85.313),
    ("Kamalpur, Odisha", 20.268, 86.001),
    ("Gopalpur, Odisha", 19.283, 84.912),
    ("Hirapur, Odisha", 20.50, 85.84),
    ("Bhograi, Odisha", 21.53, 87.07),
    ("Pipli, Odisha", 20.11, 85.73),
]

start_date = datetime.today().date() - timedelta(days=29)
dates = pd.date_range(start_date, periods=30, freq="D")

def synth_timeseries(name):
    g_base = np.random.uniform(120, 220)    # generation kWh/day
    c_base = np.random.uniform(100, 200)    # consumption kWh/day
    gen = g_base + 35*np.sin(np.linspace(0, 6.2, len(dates))) + np.random.normal(0, 10, len(dates))
    cons = c_base + 20*np.sin(np.linspace(0.4, 6.6, len(dates))) + np.random.normal(0, 12, len(dates))
    soc = np.clip(68 + 12*np.sin(np.linspace(0.1, 6.4, len(dates))) + np.random.normal(0, 5, len(dates)), 20, 100)
    # outages: when consumption>generation significantly
    diesel_hrs = np.clip((cons - gen)/45, 0, 3)  # proxy
    uptime = np.clip(100 - diesel_hrs*3 + np.random.normal(0, 1, len(dates)), 90, 100)
    # alerts: low SOC & no-signal mock
    low_soc = (soc < 35).astype(int)
    no_signal = (np.random.rand(len(dates)) < 0.03).astype(int)  # ~3% days
    return pd.DataFrame({
        "date": dates,
        "village": name,
        "generation_kWh": gen.round(1),
        "consumption_kWh": cons.round(1),
        "battery_soc_pct": soc.round(1),
        "diesel_hours": diesel_hrs.round(2),
        "uptime_pct": uptime.round(2),
        "low_soc_alert": low_soc,
        "no_signal_event": no_signal
    })

ts = pd.concat([synth_timeseries(v[0]) for v in villages], ignore_index=True)

# Daily balance & efficiency proxy
ts["net_kWh"] = (ts["generation_kWh"] - ts["consumption_kWh"]).round(1)
ts["efficiency_proxy_pct"] = (np.clip(ts["generation_kWh"] / (ts["consumption_kWh"].replace(0, np.nan)), 0, 2)*100).fillna(0).round(1)

# Village latest snapshot
latest = ts.sort_values("date").groupby("village").tail(1).reset_index(drop=True)

# ============ Sidebar Filters ============
with st.sidebar:
    st.header("🔎 Filters")
    selected_villages = st.multiselect(
        "Villages", [v[0] for v in villages],
        default=[v[0] for v in villages]
    )
    days = st.slider("Days window", 7, 30, 30, 1)
    st.markdown("---")
    st.markdown("**Alert thresholds**")
    low_soc_thr = st.slider("Low battery threshold (%)", 10, 60, 35)
    net_warn = st.slider("Net deficit warn (kWh)", -100, 0, -20)

# Filtered data
df = ts[(ts["village"].isin(selected_villages)) & (ts["date"] >= (dates[-1] - timedelta(days=days-1)))]

# ============ KPI Row ============
def kpi_card(label, value, delta=None, status="good"):
    color_class = {"good":"good","warn":"warn","bad":"bad"}.get(status,"good")
    delta_html = f"<span class='subtle'>({delta})</span>" if delta is not None else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{value} {delta_html}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

col1, col2, col3, col4 = st.columns(4)
with col1:
    total_gen = df["generation_kWh"].sum()
    kpi_card("Total Generation (kWh)", f"{total_gen:,.0f}")
with col2:
    total_cons = df["consumption_kWh"].sum()
    kpi_card("Total Consumption (kWh)", f"{total_cons:,.0f}")
with col3:
    avg_soc = df["battery_soc_pct"].mean()
    status = "good" if avg_soc >= 60 else ("warn" if avg_soc >= 35 else "bad")
    kpi_card("Avg Battery SoC (%)", f"{avg_soc:,.1f}", status=status)
with col4:
    uptime = df["uptime_pct"].mean()
    kpi_card("Avg Uptime (%)", f"{uptime:,.1f}", status="good" if uptime>96 else "warn")

st.markdown("---")

# ============ Charts ============
left, right = st.columns((2,1.2))

with left:
    st.markdown("#### 📈 Generation vs Consumption (last {} days)".format(days))
    gdf = df.groupby(["date"]).agg({"generation_kWh":"sum","consumption_kWh":"sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gdf["date"], y=gdf["generation_kWh"], name="Generation (kWh)", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=gdf["date"], y=gdf["consumption_kWh"], name="Consumption (kWh)", mode="lines+markers"))
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔋 Battery SoC Heatmap by Village")
    heat = df.pivot_table(index="village", columns="date", values="battery_soc_pct").reindex([v[0] for v in villages])
    fig_h = px.imshow(heat, color_continuous_scale="Greens", aspect="auto")
    fig_h.update_layout(template="plotly_dark", height=320, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_h, use_container_width=True)

with right:
    st.markdown("#### 🗺️ Villages – Health Snapshot")
    map_df = pd.DataFrame({
        "village":[v[0] for v in villages],
        "lat":[v[1] for v in villages],
        "lon":[v[2] for v in villages],
    }).merge(latest[["village","battery_soc_pct","uptime_pct"]], on="village", how="left")
    map_df["status"] = np.where(map_df["battery_soc_pct"]<low_soc_thr, "Low SoC", "OK")
    figm = px.scatter_mapbox(map_df, lat="lat", lon="lon", color="status",
                             size="uptime_pct", hover_data=["village","battery_soc_pct","uptime_pct"],
                             color_discrete_map={"OK":"#22C55E","Low SoC":"#EF4444"},
                             zoom=4, height=330)
    figm.update_layout(mapbox_style="carto-darkmatter", template="plotly_dark", margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(figm, use_container_width=True)

# ============ Alerts & Anomalies ============
st.markdown("---")
st.markdown("#### ⚠️ Active Alerts (last {} days)".format(days))

alerts = df.copy()
alerts["alert_type"] = np.select(
    [
        alerts["no_signal_event"]==1,
        alerts["battery_soc_pct"] < low_soc_thr,
        alerts["net_kWh"] < net_warn
    ],
    ["No Signal (Blackout risk)","Low Battery SoC","Net Deficit"],
    default="OK"
)
alerts = alerts[alerts["alert_type"]!="OK"]
alerts = alerts.sort_values(["date","village"])

if alerts.empty:
    st.success("No active alerts in the selected window. ✅")
else:
    st.dataframe(
        alerts[["date","village","generation_kWh","consumption_kWh","battery_soc_pct","net_kWh","alert_type"]]
        .rename(columns={
            "date":"Date","village":"Village","generation_kWh":"Gen (kWh)","consumption_kWh":"Load (kWh)",
            "battery_soc_pct":"SoC (%)","net_kWh":"Net (kWh)","alert_type":"Alert"
        }),
        use_container_width=True, height=240
    )

# ============ Trend Analytics ============
st.markdown("---")
st.markdown("#### 📊 Trend Analytics – Efficiency & Diesel Hours")

tleft, tright = st.columns(2)
with tleft:
    eff = df.groupby("date")["efficiency_proxy_pct"].mean().reset_index()
    fig_e = px.line(eff, x="date", y="efficiency_proxy_pct", markers=True, title="Efficiency Proxy (%)")
    fig_e.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=30,b=10), showlegend=False)
    st.plotly_chart(fig_e, use_container_width=True)

with tright:
    di = df.groupby("date")["diesel_hours"].mean().reset_index()
    fig_d = px.bar(di, x="date", y="diesel_hours", title="Avg Diesel Generator Hours")
    fig_d.update_layout(template="plotly_dark", height=300, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_d, use_container_width=True)

st.caption("Demo only: data is synthetic. Features shown – dashboards, alerts, blackout prediction (no-signal), analytics, offline-ready concept.")

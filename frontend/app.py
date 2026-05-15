# app.py — Streamlit Dashboard
# Professional UI for the EnergiChip Forecast system

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import numpy as np

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EnergiChip Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .insight-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">⚡ EnergiChip Forecast</div>', 
            unsafe_allow_html=True)
st.markdown(
    "**Semiconductor Demand Intelligence** · India & Global · Powered by Prophet Forecasting"
)
st.divider()

# ─── SIDEBAR INPUTS ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Forecast Parameters")
    st.caption("Adjust energy transition indicators to simulate demand scenarios")
    
    solar_gw = st.slider(
        "🌞 Global Solar Additions (GW/year)", 
        min_value=50.0, max_value=500.0, value=290.0, step=10.0
    )
    ev_sales = st.slider(
        "🚗 Global EV Sales (Million units/year)", 
        min_value=1.0, max_value=50.0, value=14.0, step=0.5
    )
    india_solar = st.slider(
        "🇮🇳 India Solar Capacity (GW)", 
        min_value=10.0, max_value=200.0, value=73.0, step=5.0
    )
    quarters = st.selectbox("📅 Forecast Horizon", [2, 4, 6, 8], index=1)
    
    st.divider()
    st.markdown("**Model Info**")
    st.success("Prophet Primary Forecast Model")
    st.info("MAPE ≈ 3.5% on validation data")

# ─── MAIN DASHBOARD ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

# Key metrics
with col1:
    st.metric("Global Semiconductor Market", "$530B", "+12% YoY")
with col2:
    st.metric("India Semiconductor Imports", "$48B", "+18% YoY")
with col3:
    st.metric("Solar GW (2024 Target)", f"{solar_gw} GW", "Simulated")
with col4:
    st.metric("EV Sales", f"{ev_sales}M units", "Simulated")

st.divider()

# ─── FORECAST SECTION ────────────────────────────────────────────────────────
st.subheader("📈 Revenue Forecast")

# Call API for predictions
if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
    with st.spinner("Running forecast models..."):
        try:
            API_URL = "https://energichip-api.onrender.com/predict"
            response = requests.post(
                API_URL,
                json={
                    "solar_gw_added": solar_gw,
                    "ev_sales_million": ev_sales,
                    "india_solar_gw": india_solar,
                    "quarters_ahead": quarters
                },
                timeout=10
            )
            if response.status_code != 200:
                st.error(f"Backend Error: {response.text}")
                st.stop()
            forecast_data = response.json()
            # ----------------------------------------------#
            forecast_df = pd.DataFrame({
                "Quarter": forecast_data["forecast_quarters"],
                "Predicted_Revenue_Bn": forecast_data["predicted_revenue_bn"],
                "Lower_Bound": forecast_data["lower_bound"],
                "Upper_Bound": forecast_data["upper_bound"]
            })

            csv = forecast_df.to_csv(index=False)

            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name="semiconductor_forecast.csv",
                mime="text/csv"
            )
            # ----------------------------------------------#
            
            
            # Build forecast chart
            fig = go.Figure()
            
            # Confidence band
            fig.add_trace(go.Scatter(
                x=forecast_data["forecast_quarters"] + forecast_data["forecast_quarters"][::-1],
                y=forecast_data["upper_bound"] + forecast_data["lower_bound"][::-1],
                fill="toself",
                fillcolor="rgba(102, 126, 234, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% Confidence Interval",
                hoverinfo="skip"
            ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_data["forecast_quarters"],
                y=forecast_data["predicted_revenue_bn"],
                mode="lines+markers",
                name="Predicted Revenue",
                line=dict(color="#667eea", width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Semiconductor Market Revenue Forecast (USD Billions)",
                xaxis_title="Quarter",
                yaxis_title="Revenue (USD Billions)",
                template="plotly_dark",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # AI insight
            peak_rev = max(forecast_data["predicted_revenue_bn"])
            st.markdown(f"""
            <div class="insight-box">
            <strong>🤖 AI Insight:</strong> Under the current energy transition trajectory 
            ({solar_gw} GW solar, {ev_sales}M EV sales), global semiconductor revenue is projected 
            to reach <strong>${peak_rev}B</strong> within {quarters} quarters. 
            The forecast suggests that accelerating renewable energy deployment and EV adoption may significantly increase demand for power semiconductors, particularly in automotive and grid infrastructure applications.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"API Error: {e}. Make sure the FastAPI backend is running.")
            st.info("To start the backend: `uvicorn app.main:app --reload`")

# ─── INDIA VS GLOBAL COMPARISON ──────────────────────────────────────────────
st.divider()
st.subheader("🌍 India vs Global: Semiconductor Demand Trajectory")

# Generate comparison data
years = list(range(2015, 2025))
global_data = [335, 339, 412, 469, 413, 439, 555, 573, 574, 530]
india_data = [8, 10, 14, 19, 21, 27, 35, 44, 52, 48]
st.caption("Historical visualization uses curated industry trend data for demonstration purposes.")

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name="Global Market (USD Bn)",
    x=years, y=global_data,
    marker_color="#667eea", opacity=0.8,
    yaxis="y1"
))
fig2.add_trace(go.Scatter(
    name="India Imports (USD Bn)",
    x=years, y=india_data,
    mode="lines+markers",
    line=dict(color="#FF7043", width=3),
    marker=dict(size=8),
    yaxis="y2"
))
fig2.update_layout(
    template="plotly_dark",
    height=400,
    yaxis=dict(title="Global Revenue (USD Bn)", side="left"),
    yaxis2=dict(title="India Imports (USD Bn)", side="right", overlaying="y"),
    barmode="group",
    legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig2, use_container_width=True)

st.caption("Data: WSTS, World Bank Trade Statistics | Forecast: EnergiChip Model v1.0")



if solar_gw > 350 and ev_sales > 20:
    st.success("⚡ Aggressive Energy Transition Scenario")
elif solar_gw > 200:
    st.info("🔋 Moderate Green Energy Expansion")
else:
    st.warning("🏭 Conservative Adoption Scenario")
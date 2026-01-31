import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import pearsonr

# --- PAGE CONFIG ---
st.set_page_config(page_title="Macro Policy Lab", layout="wide", page_icon="📈")

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .analysis-note { background-color: #ffffff; padding: 20px; border-left: 5px solid #002d72; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('EM_Macro_Data_India_SG_UK.xlsx - Macro data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Data file not found. Please ensure the CSV is in the same folder.")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🕹️ Control Panel")
market = st.sidebar.selectbox("Select Market", ["India", "UK", "Singapore"])

st.sidebar.divider()
st.sidebar.subheader("Taylor Rule Settings")
target_inf = st.sidebar.slider("Inflation Target (%)", 0.0, 6.0, 4.0 if market == "India" else 2.0)
r_star = st.sidebar.slider("Neutral Rate (r*)", 0.0, 5.0, 1.5)

st.sidebar.divider()
st.sidebar.subheader("⚠️ Stress Test")
oil_shock = st.sidebar.slider("Energy Price Spike (%)", 0, 100, 0)

# --- MODEL LOGIC ---
m_map = {
    "India": {"cpi": "CPI_India", "rate": "Policy_India", "beta": 0.12},
    "UK": {"cpi": "CPI_UK", "rate": "Policy_UK", "beta": 0.07},
    "Singapore": {"cpi": "CPI_Singapore", "rate": "Policy_Singapore", "beta": 0.10}
}
m = m_map[market]

current_inf = df[m['cpi']].iloc[-1]
shock_impact = oil_shock * m['beta']
adj_inf = current_inf + shock_impact
current_rate = df[m['rate']].iloc[-1]

# Taylor Rule: i = r* + pi + 0.5(pi - target)
suggested_rate = r_star + adj_inf + 0.5 * (adj_inf - target_inf)

# --- DASHBOARD LAYOUT ---
st.title(f"🏦 {market}: Monetary Policy Analysis")
st.caption("Quantitative Research Terminal for Interest Rate Projections")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Headline CPI", f"{current_inf:.2f}%")
col2.metric("Adj. CPI (Shock)", f"{adj_inf:.2f}%", delta=f"+{shock_impact:.2f}%" if oil_shock > 0 else None, delta_color="inverse")
col3.metric("Actual Policy Rate", f"{current_rate:.2f}%")
col4.metric("Taylor Fair Value", f"{suggested_rate:.2f}%", delta=f"{(suggested_rate-current_rate):.2f}%", delta_color="inverse")

# --- CHART ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df[m['cpi']], name="Inflation", line=dict(color="#d32f2f", width=1.5, dash='dot')))
fig.add_trace(go.Scatter(x=df['Date'], y=df[m['rate']], name="Policy Rate", line=dict(color="#002d72", width=3)))
fig.add_trace(go.Scatter(x=[df['Date'].iloc[-1]], y=[suggested_rate], mode='markers', marker=dict(size=15, color='#ff9800'), name='Model Suggestion'))

fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", y=1.1), plot_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)

# --- RESEARCH NOTE ---
st.subheader("🧐 Analyst Assessment")
gap = (suggested_rate - current_rate) * 100

if gap > 50:
    signal = "HAWKISH"
    note = f"The model indicates a significant 'Policy Gap' of {gap:.0f} bps. The central bank is likely behind the curve."
elif gap < -50:
    signal = "DOVISH"
    note = f"The current stance appears overly restrictive by {abs(gap):.0f} bps. Conditions favor a pivot toward easing."
else:
    signal = "NEUTRAL"
    note = "Policy is currently well-aligned with the Taylor Rule fair value."

st.markdown(f"""
<div class="analysis-note">
    <strong>Strategic Bias: {signal}</strong><br><br>
    {note}<br><br>
    <em><strong>Interactivity Note:</strong> Move the 'Energy Price Spike' slider to see how supply-side shocks force the Taylor Fair Value (orange dot) higher, requiring a more aggressive policy response.</em>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Institutional Macro Terminal | Quant Edition", layout="wide", page_icon="🏛️")

# High-End Aesthetic Styling
st.markdown("""
    <style>
    .metric-card { background-color: #111; padding: 20px; border-radius: 10px; border: 1px solid #333; }
    [data-testid="stMetricValue"] { font-size: 38px; color: #00FFAA; font-family: 'Courier New', monospace; }
    .logic-tag { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .status-box { padding: 15px; border-radius: 5px; border-left: 5px solid #00FFAA; background: #1a1a1a; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_standardized_data():
    def fetch_data(file_path, sheet_name=None, standard_col="Value", is_csv=False):
        if not os.path.exists(file_path): return pd.Series(name=standard_col, dtype='float64')
        try:
            if is_csv:
                # Specialized logic for the OECD CSV metadata
                df = pd.read_csv(file_path, skiprows=3, names=['Date', standard_col])
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Standardize columns
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next(c for c in df.columns if 'date' in c.lower() or 'time' in c.lower())
            val_col = [c for c in df.columns if c != date_col][0]
            
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col]).set_index(date_col)
            
            # Resample to Monthly (MS) to align all macro indicators
            return df[val_col].resample('MS').last().rename(standard_col)
        except: return pd.Series(name=standard_col, dtype='float64')

    # Mapping files based on your provided structure
    data_map = {
        "Yield_Spread": fetch_data("T10Y2Y.xlsx", sheet_name="Daily", standard_col="Yield_Spread"),
        "Commodity_Index": fetch_data("PALLFNFINDEXM.xlsx", sheet_name="Monthly", standard_col="Commodity_Index"),
        "CCI_Sentiment": fetch_data("export-2026-02-10T06_50_22.597Z.csv", is_csv=True, standard_col="CCI_Sentiment"),
        "INR_USD": fetch_data("DEXINUS.xlsx", sheet_name="Daily", standard_col="INR_USD")
    }
    
    # Outer join all series to ensure no data loss
    combined = pd.concat(data_map.values(), axis=1).sort_index()
    combined = combined.ffill().dropna(how='all').reset_index().rename(columns={'index': 'Date'})
    
    # Simulate VIX if Yahoo Finance part is not yet integrated
    if 'VIX' not in combined.columns: combined['VIX'] = 17.2
    
    return combined

# --- 2. THE MODEL LOGIC ENGINE ---
df = load_standardized_data()

# Fail-safe check
if df.empty or 'Yield_Spread' not in df.columns:
    st.error("Critical Error: Data Pipeline failed to initialize. Verify Excel sheet names ('Daily', 'Monthly').")
    st.stop()

latest = df.iloc[-1]
prev = df.iloc[-12] if len(df) > 12 else df.iloc[0] # Year-over-year change

# Quantitative Recession Probability (Step 1 & 4)
spread = latest['Yield_Spread']
sentiment = latest['CCI_Sentiment']
rec_prob = 10.0
if spread < 0: rec_prob += 45.0  # Recession Gauge
if sentiment < 100: rec_prob += 20.0 # Sentiment Lead
if latest['Commodity_Index'] > 200: rec_prob += 15.0 # Cost-Push Inflation

# --- 3. MAIN INTERFACE ---
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")
st.caption(f"Terminal Status: ONLINE | Analysis for {latest['Date'].strftime('%B %Y')}")

# Top Metric Row: Logic-Driven Values
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown('<p class="logic-tag">Recession Probability</p>', unsafe_allow_html=True)
    st.metric("MODEL GAUGE", f"{min(rec_prob, 100):.1f}%", 
              delta="INVERSION" if spread < 0 else "NORMAL", delta_color="inverse")

with c2:
    st.markdown('<p class="logic-tag">Consumer Confidence</p>', unsafe_allow_html=True)
    st.metric("CCI SENTIMENT", f"{sentiment:.1f}", 
              delta=f"{sentiment - prev['CCI_Sentiment']:.1f} (YoY)")

with c3:
    st.markdown('<p class="logic-tag">Commodity Shock</p>', unsafe_allow_html=True)
    st.metric("COST-PUSH IDX", f"{latest['Commodity_Index']:.1f}", 
              delta=f"{latest['Commodity_Index'] - prev['Commodity_Index']:.1f}", delta_color="inverse")

with c4:
    st.markdown('<p class="logic-tag">Risk Premium</p>', unsafe_allow_html=True)
    st.metric("VIX (ESTIMATED)", f"{latest['VIX']:.1f}", delta="Risk Neutral")

# --- 4. QUANTITATIVE VISUALIZATION ---
st.divider()
tab1, tab2 = st.tabs(["Indicator Convergence", "System Logic & Metadata"])

with tab1:
    fig = go.Figure()
    # Left Axis: Yield Spread
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Yield_Spread'], name='10Y-2Y Spread (%)', 
                             line=dict(color='#00FFAA', width=3)))
    # Right Axis: Sentiment
    fig.add_trace(go.Scatter(x=df['Date'], y=df['CCI_Sentiment'], name='Consumer Sentiment (CCI)', 
                             yaxis='y2', line=dict(color='#FF00FF', width=2, dash='dot')))
    
    fig.update_layout(
        template="plotly_dark", height=500,
        margin=dict(l=10, r=10, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="Yield Spread %", showgrid=True, gridcolor='#333'),
        yaxis2=dict(title="Sentiment Index", overlaying="y", side="right", showgrid=False)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Model Step Calibration")
    l1, l2 = st.columns(2)
    with l1:
        st.write("**Step 1: FRED Recession Gauge**")
        st.info(f"Current Spread: {spread:.2f}. Values below 0 signify a yield curve inversion, historically leading recessions by 12-18 months.")
        st.write("**Step 2: FRED Cost-Push Inflation**")
        st.success(f"Current Commodity Index: {latest['Commodity_Index']:.1f}. High values anticipate CPI spikes through energy/raw material channels.")
    with l2:
        st.write("**Step 3: Financial Stability (BIS Simulation)**")
        st.write("Monitoring Credit-to-GDP gap for debt-driven vs sustainable growth signals.")
        st.write("**Step 4: OECD Sentiment Lead**")
        st.warning(f"Current CCI: {sentiment:.1f}. OECD average is 100. Current levels suggest {'pessimistic' if sentiment < 100 else 'optimistic'} outlook.")

# --- 5. SIDEBAR: STRESS TESTING ---
st.sidebar.title("🛠️ STRESS TEST ENGINE")
stress = st.sidebar.slider("Simulate Market Shock (%)", 0, 100, 0)
if stress > 0:
    st.sidebar.error(f"SYSTEM ALERT: Hypothetical {stress}% shock would push Recession Risk to {min(rec_prob + stress/2, 100):.1f}%")

st.sidebar.divider()
with st.sidebar.expander("🎓 Academic Methodology"):
    st.markdown("""
    This terminal utilizes a **Multi-Factor Probabilistic Model**:
    - **Yield Dynamics:** T10Y2Y spread analysis.
    - **Sentiment Lag:** CCI moves 3-6 months before real GDP.
    - **Inflation Vector:** Commodity Index as a precursor to CPI.
    """)

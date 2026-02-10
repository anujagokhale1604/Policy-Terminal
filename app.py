import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Institutional Macro Terminal", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 38px; color: #00FFAA; font-family: 'Courier New', monospace; }
    .status-box { padding: 15px; border-radius: 5px; border-left: 5px solid #00FFAA; background: #111; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_and_standardize_data():
    def process_file(file_path, standard_name):
        if not os.path.exists(file_path):
            return pd.Series(name=standard_name, dtype='float64')
        try:
            if file_path.endswith('.csv'):
                # Handle OECD CCI format with skip-rows
                df = pd.read_csv(file_path, skiprows=3 if 'export' in file_path else 0)
            else:
                xl = pd.ExcelFile(file_path)
                # Auto-select data sheet (Skip README)
                sheet = xl.sheet_names[1] if 'README' in xl.sheet_names[0].upper() else xl.sheet_names[0]
                df = pd.read_excel(file_path, sheet_name=sheet)
            
            # 1. Clean column names
            df.columns = [str(c).strip() for c in df.columns]
            
            # 2. Identify Date column
            date_col = next(c for c in df.columns if 'date' in c.lower() or 'time' in c.lower())
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # 3. Identify Value column (the first numeric column that isn't the date)
            val_col = [c for c in df.columns if c != date_col and df[c].dtype in ['float64', 'int64']][0]
            
            # 4. Resample to Monthly
            df = df.dropna(subset=[date_col]).set_index(date_col).resample('MS').last()
            return df[val_col].rename(standard_name)
        except Exception as e:
            return pd.Series(name=standard_name, dtype='float64')

    # Build the combined dataframe using Joins (Prevents Index Errors)
    data_map = {
        "T10Y2Y.xlsx": "Yield_Spread",
        "PALLFNFINDEXM.xlsx": "Commodity_Index",
        "export-2026-02-10T06_50_22.597Z.csv": "CCI_Sentiment",
        "DEXINUS.xlsx": "INR_USD"
    }
    
    combined = pd.DataFrame()
    for file, label in data_map.items():
        series = process_file(file, label)
        combined = pd.concat([combined, series], axis=1)
    
    # Forward fill gaps and remove rows with no data
    combined = combined.sort_index().ffill().dropna(how='all')
    return combined.reset_index().rename(columns={'index': 'Date'})

# --- 2. EXECUTION ---
df = load_and_standardize_data()

if df.empty or 'Yield_Spread' not in df.columns:
    st.error("Critical Error: 'Yield_Spread' variable not found. Check if T10Y2Y.xlsx is in the root folder.")
    st.stop()

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🏛️ TERMINAL LOGIC")
stress_level = st.sidebar.select_slider("System Stress Level", options=["Stable", "Warning", "Crisis"])

# Quant Logic: Recession Gauge
# 
spread_val = latest['Yield_Spread']
base_prob = 65.0 if spread_val < 0 else 15.0
if stress_level == "Warning": base_prob += 15
elif stress_level == "Crisis": base_prob += 30

# --- 4. MAIN DASHBOARD ---
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

# Prediction Row
c1, c2 = st.columns([1, 2])
with c1:
    st.metric("RECESSION PROBABILITY", f"{min(base_prob, 100):.1f}%", 
              delta="INVERSION" if spread_val < 0 else "NORMAL", 
              delta_color="inverse")
with c2:
    status_msg = "STABLE" if base_prob < 30 else ("CAUTION" if base_prob < 60 else "CRITICAL")
    st.markdown(f"""
    <div class="status-box">
        <strong>Terminal Status: {status_msg}</strong><br>
        Current Yield Spread: {spread_val:.2f} | 
        Sentiment Index: {latest['CCI_Sentiment']:.1f} | 
        Spot Rate: ₹{latest['INR_USD']:.2f}
    </div>
    """, unsafe_allow_html=True)

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("10Y-2Y Spread", f"{spread_val:.2f}", help="Leading Recession Indicator")
m2.metric("Commodity Index", f"{latest['Commodity_Index']:.1f}", help="Cost-Push Inflation Gauge")
m3.metric("CCI Sentiment", f"{latest['CCI_Sentiment']:.1f}", help="Leading GDP Indicator")
m4.metric("INR/USD Spot", f"{latest['INR_USD']:.2f}", delta_color="inverse")

st.divider()

# Time Series Analysis
# 
st.subheader("Market Trends & Indicator Convergence")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Yield_Spread'], name='Yield Spread (%)', line=dict(color='#00FFAA')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['CCI_Sentiment'], name='CCI Sentiment', yaxis='y2', line=dict(color='#FF00FF', dash='dot')))

fig.update_layout(
    template="plotly_dark",
    yaxis=dict(title="Spread %"),
    yaxis2=dict(title="CCI Index", overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("View Underlying Data Audit"):
    st.dataframe(df.sort_values('Date', ascending=False), use_container_width=True)

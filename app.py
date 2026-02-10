import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from scipy.stats import norm

# --- 1. SETTINGS & UI DESIGN ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide")
st.markdown("""
    <style>
    * { font-family: 'Times New Roman', serif; }
    .stApp { background-color: #FDFCFB; }
    .metric-card { background: #FFFFFF; padding: 20px; border: 1px solid #D3C9B9; border-left: 8px solid #002366; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE RE-ENGINEERED DATA PIPELINE ---
@st.cache_data
def load_terminal_data():
    def fetch_data(filename, sheet=None, skip=0):
        if not os.path.exists(filename):
            return None
        try:
            if filename.lower().endswith('.csv'):
                return pd.read_csv(filename, skiprows=skip)
            else:
                # Read specific sheet if provided
                return pd.read_excel(filename, sheet_name=sheet)
        except Exception as e:
            st.sidebar.error(f"Error reading {filename}: {e}")
            return None

    data = {}
    
    # A. Primary Macro Workbook
    # We check for the original Excel file and the auto-converted CSV version
    macro_xlsx = "EM_Macro_Data_India_SG_UK.xlsx"
    macro_csv = "EM_Macro_Data_India_SG_UK.xlsx - Macro data.csv"
    
    if os.path.exists(macro_xlsx):
        df_m = fetch_data(macro_xlsx, sheet="Macro data")
    else:
        df_m = fetch_data(macro_csv)

    if df_m is not None:
        # Robust Date Detection: Find column with 'date' in name, or use first column
        date_col = next((c for c in df_m.columns if 'date' in str(c).lower()), df_m.columns[0])
        df_m = df_m.rename(columns={date_col: 'Date'})
        
        # FIXED: errors='coerce' turns bad dates into NaT, then dropna() removes them
        df_m['Date'] = pd.to_datetime(df_m['Date'], errors='coerce')
        data['macro'] = df_m.dropna(subset=['Date'])

    # B. Yield Spread (Daily)
    df_y = fetch_data("T10Y2Y.xlsx - Daily.csv")
    if df_y is not None:
        df_y.columns = ['Date', 'Spread']
        df_y['Date'] = pd.to_datetime(df_y['Date'], errors='coerce')
        df_y['Spread'] = pd.to_numeric(df_y['Spread'], errors='coerce')
        data['yield'] = df_y.dropna(subset=['Date']).set_index('Date').resample('MS').mean().reset_index()

    # C. Commodities (Monthly)
    df_c = fetch_data("PALLFNFINDEXM.xlsx - Monthly.csv")
    if df_c is not None:
        df_c.columns = ['Date', 'Commodities']
        df_c['Date'] = pd.to_datetime(df_c['Date'], errors='coerce')
        data['comm'] = df_c.dropna(subset=['Date'])

    # D. Consumer Confidence (OECD CSV)
    df_cci = fetch_data("export-2026-02-10T06_50_22.597Z.csv", skip=3)
    if df_cci is not None:
        df_cci.columns = ['Date', 'CCI']
        df_cci['Date'] = pd.to_datetime(df_cci['Date'], errors='coerce')
        data['cci'] = df_cci.dropna(subset=['Date'])

    return data

# --- 3. EXECUTION ---
db = load_terminal_data()

st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

if 'macro' not in db:
    st.error("🚨 **File Missing:** Please ensure 'EM_Macro_Data_India_SG_UK.xlsx' is in your GitHub folder.")
    st.stop()

# Merge Logic
df = db['macro']
if 'yield' in db: df = df.merge(db['yield'], on='Date', how='left')
if 'comm' in db: df = df.merge(db['comm'], on='Date', how='left')
if 'cci' in db: df = df.merge(db['cci'], on='Date', how='left')
df = df.sort_values('Date').ffill().bfill()

# --- 4. DASHBOARD RENDER ---
market = st.sidebar.selectbox("Jurisdiction", ["India", "UK", "Singapore"])
m_cols = {"India": ["Policy_India", "CPI_India"], "UK": ["Policy_UK", "CPI_UK"], "Singapore": ["Policy_Singapore", "CPI_Singapore"]}
p_col, c_col = m_cols[market]

# Analytics Calculations
latest_spread = df['Spread'].iloc[-1] if 'Spread' in df.columns else 0.0
rec_prob = norm.cdf(-0.5 - (1.5 * latest_spread)) * 100

# Top Row Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("RECESSION RISK", f"{rec_prob:.1f}%")
c2.metric("YIELD SPREAD", f"{latest_spread:.2f}")
c3.metric("CONS. CONFIDENCE", f"{df['CCI'].iloc[-1] if 'CCI' in df.columns else 0.0:.1f}")
c4.metric("COMMODITY INDEX", f"{df['Commodities'].iloc[-1] if 'Commodities' in df.columns else 0.0:.0f}")

# Multi-Plot Visualization
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=(f"{market} Policy vs Inflation", "Global Market Sentiment"))

# Plot 1: Policy and Inflation
fig.add_trace(go.Scatter(x=df['Date'], y=df[p_col], name="Policy Rate", line=dict(color='#002366', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df[c_col], name="CPI Inflation", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)

# Plot 2: Yield Curve and CCI
if 'Spread' in df.columns:
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Spread'], name="Yield Spread", fill='tozeroy', line=dict(color='#2E8B57')), row=2, col=1)
if 'CCI' in df.columns:
    fig.add_trace(go.Scatter(x=df['Date'], y=df['CCI'], name="Consumer Sentiment", line=dict(color='#C5A059')), row=2, col=1)

fig.update_layout(height=750, template="plotly_white", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.api import VAR

# --- 1. SETTINGS & UI DESIGN ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide")
st.markdown("""
    <style>
    * { font-family: 'Times New Roman', serif; }
    .stApp { background-color: #FDFCFB; }
    .metric-card { background: #FFFFFF; padding: 20px; border: 1px solid #D3C9B9; border-left: 8px solid #002366; }
    .status-box { padding: 10px; border-radius: 5px; margin-bottom: 15px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE RE-ENGINEERED DATA PIPELINE ---
@st.cache_data
def load_terminal_data():
    def fetch_data(filename, is_excel=True, skip=0):
        """Robustly decides whether to use read_csv or read_excel."""
        if not os.path.exists(filename):
            return None
        try:
            if filename.lower().endswith('.csv'):
                return pd.read_csv(filename, skiprows=skip)
            else:
                return pd.read_excel(filename)
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")
            return None

    data = {}
    
    # A. Primary Macro Workbook (User uploaded)
    # Checks for the Excel file or a CSV version if it was auto-converted
    macro_file = "EM_Macro_Data_India_SG_UK.xlsx"
    if not os.path.exists(macro_file):
        macro_file = "EM_Macro_Data_India_SG_UK.xlsx - Macro data.csv"
        
    df_macro = fetch_data(macro_file)
    if df_macro is not None:
        # Standardize Date column
        date_col = next((c for c in df_macro.columns if 'date' in str(c).lower()), df_macro.columns[0])
        df_macro = df_macro.rename(columns={date_col: 'Date'})
        df_macro['Date'] = pd.to_datetime(df_macro['Date'])
        data['macro'] = df_macro

    # B. Yield Spread (Sandbox converted CSV)
    df_y = fetch_data("T10Y2Y.xlsx - Daily.csv")
    if df_y is not None:
        df_y.columns = ['Date', 'Spread']
        df_y['Date'] = pd.to_datetime(df_y['Date'], errors='coerce')
        df_y['Spread'] = pd.to_numeric(df_y['Spread'], errors='coerce')
        # Resample daily to monthly to match macro data
        data['yield'] = df_y.dropna().set_index('Date').resample('MS').mean().reset_index()

    # C. Commodities (Sandbox converted CSV)
    df_c = fetch_data("PALLFNFINDEXM.xlsx - Monthly.csv")
    if df_c is not None:
        df_c.columns = ['Date', 'Commodities']
        df_c['Date'] = pd.to_datetime(df_c['Date'], errors='coerce')
        data['comm'] = df_c.dropna()

    # D. Consumer Confidence (OECD Specific CSV)
    # Note: Using skiprows=3 as the snippet shows 3 lines of metadata
    df_cci = fetch_data("export-2026-02-10T06_50_22.597Z.csv", skip=3)
    if df_cci is not None:
        df_cci.columns = ['Date', 'CCI']
        df_cci['Date'] = pd.to_datetime(df_cci['Date'], errors='coerce')
        data['cci'] = df_cci.dropna()

    return data

# --- 3. EXECUTION ---
db = load_terminal_data()

st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

if 'macro' not in db:
    st.error("🚨 **Core File Missing:** Please ensure 'EM_Macro_Data_India_SG_UK.xlsx' is uploaded to the repository.")
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
# Probit Recession Probability Formula: Φ(-0.5 - 1.5 * Spread)
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

# Predictive Insight
st.markdown("### 🔮 Quant Research Note")
if latest_spread < 0:
    st.error(f"**Warning:** The Yield Spread is currently inverted ({latest_spread:.2f}). Historically, an inverted curve suggests a high probability of recession within 12-18 months.")
else:
    st.success(f"**Stable:** The positive Yield Spread ({latest_spread:.2f}) indicates a lower immediate risk of a macro-economic downturn.")

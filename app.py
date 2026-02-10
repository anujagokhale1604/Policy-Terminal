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
    .status-active { color: #2E8B57; font-weight: bold; }
    .status-missing { color: #A52A2A; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MULTI-SOURCE DATA LOADER ---
@st.cache_data
def load_terminal_data():
    # Helper to handle the "Sandbox" naming (Excel-to-CSV conversion)
    def fetch(name, sheet=""):
        path = f"{name} - {sheet}.csv" if sheet else f"{name}.csv"
        if os.path.exists(path): return pd.read_csv(path)
        if os.path.exists(name): 
            return pd.read_excel(name, sheet_name=sheet) if sheet else pd.read_excel(name)
        return None

    data = {}
    
    # Core Macro File (from your GitHub)
    df_macro = fetch("EM_Macro_Data_India_SG_UK.xlsx", "Macro data")
    if df_macro is not None:
        df_macro['Date'] = pd.to_datetime(df_macro['Date'])
        data['macro'] = df_macro

    # Yield Spread
    df_y = fetch("T10Y2Y.xlsx", "Daily")
    if df_y is not None:
        df_y['Date'] = pd.to_datetime(df_y.iloc[:,0], errors='coerce')
        df_y['Spread'] = pd.to_numeric(df_y.iloc[:,1], errors='coerce')
        data['yield'] = df_y.dropna().set_index('Date').resample('MS').mean().reset_index()

    # Commodities
    df_c = fetch("PALLFNFINDEXM.xlsx", "Monthly")
    if df_c is not None:
        df_c['Date'] = pd.to_datetime(df_c.iloc[:,0], errors='coerce')
        df_c['Commodities'] = pd.to_numeric(df_c.iloc[:,1], errors='coerce')
        data['comm'] = df_c.dropna()

    # Consumer Confidence
    df_cci = fetch("export-2026-02-10T06_50_22.597Z.csv")
    if df_cci is not None:
        # OECD format check
        df_cci = pd.read_csv("export-2026-02-10T06_50_22.597Z.csv", skiprows=3, header=None)
        df_cci.columns = ['Date', 'CCI']
        df_cci['Date'] = pd.to_datetime(df_cci['Date'], errors='coerce')
        data['cci'] = df_cci.dropna()

    return data

# --- 3. THE ANALYTICS ENGINE ---
db = load_terminal_data()

st.title("🏛️ INSTITUTIONAL POLICY TERMINAL")

if 'macro' not in db:
    st.error("🚨 CRITICAL FILE MISSING: 'EM_Macro_Data_India_SG_UK.xlsx'. Please upload the file from GitHub to enable Policy Analysis.")
    st.stop()

# Merge all into one Master DF for the VAR Model
df = db['macro']
if 'yield' in db: df = df.merge(db['yield'], on='Date', how='left')
if 'comm' in db: df = df.merge(db['comm'], on='Date', how='left')
if 'cci' in db: df = df.merge(db['cci'], on='Date', how='left')
df = df.sort_values('Date').ffill().bfill()

# --- 4. DASHBOARD ---
market = st.sidebar.selectbox("SELECT JURISDICTION", ["India", "UK", "Singapore"])
m_cols = {"India": ["Policy_India", "CPI_India"], "UK": ["Policy_UK", "CPI_UK"], "Singapore": ["Policy_Singapore", "CPI_Singapore"]}
p_col, c_col = m_cols[market]

# Recession Probability (using Probit Model logic)
latest_spread = df['Spread'].iloc[-1]
prob = norm.cdf(-0.5 - (1.5 * latest_spread)) * 100

# Predictive Forecasting (VAR Model)
try:
    # Use Policy Rate, Inflation, and Commodities as predictors
    model_data = df[[p_col, c_col, 'Commodities']].dropna()
    model = VAR(model_data).fit(6)
    forecast = model.forecast(model_data.values[-6:], 12)
    terminal_forecast = forecast[-1, 0]
except:
    terminal_forecast = df[p_col].iloc[-1]

# Display Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("TERMINAL RATE EST.", f"{terminal_forecast:.2f}%", f"{terminal_forecast - df[p_col].iloc[-1]:.2f}")
c2.metric("RECESSION RISK", f"{prob:.1f}%")
c3.metric("YIELD SPREAD", f"{latest_spread:.2f}")
c4.metric("CONS. SENTIMENT", f"{df['CCI'].iloc[-1]:.1f}")

# Main Chart

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, 
                    subplot_titles=(f"{market} Policy vs Inflation", "Global Macro Drivers"))

# Top Chart
fig.add_trace(go.Scatter(x=df['Date'], y=df[p_col], name="Policy Rate", line=dict(color='#002366', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df[c_col], name="CPI Inflation", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)

# Bottom Chart
fig.add_trace(go.Scatter(x=df['Date'], y=df['Spread'], name="Yield Spread", fill='tozeroy', line=dict(color='#2E8B57')), row=2, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['CCI'], name="CCI (Consumer)", line=dict(color='#C5A059')), row=2, col=1)

fig.update_layout(height=800, template="plotly_white", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- 5. INTELLIGENCE REPORT ---
st.markdown("### 📝 Quant Research Note")
trend = "Hawkish" if terminal_forecast > df[p_col].iloc[-1] else "Dovish"
st.info(f"""
**Stance:** {trend} Baseline. 
The Vector Autoregression (VAR) model suggests that given current commodity prices ({df['Commodities'].iloc[-1]:.1f}) 
and the yield spread ({latest_spread:.2f}), the {market} Central Bank is projected to move towards a 
**{terminal_forecast:.2f}%** terminal rate over the next 12 months.
""")

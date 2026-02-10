import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from statsmodels.tsa.api import VAR
from scipy.stats import norm

# --- 1. THEME & UI ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide")
st.markdown("""
    <style>
    * { font-family: 'Times New Roman', serif; }
    .stApp { background-color: #F2EFE9; }
    .metric-card { background: white; padding: 15px; border: 1px solid #A39B8F; border-left: 5px solid #002366; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE DIRECT-HIT DATA PIPELINE ---
@st.cache_data
def load_data():
    try:
        # A. Primary Macro Workbook
        df_m = pd.read_excel('EM_Macro_Data_India_SG_UK.xlsx', sheet_name='Macro data')
        df_m['Date'] = pd.to_datetime(df_m['Date'])

        # B. Yield Curve (XLSX) - Skipping FRED's 10-row header
        df_y = pd.read_excel('T10Y2Y.xlsx', skiprows=10)
        df_y.columns = ['Date', 'T10Y2Y']
        df_y['Date'] = pd.to_datetime(df_y['Date'], errors='coerce')
        df_y['T10Y2Y'] = pd.to_numeric(df_y['T10Y2Y'], errors='coerce')
        # Resample daily to monthly average
        df_y = df_y.dropna().set_index('Date').resample('MS').mean().reset_index()

        # C. Commodities (XLSX) - Skipping FRED's 10-row header
        df_c = pd.read_excel('PALLFNFINDEXM.xlsx', skiprows=10)
        df_c.columns = ['Date', 'Commodities']
        df_c['Date'] = pd.to_datetime(df_c['Date'], errors='coerce')
        df_c = df_c.dropna()

        # D. Consumer Confidence (The specific CSV) - Skipping 3 lines
        df_cci = pd.read_csv('export-2026-02-10T06_50_22.597Z.csv', skiprows=3, header=None)
        df_cci.columns = ['Date', 'CCI']
        df_cci['Date'] = pd.to_datetime(df_cci['Date'], errors='coerce')
        df_cci = df_cci.dropna()

        # Merge Pipeline
        df = df_m.merge(df_y, on='Date', how='left')
        df = df.merge(df_c, on='Date', how='left')
        df = df.merge(df_cci, on='Date', how='left')
        
        return df.sort_values('Date').ffill().bfill()
    except Exception as e:
        st.error(f"Pipeline Error: {e}")
        return None

# --- 3. EXECUTION ---
df = load_data()

if df is not None:
    st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")
    
    market = st.sidebar.selectbox("Market Selection", ["India", "UK", "Singapore"])
    m_cols = {"India": ["Policy_India", "CPI_India"], "UK": ["Policy_UK", "CPI_UK"], "Singapore": ["Policy_Singapore", "CPI_Singapore"]}
    p_col, c_col = m_cols[market]

    # Analytics
    latest_spread = df['T10Y2Y'].iloc[-1]
    rec_prob = norm.cdf(-0.5 - (1.5 * latest_spread))
    
    # VAR Forecast (Lags=6, Horizon=12)
    model_df = df[[p_col, c_col, 'Commodities']].dropna()
    results = VAR(model_df).fit(6)
    forecast = results.forecast(model_df.values[-6:], 12)

    # UI Layout
    cols = st.columns(4)
    cols[0].metric("TERMINAL RATE", f"{forecast[-1, 0]:.2f}%")
    cols[1].metric("RECESSION RISK", f"{rec_prob*100:.1f}%")
    cols[2].metric("CONS. SENTIMENT", f"{df['CCI'].iloc[-1]:.1f}")
    cols[3].metric("COMMODITY INDEX", f"{df['Commodities'].iloc[-1]:.0f}")

    # Charts
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df['Date'], y=df[p_col], name="Policy Rate", line=dict(color='#002366', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df[c_col], name="Inflation", line=dict(color='red', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['T10Y2Y'], name="Yield Spread", fill='tozeroy', line=dict(color='green')), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_white", margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Analyst Intelligence:** Based on the current Yield Spread of {latest_spread:.2f}, the model estimates a {rec_prob*100:.1f}% probability of recession. Historical VAR projections suggest the {market} terminal rate should settle near {forecast[-1,0]:.2f}%.")
else:
    st.warning("Please ensure all XLSX and CSV files are in the directory and try again.")

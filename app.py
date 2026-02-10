import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from statsmodels.tsa.api import VAR
from scipy.stats import norm

# --- 1. RESEARCH UI ENGINE ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide")

st.markdown("""
    <style>
    * { font-family: 'Times New Roman', Times, serif !important; }
    .stApp { background-color: #F2EFE9; color: #2C2C2C; }
    .analyst-card { padding: 15px; border: 1px solid #A39B8F; background-color: #FFFFFF; border-left: 5px solid #002366; font-size: 0.95rem; margin-bottom: 20px; }
    .for-you-card { padding: 20px; background-color: #FDFCFB; border: 1px solid #A39B8F; border-left: 10px solid #002366; margin-bottom: 25px; }
    .main-title { font-size: 32px; font-weight: bold; color: #002366; border-bottom: 3px solid #C5A059; padding-bottom: 5px; margin-bottom: 20px;}
    .section-header { color: #7A6D5D; font-weight: bold; font-size: 1.1rem; text-transform: uppercase; margin-top: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ZERO-FAIL DATA PIPELINE ---
def robust_load(filename, search_term, is_csv=False):
    """Deep scans files to find the data start row and identifies columns."""
    try:
        if not os.path.exists(filename):
            return None
        
        # 1. Find the header row by looking for 'date'
        if is_csv:
            # CCI CSV has metadata that confuses the parser, we skip the first 3 lines
            df = pd.read_csv(filename, skiprows=3, header=None, names=['Date', 'Value'])
        else:
            # For Excel (FRED/Commodities), find the row containing 'date'
            header_scan = pd.read_excel(filename, nrows=20, header=None)
            start_row = 0
            for i, row in header_scan.iterrows():
                if any('date' in str(val).lower() for val in row.values):
                    start_row = i
                    break
            df = pd.read_excel(filename, skiprows=start_row)
        
        # 2. Identify Date and Value columns
        d_cols = [c for c in df.columns if 'date' in str(c).lower()]
        v_cols = [c for c in df.columns if c not in d_cols]
        
        if not d_cols or not v_cols:
            return None
            
        df = df.rename(columns={d_cols[0]: 'Date', v_cols[0]: search_term})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df[search_term] = pd.to_numeric(df[search_term], errors='coerce')
        return df.dropna(subset=['Date'])
    except Exception as e:
        st.sidebar.error(f"Error loading {filename}: {e}")
        return None

@st.cache_data
def load_and_merge():
    # A. Primary Macro Data (The main workbook)
    try:
        df_m = pd.read_excel('EM_Macro_Data_India_SG_UK.xlsx', sheet_name='Macro data')
        df_m['Date'] = pd.to_datetime(df_m['Date'])
    except:
        return None

    # B. Yield Curve (XLSX) - Resample Daily to Monthly
    df_y = robust_load('T10Y2Y.xlsx', 'T10Y2Y')
    if df_y is not None:
        df_y = df_y.set_index('Date').resample('MS').mean().reset_index()

    # C. Commodities (XLSX)
    df_c = robust_load('PALLFNFINDEXM.xlsx', 'Commodities')

    # D. Consumer Confidence (The specific CSV)
    df_cci = robust_load('export-2026-02-10T06_50_22.597Z.csv', 'CCI', is_csv=True)

    # Merge everything on Date
    df = df_m
    for extra_df in [df_y, df_c, df_cci]:
        if extra_df is not None:
            df = df.merge(extra_df, on='Date', how='left')
    
    return df.sort_values('Date').ffill().bfill()

# --- 3. ANALYTICS ---
def get_recession_prob(spread):
    return norm.cdf(-0.5 - (1.5 * spread))

# --- 4. RENDER TERMINAL ---
df = load_and_merge()

if df is not None:
    with st.sidebar:
        st.markdown("<h2>NAVIGATE</h2>", unsafe_allow_html=True)
        market = st.selectbox("SELECT MARKET", ["India", "UK", "Singapore"])
        f_horizon = st.slider("Forecast Horizon", 6, 24, 12)
        lags = st.number_input("Model Lags", 1, 12, 6)

    m_map = {
        "India": {"p": "Policy_India", "cpi": "CPI_India"},
        "UK": {"p": "Policy_UK", "cpi": "CPI_UK"},
        "Singapore": {"p": "Policy_Singapore", "cpi": "CPI_Singapore"}
    }
    m = m_map[market]
    
    # VAR Model (Endogenous variables)
    # Check if we have the needed columns
    needed = [m['p'], m['cpi'], 'Commodities']
    if all(col in df.columns for col in needed):
        model_df = df[needed].dropna()
        var_res = VAR(model_df).fit(lags)
        forecast = var_res.forecast(model_df.values[-var_res.k_ar:], f_horizon)
        terminal_rate = forecast[-1, 0]
    else:
        terminal_rate = df[m['p']].iloc[-1]

    # Metrics
    spread = df['T10Y2Y'].iloc[-1] if 'T10Y2Y' in df.columns else 0
    prob = get_recession_prob(spread)
    cci = df['CCI'].iloc[-1] if 'CCI' in df.columns else 100

    st.markdown(f"<div class='main-title'>{market.upper()} MACRO TERMINAL</div>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TERMINAL RATE (VAR)", f"{terminal_rate:.2f}%")
    c2.metric("RECESSION PROB.", f"{prob*100:.1f}%")
    c3.metric("CONS. CONFIDENCE", f"{cci:.1f}")
    c4.metric("YIELD SPREAD", f"{spread:.2f}")

    # Plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df[m['p']], name="Policy Rate", line=dict(color='#002366', width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df[m['cpi']], name="CPI Inflation", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)
    if 'T10Y2Y' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['T10Y2Y'], name="Yield Curve", fill='tozeroy', line=dict(color='#2E8B57')), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"<div class='analyst-card'><b>Analyst Note:</b> The 10Y-2Y spread of {spread:.2f} suggests a {prob*100:.1f}% chance of a recession. With CCI at {cci:.1f}, consumption {'is holding up' if cci > 100 else 'is showing signs of stress'}.</div>", unsafe_allow_html=True)
else:
    st.error("Data Pipeline Failure. Ensure all .xlsx files and the .csv are in the same folder as this script.")

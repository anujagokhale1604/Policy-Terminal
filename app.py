import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from statsmodels.tsa.api import VAR
from scipy.stats import norm

# --- 1. RESEARCH UI ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide")

st.markdown("""
    <style>
    * { font-family: 'Times New Roman', serif; }
    .stApp { background-color: #F2EFE9; color: #2C2C2C; }
    .analyst-card { padding: 15px; border: 1px solid #A39B8F; background: white; border-left: 5px solid #002366; margin-bottom: 20px; }
    .main-title { font-size: 32px; font-weight: bold; color: #002366; border-bottom: 3px solid #C5A059; padding-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SELF-HEALING DATA PIPELINE ---
def robust_load_macro(filename, label):
    """Detects header rows and column names automatically to prevent index errors."""
    if not os.path.exists(filename):
        return None
    try:
        # Step 1: Find where the actual data starts (look for 'date')
        # We read first 20 rows to scan for headers
        df_scan = pd.read_excel(filename, nrows=20, header=None)
        start_row = 0
        for i, row in df_scan.iterrows():
            if any('date' in str(val).lower() for val in row.values):
                start_row = i
                break
        
        # Step 2: Read file from that row
        df = pd.read_excel(filename, skiprows=start_row)
        
        # Step 3: Find Date and Data columns without using indices
        date_col = next((c for c in df.columns if 'date' in str(c).lower()), None)
        data_col = next((c for c in df.columns if c != date_col), None)
        
        if date_col is None or data_col is None:
            return None
            
        df = df[[date_col, data_col]].rename(columns={date_col: 'Date', data_col: label})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df[label] = pd.to_numeric(df[label], errors='coerce')
        return df.dropna(subset=['Date'])
    except Exception as e:
        st.sidebar.warning(f"Skipping {filename}: {e}")
        return None

@st.cache_data
def load_all_data():
    # A. Primary Macro Workbook
    try:
        df = pd.read_excel('EM_Macro_Data_India_SG_UK.xlsx', sheet_name='Macro data')
        df['Date'] = pd.to_datetime(df['Date'])
    except:
        st.error("Fatal Error: 'EM_Macro_Data_India_SG_UK.xlsx' not found or invalid.")
        return None

    # B. Yield Curve (XLSX) - Resample Daily to Monthly
    df_y = robust_load_macro('T10Y2Y.xlsx', 'T10Y2Y')
    if df_y is not None:
        df_y = df_y.set_index('Date').resample('MS').mean().reset_index()
        df = df.merge(df_y, on='Date', how='left')

    # C. Commodities (XLSX)
    df_c = robust_load_macro('PALLFNFINDEXM.xlsx', 'Commodities')
    if df_c is not None:
        df = df.merge(df_c, on='Date', how='left')

    # D. Consumer Confidence (The Specific CSV Title provided)
    try:
        csv_name = 'export-2026-02-10T06_50_22.597Z.csv'
        if os.path.exists(csv_name):
            df_cci = pd.read_csv(csv_name, skiprows=3, header=None, names=['Date', 'CCI'])
            df_cci['Date'] = pd.to_datetime(df_cci['Date'])
            df = df.merge(df_cci, on='Date', how='left')
    except:
        pass

    return df.sort_values('Date').ffill().bfill()

# --- 3. DASHBOARD RENDER ---
df = load_all_data()

if df is not None:
    st.markdown(f"<div class='main-title'>INSTITUTIONAL MACRO MONITOR</div>", unsafe_allow_html=True)
    
    market = st.sidebar.selectbox("Market Selection", ["India", "UK", "Singapore"])
    m_map = {"India": ["Policy_India", "CPI_India"], "UK": ["Policy_UK", "CPI_UK"], "Singapore": ["Policy_Singapore", "CPI_Singapore"]}
    p_col, c_col = m_map[market]

    # Metrics Calculations
    spread = df['T10Y2Y'].iloc[-1] if 'T10Y2Y' in df.columns else 0.0
    rec_prob = norm.cdf(-0.5 - (1.5 * spread))
    cci = df['CCI'].iloc[-1] if 'CCI' in df.columns else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("RECESSION PROB.", f"{rec_prob*100:.1f}%")
    c2.metric("YIELD SPREAD (10Y-2Y)", f"{spread:.2f}")
    c3.metric("CONS. CONFIDENCE", f"{cci:.1f}")

    # Main Visualization
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df[p_col], name="Policy Rate", line=dict(color='#002366', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df[c_col], name="CPI Inflation", line=dict(color='red', dash='dot')), row=1, col=1)
    
    if 'T10Y2Y' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['T10Y2Y'], name="Yield Spread", fill='tozeroy', line=dict(color='green')), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class='analyst-card'>
    <b>Macro Summary:</b> The yield spread of {spread:.2f} results in a <b>{rec_prob*100:.1f}%</b> recession probability. 
    Consumer sentiment is {'robust' if cci > 100 else 'declining'} at {cci:.1f}.
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Missing Files: Please ensure 'EM_Macro_Data_India_SG_UK.xlsx' is in your folder.")

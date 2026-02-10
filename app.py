import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# --- 2. ROBUST DATA PIPELINE ---
@st.cache_data
def load_and_merge_data():
    try:
        # A. Primary Macro Workbook
        df_m = pd.read_excel('EM_Macro_Data_India_SG_UK.xlsx', sheet_name='Macro data')
        df_m['Date'] = pd.to_datetime(df_m['Date'])
        
        # B. Yield Curve (XLSX) - Dynamic Column Finder
        df_yield = pd.read_excel('T10Y2Y.xlsx')
        # Skip header rows if FRED added them
        if "FRED" in str(df_yield.iloc[0,0]):
            df_yield = pd.read_excel('T10Y2Y.xlsx', skiprows=10)
        
        d_cols = [c for c in df_yield.columns if 'date' in str(c).lower()]
        v_cols = [c for c in df_yield.columns if 'T10' in str(c) or '2Y' in str(c)]
        
        if not d_cols or not v_cols:
            st.error("Could not find Date or Yield columns in T10Y2Y.xlsx")
            return None
            
        df_yield[d_cols[0]] = pd.to_datetime(df_yield[d_cols[0]], errors='coerce')
        df_yield = df_yield.dropna(subset=[d_cols[0]])
        df_yield = df_yield.set_index(d_cols[0]).resample('MS').mean().reset_index().rename(columns={d_cols[0]: 'Date', v_cols[0]: 'T10Y2Y'})
        
        # C. Commodity Index (XLSX)
        df_comm = pd.read_excel('PALLFNFINDEXM.xlsx')
        if "FRED" in str(df_comm.iloc[0,0]):
            df_comm = pd.read_excel('PALLFNFINDEXM.xlsx', skiprows=10)
            
        c_d_cols = [c for c in df_comm.columns if 'date' in str(c).lower()]
        c_v_cols = [c for c in df_comm.columns if 'PALL' in str(c)]
        
        df_comm['Date'] = pd.to_datetime(df_comm[c_d_cols[0]])
        df_comm = df_comm[['Date', c_v_cols[0]]].rename(columns={c_v_cols[0]: 'PALLFNFINDEXM'})

        # D. Consumer Confidence Index (CSV) - Adjusted to skip 3 lines of metadata
        df_cci = pd.read_csv('export-2026-02-10T06_50_22.597Z.csv', skiprows=3, header=None, names=['Date', 'CCI'])
        df_cci['Date'] = pd.to_datetime(df_cci['Date'])

        # Merge Pipeline
        df = df_m.merge(df_yield, on='Date', how='left')
        df = df.merge(df_comm, on='Date', how='left')
        df = df.merge(df_cci, on='Date', how='left')
        
        return df.sort_values('Date').ffill().bfill()
    except Exception as e:
        st.error(f"Data Pipeline Error: {e}")
        return None

# --- 3. ANALYTICS ---
def get_recession_prob(spread):
    return norm.cdf(-0.5 - (1.5 * spread))

# --- 4. DASHBOARD ---
df = load_and_merge_data()

if df is not None:
    with st.sidebar:
        st.markdown("<h2>NAVIGATE</h2>", unsafe_allow_html=True)
        market = st.selectbox("1. SELECT MARKET", ["India", "UK", "Singapore"])
        forecast_len = st.slider("2. VAR FORECAST HORIZON", 6, 24, 12)
        st.divider()
        st.markdown("<b>VAR SETTINGS</b>")
        lags = st.number_input("Model Lags (Months)", 1, 12, 6)

    m_map = {
        "India": {"p": "Policy_India", "cpi": "CPI_India", "sym": "INR"},
        "UK": {"p": "Policy_UK", "cpi": "CPI_UK", "sym": "GBP"},
        "Singapore": {"p": "Policy_Singapore", "cpi": "CPI_Singapore", "sym": "SGD"}
    }
    m = m_map[market]
    
    # VAR MODELING
    var_cols = [m['p'], m['cpi'], 'PALLFNFINDEXM']
    model_df = df[var_cols].dropna()
    var_model = VAR(model_df)
    results = var_model.fit(lags)
    forecast = results.forecast(model_df.values[-results.k_ar:], forecast_len)
    
    # METRICS
    curr_spread = df['T10Y2Y'].iloc[-1]
    rec_prob = get_recession_prob(curr_spread)
    
    st.markdown(f"<div class='main-title'>{market.upper()} MACRO TERMINAL</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VAR TERMINAL RATE", f"{forecast[-1, 0]:.2f}%")
    c2.metric("COMMODITY INDEX", f"{df['PALLFNFINDEXM'].iloc[-1]:.1f}")
    c3.metric("RECESSION PROB.", f"{rec_prob*100:.1f}%")
    c4.metric("CONS. CONFIDENCE", f"{df['CCI'].iloc[-1]:.1f}")

    # CHARTS
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        subplot_titles=("Monetary Policy Path", "Yield Curve (10Y-2Y)"))
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df[m['p']], name="Policy Rate", line=dict(color='#002366', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df[m['cpi']], name="CPI (YoY)", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['T10Y2Y'], name="Yield Spread", fill='tozeroy', line=dict(color='#2E8B57')), row=2, col=1)
    
    fig.update_layout(height=550, template="plotly_white", margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ANALYST NOTES
    st.markdown(f"""
    <div class='analyst-card'>
    <b>Macro Insight:</b> The yield spread is at <b>{curr_spread:.2f}</b>. 
    A negative spread (inversion) is the most reliable predictor of a recession. 
    The current probability stands at <b>{rec_prob*100:.1f}%</b>.
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Files Missing or Corrupted. Ensure 'EM_Macro_Data_India_SG_UK.xlsx', 'T10Y2Y.xlsx', 'PALLFNFINDEXM.xlsx', and 'export-2026-02-10T06_50_22.597Z.csv' are in your folder.")

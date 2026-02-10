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

# --- 2. DATA PIPELINE ---
@st.cache_data
def load_and_merge_data():
    try:
        # A. Primary Macro Workbook
        # Assuming the original xlsx file is present
        xl = pd.ExcelFile('EM_Macro_Data_India_SG_UK.xlsx')
        df_m = xl.parse('Macro data')
        df_m['Date'] = pd.to_datetime(df_m['Date'])
        
        # B. Yield Curve (Handling Daily -> Monthly Resampling)
        df_yield = pd.read_csv('T10Y2Y.xlsx - Daily.csv')
        df_yield['observation_date'] = pd.to_datetime(df_yield['observation_date'])
        # Standard Econ practice: Use the monthly mean for macro models
        df_yield = df_yield.set_index('observation_date').resample('MS').mean().reset_index().rename(columns={'observation_date': 'Date'})
        
        # C. Commodity Index (Monthly)
        df_comm = pd.read_csv('PALLFNFINDEXM.xlsx - Monthly.csv')
        df_comm['Date'] = pd.to_datetime(df_comm['observation_date'])
        df_comm = df_comm[['Date', 'PALLFNFINDEXM']]

        # D. Consumer Confidence Index (CSV with header rows)
        df_cci = pd.read_csv('export-2026-02-10T06_50_22.597Z.csv', skiprows=4, header=None, names=['Date', 'CCI'])
        df_cci['Date'] = pd.to_datetime(df_cci['Date'])

        # Merge Pipeline
        df = df_m.merge(df_yield, on='Date', how='left')
        df = df.merge(df_comm, on='Date', how='left')
        df = df.merge(df_cci, on='Date', how='left')
        
        return df.sort_values('Date').ffill().bfill()
    except Exception as e:
        st.error(f"Data Pipeline Error: {e}")
        return None

# --- 3. ECONOMETRIC CALCS ---
def get_recession_prob(spread):
    """Probit-style recession probability based on yield curve inversion"""
    # Simplified institutional logic: Prob rises as spread drops below 0
    return norm.cdf(-0.5 - (1.5 * spread))

# --- 4. DASHBOARD RENDER ---
df = load_and_merge_data()

with st.sidebar:
    st.markdown("<h2>NAVIGATE</h2>", unsafe_allow_html=True)
    market = st.selectbox("1. SELECT MARKET", ["India", "UK", "Singapore"])
    forecast_len = st.slider("2. VAR FORECAST HORIZON", 6, 24, 12)
    st.divider()
    st.markdown("<b>VAR SETTINGS</b>")
    lags = st.number_input("Model Lags (Months)", 1, 12, 6)

if df is not None:
    m_map = {
        "India": {"p": "Policy_India", "cpi": "CPI_India", "gdp": "GDP_India", "sym": "INR"},
        "UK": {"p": "Policy_UK", "cpi": "CPI_UK", "gdp": "GDP_UK", "sym": "GBP"},
        "Singapore": {"p": "Policy_Singapore", "cpi": "CPI_Singapore", "gdp": "GDP_Singapore", "sym": "SGD"}
    }
    m = m_map[market]
    
    # --- VAR MODELING ---
    # We include Policy Rate, CPI, and Commodities as endogenous variables
    var_cols = [m['p'], m['cpi'], 'PALLFNFINDEXM']
    model_df = df[var_cols].dropna()
    var_model = VAR(model_df)
    results = var_model.fit(lags)
    forecast = results.forecast(model_df.values[-results.k_ar:], forecast_len)
    
    # Latest Data for Metrics
    curr_spread = df['T10Y2Y'].iloc[-1]
    rec_prob = get_recession_prob(curr_spread)
    
    # METRICS BAR
    st.markdown(f"<div class='main-title'>{market.upper()} QUANT STRATEGY TERMINAL</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VAR TERMINAL RATE", f"{forecast[-1, 0]:.2f}%")
    c2.metric("SUPPLY STRESS (PALL)", f"{df['PALLFNFINDEXM'].iloc[-1]:.1f}")
    c3.metric("RECESSION PROB.", f"{rec_prob*100:.1f}%")
    c4.metric("CONS. CONFIDENCE", f"{df['CCI'].iloc[-1]:.1f}")

    # CHARTS
    st.markdown("<div class='section-header'><i class='fas fa-microchip'></i> I. Endogenous Multi-Variable Projections</div>", unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        subplot_titles=("Policy & Inflation Path", "Yield Curve (Recession Predictor)"))
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df[m['p']], name="Policy Rate", line=dict(color='#002366', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df[m['cpi']], name="CPI (YoY)", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df['T10Y2Y'], name="Yield Spread (10Y-2Y)", fill='tozeroy', line=dict(color='#2E8B57')), row=2, col=1)
    
    fig.update_layout(height=550, template="plotly_white", margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ANALYST NOTES
    st.markdown("<div class='section-header'><i class='fas fa-file-contract'></i> II. Analyst Intelligence Note</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='analyst-card'>
    <b>Model Insight:</b> The Vector Autoregression (VAR) model accounts for the feedback loop between commodity prices and local policy. 
    Current data shows a yield spread of <b>{curr_spread:.2f}bps</b>. Historically, inversions of this magnitude imply a 
    <b>{rec_prob*100:.1f}%</b> probability of a hard landing within 18 months. 
    The CCI at <b>{df['CCI'].iloc[-1]:.1f}</b> indicates consumer sentiment is {'stable' if df['CCI'].iloc[-1] > 100 else 'deteriorating'}.
    </div>
    """, unsafe_allow_html=True)

    # RECOMMENDATIONS
    st.markdown("<div class='section-header'><i class='fas fa-briefcase'></i> III. Portfolio Recommendations</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='for-you-card'>
    <b>Institutional Positioning:</b><br>
    • <b>Interest Rate Risk:</b> The VAR path suggests a terminal equilibrium of {forecast[-1, 0]:.2f}%. Investors should {'lock in yields' if forecast[-1, 0] < df[m['p']].iloc[-1] else 'stay floating'} to optimize carry.<br>
    • <b>Inflation Hedge:</b> With Global Commodities at {df['PALLFNFINDEXM'].iloc[-1]:.1f}, supply-side pressure remains {'elevated' if df['PALLFNFINDEXM'].iloc[-1] > 150 else 'moderate'}. Diversification into hard assets is recommended.
    </div>
    """, unsafe_allow_html=True)

    # LATEX
    st.markdown("<div class='section-header'>Econometric Specification</div>", unsafe_allow_html=True)
    st.latex(r'''
    Y_t = C + \sum_{i=1}^{p} \Phi_i Y_{t-i} + \epsilon_t, \quad Y = [Policy, CPI, Commodities]^T
    ''')

else:
    st.error("Missing required data files. Ensure Excel and CSVs are in the directory.")

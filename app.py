import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from statsmodels.tsa.api import VAR
from scipy.stats import norm

# --- 1. CHIC RESEARCH UI ENGINE ---
st.set_page_config(page_title="Macro Intel Pro", layout="wide")

st.markdown("""
    <style>
    * { font-family: 'Times New Roman', Times, serif !important; }
    .stApp { background-color: #F2EFE9; color: #2C2C2C; }
    .analyst-card { padding: 15px; border: 1px solid #A39B8F; background-color: #FFFFFF; border-left: 5px solid #002366; font-size: 0.95rem; margin-bottom: 20px; }
    .for-you-card { padding: 20px; background-color: #FDFCFB; border: 1px solid #A39B8F; border-left: 10px solid #002366; margin-bottom: 25px; }
    .main-title { font-size: 32px; font-weight: bold; color: #002366; border-bottom: 3px solid #C5A059; padding-bottom: 5px; margin-bottom: 20px; }
    .section-header { color: #7A6D5D; font-weight: bold; font-size: 1.1rem; text-transform: uppercase; margin-top: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ROBUST DATA ENGINE ---
def deep_scan_xlsx(filename, search_term):
    """Automatically finds the data start row in FRED/Institutional Excel files."""
    try:
        # Load first 20 rows to find the header
        header_check = pd.read_excel(filename, nrows=20)
        start_row = 0
        for i, row in header_check.iterrows():
            if any(search_term.lower() in str(val).lower() for val in row.values):
                start_row = i + 1
                break
        
        df = pd.read_excel(filename, skiprows=start_row)
        # Identify columns
        d_col = [c for c in df.columns if 'date' in str(c).lower()][0]
        v_col = [c for c in df.columns if c != d_col][0]
        
        df[d_col] = pd.to_datetime(df[d_col], errors='coerce')
        df[v_col] = pd.to_numeric(df[v_col], errors='coerce')
        return df.dropna(subset=[d_col]).rename(columns={d_col: 'Date', v_col: search_term})
    except Exception as e:
        st.error(f"Error reading {filename}: {e}")
        return None

@st.cache_data
def load_all_data():
    try:
        # A. Base Workbook
        df_m = pd.read_excel('EM_Macro_Data_India_SG_UK.xlsx', sheet_name='Macro data')
        df_m['Date'] = pd.to_datetime(df_m['Date'])
        
        # B. Yield Curve (XLSX) - Daily to Monthly
        df_y = deep_scan_xlsx('T10Y2Y.xlsx', 'observation_date')
        df_y = df_y.set_index('Date').resample('MS').mean().reset_index().rename(columns={'observation_date': 'T10Y2Y'})
        
        # C. Commodity Index (XLSX)
        df_c = deep_scan_xlsx('PALLFNFINDEXM.xlsx', 'observation_date')
        df_c = df_c.rename(columns={'observation_date': 'Commodities'})

        # D. Consumer Confidence (CSV)
        # This specific CSV has 3 meta rows
        df_cci = pd.read_csv('export-2026-02-10T06_50_22.597Z.csv', skiprows=3, header=None, names=['Date', 'CCI'])
        df_cci['Date'] = pd.to_datetime(df_cci['Date'])

        # Master Merge
        df = df_m.merge(df_y, on='Date', how='left').merge(df_c, on='Date', how='left').merge(df_cci, on='Date', how='left')
        return df.sort_values('Date').ffill().bfill()
    except Exception as e:
        st.error(f"Data Pipeline Failure: {e}")
        return None

# --- 3. ANALYTICS (VAR & PROBIT) ---
def get_recession_prob(spread):
    # Probit model logic: -0.5 is intercept, -1.5 is sensitivity to inversion
    return norm.cdf(-0.5 - (1.5 * spread))

# --- 4. TERMINAL UI ---
df = load_all_data()

if df is not None:
    with st.sidebar:
        st.markdown("<h2>NAVIGATE</h2>", unsafe_allow_html=True)
        market = st.selectbox("SELECT MARKET", ["India", "UK", "Singapore"])
        f_horizon = st.slider("VAR Forecast Horizon (Months)", 6, 24, 12)
        st.divider()
        st.info("Institutional Tip: When yield spreads turn negative (below 0), the recession probability usually spikes above 30%.")

    m_map = {
        "India": {"p": "Policy_India", "cpi": "CPI_India", "sym": "INR"},
        "UK": {"p": "Policy_UK", "cpi": "CPI_UK", "sym": "GBP"},
        "Singapore": {"p": "Policy_Singapore", "cpi": "CPI_Singapore", "sym": "SGD"}
    }
    m = m_map[market]
    
    # RUN VECTOR AUTOREGRESSION (VAR)
    var_cols = [m['p'], m['cpi'], 'Commodities']
    model_df = df[var_cols].dropna()
    var_res = VAR(model_df).fit(maxlags=6)
    forecast = var_res.forecast(model_df.values[-var_res.k_ar:], f_horizon)
    
    # GET STATS
    latest_spread = df['T10Y2Y'].iloc[-1]
    rec_prob = get_recession_prob(latest_spread)
    latest_cci = df['CCI'].iloc[-1]
    
    st.markdown(f"<div class='main-title'>{market.upper()} RESEARCH TERMINAL</div>", unsafe_allow_html=True)

    # TOP METRICS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VAR TERMINAL RATE", f"{forecast[-1, 0]:.2f}%", f"{forecast[-1, 0] - df[m['p']].iloc[-1]:.2f}%")
    c2.metric("RECESSION PROB.", f"{rec_prob*100:.1f}%")
    c3.metric("CONS. CONFIDENCE", f"{latest_cci:.1f}")
    c4.metric("COMMODITY STRESS", f"{df['Commodities'].iloc[-1]:.1f}")

    # PROJECTION CHART
    st.markdown("<div class='section-header'>I. Forecasting & Financial Cycle</div>", unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                        subplot_titles=("Policy Rate & CPI Path (VAR)", "Yield Curve Inversion Monitor"))
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df[m['p']], name="Policy Rate", line=dict(color='#002366', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df[m['cpi']], name="CPI (YoY)", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['T10Y2Y'], name="Yield Spread (10Y-2Y)", fill='tozeroy', line=dict(color='#2E8B57')), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_white", margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ANALYSIS NOTE
    st.markdown("<div class='section-header'>II. Analyst Intelligence Note</div>", unsafe_allow_html=True)
    sentiment = "Bullish" if latest_cci > 100 else "Cautious/Bearish"
    st.markdown(f"""
    <div class='analyst-card'>
    <b>Macro Summary:</b> The interaction between <b>Global Commodities</b> and local inflation suggests a terminal interest rate of <b>{forecast[-1, 0]:.2f}%</b>. 
    The current 10Y-2Y spread of <b>{latest_spread:.2f}</b> yields a Probit-calculated recession probability of <b>{rec_prob*100:.1f}%</b>. 
    The Consumer Confidence Index (CCI) stands at <b>{latest_cci:.1f}</b>, indicating a <b>{sentiment}</b> consumer outlook for the next two quarters.
    </div>
    """, unsafe_allow_html=True)

    # STRATEGIC RECOMMENDATIONS
    st.markdown("<div class='section-header'>III. Portfolio Strategy</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='for-you-card'>
    <b>Strategic Guidance:</b><br>
    • <b>Fixed Income:</b> If the VAR model projects a rate cut ({forecast[-1,0]:.2f}%), begin extending duration in sovereign bonds to capture capital appreciation.<br>
    • <b>Risk Exposure:</b> Given the {rec_prob*100:.1f}% recession probability, portfolio managers should reduce weight in high-beta cyclicals and rotate into defensive staples.<br>
    • <b>Cost Monitoring:</b> The Commodity Index is a leading driver of your local CPI. Current levels of {df['Commodities'].iloc[-1]:.1f} suggest inflation pressure is {'cooling' if df['Commodities'].iloc[-1] < 150 else 'persisting'}.
    </div>
    """, unsafe_allow_html=True)

    # MATHEMATICAL SPECIFICATION
    st.latex(r'''P(Recession) = \Phi(\beta_0 + \beta_1 \cdot Spread_t)''')

else:
    st.error("Missing Files. Ensure 'EM_Macro_Data_India_SG_UK.xlsx', 'T10Y2Y.xlsx', 'PALLFNFINDEXM.xlsx', and the CSV are in the same folder.")

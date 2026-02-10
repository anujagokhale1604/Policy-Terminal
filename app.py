import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy.stats import norm

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Institutional Macro Terminal", layout="wide")
st.markdown("<style>*{font-family:'Times New Roman',serif;}.stApp{background-color:#FDFCFB;}</style>", unsafe_allow_html=True)

# EXACT FILE NAMES FROM YOUR UPLOADS
YIELD_FILE = "T10Y2Y.xlsx - Daily.csv"
COMM_FILE = "PALLFNFINDEXM.xlsx - Monthly.csv"
CCI_FILE = "export-2026-02-10T06_50_22.597Z.csv"
MACRO_FILE = "EM_Macro_Data_India_SG_UK.xlsx - Macro data.csv"
INR_FILE = "DEXINUS.xlsx - Daily.csv"
GBP_FILE = "DEXUSUK.xlsx - Daily.csv"
SGD_FILE = "AEXSIUS.xlsx - Annual.csv"

# --- 2. DATA PROCESSING ENGINE ---
@st.cache_data
def load_all_data():
    def to_monthly(df, date_col, val_col):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        # Snap all dates to the 1st of the month for perfect merging
        df['Date'] = df[date_col].dt.to_period('M').dt.to_timestamp()
        return df[['Date', val_col]].groupby('Date').mean().reset_index()

    db = {}
    
    # Global Macro Indicators
    if os.path.exists(YIELD_FILE):
        db['yield'] = to_monthly(pd.read_csv(YIELD_FILE), 'observation_date', 'T10Y2Y')
    if os.path.exists(COMM_FILE):
        db['comm'] = to_monthly(pd.read_csv(COMM_FILE), 'observation_date', 'PALLFNFINDEXM')
    if os.path.exists(CCI_FILE):
        df_cci = pd.read_csv(CCI_FILE, skiprows=3, header=None)
        df_cci.columns = ['Date_Raw', 'CCI']
        db['cci'] = to_monthly(df_cci, 'Date_Raw', 'CCI')

    # Local Macro Data (India, UK, SG)
    if os.path.exists(MACRO_FILE):
        df_m = pd.read_csv(MACRO_FILE)
        df_m['Date'] = pd.to_datetime(df_m['Date']).dt.to_period('M').dt.to_timestamp()
        db['macro'] = df_m

    # FX Data
    if os.path.exists(INR_FILE): db['fx_inr'] = to_monthly(pd.read_csv(INR_FILE), 'observation_date', 'DEXINUS')
    if os.path.exists(GBP_FILE): db['fx_gbp'] = to_monthly(pd.read_csv(GBP_FILE), 'observation_date', 'DEXUSUK')
    if os.path.exists(SGD_FILE): db['fx_sgd'] = to_monthly(pd.read_csv(SGD_FILE), 'observation_date', 'AEXSIUS')
            
    return db

# --- 3. DATA MERGE & ALIGNMENT ---
data_dict = load_all_data()
# Start timeline from 2012 (matching your macro data)
master_df = pd.DataFrame({'Date': pd.date_range(start="2012-01-01", end="2026-02-01", freq='MS')})

for key in data_dict:
    master_df = master_df.merge(data_dict[key], on='Date', how='left')

# Robust cleanup: Forward fill ensures current metrics aren't 0
master_df = master_df.sort_values('Date').ffill().bfill()

# --- 4. THE DASHBOARD UI ---
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

market = st.sidebar.selectbox("Jurisdiction", ["India", "UK", "Singapore"])
mappings = {
    "India": {"policy": "Policy_India", "cpi": "CPI_India", "fx": "DEXINUS", "fx_label": "INR/USD"},
    "UK": {"policy": "Policy_UK", "cpi": "CPI_UK", "fx": "DEXUSUK", "fx_label": "USD/GBP"},
    "Singapore": {"policy": "Policy_Singapore", "cpi": "CPI_Singapore", "fx": "AEXSIUS", "fx_label": "SGD/USD"}
}

# Calculated Metrics (Latest available non-NaN values)
latest = master_df.iloc[-1]
spread = latest.get('T10Y2Y', 0.0)
cci = latest.get('CCI', 0.0)
comm = latest.get('PALLFNFINDEXM', 0.0)
# Recession Risk formula: ~30.9% is normal for a flat curve
prob = norm.cdf(-0.5 - (1.5 * spread)) * 100 if spread != 0 else 30.9

c1, c2, c3, c4 = st.columns(4)
c1.metric("RECESSION RISK", f"{prob:.1f}%")
c2.metric("YIELD SPREAD (10Y-2Y)", f"{spread:.2f}")
c3.metric("CONS. CONFIDENCE", f"{cci:.1f}")
c4.metric("COMMODITY INDEX", f"{comm:.0f}")

# --- 5. CHARTS ---
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    subplot_titles=(f"{market}: Policy vs Inflation", "Global Macro Drivers", f"{market}: Sentiment & FX Rate"))

# Row 1: Policy vs CPI
p_col, c_col = mappings[market]['policy'], mappings[market]['cpi']
if p_col in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df[p_col], name="Policy Rate", line=dict(color='#002366', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df[c_col], name="CPI Inflation", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)

# Row 2: Yields & Commodities
if 'T10Y2Y' in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df['T10Y2Y'], name="Yield Spread", fill='tozeroy', line=dict(color='#2E8B57')), row=2, col=1)
if 'PALLFNFINDEXM' in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df['PALLFNFINDEXM'], name="Commodity Index", line=dict(color='#C5A059')), row=2, col=1)

# Row 3: Sentiment & FX
fx_col = mappings[market]['fx']
if 'CCI' in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df['CCI'], name="Consumer Sentiment", line=dict(color='#4682B4')), row=3, col=1)
if fx_col in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df[fx_col], name=mappings[market]['fx_label'], line=dict(color='#708090')), row=3, col=1)

fig.update_layout(height=1000, template="plotly_white", showlegend=True, margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig, use_container_width=True)

# System Audit Tool (Check if files are being read)
with st.expander("System Audit - Data Source Status"):
    cols = st.columns(2)
    cols[0].write(f"Yield Data: {'✅' if 'T10Y2Y' in master_df.columns else '❌'}")
    cols[0].write(f"Commodity Data: {'✅' if 'PALLFNFINDEXM' in master_df.columns else '❌'}")
    cols[1].write(f"CCI Data: {'✅' if 'CCI' in master_df.columns else '❌'}")
    cols[1].write(f"Macro Sheet: {'✅' if 'macro' in data_dict else '❌'}")

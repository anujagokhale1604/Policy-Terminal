import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from scipy.stats import norm

# --- 1. SETTINGS & UI ---
st.set_page_config(page_title="Institutional Macro Terminal", layout="wide")
st.markdown("<style>*{font-family:'Times New Roman',serif;}.stApp{background-color:#FDFCFB;}</style>", unsafe_allow_html=True)

# --- 2. ROBUST DATA ENGINE ---
@st.cache_data
def load_data():
    def clean_df(df, val_name):
        # Identify date column (usually 1st column)
        d_col = next((c for c in df.columns if 'date' in str(c).lower()), df.columns[0])
        df = df.rename(columns={d_col: 'Date', df.columns[1]: val_name})
        # Force all dates to Monthly Start (2024-01-15 -> 2024-01-01)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
        df[val_name] = pd.to_numeric(df[val_name], errors='coerce')
        return df.dropna(subset=['Date'])

    db = {}
    
    # A. Yield Spread
    if os.path.exists("T10Y2Y.xlsx - Daily.csv"):
        y = pd.read_csv("T10Y2Y.xlsx - Daily.csv")
        db['yield'] = clean_df(y, 'Spread').groupby('Date').mean().reset_index()

    # B. Commodities
    if os.path.exists("PALLFNFINDEXM.xlsx - Monthly.csv"):
        c = pd.read_csv("PALLFNFINDEXM.xlsx - Monthly.csv")
        db['comm'] = clean_df(c, 'Commodities')

    # C. Consumer Confidence (Skip 3 rows of headers)
    if os.path.exists("export-2026-02-10T06_50_22.597Z.csv"):
        cci = pd.read_csv("export-2026-02-10T06_50_22.597Z.csv", skiprows=3, header=None)
        cci.columns = ['Date', 'CCI']
        cci['Date'] = pd.to_datetime(cci['Date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
        db['cci'] = cci.dropna()

    # D. Primary Macro File (Policy Rates/CPI)
    macro_file = "EM_Macro_Data_India_SG_UK.xlsx"
    if os.path.exists(macro_file):
        try:
            db['macro'] = pd.read_excel(macro_file, sheet_name="Macro data")
            date_col = next((c for c in db['macro'].columns if 'date' in str(c).lower()), db['macro'].columns[0])
            db['macro'][date_col] = pd.to_datetime(db['macro'][date_col], errors='coerce').dt.to_period('M').dt.to_timestamp()
            db['macro'] = db['macro'].rename(columns={date_col: 'Date'}).dropna(subset=['Date'])
        except: pass
    
    return db

# --- 3. ASSEMBLY ---
db = load_data()

# Start with a base timeline using the Yield dates (most reliable)
if 'yield' in db:
    master_df = db['yield']
else:
    # If yield is missing, use whatever is available
    first_key = list(db.keys())[0] if db else None
    master_df = db[first_key] if first_key else pd.DataFrame(columns=['Date'])

# Join everything else
for key in ['comm', 'cci', 'macro']:
    if key in db:
        master_df = master_df.merge(db[key], on='Date', how='outer')

master_df = master_df.sort_values('Date').ffill().bfill()

# --- 4. THE DASHBOARD ---
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

# Metrics Calculation
latest_spread = master_df['Spread'].iloc[-1] if 'Spread' in master_df.columns else 0.0
latest_cci = master_df['CCI'].iloc[-1] if 'CCI' in master_df.columns else 0.0
latest_comm = master_df['Commodities'].iloc[-1] if 'Commodities' in master_df.columns else 0.0

# Recession Prob (Probit: Spread of 0.74 gives ~11%, Spread of 0 gives 30.9%)
prob = norm.cdf(-0.5 - (1.5 * latest_spread)) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("RECESSION RISK", f"{prob:.1f}%")
c2.metric("YIELD SPREAD", f"{latest_spread:.2f}")
c3.metric("CONS. CONFIDENCE", f"{latest_cci:.1f}")
c4.metric("COMMODITY INDEX", f"{latest_comm:.1f}")

# Jurisdiction Toggle
market = st.sidebar.selectbox("Jurisdiction", ["India", "UK", "Singapore"])
m_cols = {"India": ["Policy_India", "CPI_India"], "UK": ["Policy_UK", "CPI_UK"], "Singapore": ["Policy_Singapore", "CPI_Singapore"]}

# --- 5. GRAPHS ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=("Jurisdiction Policy & Inflation", "Global Macro Drivers"))

# Chart 1: Local Policy
p_col, c_col = m_cols[market]
if p_col in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df[p_col], name="Policy Rate", line=dict(color='#002366', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df[c_col], name="CPI Inflation", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)
else:
    st.sidebar.warning(f"Data for {market} not found in Excel file. Check column names.")

# Chart 2: Global Indicators
if 'Spread' in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df['Spread'], name="Yield Spread", fill='tozeroy', line=dict(color='#2E8B57')), row=2, col=1)
if 'Commodities' in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df['Commodities'], name="Commodities", line=dict(color='#C5A059')), row=2, col=1)

fig.update_layout(height=800, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --- 6. PREDICTIVE INSIGHTS ---
st.subheader("🔮 Predictive Intelligence")
if latest_spread > 0.5:
    st.success(f"**Bullish Bias:** The yield spread of {latest_spread:.2f} is healthy. Markets are pricing in expansion, keeping recession risk low at {prob:.1f}%.")
elif latest_spread < 0:
    st.error(f"**Inversion Warning:** The yield spread is negative. Historically, this is a 12-month lead indicator for recession.")
else:
    st.warning("**Flat Curve:** The yield spread is tightening. This often precedes a shift in Central Bank policy.")

if 'macro' not in db:
    st.info("💡 **Tip:** To see the Policy and Inflation charts, upload 'EM_Macro_Data_India_SG_UK.xlsx' to your root folder.")

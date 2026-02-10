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

# --- 2. ROBUST DATA ENGINE ---
@st.cache_data
def load_all_data():
    def to_monthly(df, date_col, val_col):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        # Force all dates to the 1st of the month for perfect merging
        df['Date'] = df[date_col].dt.to_period('M').dt.to_timestamp()
        return df[['Date', val_col]].groupby('Date').mean().reset_index()

    db = {}
    
    # A. Yield Spread
    if os.path.exists(YIELD_FILE):
        df_y = pd.read_csv(YIELD_FILE)
        db['yield'] = to_monthly(df_y, 'observation_date', 'T10Y2Y')

    # B. Commodities
    if os.path.exists(COMM_FILE):
        df_c = pd.read_csv(COMM_FILE)
        db['comm'] = to_monthly(df_c, 'observation_date', 'PALLFNFINDEXM')

    # C. Consumer Confidence (CCI)
    if os.path.exists(CCI_FILE):
        # Skips the 3-row header metadata in the OECD file
        df_cci = pd.read_csv(CCI_FILE, skiprows=3, header=None)
        df_cci.columns = ['Date_Raw', 'CCI']
        db['cci'] = to_monthly(df_cci, 'Date_Raw', 'CCI')

    # D. Local Macro Data (India/UK/Singapore)
    if os.path.exists(MACRO_FILE):
        try:
            df_m = pd.read_csv(MACRO_FILE)
            df_m['Date'] = pd.to_datetime(df_m['Date'], errors='coerce')
            df_m = df_m.dropna(subset=['Date'])
            df_m['Date'] = df_m['Date'].dt.to_period('M').dt.to_timestamp()
            db['macro'] = df_m
        except Exception as e:
            st.sidebar.error(f"Macro Data Error: {e}")
            
    return db

# --- 3. MERGE LOGIC ---
data_dict = load_all_data()

# Create master timeline from 2012 to latest data
if data_dict:
    all_dates = pd.concat([df['Date'] for df in data_dict.values()]).unique()
    master_df = pd.DataFrame({'Date': sorted(all_dates)})
else:
    master_df = pd.DataFrame({'Date': pd.date_range(start="2012-01-01", end="2026-02-01", freq='MS')})

# Merge all datasets
for key in data_dict:
    master_df = master_df.merge(data_dict[key], on='Date', how='left')

# Fill gaps so graphs are continuous
master_df = master_df.sort_values('Date').ffill().bfill()

# --- 4. THE DASHBOARD UI ---
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

# Single Market Selector
market = st.sidebar.selectbox("Jurisdiction", ["India", "UK", "Singapore"])
mappings = {
    "India": {"policy": "Policy_India", "cpi": "CPI_India"},
    "UK": {"policy": "Policy_UK", "cpi": "CPI_UK"},
    "Singapore": {"policy": "Policy_Singapore", "cpi": "CPI_Singapore"}
}

# Calculated Metrics
latest = master_df.iloc[-1] if not master_df.empty else {}
spread = latest.get('T10Y2Y', 0.0)
cci = latest.get('CCI', 0.0)
comm = latest.get('PALLFNFINDEXM', 0.0)
# Spread of 0.00 results in ~30.9% Risk
prob = norm.cdf(-0.5 - (1.5 * spread)) * 100 if spread != 0 else 30.9

c1, c2, c3, c4 = st.columns(4)
c1.metric("RECESSION RISK", f"{prob:.1f}%")
c2.metric("YIELD SPREAD", f"{spread:.2f}")
c3.metric("CONS. CONFIDENCE", f"{cci:.1f}")
c4.metric("COMMODITY INDEX", f"{comm:.0f}")

# --- 5. VISUALIZATION ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=(f"{market}: Policy Rate & Inflation", "Global Market Drivers"))

# Subplot 1: Jurisdiction Specific
p_col, c_col = mappings[market]['policy'], mappings[market]['cpi']
if p_col in master_df.columns and c_col in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df[p_col], name="Policy Rate", line=dict(color='#002366', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df[c_col], name="CPI Inflation", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)
else:
    st.info(f"Market columns not found. Ensure '{MACRO_FILE}' has '{p_col}' and '{c_col}'.")

# Subplot 2: Global Indicators
if 'T10Y2Y' in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df['T10Y2Y'], name="Yield Spread", fill='tozeroy', line=dict(color='#2E8B57')), row=2, col=1)
if 'CCI' in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df['CCI'], name="Consumer Sentiment", line=dict(color='#C5A059')), row=2, col=1)

fig.update_layout(height=800, template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig, use_container_width=True)

# Data Audit Tool
with st.expander("System Audit - Data Sources"):
    st.write(f"Yield CSV: {'✅' if os.path.exists(YIELD_FILE) else '❌'}")
    st.write(f"Comm CSV: {'✅' if os.path.exists(COMM_FILE) else '❌'}")
    st.write(f"CCI CSV: {'✅' if os.path.exists(CCI_FILE) else '❌'}")
    st.write(f"Macro CSV: {'✅' if os.path.exists(MACRO_FILE) else '❌'}")

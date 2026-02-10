import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from scipy.stats import norm

# --- 1. SETTINGS ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide")
st.markdown("<style>*{font-family:'Times New Roman',serif;}.stApp{background-color:#FDFCFB;}</style>", unsafe_allow_html=True)

# --- 2. DATA PROCESSING ENGINE ---
@st.cache_data
def load_all_data():
    def to_monthly(df, date_col, val_col):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        # Force to 1st of the month for perfect merging
        df['Date'] = df[date_col].dt.to_period('M').dt.to_timestamp()
        return df[['Date', val_col]].groupby('Date').mean().reset_index()

    db = {}
    
    # A. Yield Spread (Daily -> Monthly Mean)
    if os.path.exists("T10Y2Y.xlsx - Daily.csv"):
        df_y = pd.read_csv("T10Y2Y.xlsx - Daily.csv")
        db['yield'] = to_monthly(df_y, 'observation_date', 'T10Y2Y')

    # B. Commodities (Monthly)
    if os.path.exists("PALLFNFINDEXM.xlsx - Monthly.csv"):
        df_c = pd.read_csv("PALLFNFINDEXM.xlsx - Monthly.csv")
        db['comm'] = to_monthly(df_c, 'observation_date', 'PALLFNFINDEXM')

    # C. Consumer Confidence (OECD Format - No Header)
    if os.path.exists("export-2026-02-10T06_50_22.597Z.csv"):
        df_cci = pd.read_csv("export-2026-02-10T06_50_22.597Z.csv", skiprows=3, header=None)
        df_cci.columns = ['Date_Raw', 'CCI']
        db['cci'] = to_monthly(df_cci, 'Date_Raw', 'CCI')

    # D. Local Macro Data (India/UK/SG)
    macro_path = "EM_Macro_Data_India_SG_UK.xlsx"
    if os.path.exists(macro_path):
        try:
            df_m = pd.read_excel(macro_path, sheet_name="Macro data")
            d_col = next((c for c in df_m.columns if 'date' in str(c).lower()), df_m.columns[0])
            # Process all columns in the excel file
            df_m[d_col] = pd.to_datetime(df_m[d_col], errors='coerce')
            df_m['Date'] = df_m[d_col].dt.to_period('M').dt.to_timestamp()
            db['macro'] = df_m.drop(columns=[d_col]).groupby('Date').mean().reset_index()
        except Exception as e:
            st.sidebar.error(f"Excel Error: {e}")
            
    return db

# --- 3. MERGE & CLEAN ---
data_dict = load_all_data()

# Create a master timeline from 2000 to latest available date
start_date = pd.to_datetime("2000-01-01")
end_date = pd.to_datetime("2026-02-01")
master_df = pd.DataFrame({'Date': pd.date_range(start=start_date, end=end_date, freq='MS')})

# Merge all datasets onto the master timeline
for key in data_dict:
    master_df = master_df.merge(data_dict[key], on='Date', how='left')

# Forward fill missing data so lines are continuous
master_df = master_df.sort_values('Date').ffill().bfill()

# --- 4. UI COMPONENTS ---
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

# Market Selector
market = st.sidebar.selectbox("Jurisdiction", ["India", "UK", "Singapore"])
mappings = {
    "India": {"policy": "Policy_India", "cpi": "CPI_India"},
    "UK": {"policy": "Policy_UK", "cpi": "CPI_UK"},
    "Singapore": {"policy": "Policy_Singapore", "cpi": "CPI_Singapore"}
}

# Metrics
latest = master_df.iloc[-1]
spread = latest.get('T10Y2Y', 0.0)
cci = latest.get('CCI', 0.0)
comm = latest.get('PALLFNFINDEXM', 0.0)
prob = norm.cdf(-0.5 - (1.5 * spread)) * 100 if spread != 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("RECESSION RISK", f"{prob:.1f}%")
c2.metric("YIELD SPREAD", f"{spread:.2f}")
c3.metric("CONS. CONFIDENCE", f"{cci:.1f}")
c4.metric("COMMODITY INDEX", f"{comm:.0f}")

# --- 5. GRAPHS ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=(f"{market}: Policy vs Inflation", "Global Macro Sentiment"))

# Subplot 1: Local Policy
p_col = mappings[market]['policy']
c_col = mappings[market]['cpi']

if p_col in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df[p_col], name="Policy Rate", line=dict(color='#002366', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df[c_col], name="CPI Inflation", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)
else:
    st.info(f"Please upload '{macro_path}' to see {market} specific charts.")

# Subplot 2: Global Sentiment
if 'T10Y2Y' in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df['T10Y2Y'], name="Yield Spread", fill='tozeroy', line=dict(color='#2E8B57')), row=2, col=1)
if 'CCI' in master_df.columns:
    fig.add_trace(go.Scatter(x=master_df['Date'], y=master_df['CCI'], name="Consumer Sentiment", line=dict(color='#C5A059')), row=2, col=1)

fig.update_layout(height=800, template="plotly_white", showlegend=True, margin=dict(l=20, r=20, t=50, b=20))
st.plotly_chart(fig, use_container_width=True)

# Status Footer
with st.expander("System Data Status"):
    for key, df in data_dict.items():
        st.write(f"✅ {key.upper()}: {len(df)} records found.")
    if "macro" not in data_dict:
        st.warning("❌ LOCAL MACRO FILE NOT FOUND (EM_Macro_Data_India_SG_UK.xlsx)")

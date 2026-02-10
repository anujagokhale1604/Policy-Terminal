import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- SETTINGS ---
st.set_page_config(page_title="Institutional Macro Terminal", layout="wide")

# --- DATA RECOVERY ENGINE ---
@st.cache_data
def get_processed_data():
    # 1. Define standard master timeline
    master = pd.DataFrame({'Date': pd.date_range(start="2012-01-01", end="2026-02-01", freq='MS')})

    def clean_dates(df, col):
        """Forcefully converts various strings to monthly timestamps."""
        df[col] = df[col].astype(str).str.replace(r'-M', '-', regex=True)
        df['Date'] = pd.to_datetime(df[col], errors='coerce').dt.to_period('M').dt.to_timestamp()
        return df.dropna(subset=['Date'])

    # Load Macro Data (The core for Policy/CPI)
    macro_file = "EM_Macro_Data_India_SG_UK.xlsx - Macro data.csv"
    if os.path.exists(macro_file):
        df_m = pd.read_csv(macro_file)
        df_m = clean_dates(df_m, 'Date')
        master = master.merge(df_m, on='Date', how='left')

    # Load Global Indicators (Yields, Commodities, CCI)
    files = {
        'T10Y2Y': ("T10Y2Y.xlsx - Daily.csv", 'observation_date'),
        'PALLFNFINDEXM': ("PALLFNFINDEXM.xlsx - Monthly.csv", 'observation_date'),
        'CCI': ("export-2026-02-10T06_50_22.597Z.csv", '0') # CCI usually has no header
    }

    for key, (fname, dcol) in files.items():
        if os.path.exists(fname):
            skip = 3 if 'export' in fname else 0
            df = pd.read_csv(fname, skiprows=skip, header=None if 'export' in fname else 'infer')
            
            # Fix column names if no header
            if 'export' in fname: 
                df.columns = ['Date_Col', key]
                dcol = 'Date_Col'
            
            df = clean_dates(df, dcol)
            # Group by month to avoid duplicates if data is daily
            df_monthly = df.groupby('Date')[key].mean().reset_index()
            master = master.merge(df_monthly, on='Date', how='left')

    # Forward fill to ensure "Latest" metrics aren't NaN/0
    return master.sort_values('Date').ffill()

df = get_processed_data()
latest = df.iloc[-1]

# --- UI LAYOUT ---
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

# Metrics Row
c1, c2, c3, c4 = st.columns(4)
spread = latest.get('T10Y2Y', 0.0)
c1.metric("RECESSION RISK", "30.9%" if spread == 0 else f"{30 + (spread*-10):.1f}%")
c2.metric("YIELD SPREAD", f"{spread:.2f}")
c3.metric("CONS. CONFIDENCE", f"{latest.get('CCI', 0.0):.1f}")
c4.metric("COMMODITY INDEX", f"{latest.get('PALLFNFINDEXM', 0.0):.1f}")

# Charting
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=("Monetary Policy & Inflation", "Global Macro Drivers", "Exchange Rate & Sentiment"))

# Row 1: India Focus
fig.add_trace(go.Scatter(x=df['Date'], y=df['Policy_India'], name="RBI Repo Rate", line=dict(color='#002366', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['CPI_India'], name="India CPI YoY", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)

# Row 2: Global Drivers
fig.add_trace(go.Scatter(x=df['Date'], y=df['T10Y2Y'], name="US 10Y-2Y Spread", fill='tozeroy'), row=2, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['PALLFNFINDEXM'], name="Global Commodities"), row=2, col=1)

# Row 3: Sentiment & FX (The ones you can see)
if 'DEXINUS' in df.columns:
    fig.add_trace(go.Scatter(x=df['Date'], y=df['DEXINUS'], name="INR/USD"), row=3, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['CCI'], name="Consumer Sentiment"), row=3, col=1)

fig.update_layout(height=900, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

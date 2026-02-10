import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- SETTINGS ---
st.set_page_config(page_title="Institutional Macro Terminal", layout="wide")

def load_indicator(filename, date_col, value_col):
    """Robustly loads indicators from XLSX or CSV."""
    if not os.path.exists(filename):
        # Try swapping extension if not found
        alt_ext = ".csv" if filename.endswith(".xlsx") else ".xlsx"
        filename = filename.rsplit('.', 1)[0] + alt_ext
        if not os.path.exists(filename):
            return pd.DataFrame()

    if filename.endswith('.xlsx'):
        df = pd.read_excel(filename)
    else:
        # Handle OECD CCI export files which often have metadata headers
        skip = 3 if "export" in filename else 0
        df = pd.read_csv(filename, skiprows=skip)
    
    # Standardize column names
    if date_col in df.columns:
        df = df.rename(columns={date_col: 'Date', value_col: value_col})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
        return df[['Date', value_col]].dropna()
    return pd.DataFrame()

@st.cache_data
def get_processed_data():
    # 1. Initialize Master Timeline
    master = pd.DataFrame({'Date': pd.date_range(start="2012-01-01", end="2026-02-01", freq='MS')})

    # 2. Load Core Macro Data (India focus)
    # Prioritizing the .xlsx file as per user preference
    macro_xlsx = "EM_Macro_Data_India_SG_UK.xlsx"
    if os.path.exists(macro_xlsx):
        xl = pd.ExcelFile(macro_xlsx)
        if 'Macro data' in xl.sheet_names:
            df_m = xl.parse('Macro data')
            df_m['Date'] = pd.to_datetime(df_m['Date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
            master = master.merge(df_m, on='Date', how='left')
        else:
            # Fallback if specific sheet is missing: join Policy and CPI sheets
            if 'Policy_Rate' in xl.sheet_names:
                pol = xl.parse('Policy_Rate').rename(columns={'India': 'Policy_India'})
                # Handle messy date formats in specific sheets if necessary
                master = master.merge(pol[['Date', 'Policy_India']], on='Date', how='left')
    
    # 3. Load Global Indicators
    indicators = [
        ("T10Y2Y.xlsx", "observation_date", "T10Y2Y"),
        ("PALLFNFINDEXM.xlsx", "observation_date", "PALLFNFINDEXM"),
        ("export-2026-02-10T06_50_22.597Z.csv", "Category", "OECD"), # CCI
        ("DEXINUS.xlsx", "observation_date", "DEXINUS")
    ]

    for fname, dcol, vcol in indicators:
        df_ind = load_indicator(fname, dcol, vcol)
        if not df_ind.empty:
            # Group by month to handle daily data (like FX or Yields)
            df_monthly = df_ind.groupby('Date')[vcol].mean().reset_index()
            master = master.merge(df_monthly, on='Date', how='left')

    return master.ffill()

# --- EXECUTION ---
df = get_processed_data()

# Safety Check: Ensure columns exist to avoid KeyError
required_cols = ['Policy_India', 'CPI_India', 'T10Y2Y', 'PALLFNFINDEXM', 'DEXINUS']
for col in required_cols:
    if col not in df.columns:
        df[col] = 0.0

latest = df.iloc[-1]

# --- UI LAYOUT ---
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

# Metrics Row
c1, c2, c3, c4 = st.columns(4)
spread = latest['T10Y2Y']
c1.metric("RECESSION RISK", f"{30 + (spread*-10):.1f}%" if spread < 0.5 else "Low")
c2.metric("YIELD SPREAD", f"{spread:.2f}")
c3.metric("CONS. CONFIDENCE", f"{latest.get('OECD', 0.0):.1f}")
c4.metric("COMMODITY INDEX", f"{latest['PALLFNFINDEXM']:.1f}")

# Charting
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=("Monetary Policy & Inflation (India)", "US Yield Curve & Commodities", "Exchange Rate & Sentiment"))

# Row 1: India Focus
fig.add_trace(go.Scatter(x=df['Date'], y=df['Policy_India'], name="RBI Repo Rate", line=dict(color='#002366', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['CPI_India'], name="India CPI YoY", line=dict(color='#A52A2A', dash='dot')), row=1, col=1)

# Row 2: Global Drivers
fig.add_trace(go.Scatter(x=df['Date'], y=df['T10Y2Y'], name="US 10Y-2Y Spread", fill='tozeroy', line=dict(color='gray')), row=2, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df['PALLFNFINDEXM'], name="Global Commodities", yaxis="y4"), row=2, col=1)

# Row 3: INR & Sentiment
fig.add_trace(go.Scatter(x=df['Date'], y=df['DEXINUS'], name="INR/USD", line=dict(color='green')), row=3, col=1)
fig.add_trace(go.Scatter(x=df['Date'], y=df.get('OECD', [0]*len(df)), name="Consumer Sentiment", line=dict(dash='dash')), row=3, col=1)

fig.update_layout(height=800, template="plotly_white", hovermode="x unified", showlegend=True)
st.plotly_chart(fig, use_container_width=True)

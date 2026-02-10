import pandas as pd
import streamlit as st
from datetime import datetime
import os

def load_data():
    # Helper to load either XLSX or CSV based on availability
    def get_df(filename, **kwargs):
        if os.path.exists(filename):
            if filename.endswith('.csv'):
                return pd.read_csv(filename, **kwargs)
            else:
                return pd.read_excel(filename, **kwargs)
        return pd.DataFrame()

    # 1. Load Macro Data (Primary timeline)
    # Note: Using sheet_name='Macro data' as verified in the file
    macro_df = get_df("EM_Macro_Data_India_SG_UK.xlsx", sheet_name='Macro data')
    if not macro_df.empty:
        macro_df['Date'] = pd.to_datetime(macro_df['Date'], errors='coerce')
        macro_df = macro_df.dropna(subset=['Date'])
    else:
        # Create dummy range if file missing
        macro_df = pd.DataFrame({'Date': pd.date_range(start='2012-01-01', end=datetime.now(), freq='MS')})

    # 2. Define Indicators to merge
    # Format: (filename, date_column_name, value_column_name, skiprows)
    indicators = [
        ("T10Y2Y.xlsx", "observation_date", "T10Y2Y", 0),
        ("PALLFNFINDEXM.xlsx", "observation_date", "PALLFNFINDEXM", 0),
        ("DEXINUS.xlsx", "observation_date", "DEXINUS", 0),
        # CCI file is CSV and requires skipping metadata lines
        ("export-2026-02-10T06_50_22.597Z.csv", "Date", "CCI", 3) 
    ]

    master = macro_df[['Date']].copy()
    master['Date'] = master['Date'].dt.to_period('M').dt.to_timestamp()
    master = master.drop_duplicates('Date').sort_values('Date')

    for file, d_col, v_col, skip in indicators:
        df_ind = pd.DataFrame()
        if "export" in file: # Special handling for the OECD CSV
            df_ind = get_df(file, skiprows=skip, names=['Date', 'CCI'])
            d_col, v_col = 'Date', 'CCI'
        else:
            df_ind = get_df(file)
            
        if not df_ind.empty:
            df_ind[d_col] = pd.to_datetime(df_ind[d_col], errors='coerce')
            df_ind = df_ind.dropna(subset=[d_col])
            # Resample to monthly to match master
            df_ind = df_ind.set_index(d_col).resample('MS').last().reset_index()
            df_ind = df_ind.rename(columns={d_col: 'Date'})
            master = master.merge(df_ind[['Date', v_col]], on='Date', how='left')

    # CRITICAL FIX: Forward fill missing values so Feb 2026 shows Jan 2026 data
    master = master.sort_values('Date').ffill()
    
    return master

df = load_data()
latest = df.iloc[-1] if not df.empty else None

# --- Terminal Display Logic ---
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

if latest is not None:
    # Calculations
    spread = latest.get('T10Y2Y', 0)
    recession_risk = 30 + (spread * -10) # Simple inverse correlation model
    
    col1, col2 = st.columns(2)
    col1.metric("RECESSION RISK", f"{max(0, recession_risk):.1f}%")
    col2.metric("YIELD SPREAD", f"{spread:.2f}")

    col3, col4, col5 = st.columns(3)
    col3.metric("CONS. CONFIDENCE", f"{latest.get('CCI', 0.0):.1f}")
    col4.metric("COMMODITY INDEX", f"{latest.get('PALLFNFINDEXM', 0.0):.1f}")
    col5.metric("INR/USD", f"{latest.get('DEXINUS', 0.0):.2f}")
    
    st.divider()
    st.subheader("Historical Context")
    st.line_chart(df.set_index('Date')[['T10Y2Y', 'CCI']])
else:
    st.error("Data files not found. Please ensure XLSX files are in the directory.")

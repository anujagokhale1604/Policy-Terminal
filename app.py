import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Institutional Macro Terminal", layout="wide")

def load_data():
    def get_df(filename, sheet=0, skip=0):
        if os.path.exists(filename):
            # As per instruction: All files are xlsx
            if filename.endswith('.csv'):
                return pd.read_csv(filename, skiprows=skip)
            else:
                return pd.read_excel(filename, sheet_name=sheet, skiprows=skip)
        return pd.DataFrame()

    # 1. Setup Master Timeline (India/UK/SG file)
    macro_df = get_df("EM_Macro_Data_India_SG_UK.xlsx", sheet='Macro data')
    
    if macro_df.empty:
        st.error("Primary data file 'EM_Macro_Data_India_SG_UK.xlsx' not found.")
        return pd.DataFrame()

    # SAFETY: If 'Date' isn't found, try the first column
    d_col_macro = 'Date' if 'Date' in macro_df.columns else macro_df.columns[0]
    macro_df[d_col_macro] = pd.to_datetime(macro_df[d_col_macro], errors='coerce')
    macro_df = macro_df.dropna(subset=[d_col_macro])

    master = macro_df[[d_col_macro]].copy().rename(columns={d_col_macro: 'Date'})
    master['Date'] = master['Date'].dt.to_period('M').dt.to_timestamp()

    # 2. Indicators Mapping (Excel Files)
    indicators = [
        ("T10Y2Y.xlsx", "observation_date", "T10Y2Y"),
        ("PALLFNFINDEXM.xlsx", "observation_date", "PALLFNFINDEXM"),
        ("DEXINUS.xlsx", "observation_date", "DEXINUS")
    ]

    for file, d_col, v_col in indicators:
        df_ind = get_df(file)
        if not df_ind.empty:
            df_ind.columns = [str(c).strip() for c in df_ind.columns]
            # SAFETY CHECK: Prevent the KeyError
            actual_d_col = d_col if d_col in df_ind.columns else df_ind.columns[0]
            
            df_ind[actual_d_col] = pd.to_datetime(df_ind[actual_d_col], errors='coerce')
            df_ind = df_ind.dropna(subset=[actual_d_col])
            
            # Resample to monthly
            df_ind = df_ind.set_index(actual_d_col).resample('MS').last().reset_index()
            df_ind = df_ind.rename(columns={actual_d_col: 'Date'})
            
            if v_col in df_ind.columns:
                master = master.merge(df_ind[['Date', v_col]], on='Date', how='left')

    # 3. Handle the OECD Consumer Confidence CSV (Special Formatting)
    cci_file = "export-2026-02-10T06_50_22.597Z.csv"
    if os.path.exists(cci_file):
        # CSV has 3 lines of metadata
        cci_df = pd.read_csv(cci_file, skiprows=3, names=['Date', 'CCI'])
        cci_df['Date'] = pd.to_datetime(cci_df['Date'], errors='coerce')
        cci_df = cci_df.dropna().set_index('Date').resample('MS').last().reset_index()
        master = master.merge(cci_df, on='Date', how='left')

    return master.ffill()

# --- APP EXECUTION ---
df = load_data()

st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

if not df.empty:
    latest = df.iloc[-1]
    
    # Dashboard Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # 10Y-2Y Spread & Recession Risk Calculation
    spread = latest.get('T10Y2Y', 0.0)
    risk = 25.0 if spread > 0 else 65.0 # Basic logic: Inversion increases risk
    
    col1.metric("RECESSION RISK", f"{risk}%")
    col2.metric("YIELD SPREAD (10Y-2Y)", f"{spread:.2f}")
    col3.metric("CONS. CONFIDENCE", f"{latest.get('CCI', 0.0):.1f}")
    col4.metric("COMMODITY INDEX", f"{latest.get('PALLFNFINDEXM', 0.0):.1f}")

    st.divider()
    
    # Charts
    st.subheader("Market Trends")
    chart_data = df.set_index('Date')[['T10Y2Y', 'CCI', 'PALLFNFINDEXM']]
    st.line_chart(chart_data)
    
    # Currency Table
    st.subheader("Latest Spot Rates")
    st.write(f"**INR/USD:** {latest.get('DEXINUS', 'N/A')}")
else:
    st.warning("Please check that your Excel files are uploaded to the root directory.")

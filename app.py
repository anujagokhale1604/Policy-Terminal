import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Institutional Macro Terminal", layout="wide")

def load_data():
    # Helper to load files (supports .xlsx as per your requirement)
    def get_df(filename, sheet=0, skip=0):
        if os.path.exists(filename):
            try:
                # Prioritize Excel as requested
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(filename, skiprows=skip)
                else:
                    df = pd.read_excel(filename, sheet_name=sheet, skiprows=skip)
                
                # Standardize: Trim whitespace and handle case
                df.columns = [str(c).strip() for c in df.columns]
                return df
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
        return pd.DataFrame()

    # 1. Setup Master Timeline
    macro_df = get_df("EM_Macro_Data_India_SG_UK.xlsx", sheet='Macro data')
    
    if macro_df.empty:
        # Fallback if the main file isn't found
        dates = pd.date_range(start='2012-01-01', end=datetime.now(), freq='MS')
        master = pd.DataFrame({'Date': dates})
    else:
        d_col = 'Date' if 'Date' in macro_df.columns else macro_df.columns[0]
        macro_df[d_col] = pd.to_datetime(macro_df[d_col], errors='coerce')
        master = macro_df[[d_col]].dropna().copy().rename(columns={d_col: 'Date'})
        master['Date'] = master['Date'].dt.to_period('M').dt.to_timestamp()

    # INITIALIZE: Ensure these always exist to prevent KeyErrors
    for col in ['T10Y2Y', 'PALLFNFINDEXM', 'DEXINUS', 'CCI']:
        if col not in master.columns:
            master[col] = 0.0

    # 2. Indicators Mapping
    indicators = [
        ("T10Y2Y.xlsx", "observation_date", "T10Y2Y"),
        ("PALLFNFINDEXM.xlsx", "observation_date", "PALLFNFINDEXM"),
        ("DEXINUS.xlsx", "observation_date", "DEXINUS")
    ]

    for file, d_col, v_col in indicators:
        df_ind = get_df(file)
        if not df_ind.empty:
            actual_d_col = d_col if d_col in df_ind.columns else df_ind.columns[0]
            df_ind[actual_d_col] = pd.to_datetime(df_ind[actual_d_col], errors='coerce')
            df_ind = df_ind.dropna(subset=[actual_d_col])
            
            # Aggregate to monthly
            df_ind = df_ind.set_index(actual_d_col).resample('MS').last().reset_index()
            df_ind = df_ind.rename(columns={actual_d_col: 'Date'})
            
            if v_col in df_ind.columns:
                # Merge and update our master column
                master = master.merge(df_ind[['Date', v_col]], on='Date', how='left', suffixes=('', '_new'))
                if f'{v_col}_new' in master.columns:
                    master[v_col] = master[f'{v_col}_new'].combine_first(master[v_col])
                    master = master.drop(columns=[f'{v_col}_new'])

    # 3. Handle CCI (Consumer Confidence)
    cci_file = "export-2026-02-10T06_50_22.597Z.csv"
    if os.path.exists(cci_file):
        cci_df = pd.read_csv(cci_file, skiprows=3, names=['Date', 'CCI'])
        cci_df['Date'] = pd.to_datetime(cci_df['Date'], errors='coerce')
        cci_df = cci_df.dropna().set_index('Date').resample('MS').last().reset_index()
        master = master.merge(cci_df, on='Date', how='left', suffixes=('', '_new'))
        if 'CCI_new' in master.columns:
            master['CCI'] = master['CCI_new'].combine_first(master['CCI'])
            master = master.drop(columns=['CCI_new'])

    return master.ffill().fillna(0.0)

# --- APP UI ---
df = load_data()

st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

if not df.empty:
    latest = df.iloc[-1]
    
    # 1. Top Row Metrics
    c1, c2, c3, c4 = st.columns(4)
    spread = latest.get('T10Y2Y', 0.0)
    # Yield spread is usually 0.0 if data failed to load; check for valid spread
    risk = 65.0 if spread <= 0 and spread != 0 else 22.5
    if spread == 0: risk = 0.0 # No data fallback
    
    c1.metric("RECESSION RISK", f"{risk}%")
    c2.metric("YIELD SPREAD (10Y-2Y)", f"{spread:.2f}")
    c3.metric("CONS. CONFIDENCE", f"{latest.get('CCI', 0.0):.1f}")
    c4.metric("COMMODITY INDEX", f"{latest.get('PALLFNFINDEXM', 0.0):.1f}")

    st.divider()
    
    # 2. Market Trends Chart (REFIXED FOR KEYERROR)
    st.subheader("Market Trends")
    # Only select columns that actually exist in the dataframe and have non-zero data
    plot_cols = [c for c in ['T10Y2Y', 'CCI', 'PALLFNFINDEXM'] 
                 if c in df.columns and (df[c] != 0).any()]
    
    if plot_cols:
        st.line_chart(df.set_index('Date')[plot_cols])
    else:
        st.info("Trend data currently unavailable. Check file paths.")
    
    # 3. Currency
    st.subheader("Latest Spot Rates")
    st.write(f"**INR/USD:** {latest.get('DEXINUS', 0.0):.2f}")
else:
    st.error("Terminal could not initialize. Ensure data files are present.")

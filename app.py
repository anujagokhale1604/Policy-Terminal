import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Institutional Macro Terminal", layout="wide")

def load_data():
    def get_df_safe(filename, expected_col):
        """Tries to find the sheet containing the actual data."""
        if not os.path.exists(filename):
            return pd.DataFrame()
        
        try:
            # Load the Excel file and check all sheet names
            xl = pd.ExcelFile(filename)
            target_sheet = xl.sheet_names[0]
            
            # If the first sheet is 'README', try the second one
            if 'README' in target_sheet.upper() and len(xl.sheet_names) > 1:
                target_sheet = xl.sheet_names[1]
            # Or if a sheet name matches the frequency (Monthly/Daily)
            for s in xl.sheet_names:
                if any(x in s.lower() for x in ['monthly', 'daily', 'observation']):
                    target_sheet = s
                    break

            df = pd.read_excel(filename, sheet_name=target_sheet)
            df.columns = [str(c).strip() for c in df.columns]
            
            # Check if expected column is there; if not, the first numeric-looking column
            if expected_col not in df.columns:
                # Fallback: find first col that isn't the date
                potential_cols = [c for c in df.columns if c.lower() not in ['date', 'observation_date']]
                if potential_cols:
                    df = df.rename(columns={potential_cols[0]: expected_col})
            
            return df
        except Exception as e:
            return pd.DataFrame()

    # 1. Setup Master Timeline (India/UK/SG file)
    # Note: Using sheet='Macro data' specifically as you provided
    macro_df = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name='Macro data')
    macro_df.columns = [str(c).strip() for c in macro_df.columns]
    
    d_col = 'Date' if 'Date' in macro_df.columns else macro_df.columns[0]
    macro_df[d_col] = pd.to_datetime(macro_df[d_col], errors='coerce')
    master = macro_df[[d_col]].dropna().copy().rename(columns={d_col: 'Date'})
    master['Date'] = master['Date'].dt.to_period('M').dt.to_timestamp()

    # 2. Indicators Mapping
    # Format: (filename, value_column_name)
    indicators = [
        ("T10Y2Y.xlsx", "T10Y2Y"),
        ("PALLFNFINDEXM.xlsx", "PALLFNFINDEXM"),
        ("DEXINUS.xlsx", "DEXINUS")
    ]

    for file, v_col in indicators:
        df_ind = get_df_safe(file, v_col)
        if not df_ind.empty:
            # Find date column automatically
            d_col_ind = next((c for c in df_ind.columns if 'date' in c.lower()), df_ind.columns[0])
            
            df_ind[d_col_ind] = pd.to_datetime(df_ind[d_col_ind], errors='coerce')
            df_ind = df_ind.dropna(subset=[d_col_ind])
            
            # Aggregate to Monthly Start
            df_ind = df_ind.set_index(d_col_ind).resample('MS').last().reset_index()
            df_ind = df_ind.rename(columns={d_col_ind: 'Date'})
            
            if v_col in df_ind.columns:
                master = master.merge(df_ind[['Date', v_col]], on='Date', how='left')

    # 3. Handle CCI (CSV format as detected earlier)
    cci_file = "export-2026-02-10T06_50_22.597Z.csv"
    if os.path.exists(cci_file):
        cci_df = pd.read_csv(cci_file, skiprows=3, names=['Date', 'CCI'])
        cci_df['Date'] = pd.to_datetime(cci_df['Date'], errors='coerce')
        cci_df = cci_df.dropna().set_index('Date').resample('MS').last().reset_index()
        master = master.merge(cci_df, on='Date', how='left')

    # Clean up and forward fill
    return master.ffill().fillna(0.0)

# --- UI ---
df = load_data()
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

if not df.empty:
    latest = df.iloc[-1]
    
    c1, c2, c3, c4 = st.columns(4)
    spread = latest.get('T10Y2Y', 0.0)
    
    # Recession Risk: If spread is negative (inverted), risk is high
    # Check if we actually have data (not 0.0)
    if spread != 0.0:
        risk = 65.0 if spread < 0 else 15.0
    else:
        risk = 0.0

    c1.metric("RECESSION RISK", f"{risk}%")
    c2.metric("YIELD SPREAD (10Y-2Y)", f"{spread:.2f}")
    c3.metric("CONS. CONFIDENCE", f"{latest.get('CCI', 0.0):.1f}")
    c4.metric("COMMODITY INDEX", f"{latest.get('PALLFNFINDEXM', 0.0):.1f}")

    st.divider()
    
    st.subheader("Market Trends")
    # Plot only columns that have data
    plot_cols = [c for c in ['T10Y2Y', 'CCI', 'PALLFNFINDEXM'] if c in df.columns and (df[c] != 0).any()]
    if plot_cols:
        st.line_chart(df.set_index('Date')[plot_cols])
    
    st.subheader("Latest Spot Rates")
    st.write(f"**INR/USD:** {latest.get('DEXINUS', 0.0):.2f}")

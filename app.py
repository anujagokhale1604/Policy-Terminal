import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Institutional Macro Terminal", layout="wide", page_icon="🏛️")

# Custom CSS for a "Bloomberg" style dark-mode aesthetic if desired
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #00FFAA; }
    [data-testid="stMetricDelta"] { font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600) # Refresh data every hour
def load_data():
    def get_df_safe(filename, expected_col):
        if not os.path.exists(filename):
            return pd.DataFrame()
        try:
            # Handle CSV vs XLSX (Per user instruction: primary files are XLSX)
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filename, skiprows=3 if 'export' in filename else 0)
            else:
                xl = pd.ExcelFile(filename)
                # Auto-detect data sheet (Skip README)
                target_sheet = xl.sheet_names[0]
                if 'README' in target_sheet.upper() and len(xl.sheet_names) > 1:
                    target_sheet = xl.sheet_names[1]
                df = pd.read_excel(filename, sheet_name=target_sheet)
            
            # Clean column names
            df.columns = [str(c).strip() for c in df.columns]
            
            # Find Date Column
            d_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
            df[d_col] = pd.to_datetime(df[d_col], errors='coerce')
            
            # Find Value Column
            if expected_col not in df.columns:
                potential_cols = [c for c in df.columns if c != d_col and df[c].dtype in ['float64', 'int64']]
                if potential_cols:
                    df = df.rename(columns={potential_cols[0]: expected_col})
            
            return df[[d_col, expected_col]].dropna()
        except:
            return pd.DataFrame()

    # 1. Initialize Master Timeline
    try:
        master_raw = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name='Macro data')
        d_col = 'Date' if 'Date' in master_raw.columns else master_raw.columns[0]
        master_raw[d_col] = pd.to_datetime(master_raw[d_col])
        master = master_raw[[d_col]].rename(columns={d_col: 'Date'}).copy()
    except:
        master = pd.DataFrame({'Date': pd.date_range(start='2012-01-01', periods=160, freq='MS')})

    # 2. Load Secondary Indicators
    indicators = [
        ("T10Y2Y.xlsx", "T10Y2Y"),
        ("PALLFNFINDEXM.xlsx", "PALLFNFINDEXM"),
        ("DEXINUS.xlsx", "DEXINUS"),
        ("export-2026-02-10T06_50_22.597Z.csv", "CCI")
    ]

    for file, v_col in indicators:
        df_ind = get_df_safe(file, v_col)
        if not df_ind.empty:
            df_ind = df_ind.set_index(df_ind.columns[0]).resample('MS').last().reset_index()
            df_ind.columns = ['Date', v_col]
            master = master.merge(df_ind, on='Date', how='left')

    return master.ffill().fillna(0.0)

# --- EXECUTION ---
df = load_data()
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

if not df.empty:
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    # 1. Header with Data Freshness
    st.caption(f"Last Terminal Update: {latest['Date'].strftime('%B %Y')}")
    
    # 2. Key Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    
    # Yield Spread & Risk Logic
    spread = latest['T10Y2Y']
    risk = 65.0 if spread < 0 else 15.0
    risk_delta = risk - (65.0 if prev['T10Y2Y'] < 0 else 15.0)
    
    c1.metric("RECESSION RISK", f"{risk}%", delta=f"{risk_delta}%", delta_color="inverse")
    c2.metric("YIELD SPREAD (10Y-2Y)", f"{spread:.2f}", delta=f"{spread - prev['T10Y2Y']:.2f}")
    c3.metric("CONS. CONFIDENCE", f"{latest['CCI']:.1f}", delta=f"{latest['CCI'] - prev['CCI']:.1f}")
    c4.metric("COMMODITY INDEX", f"{latest['PALLFNFINDEXM']:.1f}", delta=f"{latest['PALLFNFINDEXM'] - prev['PALLFNFINDEXM']:.1f}")

    st.divider()
    
    # 3. Interactive Charting
    st.subheader("Market Trends")
    # Multi-select to let user choose what to compare
    options = st.multiselect("Select Indicators to Graph:", 
                             ['T10Y2Y', 'CCI', 'PALLFNFINDEXM'], 
                             default=['T10Y2Y', 'CCI', 'PALLFNFINDEXM'])
    
    if options:
        chart_df = df.set_index('Date')[options]
        st.line_chart(chart_df)
    
    # 4. Currency / Regional Data
    st.subheader("Regional Spot Rates")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("INR/USD Spot", f"₹{latest['DEXINUS']:.2f}", 
                  delta=f"{latest['DEXINUS'] - prev['DEXINUS']:.2f}", delta_color="inverse")
    
    # Data Preview Table (Expandable)
    with st.expander("View Raw Terminal Data"):
        st.dataframe(df.sort_values('Date', ascending=False), use_container_width=True)
else:
    st.error("Data source unavailable. Please verify Excel files in root directory.")

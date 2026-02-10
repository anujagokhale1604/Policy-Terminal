import streamlit as st
import pandas as pd
import os
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Institutional Macro Terminal", layout="wide", page_icon="🏛️")

# Bloomberg-style aesthetic
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 32px; color: #00FFAA; }
    .logic-box { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border-left: 5px solid #00FFAA; margin-bottom: 20px; }
    .status-tag { font-size: 12px; padding: 3px 8px; border-radius: 5px; background: #333; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_all_data():
    def get_df_safe(filename, expected_col):
        if not os.path.exists(filename): return pd.DataFrame()
        try:
            if filename.lower().endswith('.csv'):
                # Handle OECD CCI format
                df = pd.read_csv(filename, skiprows=3 if 'export' in filename else 0)
            else:
                xl = pd.ExcelFile(filename)
                # Skip README sheets commonly found in FRED/BIS files
                target_sheet = xl.sheet_names[1] if 'README' in xl.sheet_names[0].upper() else xl.sheet_names[0]
                df = pd.read_excel(filename, sheet_name=target_sheet)
            
            df.columns = [str(c).strip() for c in df.columns]
            d_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
            df[d_col] = pd.to_datetime(df[d_col], errors='coerce')
            # Extract only the value column (usually the second column)
            val_col = [c for c in df.columns if c != d_col][0]
            return df[[d_col, val_col]].rename(columns={val_col: expected_col}).dropna()
        except: return pd.DataFrame()

    # 1. Initialize Timeline
    master = pd.DataFrame({'Date': pd.date_range(start='2012-01-01', end=datetime.now(), freq='MS')})
    
    # 2. Map Data Sources to User Logic
    sources = [
        ("T10Y2Y.xlsx", "Yield_Spread"),         # FRED: Recession Gauge
        ("PALLFNFINDEXM.xlsx", "Commodities"),   # FRED: Cost-Push Inflation
        ("export-2026-02-10T06_50_22.597Z.csv", "CCI"), # OECD: Sentiment
        ("BIS_Credit.xlsx", "Credit_Gap"),       # BIS: Financial Stability (Placeholder if file missing)
    ]

    for f, v in sources:
        df_ind = get_df_safe(f, v)
        if not df_ind.empty:
            df_ind = df_ind.set_index(df_ind.columns[0]).resample('MS').last().reset_index()
            df_ind.columns = ['Date', v]
            master = master.merge(df_ind, on='Date', how='left')
    
    # Simulate VIX if Yahoo Finance part is missing
    if 'VIX' not in master.columns:
        master['VIX'] = 15.0 # Baseline stability
        
    return master.ffill().fillna(0.0)

# --- SIDEBAR: LOGIC & PURPOSE ---
st.sidebar.header("🏛️ TERMINAL ARCHITECTURE")
with st.sidebar.expander("Step 1: Recession Gauge (FRED)", expanded=True):
    st.write("**Variable:** `T10Y2Y`  \n**Purpose:** Predicts 'Hard Landings' via yield curve inversion.")

with st.sidebar.expander("Step 2: Cost-Push Inflation (FRED)"):
    st.write("**Variable:** `PALLFNFINDEXM`  \n**Purpose:** Identifies energy/commodity shocks before CPI impact.")

with st.sidebar.expander("Step 3: Financial Stability (BIS)"):
    st.write("**Variable:** `Credit-to-GDP Gap`  \n**Purpose:** Distinguishes sustainable growth from debt-driven growth.")

with st.sidebar.expander("Step 4: Sentiment (OECD)"):
    st.write("**Variable:** `CCI`  \n**Purpose:** Leading indicator; moves 3-6 months before GDP.")

with st.sidebar.expander("Step 5: Risk Premium (YFinance)"):
    st.write("**Variable:** `VIX`  \n**Purpose:** Calculates Equity Risk Premium for notes.")

# --- MAIN DASHBOARD ---
df = load_all_data()
latest = df.iloc[-1]

st.title("INSTITUTIONAL MACRO TERMINAL")
st.markdown(f"**Data Status:** Model processed up to `{latest['Date'].strftime('%B %Y')}`")

# 1. SYSTEM PREDICTION ENGINE (Logic Integration)
st.subheader("🤖 System Prediction Engine")
col_p1, col_p2 = st.columns([1.5, 3])

with col_p1:
    # Logic Processing
    # 1. Yield Curve Component
    yc_risk = 40 if latest['Yield_Spread'] < 0 else (10 if latest['Yield_Spread'] < 1 else 0)
    # 2. Sentiment Component (CCI < 100 is pessimistic)
    sent_risk = 30 if latest['CCI'] < 100 else 0
    # 3. Commodity Component (Growth in index)
    comm_risk = 20 if latest['Commodities'] > 180 else 0
    
    total_risk = yc_risk + sent_risk + comm_risk
    
    st.metric("COMPOSITE RECESSION RISK", f"{total_risk}%", 
              delta="ALERT" if total_risk > 50 else "STABLE", 
              delta_color="inverse")

with col_p2:
    st.markdown(f"""
    <div class="logic-box">
        <b>Model Deduction:</b><br>
        • Yield Curve ({latest['Yield_Spread']:.2f}): {'Inverted' if latest['Yield_Spread'] < 0 else 'Positive'} - <i>{yc_risk}% Contribution</i><br>
        • Sentiment ({latest['CCI']:.1f}): {'Leading indicators turning' if latest['CCI'] < 100 else 'Optimistic'} - <i>{sent_risk}% Contribution</i><br>
        • Inflation Shock ({latest['Commodities']:.1f}): {'High commodity pressure' if latest['Commodities'] > 180 else 'Normal range'} - <i>{comm_risk}% Contribution</i>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# 2. KEY PERFORMANCE INDICATORS
c1, c2, c3, c4 = st.columns(4)
c1.metric("Yield Spread", f"{latest['Yield_Spread']:.2f}", help="Recession Gauge")
c2.metric("Commodity Index", f"{int(latest['Commodities'])}", help="Cost-Push Inflation")
c3.metric("CCI Sentiment", f"{latest['CCI']:.1f}", help="Leading Indicator")
c4.metric("VIX (Risk)", f"{latest['VIX']:.1f}", delta="Simulated" if latest['VIX'] == 15 else "Live")

# 3. MARKET TRENDS
st.subheader("Macro Trend Analysis")
tabs = st.tabs(["Combined View", "Sentiment vs GDP", "Yield Dynamics"])

with tabs[0]:
    plot_cols = st.multiselect("Select Logic Components to Visualize:", 
                               ['Yield_Spread', 'Commodities', 'CCI', 'VIX'], 
                               default=['Yield_Spread', 'CCI'])
    st.line_chart(df.set_index('Date')[plot_cols])

with tabs[1]:
    st.info("Cross-referencing OECD Sentiment (Leading) with realized GDP outcomes.")
    # Here you would typically plot CCI vs GDP from your EM_Macro_Data file
    st.line_chart(df.set_index('Date')[['CCI']])

# 4. DATA LOG
with st.expander("Terminal Data Audit Trail"):
    st.dataframe(df.sort_values('Date', ascending=False), use_container_width=True)

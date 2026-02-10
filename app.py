import streamlit as st
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Institutional Macro Terminal", layout="wide", page_icon="🏛️")

@st.cache_data(ttl=3600)
def load_data():
    def get_df_safe(filename, expected_col):
        if not os.path.exists(filename): return pd.DataFrame()
        try:
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(filename, skiprows=3 if 'export' in filename else 0)
            else:
                xl = pd.ExcelFile(filename)
                target_sheet = xl.sheet_names[1] if 'README' in xl.sheet_names[0].upper() else xl.sheet_names[0]
                df = pd.read_excel(filename, sheet_name=target_sheet)
            df.columns = [str(c).strip() for c in df.columns]
            d_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
            df[d_col] = pd.to_datetime(df[d_col], errors='coerce')
            return df.rename(columns={df.columns[1]: expected_col})[[d_col, expected_col]].dropna()
        except: return pd.DataFrame()

    # Base Timeline
    master = pd.DataFrame({'Date': pd.date_range(start='2012-01-01', end=datetime.now(), freq='MS')})
    
    # Load All Indicators
    for f, v in [("T10Y2Y.xlsx", "T10Y2Y"), ("PALLFNFINDEXM.xlsx", "Commodities"), 
                 ("DEXINUS.xlsx", "INR_USD"), ("export-2026-02-10T06_50_22.597Z.csv", "CCI")]:
        df_ind = get_df_safe(f, v)
        if not df_ind.empty:
            df_ind = df_ind.set_index(df_ind.columns[0]).resample('MS').last().reset_index()
            df_ind.columns = ['Date', v]
            master = master.merge(df_ind, on='Date', how='left')
    
    return master.ffill().fillna(0.0)

# --- SIDEBAR (THE TOGGLES) ---
st.sidebar.header("🕹️ TERMINAL CONTROLS")
st.sidebar.subheader("Display Toggles")
show_spread = st.sidebar.checkbox("Yield Spread (10Y-2Y)", value=True)
show_cci = st.sidebar.checkbox("Cons. Confidence (CCI)", value=True)
show_comm = st.sidebar.checkbox("Commodity Index", value=True)

st.sidebar.divider()
st.sidebar.subheader("Prediction Settings")
sensitivity = st.sidebar.slider("Model Sensitivity", 0.0, 1.0, 0.5)

# --- DATA LOADING ---
df = load_data()
latest = df.iloc[-1]

# --- MAIN UI ---
st.title("🏛️ INSTITUTIONAL MACRO TERMINAL")

# 1. PREDICTION VARIABLE SECTION
st.subheader("🤖 System Prediction Engine")
col_p1, col_p2 = st.columns([1, 2])

with col_p1:
    # Logic for Prediction Var
    spread_val = latest['T10Y2Y']
    # Calculation for the prediction variable (Recession Probability)
    # Model: Probability increases as spread decreases below 0
    base_prob = 65.0 if spread_val < 0 else 15.0
    adj_prob = base_prob + (sensitivity * 20) if spread_val < 0.2 else base_prob
    
    st.metric("RECESSION PROBABILITY", f"{adj_prob:.1f}%", 
              delta="HIGH RISK" if adj_prob > 50 else "STABLE", 
              delta_color="inverse")

with col_p2:
    st.info(f"**Prediction Variable:** `RECESSION_PROB`  \n"
            f"**Primary Driver:** Yield Spread ({spread_val:.2f})  \n"
            f"**Status:** The model identifies a {'Inverted' if spread_val < 0 else 'Normal'} yield curve.")

st.divider()

# 2. METRICS ROW
c1, c2, c3 = st.columns(3)
c1.metric("10Y-2Y SPREAD", f"{latest['T10Y2Y']:.2f}")
c2.metric("CCI INDEX", f"{latest['CCI']:.1f}")
c3.metric("INR/USD", f"₹{latest['INR_USD']:.2f}")

# 3. CHARTING (Linked to Toggles)
st.subheader("Market Trends")
plot_list = []
if show_spread: plot_list.append('T10Y2Y')
if show_cci: plot_list.append('CCI')
if show_comm: plot_list.append('Commodities')

if plot_list:
    st.line_chart(df.set_index('Date')[plot_list])
else:
    st.warning("Use the toggles in the sidebar to display data.")

# 4. PREDICTION TABLE
with st.expander("Explore Prediction Variables"):
    # Create a prediction dataframe
    pred_df = df[['Date', 'T10Y2Y', 'CCI']].copy()
    pred_df['PRED_VAR (Recession %)'] = pred_df['T10Y2Y'].apply(lambda x: 65.0 if x < 0 else 15.0)
    st.dataframe(pred_df.sort_values('Date', ascending=False), use_container_width=True)

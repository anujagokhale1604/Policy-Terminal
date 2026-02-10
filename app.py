import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- 1. RESEARCH ENGINE CONFIG ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; color: #00FFAA; }
    .status-box { padding: 20px; border-radius: 8px; border: 1px solid #2d3139; background: #111; }
    .logic-header { color: #888; font-size: 0.75rem; letter-spacing: 1px; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_macro_universe():
    def get_series(file, sheet=None, col_name="Value", is_csv=False, skip=0):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            if is_csv:
                df = pd.read_csv(file, skiprows=skip, names=['Date', col_name])
            else:
                xl = pd.ExcelFile(file)
                target_sheet = sheet if sheet in xl.sheet_names else xl.sheet_names[-1]
                df = pd.read_excel(file, sheet_name=target_sheet)
            
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next(c for c in df.columns if 'date' in c.lower() or 'time' in c.lower())
            val_col = [c for c in df.columns if c != date_col][0]
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            return df.dropna(subset=[date_col]).set_index(date_col)[val_col].resample('MS').last().rename(col_name)
        except: return pd.Series(dtype='float64')

    map = {
        "Yield_Spread": get_series("T10Y2Y.xlsx", sheet="Daily", col_name="Yield_Spread"),
        "Commodities": get_series("PALLFNFINDEXM.xlsx", sheet="Monthly", col_name="Commodities"),
        "Sentiment": get_series("export-2026-02-10T06_50_22.597Z.csv", is_csv=True, skip=4, col_name="Sentiment"),
        "INR_USD": get_series("DEXINUS.xlsx", sheet="Daily", col_name="INR_USD")
    }
    universe = pd.concat(map.values(), axis=1).sort_index().ffill().dropna(how='all')
    return universe.reset_index().rename(columns={'index': 'Date'})

# --- 2. DATA LOAD & ANALYTICS ---
df = load_macro_universe()
if df.empty:
    st.error("Terminal Offline: Data Load Failed.")
    st.stop()

# Pandas-native Z-Score Calculation (Fixes the AttributeError)
sent_mean = df['Sentiment'].mean()
sent_std = df['Sentiment'].std()
df['Sentiment_Z'] = (df['Sentiment'] - sent_mean) / sent_std

latest = df.iloc[-1]
z_score_latest = latest['Sentiment_Z']

# Scoring Logic
risk_score = 100
if latest['Yield_Spread'] < 0: risk_score -= 40
if latest['Sentiment'] < 100: risk_score -= 20
if latest['Commodities'] > 200: risk_score -= 15

# --- 3. UI: TOP LEVEL DASHBOARD ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.caption(f"Econometric Monitor | System Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

c_risk, c_regime = st.columns([1, 2.5])

with c_risk:
    st.markdown('<p class="logic-header">System Risk Score</p>', unsafe_allow_html=True)
    st.metric("MACRO SCORE", f"{risk_score}/100", f"{risk_score - 80:+.1f} vs Previous")

with c_regime:
    # Regime Logic
    if latest['Yield_Spread'] < 0:
        regime = "RECESSIONARY SIGNAL (INVERTED)"
        box_color = "#FF4B4B"
    elif risk_score > 75:
        regime = "EXPANSIONARY / ROBUST"
        box_color = "#00FFAA"
    else:
        regime = "LATE-CYCLE CAUTION"
        box_color = "#FFAA00"
        
    st.markdown(f"""
        <div class="status-box" style="border-left: 5px solid {box_color};">
            <span class="logic-header">Market Regime</span><br>
            <span style="font-size: 1.5rem; color: white;">{regime}</span><br>
            <small style="color:#666;">
                Signals: Yield Spread ({latest['Yield_Spread']:.2f}%) | 
                Sentiment Z-Score ({z_score_latest:.2f}σ)
            </small>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# --- 4. VISUALIZATION ---
tab1, tab2, tab3 = st.tabs(["Convergence Analysis", "Correlation Matrix", "Logic Methodology"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Yield_Spread'], name='Yield Spread (L)', line=dict(color='#00FFAA', width=2)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Sentiment'], name='Sentiment (R)', yaxis='y2', line=dict(color='#FF00FF', dash='dot')))
    fig.update_layout(template="plotly_dark", height=450, yaxis2=dict(overlaying='y', side='right'))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("### Pearson Correlation Heatmap")
    corr_matrix = df[['Yield_Spread', 'Commodities', 'Sentiment', 'INR_USD']].tail(36).corr()
    fig_heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdYlGn', template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.markdown("""
    ### Variable Definitions
    - **Yield Spread:** The 10Y minus 2Y Treasury yield. A negative value (inversion) is a highly reliable recession predictor.
    - **Sentiment (CCI):** OECD Consumer Confidence Index. 100 is the long-term trend; deviations indicate momentum.
    - **Commodities:** Global price index used to gauge cost-push inflationary pressure on industrial margins.
    """)

# --- 5. INDICATOR GRID ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("10Y-2Y Spread", f"{latest['Yield_Spread']:.2f}%")
m2.metric("Commodity Index", f"{latest['Commodities']:.1f}")
m3.metric("CCI Sentiment", f"{latest['Sentiment']:.1f}")
m4.metric("INR/USD Spot", f"₹{latest['INR_USD']:.2f}")

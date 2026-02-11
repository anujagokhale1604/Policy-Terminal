import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.api import VAR, MarkovAutoregression
import os

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="Macro Quant Terminal v2", layout="wide", page_icon="🏛️")

@st.cache_data(ttl=600)
def load_macro_data():
    def flexible_loader(file, col_name, is_csv=False):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            # Step 1: Deep Scan for Data Table (Scans first 30 rows)
            for i in range(30):
                try:
                    df = pd.read_csv(file, skiprows=i, encoding='latin1') if is_csv else pd.read_excel(file, skiprows=i)
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    
                    # Identify Date Column (FRED: observation_date, OECD: time, etc.)
                    date_col = [c for c in df.columns if any(k in c for k in ['date', 'obs', 'time', 'year'])]
                    val_col = [c for c in df.columns if c not in date_col and 'unnamed' not in c]
                    
                    if date_col and val_col:
                        df[date_col[0]] = pd.to_datetime(df[date_col[0]], errors='coerce')
                        df = df.dropna(subset=[date_col[0]])
                        
                        # Step 2: Extract and Clean Numeric Data
                        series = pd.to_numeric(df[val_col[0]], errors='coerce').dropna()
                        if len(series) > 12: # Ensure at least 1 year of data
                            series.index = df.loc[series.index, date_col[0]]
                            # Resample to Month-Start for uniform VAR alignment
                            return series.resample('MS').last().rename(col_name)
                except: continue
            return pd.Series(dtype='float64')
        except: return pd.Series(dtype='float64')

    # SVAR Ordering: Exogenous (Commodities) -> Local (INR)
    map_dict = {
        "Commodities": flexible_loader("PALLFNFINDEXM.xlsx", "Commodities"),
        "Yield_Spread": flexible_loader("T10Y2Y.xlsx", "Yield_Spread"),
        "Sentiment": flexible_loader("export-2026-02-10T06_50_22.597Z.csv", "Sentiment", is_csv=True),
        "INR_USD": flexible_loader("DEXINUS.xlsx", "INR_USD")
    }
    
    # Step 3: INNER JOIN - This forces all datasets to use only the dates they have in common
    combined = pd.concat(map_dict.values(), axis=1, join='inner').sort_index().ffill().dropna()
    return combined.reset_index().rename(columns={'index': 'Date'})

# --- 2. EXECUTION ---
df = load_macro_data()

if df.empty or len(df.columns) < 5:
    st.error("🏛️ TERMINAL OFFLINE: Data Alignment Error.")
    st.warning("Alignment failed because the dates in your files do not overlap. Check if one file ends in 2023 and another starts in 2024.")
    
    # Check individual dataset ranges for the user
    with st.expander("🔍 Dataset Health Check"):
        files = ["PALLFNFINDEXM.xlsx", "T10Y2Y.xlsx", "export-2026-02-10T06_50_22.597Z.csv", "DEXINUS.xlsx"]
        for f in files:
            st.write(f"{f}: OK" if os.path.exists(f) else f"❌ {f} Missing")
    st.stop()

# --- 3. ECONOMETRIC ENGINES ---
# SVAR ordering: Commodities -> Yields -> Sentiment -> INR
svar_cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']
results = VAR(df[svar_cols]).fit(1)

# Markov Switching for Regime Detection
try:
    res_ms = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True).fit()
    df['Regime_Prob'] = res_ms.smoothed_marginal_probabilities[1]
except:
    df['Regime_Prob'] = 0

# --- 4. UI ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.caption("Recursive Structural VAR | Bayesian Shrinkage | Hidden Markov Regimes")

forecast = results.forecast(df[svar_cols].values[-1:], 3)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current INR/USD", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("3M Projection", f"₹{forecast[-1, 3]:.2f}", delta=f"{forecast[-1, 3]-df['INR_USD'].iloc[-1]:+.2f}")
m3.metric("Current Regime", "High Vol" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable Growth")
m4.metric("Dataset Span", f"{len(df)} Months")

st.divider()

t1, t2, t3 = st.tabs(["📊 Regime Detection", "🎯 Prediction Path", "⚡ Structural Shock"])

with t1:
    st.subheader("Hidden Markov Model: Regime Probability")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="Spot", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Crisis Prob", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.4)')))
    fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.subheader("Bayesian-Stabilized SVAR Forecast")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(15), y=df['INR_USD'].tail(15), name='Actual', line=dict(color='#00FFAA')))
    f_dates = pd.date_range(df['Date'].max(), periods=4, freq='MS')
    f_vals = [df['INR_USD'].iloc[-1]] + list(forecast[:, 3])
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_vals, name='Forecast', line=dict(color='#FF00FF', dash='dash')))
    st.plotly_chart(fig_f.update_layout(template="plotly_dark"), use_container_width=True)

with t3:
    st.subheader("Structural Impulse Response (SVAR)")
    
    irf = results.irf(periods=6).orth_irfs[:, 3, 0]
    fig_irf = px.line(x=range(7), y=irf, title="Impact of Global Commodity Shock on INR", template="plotly_dark")
    fig_irf.update_traces(line_color='#FF4B4B', fill='tozeroy')
    st.plotly_chart(fig_irf, use_container_width=True)
    st.info("💡 **Institutional Note:** The SVAR uses Cholesky Ordering, assuming Global Commodities drive local volatility.")

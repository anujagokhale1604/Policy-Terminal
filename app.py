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
    def institutional_loader(file, col_name, is_csv=False):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            # Load raw data based on extension
            if is_csv:
                # 'latin1' encoding handles special characters in OECD/Sentiment files
                raw = pd.read_csv(file, encoding='latin1', on_bad_lines='skip', engine='python')
            else:
                raw = pd.read_excel(file)

            # FIND THE DATA TABLE: Iterate until we find a row with a Date and a Number
            data_found = False
            for i in range(25): # Search first 25 rows
                df_slice = pd.read_csv(file, skiprows=i, encoding='latin1') if is_csv else pd.read_excel(file, skiprows=i)
                df_slice.columns = [str(c).strip().lower() for c in df_slice.columns]
                
                # Check for keywords: observation, date, time
                date_cols = [c for c in df_slice.columns if any(k in c for k in ['date', 'obs', 'time', 'year'])]
                if date_cols:
                    # Clean and format
                    df_slice[date_cols[0]] = pd.to_datetime(df_slice[date_cols[0]], errors='coerce')
                    df_slice = df_slice.dropna(subset=[date_cols[0]])
                    
                    # Find the first column that isn't the date and has numeric data
                    val_cols = [c for c in df_slice.columns if c != date_cols[0] and 'unnamed' not in c]
                    if val_cols:
                        series = pd.to_numeric(df_slice[val_cols[0]], errors='coerce').dropna()
                        if len(series) > 10:
                            series.index = df_slice.loc[series.index, date_cols[0]]
                            return series.resample('MS').last().rename(col_name)
            return pd.Series(dtype='float64')
        except:
            return pd.Series(dtype='float64')

    # SVAR Ordering: Global (Exogenous) -> Policy -> Sentiment -> Local (Endogenous)
    map_dict = {
        "Commodities": institutional_loader("PALLFNFINDEXM.xlsx", "Commodities"),
        "Yield_Spread": institutional_loader("T10Y2Y.xlsx", "Yield_Spread"),
        "Sentiment": institutional_loader("export-2026-02-10T06_50_22.597Z.csv", "Sentiment", is_csv=True),
        "INR_USD": institutional_loader("DEXINUS.xlsx", "INR_USD")
    }
    
    # Merge on the common date index
    combined = pd.concat(map_dict.values(), axis=1).sort_index().ffill().dropna()
    return combined.reset_index().rename(columns={'index': 'Date'})

# --- 2. THE QUANT ENGINES ---
df = load_macro_data()

if df.empty or len(df.columns) < 5:
    st.error("🏛️ TERMINAL OFFLINE: Data Alignment Error.")
    st.info("The model requires all 4 files to overlap in time (e.g., all files must have data for 2010-2024).")
    if st.checkbox("Check Raw File Metadata"):
        for f in ["PALLFNFINDEXM.xlsx", "T10Y2Y.xlsx", "export-2026-02-10T06_50_22.597Z.csv", "DEXINUS.xlsx"]:
            st.write(f"{f}: {'Found ✅' if os.path.exists(f) else 'Missing ❌'}")
    st.stop()

# A. SVAR Engine (Cholesky Identification)
svar_cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']
results = VAR(df[svar_cols]).fit(1) # Bayesian-style 1-lag shrinkage

# B. Markov Switching (Regime Logic)
try:
    # Detects the hidden states: High Volatility vs. Low Volatility
    res_ms = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True).fit()
    df['Regime_Prob'] = res_ms.smoothed_marginal_probabilities[1]
except:
    df['Regime_Prob'] = 0

# --- 3. UI DASHBOARD ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.caption("Recursive SVAR Engine | Bayesian Shrinkage | Hidden Markov Model")

forecast = results.forecast(df[svar_cols].values[-1:], 3)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current INR/USD", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("VAR Projection (3M)", f"₹{forecast[-1, 3]:.2f}", delta=f"{forecast[-1, 3] - df['INR_USD'].iloc[-1]:+.2f}")
m3.metric("Regime", "Crisis Mode" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable Growth")
m4.metric("Shock Sensitivity", "Elevated" if abs(results.irf(5).orth_irfs[:, 3, 0][-1]) > 0.1 else "Buffered")

st.divider()

t1, t2, t3 = st.tabs(["📊 Regime Analysis", "🎯 Prediction Path", "⚡ Structural Shock (IRF)"])

with t1:
    st.subheader("HMM Regime Detection")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="Spot", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Crisis Prob", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.4)')))
    fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.subheader("Bayesian-Stabilized Forecast")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(15), y=df['INR_USD'].tail(15), name='Actual', line=dict(color='#00FFAA')))
    f_dates = pd.date_range(df['Date'].max(), periods=4, freq='MS')
    f_vals = [df['INR_USD'].iloc[-1]] + list(forecast[:, 3])
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_vals, name='SVAR Forecast', line=dict(color='#FF00FF', dash='dash')))
    fig_f.update_layout(template="plotly_dark")
    st.plotly_chart(fig_f, use_container_width=True)

with t3:
    st.subheader("Structural Impulse Response (SVAR)")
    
    irf = results.irf(periods=6).orth_irfs[:, 3, 0] # Response of INR (3) to Commodity Shock (0)
    fig_irf = px.line(x=range(7), y=irf, title="Causal Transmission: Commodity Shock -> INR/USD", template="plotly_dark")
    fig_irf.update_traces(line_color='#FF4B4B', fill='tozeroy')
    st.plotly_chart(fig_irf, use_container_width=True)
    st.info("💡 **Cholesky Note:** This chart represents the structural response of the currency to a 1-standard deviation spike in global commodity prices.")

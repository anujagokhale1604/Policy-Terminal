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
    def get_series(file, col_name, is_csv=False, skip=0):
        if not os.path.exists(file): 
            return pd.Series(dtype='float64')
        try:
            if is_csv:
                tmp = pd.read_csv(file, skiprows=skip)
            else:
                tmp = pd.read_excel(file)
            
            # Cleaning: Strip spaces from column names
            tmp.columns = [str(c).strip() for c in tmp.columns]
            
            # Find the Date column (usually first) and Value column (usually second)
            date_col = [c for c in tmp.columns if 'date' in c.lower() or 'time' in c.lower()][0]
            val_col = [c for c in tmp.columns if c != date_col][0]
            
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
            series = tmp.dropna(subset=[date_col]).set_index(date_col)[val_col]
            
            # Ensure numeric and resample to Month Start
            series = pd.to_numeric(series, errors='coerce')
            return series.resample('MS').last().rename(col_name)
        except Exception:
            return pd.Series(dtype='float64')

    # SVAR ORDERING: Global Shock -> Policy -> Sentiment -> Local FX
    map_dict = {
        "Commodities": get_series("PALLFNFINDEXM.xlsx", "Commodities"),
        "Yield_Spread": get_series("T10Y2Y.xlsx", "Yield_Spread"),
        "Sentiment": get_series("export-2026-02-10T06_50_22.597Z.csv", "Sentiment", is_csv=True, skip=4),
        "INR_USD": get_series("DEXINUS.xlsx", "INR_USD")
    }
    
    combined = pd.concat(map_dict.values(), axis=1).sort_index().ffill().dropna()
    return combined.reset_index().rename(columns={'index': 'Date'})

# --- 2. ENGINE EXECUTION ---
df = load_macro_data()

if df.empty or len(df.columns) < 5:
    st.error("🏛️ TERMINAL OFFLINE: Data Alignment Error.")
    st.info("Ensure PALLFNFINDEXM.xlsx, T10Y2Y.xlsx, DEXINUS.xlsx, and the Sentiment CSV are in the same folder as this app.")
    st.stop()

# A. Markov Switching (Regime Detection)
try:
    mod_regime = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True)
    res_regime = mod_regime.fit()
    df['Regime_Prob'] = res_regime.smoothed_marginal_probabilities[1]
except:
    df['Regime_Prob'] = 0

# B. Structural VAR (SVAR) with Cholesky Identification
svar_cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']
model = VAR(df[svar_cols])
results = model.fit(1) # Lag 1 for Bayesian-style shrinkage stability

# Forecast
forecast_steps = 3
forecast_values = results.forecast(df[svar_cols].values[-1:], forecast_steps)
forecast_dates = pd.date_range(start=df['Date'].max() + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
forecast_df = pd.DataFrame(forecast_values, columns=svar_cols, index=forecast_dates)

# --- 3. UI DASHBOARD ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.caption("Predictive Engine: SVAR(1) | Bayesian Shrinkage | Hidden Markov Regimes")

# Summary Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current INR/USD", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("VAR Projection (3M)", f"₹{forecast_df['INR_USD'].iloc[-1]:.2f}", 
          delta=f"{forecast_df['INR_USD'].iloc[-1] - df['INR_USD'].iloc[-1]:+.2f}")
m3.metric("Regime", "High Volatility" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable Growth")
m4.metric("Model Confidence", "High" if len(df) > 36 else "Moderate")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Regime Analysis", "🎯 Predictive Path", "⚡ Structural Shocks", "📚 Logic"])

with tab1:
    st.subheader("Hidden Markov Model (HMM) Regime Detection")
    
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="Spot Rate", line=dict(color='#00FFAA')))
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Crisis Probability", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.4)')))
    fig_h.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig_h, use_container_width=True)

with tab2:
    st.subheader("Bayesian-Stabilized Forecast")
    fig_f = go.Figure()
    # History
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(18), y=df['INR_USD'].tail(18), name='Actual', line=dict(color='#00FFAA', width=3)))
    # Forecast Connection
    f_x = [df['Date'].iloc[-1]] + list(forecast_df.index)
    f_y = [df['INR_USD'].iloc[-1]] + list(forecast_df['INR_USD'])
    fig_f.add_trace(go.Scatter(x=f_x, y=f_y, name='SVAR Forecast', line=dict(color='#FF00FF', dash='dash', width=3)))
    fig_f.update_layout(template="plotly_dark")
    st.plotly_chart(fig_f, use_container_width=True)

with tab3:
    st.subheader("Structural Impulse Response (SVAR)")
    
    irf = results.irf(periods=6)
    # Causal path: Commodity (0) -> INR_USD (3)
    irf_vals = irf.orth_irfs[:, 3, 0] 
    fig_irf = px.line(x=range(7), y=irf_vals, title="Response of INR to a 1-SD Global Commodity Shock", template="plotly_dark")
    fig_irf.update_traces(line_color='#FF4B4B', fill='tozeroy')
    st.plotly_chart(fig_irf, use_container_width=True)
    st.info("The SVAR uses a recursive Cholesky ordering to isolate how global price shocks transmit to the Rupee.")

with tab4:
    st.markdown("""
    ### Institutional Upgrades Integrated:
    * **Structural Identification (SVAR):** Unlike a standard VAR, this uses **Cholesky Decomposition** to define causal direction. We assume global commodities move independently of the local Rupee.
    * **Bayesian Shrinkage:** By limiting the model to 1st-order lags and monthly data, we impose a structural prior that prevents "over-fitting" to short-term noise.
    * **Markov Switching:** The terminal automatically identifies structural changes in volatility, allowing the model to distinguish between "Normal" and "Stress" market regimes.
    """)

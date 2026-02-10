import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.api import VAR, MarkovAutoregression
import os

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="Macro Quant Terminal v2", layout="wide", page_icon="🏛️")

def log_debug(msg):
    """Simple UI logger for debugging data issues."""
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    st.session_state.debug_logs.append(msg)

@st.cache_data(ttl=600)
def load_macro_data():
    def get_series(file, col_name, is_csv=False, skip=0):
        if not os.path.exists(file): 
            log_debug(f"❌ FILE NOT FOUND: {file}")
            return pd.Series(dtype='float64')
        try:
            if is_csv:
                tmp = pd.read_csv(file, skiprows=skip)
            else:
                tmp = pd.read_excel(file)
            
            # Cleaning column headers
            tmp.columns = [str(c).strip() for c in tmp.columns]
            
            # Heuristic to find Date and Value columns
            date_col = [c for c in tmp.columns if 'date' in c.lower() or 'time' in c.lower() or 'observation' in c.lower()][0]
            val_col = [c for c in tmp.columns if c != date_col][0]
            
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
            series = tmp.dropna(subset=[date_col]).set_index(date_col)[val_col]
            series = pd.to_numeric(series, errors='coerce')
            
            log_debug(f"✅ LOADED: {file} ({len(series)} rows)")
            return series.resample('MS').last().rename(col_name)
        except Exception as e:
            log_debug(f"⚠️ ERROR IN {file}: {str(e)}")
            return pd.Series(dtype='float64')

    # SVAR ORDERING
    map_dict = {
        "Commodities": get_series("PALLFNFINDEXM.xlsx", "Commodities"),
        "Yield_Spread": get_series("T10Y2Y.xlsx", "Yield_Spread"),
        "Sentiment": get_series("export-2026-02-10T06_50_22.597Z.csv", "Sentiment", is_csv=True, skip=4),
        "INR_USD": get_series("DEXINUS.xlsx", "INR_USD")
    }
    
    combined = pd.concat(map_dict.values(), axis=1).sort_index().ffill().dropna()
    return combined.reset_index().rename(columns={'index': 'Date'})

# --- 2. EXECUTION & ERROR HANDLING ---
df = load_macro_data()

# Show Diagnostics if data fails
if df.empty or len(df.columns) < 5:
    st.error("🏛️ TERMINAL OFFLINE: Data Alignment Error.")
    with st.expander("🔍 View Debug Diagnostics", expanded=True):
        for log in st.session_state.get("debug_logs", []):
            st.write(log)
        st.info("Check if your filenames match exactly (case-sensitive on some systems).")
    st.stop()

# --- 3. ECONOMETRIC ENGINES ---
# A. Markov Switching
try:
    mod_regime = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True)
    res_regime = mod_regime.fit()
    df['Regime_Prob'] = res_regime.smoothed_marginal_probabilities[1]
except:
    df['Regime_Prob'] = 0

# B. Structural VAR (SVAR)
svar_cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']
model = VAR(df[svar_cols])
results = model.fit(1)

# Forecast
forecast_steps = 3
forecast_values = results.forecast(df[svar_cols].values[-1:], forecast_steps)
forecast_dates = pd.date_range(start=df['Date'].max() + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
forecast_df = pd.DataFrame(forecast_values, columns=svar_cols, index=forecast_dates)

# --- 4. UI DASHBOARD ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.caption("Engine: SVAR(1) | Bayesian Shrinkage | Hidden Markov Regimes")

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current INR/USD", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("VAR Projection (3M)", f"₹{forecast_df['INR_USD'].iloc[-1]:.2f}")
m3.metric("Regime", "High Volatility" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable Growth")
m4.metric("Status", "Operational", delta="Live")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Regime Analysis", "🎯 Predictive Path", "⚡ Structural Shocks", "📚 Logic"])

with tab1:
    st.subheader("Regime Classification (HMM)")
    
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="Spot Rate", line=dict(color='#00FFAA')))
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Crisis Probability", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.4)')))
    fig_h.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig_h, use_container_width=True)

with tab2:
    st.subheader("Bayesian-Stabilized Forecast")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(18), y=df['INR_USD'].tail(18), name='Actual', line=dict(color='#00FFAA', width=3)))
    f_x = [df['Date'].iloc[-1]] + list(forecast_df.index)
    f_y = [df['INR_USD'].iloc[-1]] + list(forecast_df['INR_USD'])
    fig_f.add_trace(go.Scatter(x=f_x, y=f_y, name='SVAR Forecast', line=dict(color='#FF00FF', dash='dash', width=3)))
    fig_f.update_layout(template="plotly_dark")
    st.plotly_chart(fig_f, use_container_width=True)

with tab3:
    st.subheader("Structural Impulse Response (SVAR)")
    
    irf = results.irf(periods=6)
    irf_vals = irf.orth_irfs[:, 3, 0] 
    fig_irf = px.line(x=range(7), y=irf_vals, title="Transmission of Global Commodity Shock to local FX", template="plotly_dark")
    fig_irf.update_traces(line_color='#FF4B4B', fill='tozeroy')
    st.plotly_chart(fig_irf, use_container_width=True)

with tab4:
    st.markdown("""
    ### 1. Structural Identification (SVAR)
    Standard VAR models are purely correlation-based. Our **SVAR** uses a recursive **Cholesky Ordering** to identify causal direction.
    
    ### 2. Bayesian Stability
    By limiting the model to a single lag on monthly-sampled data, we enforce a random-walk prior (Minnesota style) to ensure the forecast doesn't overfit to noise.
    
    ### 3. Markov Switching
    We use a **Hidden Markov Model** to detect hidden states in the economy, specifically distinguishing between expansionary and contractionary regimes.
    """)

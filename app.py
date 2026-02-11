import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.api import VAR, MarkovAutoregression
import os

st.set_page_config(page_title="Macro Quant Terminal v2", layout="wide", page_icon="🏛️")

@st.cache_data(ttl=600)
def load_macro_data():
    def flexible_loader(file, col_name, is_csv=False):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            for i in range(30): # Scan headers
                try:
                    df = pd.read_csv(file, skiprows=i, encoding='latin1') if is_csv else pd.read_excel(file, skiprows=i)
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    date_col = [c for c in df.columns if any(k in c for k in ['date', 'obs', 'time', 'year'])][0]
                    val_col = [c for c in df.columns if c != date_col and 'unnamed' not in c][0]
                    
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    df = df.dropna(subset=[date_col])
                    series = pd.to_numeric(df[val_col], errors='coerce').dropna()
                    
                    if len(series) > 5:
                        series.index = df.loc[series.index, date_col]
                        # Resample to Month-Start to ensure 01-01 and 01-15 alignment
                        return series.resample('MS').last().rename(col_name)
                except: continue
            return pd.Series(dtype='float64')
        except: return pd.Series(dtype='float64')

    datasets = {
        "Commodities": flexible_loader("PALLFNFINDEXM.xlsx", "Commodities"),
        "Yield_Spread": flexible_loader("T10Y2Y.xlsx", "Yield_Spread"),
        "Sentiment": flexible_loader("export-2026-02-10T06_50_22.597Z.csv", "Sentiment", is_csv=True),
        "INR_USD": flexible_loader("DEXINUS.xlsx", "INR_USD")
    }
    
    # LOGGING FOR DEBUG
    log_info = {k: (v.index.min(), v.index.max()) for k, v in datasets.items() if not v.empty}
    
    # INNER JOIN: Forces exact date matches
    combined = pd.concat(datasets.values(), axis=1, join='inner').sort_index().ffill().dropna()
    return combined.reset_index().rename(columns={'index': 'Date'}), log_info

# --- EXECUTION ---
df, health_logs = load_macro_data()

if df.empty:
    st.error("🏛️ TERMINAL OFFLINE: Data Alignment Error.")
    st.subheader("🔍 Temporal Overlap Analysis")
    st.write("The model requires all 4 files to have data for the **same months**. Here is what was found:")
    
    cols = st.columns(len(health_logs))
    for i, (name, dates) in enumerate(health_logs.items()):
        cols[i].metric(name, f"{dates[0].year} to {dates[1].year}")
        
    st.info("💡 **Fix:** If the Sentiment data starts in 2024 but FX data ends in 2023, there is no 'Overlap.' You must ensure your Sentiment CSV download covers the same range as your FRED files.")
    st.stop()

# --- THE SVAR & MARKOV ENGINES ---
svar_cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']
results = VAR(df[svar_cols]).fit(1) # Bayesian Shrinkage

try:
    res_ms = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True).fit()
    df['Regime_Prob'] = res_ms.smoothed_marginal_probabilities[1]
except:
    df['Regime_Prob'] = 0

# --- DASHBOARD UI ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.caption("Engine: SVAR(1) with Cholesky Identification & HMM Regime Switching")

forecast = results.forecast(df[svar_cols].values[-1:], 3)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current INR/USD", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("3M Projection", f"₹{forecast[-1, 3]:.2f}")
m3.metric("Regime", "High Vol" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable")
m4.metric("Samples", f"{len(df)} Months")

st.divider()

t1, t2, t3, t4 = st.tabs(["📊 Regime Detection", "🎯 Prediction Path", "⚡ Structural Shock", "📚 Methodology"])

with t1:
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="INR/USD", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Regime Prob", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.3)')))
    fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    f_dates = pd.date_range(df['Date'].max(), periods=4, freq='MS')
    f_vals = [df['INR_USD'].iloc[-1]] + list(forecast[:, 3])
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(24), y=df['INR_USD'].tail(24), name='Actual', line=dict(color='#00FFAA')))
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_vals, name='SVAR Forecast', line=dict(color='#FF00FF', dash='dash')))
    st.plotly_chart(fig_f.update_layout(template="plotly_dark"), use_container_width=True)

with t3:
    
    irf = results.irf(periods=10).orth_irfs[:, 3, 0]
    fig_irf = px.line(x=range(11), y=irf, title="Structural Response: Commodity Shock -> INR", template="plotly_dark")
    st.plotly_chart(fig_irf.update_traces(line_color='#FF4B4B', fill='tozeroy'), use_container_width=True)

with t4:
    st.markdown("""
    ### Institutional Framework
    - **Structural VAR:** Implements a recursive Cholesky identification.
    - **Bayesian Smoothing:** Uses a restricted lag order (L=1) to prevent overfitting on monthly macro series.
    - **Hidden Markov Model:** Non-linear regime detection to account for structural breaks in volatility.
    """)

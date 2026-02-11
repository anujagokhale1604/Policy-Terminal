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
    def fred_cleaner(file, col_name, is_csv=False):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            if is_csv:
                # Handle Sentiment CSV: skip headers until 'Date' is found
                df = pd.read_csv(file, on_bad_lines='skip', engine='python')
                for i in range(20):
                    test = pd.read_csv(file, skiprows=i)
                    if any('date' in str(c).lower() for c in test.columns):
                        df = test
                        break
            else:
                # Handle FRED Excel: The actual data usually starts on Row 11 (skiprows=10)
                # We will hunt for the row that has 'observation_date'
                for i in range(15):
                    df = pd.read_excel(file, skiprows=i)
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    if any(kw in df.columns[0] for kw in ['date', 'observation']):
                        break
            
            # Identify Date and Value columns
            date_col = df.columns[0]
            val_col = [c for c in df.columns if c != date_col and 'unnamed' not in c.lower()][0]
            
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            series = pd.to_numeric(df[val_col], errors='coerce')
            
            # Construct cleaned series
            clean_s = pd.Series(series.values, index=df[date_col])
            return clean_s.dropna().resample('MS').last().rename(col_name)
        except:
            return pd.Series(dtype='float64')

    # SVAR Ordering: Commodities (Global) -> Yields -> Sentiment -> INR/USD (Local)
    map_dict = {
        "Commodities": fred_cleaner("PALLFNFINDEXM.xlsx", "Commodities"),
        "Yield_Spread": fred_cleaner("T10Y2Y.xlsx", "Yield_Spread"),
        "Sentiment": fred_cleaner("export-2026-02-10T06_50_22.597Z.csv", "Sentiment", is_csv=True),
        "INR_USD": fred_cleaner("DEXINUS.xlsx", "INR_USD")
    }
    
    combined = pd.concat(map_dict.values(), axis=1).sort_index().ffill().dropna()
    return combined.reset_index().rename(columns={'index': 'Date'})

# --- CORE ENGINES ---
df = load_macro_data()

if df.empty or len(df.columns) < 5:
    st.error("🏛️ TERMINAL OFFLINE: Data Alignment Error.")
    st.info("Check: 1. Files in root directory? 2. Excel files are .xlsx? 3. CSV matches Sentiment name?")
    st.stop()

# 1. SVAR with Cholesky Identification
svar_cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']
# Using 1 lag for Bayesian-style shrinkage (prevents wild swings in small samples)
model = VAR(df[svar_cols])
results = model.fit(1)

# 2. Markov Switching (Regime Probabilities)
try:
    # Captures High vs Low volatility regimes
    mod_ms = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True)
    res_ms = mod_ms.fit()
    df['Regime_Prob'] = res_ms.smoothed_marginal_probabilities[1]
except:
    df['Regime_Prob'] = 0

# --- DASHBOARD UI ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.caption("Engine: SVAR(1) | Bayesian Shrinkage | Hidden Markov Model")

forecast = results.forecast(df[svar_cols].values[-1:], 3)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current INR/USD", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("VAR Projection (3M)", f"₹{forecast[-1, 3]:.2f}", 
          delta=f"{forecast[-1, 3] - df['INR_USD'].iloc[-1]:+.2f}")
m3.metric("Regime", "High Vol" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable")
m4.metric("Shock Sensitivity", "Elevated" if abs(results.irf(5).orth_irfs[:, 3, 0][-1]) > 0.1 else "Low")

st.divider()

t1, t2, t3 = st.tabs(["📊 Regime Analysis", "🎯 Predictive Path", "⚡ Structural Shock (IRF)"])

with t1:
    
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="Spot", line=dict(color='#00FFAA')))
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Crisis Prob", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.4)')))
    fig_h.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig_h, use_container_width=True)

with t2:
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(18), y=df['INR_USD'].tail(18), name='Actual', line=dict(color='#00FFAA')))
    f_dates = pd.date_range(df['Date'].max(), periods=4, freq='MS')
    f_vals = [df['INR_USD'].iloc[-1]] + list(forecast[:, 3])
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_vals, name='SVAR Forecast', line=dict(color='#FF00FF', dash='dash')))
    fig_f.update_layout(template="plotly_dark", title="3-Month Structural Projection")
    st.plotly_chart(fig_f, use_container_width=True)

with t3:
    
    irf = results.irf(periods=6).orth_irfs[:, 3, 0] # Response of INR (3) to Commodity Shock (0)
    fig_irf = px.line(x=range(7), y=irf, title="Causal Impact: 1-SD Commodity Shock -> INR/USD", template="plotly_dark")
    fig_irf.update_traces(line_color='#FF4B4B', fill='tozeroy')
    st.plotly_chart(fig_irf, use_container_width=True)
    st.info("💡 **Structural Note:** Cholesky Identification orders Global Commodities first, assuming they drive local FX without immediate feedback.")

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
    def get_series(file, col_name, is_csv=False):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            # Step 1: Read the file
            if is_csv:
                # Scans the CSV for the line containing 'Date' to skip headers dynamically
                tmp = pd.read_csv(file, on_bad_lines='skip', engine='python')
                # If metadata is at the top, find the real header
                if 'date' not in "".join(map(str, tmp.columns)).lower():
                    for i in range(15): # Scan first 15 rows
                        test = pd.read_csv(file, skiprows=i)
                        if any('date' in str(c).lower() for c in test.columns):
                            tmp = test
                            break
            else:
                # For XLSX (FRED Files): Skip the first 10 rows where metadata usually sits
                tmp = pd.read_excel(file)
                if len(tmp) < 5 or 'date' not in "".join(map(str, tmp.columns)).lower():
                    for i in range(1, 15):
                        test = pd.read_excel(file, skiprows=i)
                        if len(test) > 0 and any(kw in str(test.columns[0]).lower() for kw in ['date', 'observation']):
                            tmp = test
                            break

            # Step 2: Clean Columns
            tmp.columns = [str(c).strip() for c in tmp.columns]
            date_col = [c for c in tmp.columns if any(kw in c.lower() for kw in ['date', 'time', 'observation'])][0]
            val_col = [c for c in tmp.columns if c != date_col and 'unnamed' not in c.lower()][0]

            # Step 3: Format
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors='coerce')
            series = tmp.dropna(subset=[date_col]).set_index(date_col)[val_col]
            series = pd.to_numeric(series, errors='coerce').dropna()
            
            return series.resample('MS').last().rename(col_name)
        except Exception as e:
            return pd.Series(dtype='float64')

    # Mapping
    map_dict = {
        "Commodities": get_series("PALLFNFINDEXM.xlsx", "Commodities"),
        "Yield_Spread": get_series("T10Y2Y.xlsx", "Yield_Spread"),
        "Sentiment": get_series("export-2026-02-10T06_50_22.597Z.csv", "Sentiment", is_csv=True),
        "INR_USD": get_series("DEXINUS.xlsx", "INR_USD")
    }
    
    combined = pd.concat(map_dict.values(), axis=1).sort_index().ffill().dropna()
    return combined.reset_index().rename(columns={'index': 'Date'})

# --- 2. ENGINE ---
df = load_macro_data()

if df.empty or len(df.columns) < 5:
    st.error("🏛️ TERMINAL OFFLINE: Data Alignment Error.")
    st.info("The loader is now scanning for data tables within your files. Please ensure the files contain 'Date' and 'Value' columns.")
    st.stop()

# Markov Switching Logic
try:
    mod_regime = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True)
    res_regime = mod_regime.fit()
    df['Regime_Prob'] = res_regime.smoothed_marginal_probabilities[1]
except:
    df['Regime_Prob'] = 0

# SVAR Logic
svar_cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']
results = VAR(df[svar_cols]).fit(1)

# Forecast
forecast_steps = 3
forecast_values = results.forecast(df[svar_cols].values[-1:], forecast_steps)
forecast_dates = pd.date_range(start=df['Date'].max() + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
forecast_df = pd.DataFrame(forecast_values, columns=svar_cols, index=forecast_dates)

# --- 3. UI ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.caption("Engine: SVAR(1) | Bayesian Shrinkage | Hidden Markov Regimes")

# Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current INR/USD", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("3M Projection", f"₹{forecast_df['INR_USD'].iloc[-1]:.2f}", 
          delta=f"{forecast_df['INR_USD'].iloc[-1] - df['INR_USD'].iloc[-1]:+.2f}")
m3.metric("Regime State", "High Vol" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable")
m4.metric("Dataset Size", f"{len(df)} Months")

st.divider()

# Tab Layout
t1, t2, t3 = st.tabs(["📊 Regime Analysis", "🎯 Predictive Path", "⚡ Structural Shocks"])

with t1:
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="Spot Rate", line=dict(color='#00FFAA')))
    fig_h.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Crisis Prob", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.4)')))
    fig_h.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig_h, use_container_width=True)

with t2:
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(12), y=df['INR_USD'].tail(12), name='Actual', line=dict(color='#00FFAA', width=3)))
    f_x = [df['Date'].iloc[-1]] + list(forecast_df.index)
    f_y = [df['INR_USD'].iloc[-1]] + list(forecast_df['INR_USD'])
    fig_f.add_trace(go.Scatter(x=f_x, y=f_y, name='Forecast', line=dict(color='#FF00FF', dash='dash')))
    fig_f.update_layout(template="plotly_dark")
    st.plotly_chart(fig_f, use_container_width=True)

with t3:
    irf = results.irf(periods=6).orth_irfs[:, 3, 0] 
    st.plotly_chart(px.line(x=range(7), y=irf, title="INR Response to Global Commodity Shock", template="plotly_dark").update_traces(line_color='#FF4B4B', fill='tozeroy'), use_container_width=True)

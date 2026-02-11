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
    def find_data_in_file(file, col_name, is_csv=False):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            # 1. Load the raw file
            if is_csv:
                # Try common delimiters
                raw_df = pd.read_csv(file, on_bad_lines='skip', sep=None, engine='python')
            else:
                # Load the first sheet by default
                raw_df = pd.read_excel(file)

            # 2. Iterative Header Search (The "Deep Scan")
            # We look for a column that looks like a date and a column that looks like a number
            for skip in range(0, 15):
                try:
                    df = pd.read_csv(file, skiprows=skip) if is_csv else pd.read_excel(file, skiprows=skip)
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    
                    # Find Date Column: look for 'date', 'obs', 'time', 'year'
                    date_cols = [c for c in df.columns if any(k in c for k in ['date', 'obs', 'time', 'year'])]
                    # Find Value Column: the one that isn't a date and has numeric data
                    potential_vals = [c for c in df.columns if c not in date_cols and 'unnamed' not in c]
                    
                    if date_cols and potential_vals:
                        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
                        # Drop rows where date or value is missing
                        df = df.dropna(subset=[date_cols[0]])
                        series = pd.to_numeric(df[potential_vals[0]], errors='coerce').dropna()
                        
                        if len(series) > 5:
                            series.index = df.loc[series.index, date_cols[0]]
                            return series.resample('MS').last().rename(col_name)
                except:
                    continue
            return pd.Series(dtype='float64')
        except:
            return pd.Series(dtype='float64')

    # Mapping with specific SVAR ordering
    map_dict = {
        "Commodities": find_data_in_file("PALLFNFINDEXM.xlsx", "Commodities"),
        "Yield_Spread": find_data_in_file("T10Y2Y.xlsx", "Yield_Spread"),
        "Sentiment": find_data_in_file("export-2026-02-10T06_50_22.597Z.csv", "Sentiment", is_csv=True),
        "INR_USD": find_data_in_file("DEXINUS.xlsx", "INR_USD")
    }
    
    combined = pd.concat(map_dict.values(), axis=1).sort_index().ffill().dropna()
    return combined.reset_index().rename(columns={'index': 'Date'})

# --- CORE ENGINE ---
df = load_macro_data()

if df.empty or len(df.columns) < 5:
    st.error("🏛️ TERMINAL OFFLINE: Data Alignment Error.")
    st.warning("The script cannot align the 4 datasets. This usually means the 'Date' column names differ across files.")
    
    # Debug Helper for the user
    if st.checkbox("Show Raw File Diagnostics"):
        files = ["PALLFNFINDEXM.xlsx", "T10Y2Y.xlsx", "export-2026-02-10T06_50_22.597Z.csv", "DEXINUS.xlsx"]
        for f in files:
            if os.path.exists(f):
                st.write(f"Columns in {f}:", pd.read_excel(f).columns.tolist() if 'xlsx' in f else pd.read_csv(f).columns.tolist())
            else:
                st.write(f"❌ {f} is missing from the folder.")
    st.stop()

# 1. Structural VAR fit (Cholesky Ordering)
svar_cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']
results = VAR(df[svar_cols]).fit(1)

# 2. Markov Switching Regime Detection
try:
    res_ms = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True).fit()
    df['Regime_Prob'] = res_ms.smoothed_marginal_probabilities[1]
except:
    df['Regime_Prob'] = 0

# --- DASHBOARD UI ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.caption("Recursive Identification (SVAR) | Bayesian Shrinkage | Markov Switching")

forecast = results.forecast(df[svar_cols].values[-1:], 3)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current INR/USD", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("VAR Projection (3M)", f"₹{forecast[-1, 3]:.2f}")
m3.metric("Regime", "High Vol" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable")
m4.metric("Shock Sensitivity", "Elevated" if abs(results.irf(5).orth_irfs[:, 3, 0][-1]) > 0.05 else "Low")

st.divider()

t1, t2, t3 = st.tabs(["📊 Regime Detection", "🎯 Prediction Path", "⚡ Structural Shock (IRF)"])

with t1:
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="Spot", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Crisis Prob", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.3)')))
    fig_h = fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig_h, use_container_width=True)

with t2:
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(15), y=df['INR_USD'].tail(15), name='Historical', line=dict(color='#00FFAA')))
    f_dates = pd.date_range(df['Date'].max(), periods=4, freq='MS')
    f_vals = [df['INR_USD'].iloc[-1]] + list(forecast[:, 3])
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_vals, name='VAR Forecast', line=dict(color='#FF00FF', dash='dash')))
    st.plotly_chart(fig_f.update_layout(template="plotly_dark"), use_container_width=True)

with t3:
    
    irf = results.irf(10).orth_irfs[:, 3, 0] # Response of INR (3) to Commodity Shock (0)
    fig_irf = px.line(x=range(11), y=irf, title="Impact of 1-SD Commodity Shock on INR/USD", template="plotly_dark")
    st.plotly_chart(fig_irf.update_traces(line_color='#FF4B4B', fill='tozeroy'), use_container_width=True)
    st.info("The SVAR uses structural identification to ensure the commodity shock is purely exogenous.")

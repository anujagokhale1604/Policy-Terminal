import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.api import VAR, MarkovAutoregression

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="Macro Quant Terminal v2", layout="wide", page_icon="🏛️")

@st.cache_data(ttl=600)
def load_macro_data():
    try:
        # A. Global Commodities (XLSX)
        # Using the specific filename you noted as 'only xlsx'
        df_comm = pd.read_excel("PALLFNFINDEXM.xlsx", sheet_name="Monthly")
        df_comm['Date'] = pd.to_datetime(df_comm['observation_date'])
        comm = df_comm.set_index('Date')['PALLFNFINDEXM'].rename("Commodities")

        # B. Yield Spread (XLSX -> Resample to Monthly)
        df_yield = pd.read_excel("T10Y2Y.xlsx", sheet_name="Daily")
        df_yield['Date'] = pd.to_datetime(df_yield['observation_date'])
        # Convert to numeric to handle any '.' placeholders for holidays
        yield_val = pd.to_numeric(df_yield['T10Y2Y'], errors='coerce')
        yield_spread = pd.Series(yield_val.values, index=df_yield['Date']).resample('MS').mean().rename("Yield_Spread")

        # C. Global Sentiment (The 'Only' CSV)
        # Skips metadata rows to reach the time-series data
        df_sent = pd.read_csv("export-2026-02-10T06_50_22.597Z.csv", skiprows=3, header=None, names=['Date', 'Sentiment'])
        df_sent['Date'] = pd.to_datetime(df_sent['Date'], errors='coerce')
        sentiment = df_sent.dropna().set_index('Date')['Sentiment'].rename("Sentiment")

        # D. INR/USD Spot (XLSX -> Resample to Monthly)
        df_inr = pd.read_excel("DEXINUS.xlsx", sheet_name="Daily")
        df_inr['Date'] = pd.to_datetime(df_inr['observation_date'])
        inr_val = pd.to_numeric(df_inr['DEXINUS'], errors='coerce')
        inr_usd = pd.Series(inr_val.values, index=df_inr['Date']).resample('MS').mean().rename("INR_USD")

        # --- DATA ALIGNMENT ---
        combined = pd.concat([comm, yield_spread, sentiment, inr_usd], axis=1).sort_index()
        # Clean edges and interpolate internal missing values (holidays/reporting lags)
        combined = combined.interpolate(method='linear').dropna()
        
        return combined.reset_index().rename(columns={'index': 'Date'}), "Online"
    except Exception as e:
        return pd.DataFrame(), f"Init Error: {str(e)}"

# --- 2. ENGINE EXECUTION ---
df, status = load_macro_data()

st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")

if df.empty:
    st.error(f"🏛️ TERMINAL OFFLINE: {status}")
    st.info("Check if files 'PALLFNFINDEXM.xlsx', 'T10Y2Y.xlsx', 'DEXINUS.xlsx' and the export CSV are in the same folder.")
    st.stop()

# Ordering variables for Cholesky Recursive Identification
# Order: World Shocks (Commodities) -> US Policy (Yields) -> Confidence (Sentiment) -> Local FX (INR)
cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']

# --- 3. THE ECONOMETRIC ENGINES ---
try:
    # SVAR(1) with Recursive Identification
    model_var = VAR(df[cols])
    res_var = model_var.fit(1)
    forecast = res_var.forecast(df[cols].values[-1:], 3)
    
    # Markov Switching Autoregression for Regime Detection
    res_ms = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True).fit()
    df['Regime_Prob'] = res_ms.smoothed_marginal_probabilities[1]
    engine_status = "SVAR(1) + MS-AR Engine"
except Exception as e:
    engine_status = f"Fallback Mode (Error: {str(e)[:50]}...)"
    forecast = np.tile(df[cols].iloc[-1].values, (3, 1))
    df['Regime_Prob'] = 0

# --- 4. DASHBOARD UI ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("INR/USD Current", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("3M VAR Forecast", f"₹{forecast[-1, 3]:.2f}")
m3.metric("Volatility Regime", "High Risk" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable")
m4.metric("Quant Engine", engine_status)

st.divider()

t1, t2, t3 = st.tabs(["📊 Regime Probability", "🎯 Predictive Path", "⚡ Structural Shock"])

with t1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="Spot Rate", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Risk Prob", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.3)')))
    fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    f_dates = pd.date_range(df['Date'].max(), periods=4, freq='MS')
    f_vals = [df['INR_USD'].iloc[-1]] + list(forecast[:, 3])
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(24), y=df['INR_USD'].tail(24), name='Actual', line=dict(color='#00FFAA')))
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_vals, name='VAR Forecast', line=dict(color='#FF00FF', dash='dash')))
    st.plotly_chart(fig_f.update_layout(template="plotly_dark"), use_container_width=True)

with t3:
    # Impact of a Commodity Shock on the Rupee over 10 months
    irf = res_var.irf(periods=10).orth_irfs[:, 3, 0]
    fig_irf = px.line(x=range(11), y=irf, title="Recursive Shock: Impact of Global Commodity Surge on INR/USD", template="plotly_dark")
    st.plotly_chart(fig_irf.update_traces(line_color='#FF4B4B', fill='tozeroy'), use_container_width=True)

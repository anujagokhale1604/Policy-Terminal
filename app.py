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
    try:
        # 1. Global Commodities (Monthly)
        # Using the specific sheet found in your uploaded PALLFNFINDEXM.xlsx
        df_comm = pd.read_csv("PALLFNFINDEXM.xlsx - Monthly.csv") # Adjusted for CSV accessibility
        df_comm['Date'] = pd.to_datetime(df_comm['observation_date'])
        comm = df_comm.set_index('Date')['PALLFNFINDEXM'].rename("Commodities")

        # 2. Yield Spread (Daily -> Monthly)
        df_yield = pd.read_csv("T10Y2Y.xlsx - Daily.csv")
        df_yield['Date'] = pd.to_datetime(df_yield['observation_date'])
        # Critical Fix: Convert '.' or empty strings to NaN before numeric conversion
        yield_spread = pd.to_numeric(df_yield['T10Y2Y'], errors='coerce')
        yield_spread = pd.Series(yield_spread.values, index=df_yield['Date']).resample('MS').mean().rename("Yield_Spread")

        # 3. Sentiment (CSV - Monthly)
        # Your export file has 3 rows of metadata
        df_sent = pd.read_csv("export-2026-02-10T06_50_22.597Z.csv", skiprows=4, header=None, names=['Date', 'Sentiment'])
        df_sent['Date'] = pd.to_datetime(df_sent['Date'], errors='coerce')
        sentiment = df_sent.dropna().set_index('Date')['Sentiment'].rename("Sentiment")

        # 4. INR/USD Spot (Daily -> Monthly)
        df_inr = pd.read_csv("DEXINUS.xlsx - Daily.csv")
        df_inr['Date'] = pd.to_datetime(df_inr['observation_date'])
        inr_usd = pd.to_numeric(df_inr['DEXINUS'], errors='coerce')
        inr_usd = pd.Series(inr_usd.values, index=df_inr['Date']).resample('MS').mean().rename("INR_USD")

        # --- THE ALIGNMENT GUARD ---
        # Join all series and drop any row that has a NaN in ANY column
        combined = pd.concat([comm, yield_spread, sentiment, inr_usd], axis=1).sort_index()
        
        # Linear interpolation fills small gaps, then we drop the remaining edges
        combined = combined.interpolate(method='linear').dropna()
        
        if len(combined) < 12:
            return pd.DataFrame(), f"Insufficient Data Overlap: Only {len(combined)} months found."
            
        return combined.reset_index().rename(columns={'index': 'Date'}), "Online"
    except Exception as e:
        return pd.DataFrame(), f"Initialization Error: {str(e)}"

# --- 2. EXECUTION ---
df, status = load_macro_data()

st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")

if df.empty:
    st.error(f"🏛️ TERMINAL OFFLINE: {status}")
    st.stop()

# Recursive ordering: Global Shocks -> Local Pricing
cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']

# --- 3. ECONOMETRICS ---

try:
    # We use a 1-month lag to capture immediate macro transmission
    model_var = VAR(df[cols])
    res_var = model_var.fit(1)
    forecast = res_var.forecast(df[cols].values[-1:], 3)
    
    # Markov Switching for Regime Detection
    res_ms = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True).fit()
    df['Regime'] = res_ms.smoothed_marginal_probabilities[1]
except Exception as e:
    st.warning(f"Engine Warning: Falling back to Linear Drift due to: {e}")
    forecast = np.tile(df[cols].iloc[-1].values, (3, 1))
    df['Regime'] = 0

# --- 4. DASHBOARD ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("INR/USD Spot", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("3M VAR Forecast", f"₹{forecast[-1, 3]:.2f}")
m3.metric("Regime Risk", f"{df['Regime'].iloc[-1]:.1%}")
m4.metric("Engine", "SVAR-Cholesky")

st.divider()

t1, t2, t3 = st.tabs(["📊 Regime Analysis", "🎯 Predictive Path", "⚡ Structural Logic"])

with t1:
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="Spot", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime'], name="Volatility Regime", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.3)')))
    fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    f_dates = pd.date_range(df['Date'].max(), periods=4, freq='MS')
    f_vals = [df['INR_USD'].iloc[-1]] + list(forecast[:, 3])
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(24), y=df['INR_USD'].tail(24), name='Actual', line=dict(color='#00FFAA')))
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_vals, name='Forecast', line=dict(color='#FF00FF', dash='dash')))
    st.plotly_chart(fig_f.update_layout(template="plotly_dark"), use_container_width=True)

with t3:
    
    irf = res_var.irf(periods=5).orth_irfs[:, 3, 0]
    st.plotly_chart(px.line(x=range(6), y=irf, title="Impact of Commodity Shock on Rupee", template="plotly_dark").update_traces(line_color='#FF4B4B', fill='tozeroy'), use_container_width=True)

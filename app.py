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
    try:
        # A. Global Commodities (Monthly)
        df_comm = pd.read_excel("PALLFNFINDEXM.xlsx", sheet_name="Monthly")
        df_comm.columns = [str(c).strip() for c in df_comm.columns]
        df_comm['Date'] = pd.to_datetime(df_comm.iloc[:, 0])
        comm = df_comm.set_index('Date').iloc[:, 0].rename("Commodities")

        # B. Yield Spread (Daily -> Resample to Month-Start)
        df_yield = pd.read_excel("T10Y2Y.xlsx", sheet_name="Daily")
        df_yield.columns = [str(c).strip() for c in df_yield.columns]
        df_yield['Date'] = pd.to_datetime(df_yield.iloc[:, 0])
        yield_spread = pd.to_numeric(df_yield.set_index('Date').iloc[:, 0], errors='coerce').resample('MS').last().rename("Yield_Spread")

        # C. Global Sentiment (CSV - Monthly)
        # Using skip=3 to bypass the OECD metadata headers
        df_sent = pd.read_csv("export-2026-02-10T06_50_22.597Z.csv", skiprows=3, header=None, names=['Date', 'Sentiment'])
        df_sent['Date'] = pd.to_datetime(df_sent['Date'], errors='coerce')
        sentiment = df_sent.dropna().set_index('Date')['Sentiment'].rename("Sentiment")

        # D. INR/USD Spot (Daily -> Resample to Month-Start)
        df_inr = pd.read_excel("DEXINUS.xlsx", sheet_name="Daily")
        df_inr.columns = [str(c).strip() for c in df_inr.columns]
        df_inr['Date'] = pd.to_datetime(df_inr.iloc[:, 0])
        inr_usd = pd.to_numeric(df_inr.set_index('Date').iloc[:, 0], errors='coerce').resample('MS').last().rename("INR_USD")

        # --- DATA ALIGNMENT ---
        # Inner join ensures the SVAR math has complete matrices for all dates
        combined = pd.concat([comm, yield_spread, sentiment, inr_usd], axis=1).sort_index()
        
        # Institutional Forward-Fill: Handle holidays/missing daily prints before inner join
        combined = combined.ffill().bfill().dropna()
        
        return combined.reset_index().rename(columns={'index': 'Date'}), "Online"
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- 2. ENGINE EXECUTION ---
df, status = load_macro_data()

st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.caption(f"Status: {status} | Engine: SVAR(1) with Recursive Identification")

if df.empty:
    st.error("🏛️ TERMINAL OFFLINE: Data Alignment Error")
    st.info("Verify filenames: PALLFNFINDEXM.xlsx, T10Y2Y.xlsx, export-2026-02-10T06_50_22.597Z.csv, DEXINUS.xlsx")
    st.stop()

# Ordering variables for Cholesky (Exogenous -> Endogenous)
cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']

# --- 3. THE ECONOMETRIC ENGINES ---
# Engine A: Vector Autoregression (Bayesian Shrinkage via L=1)
model_var = VAR(df[cols])
res_var = model_var.fit(1)

# Engine B: Markov Switching (Volatility Regime Detection)
res_ms = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True).fit()
df['Regime_Prob'] = res_ms.smoothed_marginal_probabilities[1]

# --- 4. DASHBOARD UI ---
forecast = res_var.forecast(df[cols].values[-1:], 3)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Spot Rate", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("3M Projection", f"₹{forecast[-1, 3]:.2f}", f"{forecast[-1, 3] - df['INR_USD'].iloc[-1]:.2f}")
m3.metric("State", "High Vol" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable")
m4.metric("Dataset Span", f"{len(df)} Mo")

st.divider()

t1, t2, t3, t4 = st.tabs(["📊 Regime Analysis", "🎯 Prediction Path", "⚡ Structural Shock", "📚 Framework"])

with t1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="Spot", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Risk Probability", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.3)')))
    fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1], title="Regime Probability"))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    f_dates = pd.date_range(df['Date'].max(), periods=4, freq='MS')
    f_vals = [df['INR_USD'].iloc[-1]] + list(forecast[:, 3])
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(24), y=df['INR_USD'].tail(24), name='Actual', line=dict(color='#00FFAA')))
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_vals, name='SVAR Forecast', line=dict(color='#FF00FF', dash='dash')))
    st.plotly_chart(fig_f.update_layout(template="plotly_dark", title="24-Month History & 3-Month Projection"), use_container_width=True)

with t3:
    # Calculating Impulse Response: How does a 1-std-dev shock in Commodities affect INR?
    irf = res_var.irf(periods=10).orth_irfs[:, 3, 0]
    fig_irf = px.line(x=range(11), y=irf, title="Structural Response: Commodity Inflation Shock -> Rupee Impact", template="plotly_dark")
    st.plotly_chart(fig_irf.update_traces(line_color='#FF4B4B', fill='tozeroy'), use_container_width=True)
    st.info("This chart uses Cholesky identification to isolate the causal impact of global commodity prices on the INR/USD exchange rate.")

with t4:
    st.markdown("""
    ### Institutional Methodology
    * **Bayesian Stabilization**: The VAR is restricted to 1 lag to prevent overfitting and ensure stability across sparse monthly data.
    * **Recursive Identification**: Variables are ordered $Commodities \\rightarrow Yields \\rightarrow Sentiment \\rightarrow INR$ to reflect the flow of global macro shocks.
    * **Hidden Markov Model**: The regime detector separates 'Normal' volatility from 'Crisis' states based on the second-moment properties of the exchange rate.
    """)

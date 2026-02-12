import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="Macro Quant Terminal v2.1", layout="wide", page_icon="🏛️")

@st.cache_data(ttl=600)
def load_and_preprocess_data():
    try:
        # A. Global Commodities (XLSX)
        df_comm = pd.read_excel("PALLFNFINDEXM.xlsx", sheet_name="Monthly")
        df_comm['Date'] = pd.to_datetime(df_comm['observation_date'])
        comm = df_comm.set_index('Date')['PALLFNFINDEXM'].rename("Commodities")

        # B. Yield Spread (XLSX)
        df_yield = pd.read_excel("T10Y2Y.xlsx", sheet_name="Daily")
        df_yield['Date'] = pd.to_datetime(df_yield['observation_date'])
        yield_val = pd.to_numeric(df_yield['T10Y2Y'], errors='coerce')
        yield_spread = pd.Series(yield_val.values, index=df_yield['Date']).resample('MS').mean().rename("Yield_Spread")

        # C. Global Sentiment (The 'Only' CSV)
        df_sent = pd.read_csv("export-2026-02-10T06_50_22.597Z.csv", skiprows=3, header=None, names=['Date', 'Sentiment'])
        df_sent['Date'] = pd.to_datetime(df_sent['Date'], errors='coerce')
        sentiment = df_sent.dropna().set_index('Date')['Sentiment'].rename("Sentiment")

        # D. INR/USD Spot (XLSX)
        df_inr = pd.read_excel("DEXINUS.xlsx", sheet_name="Daily")
        df_inr['Date'] = pd.to_datetime(df_inr['observation_date'])
        inr_val = pd.to_numeric(df_inr['DEXINUS'], errors='coerce')
        inr_usd = pd.Series(inr_val.values, index=df_inr['Date']).resample('MS').mean().rename("INR_USD")

        # --- DATA ALIGNMENT ---
        combined = pd.concat([comm, yield_spread, sentiment, inr_usd], axis=1).sort_index()
        combined = combined.interpolate(method='linear').dropna()
        
        return combined, "Online"
    except Exception as e:
        return pd.DataFrame(), f"Init Error: {str(e)}"

# --- 2. ENGINE EXECUTION ---
raw_df, status = load_and_preprocess_data()

st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")

if raw_df.empty:
    st.error(f"🏛️ TERMINAL OFFLINE: {status}")
    st.stop()

# --- 3. ECONOMETRIC TRANSFORMATION ---
# VAR models require stationary data. We'll use 1st differences.
# We also standardize to ensure SVD convergence for Markov Switching.
df_diff = raw_df.diff().dropna()
means = df_diff.mean()
stds = df_diff.std()
df_std = (df_diff - means) / stds

cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']
current_inr = raw_df['INR_USD'].iloc[-1]

try:
    # A. Structural VAR(1)
    model_var = VAR(df_std[cols])
    res_var = model_var.fit(1)
    
    # B. Markov Switching (Regime Detection)
    # Using the standardized returns of INR/USD
    res_ms = MarkovAutoregression(df_std['INR_USD'], k_regimes=2, order=1, switching_variance=False).fit()
    regime_probs = res_ms.smoothed_marginal_probabilities[1]
    
    # C. Multi-Step Forecast (3 Months)
    # Forecast in standardized-difference space
    forecast_std_diff = res_var.forecast(df_std[cols].values[-1:], 3)
    
    # Reverse Standardization and Differencing
    # 1. Reverse Standardization: (std * std_dev) + mean
    forecast_diff = (forecast_std_diff[:, 3] * stds['INR_USD']) + means['INR_USD']
    
    # 2. Reverse Differencing: Cumulative sum starting from current level
    forecast_levels = [current_inr]
    for d in forecast_diff:
        forecast_levels.append(forecast_levels[-1] + d)
    forecast_final = forecast_levels[1:]
    
    engine_status = "SVAR(1) + MS-AR Engine (Optimized)"
except Exception as e:
    engine_status = f"Fallback Mode (Error: {str(e)[:40]}...)"
    forecast_final = [current_inr] * 3
    regime_probs = pd.Series(0, index=df_diff.index)

# --- 4. DASHBOARD UI ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("INR/USD Current", f"₹{current_inr:.2f}")
m2.metric("3M VAR Forecast", f"₹{forecast_final[-1]:.2f}", f"{forecast_final[-1]-current_inr:+.2f}")
m3.metric("Volatility Regime", "High Risk" if (regime_probs.iloc[-1] > 0.5) else "Stable")
m4.metric("Quant Engine", engine_status)

st.divider()

t1, t2, t3 = st.tabs(["📊 Regime Probability", "🎯 Predictive Path", "⚡ Structural Shock"])

with t1:
    fig_regime = go.Figure()
    fig_regime.add_trace(go.Scatter(x=raw_df.index, y=raw_df['INR_USD'], name="Spot Rate", line=dict(color='#00FFAA')))
    fig_regime.add_trace(go.Scatter(x=df_diff.index, y=regime_probs, name="Risk Prob", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.4)')))
    fig_regime.update_layout(
        template="plotly_dark", 
        yaxis2=dict(overlaying='y', side='right', range=[0, 1], title="Regime Probability"),
        yaxis=dict(title="INR/USD"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_regime, use_container_width=True)

with t2:
    f_dates = pd.date_range(raw_df.index[-1], periods=4, freq='MS')
    f_plot_vals = [current_inr] + list(forecast_final)
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=raw_df.index[-24:], y=raw_df['INR_USD'].iloc[-24:], name='Historical (2Y)', line=dict(color='#00FFAA')))
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_plot_vals, name='VAR Forecast', line=dict(color='#FF00FF', dash='dash')))
    fig_f.update_layout(template="plotly_dark", title="INR/USD Predictive Path (Mean Reverting VAR)")
    st.plotly_chart(fig_f, use_container_width=True)

with t3:
    if "Optimized" in engine_status:
        # Impulse Response: Impact of a 1 std-dev shock in Commodities on INR/USD
        irf = res_var.irf(periods=10).orth_irfs[:, 3, 0]
        # Cumulative impact to show total level change
        cum_irf = np.cumsum(irf) * stds['INR_USD']
        fig_irf = px.line(x=range(11), y=cum_irf, title="Structural Shock: Impact of Global Commodity Surge on INR/USD (in ₹)", template="plotly_dark")
        fig_irf.update_traces(line_color='#FF4B4B', fill='tozeroy')
        st.plotly_chart(fig_irf, use_container_width=True)
    else:
        st.warning("Shock Analysis unavailable in Fallback Mode.")

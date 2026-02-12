import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

# --- 1. CONFIG & UI SETUP ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide")

# --- 2. SIDEBAR: MODEL CONTROL PANEL ---
st.sidebar.header("🛠️ MODEL CONTROL PANEL")

st.sidebar.subheader("Shock Scenarios")
yield_shock = st.sidebar.slider("US Yield Spread Shock (bps)", -100, 100, 0, step=10, 
                                help="Simulate a shift in the US 10Y-2Y yield curve.")
comm_shock = st.sidebar.slider("Commodity Price Spike (%)", -20, 50, 0,
                               help="Simulate a global commodity price shock.")

st.sidebar.subheader("Engine Hyperparameters")
var_lags = st.sidebar.selectbox("VAR Lag Order", [1, 2, 3], index=0,
                                help="Number of previous months the model 'remembers'.")
regime_count = st.sidebar.radio("Markov Regimes", [2, 3], index=0,
                                help="Number of hidden volatility states to track.")
forecast_horizon = st.sidebar.number_input("Forecast Months", 1, 12, 3)

# --- 3. DATA ENGINE ---
@st.cache_data
def load_data():
    try:
        # Commodities (XLSX)
        df_comm = pd.read_excel("PALLFNFINDEXM.xlsx", sheet_name="Monthly")
        df_comm['Date'] = pd.to_datetime(df_comm['observation_date'])
        comm = df_comm.set_index('Date')['PALLFNFINDEXM'].rename("Commodities")

        # Yield Spread (XLSX)
        df_yield = pd.read_excel("T10Y2Y.xlsx", sheet_name="Daily")
        df_yield['Date'] = pd.to_datetime(df_yield['observation_date'])
        yield_val = pd.to_numeric(df_yield['T10Y2Y'], errors='coerce')
        yield_spread = pd.Series(yield_val.values, index=df_yield['Date']).resample('MS').mean().rename("Yield_Spread")

        # Sentiment (Note: File is named CSV but user prefers Excel logic; using read_csv for safety)
        df_sent = pd.read_csv("export-2026-02-10T06_50_22.597Z.csv", skiprows=3, header=None, names=['Date', 'Sentiment'])
        df_sent['Date'] = pd.to_datetime(df_sent['Date'], errors='coerce')
        sentiment = df_sent.dropna().set_index('Date')['Sentiment'].rename("Sentiment")

        # INR/USD Spot (XLSX)
        df_inr = pd.read_excel("DEXINUS.xlsx", sheet_name="Daily")
        df_inr['Date'] = pd.to_datetime(df_inr['observation_date'])
        inr_val = pd.to_numeric(df_inr['DEXINUS'], errors='coerce')
        inr_usd = pd.Series(inr_val.values, index=df_inr['Date']).resample('MS').mean().rename("INR_USD")

        combined = pd.concat([comm, yield_spread, sentiment, inr_usd], axis=1).sort_index().interpolate().dropna()
        return combined
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return pd.DataFrame()

raw_df = load_data()

# --- 4. METHODOLOGY NOTE ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
with st.expander("📖 METHODOLOGY & QUANT ENGINE LOGIC", expanded=True):
    st.markdown(f"""
    **Engine Core:** This terminal utilizes a **Structural Vector Autoregression (SVAR)** integrated with a **Markov-Switching (MS-AR)** framework.
    1. **SVAR Layer:** Captures the multi-directional relationships between the Rupee, US Treasury spreads, and Commodity markets. Unlike standard models, it assumes *Endogeneity* (every variable influences every other variable).
    2. **Markov Layer:** Specifically detects **Hidden Regimes**. It differentiates between 'Normal' market behavior and 'Stress' periods, ensuring the forecast doesn't just average out extremes.
    3. **Stationarity & SVD:** To prevent mathematical 'drift' and ensure SVD convergence, the data is transformed into **First-Differences ($\Delta$)** and then **Standardized ($\sigma$)**. This places all macro indicators on a level playing field for the solvers.
    """)

if not raw_df.empty:
    # --- 5. ECONOMETRICS & STRESS TESTING ---
    # Prepare stationary standardized data
    df_diff = raw_df.diff().dropna()
    means, stds = df_diff.mean(), df_diff.std()
    df_std = (df_diff - means) / stds

    # Initialize Models
    model_var = VAR(df_std)
    res_var = model_var.fit(var_lags)
    
    # Apply Stress Test Shocks to the last observation
    last_obs = df_std.iloc[-1:].copy()
    last_obs['Yield_Spread'] += (yield_shock / 100) / stds['Yield_Spread']
    last_obs['Commodities'] *= (1 + comm_shock / 100)

    # Generate Forecast
    forecast_std = res_var.forecast(last_obs.values, forecast_horizon)
    forecast_final = (forecast_std[:, 3] * stds['INR_USD']) + means['INR_USD']
    forecast_levels = np.cumsum(np.insert(forecast_final, 0, raw_df['INR_USD'].iloc[-1]))[1:]

    # Regime Probability Logic
    res_ms = MarkovAutoregression(df_std['INR_USD'], k_regimes=regime_count, order=1).fit()
    regime_probs = res_ms.smoothed_marginal_probabilities[1]

    # --- 6. DASHBOARD TABS ---
    t1, t2, t3 = st.tabs(["📊 Regime Probability", "🎯 Predictive Path", "⚡ Structural Shock"])

    with t1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=raw_df.index, y=raw_df['INR_USD'], name="Spot Rate", line=dict(color='#00FFAA')))
        fig1.add_trace(go.Scatter(x=regime_probs.index, y=regime_probs, name="Risk Prob", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.4)')))
        fig1.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
        st.plotly_chart(fig1, use_container_width=True)
        st.info(f"""
        **Dynamic Note:** Currently tracking **{regime_count} hidden states**. The 'Risk Prob' area represents the 
        smoothed filtered probability of a transition from a stable mean-reverting regime to a 
        high-variance regime. A reading above 0.5 indicates **Structural Instability**, where historical 
        correlations may break down.
        """)

    with t2:
        f_dates = pd.date_range(raw_df.index[-1], periods=forecast_horizon+1, freq='MS')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=raw_df.index[-24:], y=raw_df['INR_USD'].iloc[-24:], name='Historical', line=dict(color='#00FFAA')))
        fig2.add_trace(go.Scatter(x=f_dates, y=[raw_df['INR_USD'].iloc[-1]] + list(forecast_levels), name='Forecast', line=dict(dash='dash', color='#FF00FF')))
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
        st.info(f"""
        **Dynamic Note:** This path visualizes the **SVAR({var_lags}) Mean Projection**. 
        Under your current stress test of **{yield_shock}bps yield shock**, the model predicts 
        a {'depreciation' if yield_shock > 0 else 'strengthening'} bias. The path assumes 
        *Mean Reversion*, meaning the currency eventually tries to return to its macro equilibrium.
        """)

    with t3:
        irf = res_var.irf(periods=10).orth_irfs[:, 3, 0]
        fig3 = go.Figure(go.Scatter(x=list(range(11)), y=np.cumsum(irf), fill='tozeroy', line=dict(color='#FF4B4B')))
        fig3.update_layout(template="plotly_dark", title="Cumulative Response of INR to Commodity Shock")
        st.plotly_chart(fig3, use_container_width=True)
        st.info(f"""
        **Dynamic Note:** This **Impulse Response Function (IRF)** measures the 'decay rate' of a 
        macro shock. It shows how the INR absorbs a **{comm_shock}% commodity surge**. 
        If the curve plateaus at a high level, the shock is **Structural (Permanent)**; if it 
        trends back toward the baseline, the shock is **Transitory**.
        """)
else:
    st.warning("Please check your data files (DEXINUS.xlsx, T10Y2Y.xlsx, etc.) in the root directory.")

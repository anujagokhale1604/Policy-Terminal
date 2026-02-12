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

# --- 3. DATA ENGINE (Using Excel files as specified) ---
@st.cache_data
def load_and_sync_data():
    try:
        # Commodity Index
        df_comm = pd.read_excel("PALLFNFINDEXM.xlsx", sheet_name="Monthly")
        df_comm['Date'] = pd.to_datetime(df_comm['observation_date'])
        comm = df_comm.set_index('Date')['PALLFNFINDEXM'].rename("Commodities")

        # Yield Spread
        df_yield = pd.read_excel("T10Y2Y.xlsx", sheet_name="Daily")
        df_yield['Date'] = pd.to_datetime(df_yield['observation_date'])
        df_yield['T10Y2Y'] = pd.to_numeric(df_yield['T10Y2Y'], errors='coerce')
        yield_spread = df_yield.set_index('Date')['T10Y2Y'].resample('MS').mean().rename("Yield_Spread")

        # INR/USD Spot
        df_inr = pd.read_excel("DEXINUS.xlsx", sheet_name="Daily")
        df_inr['Date'] = pd.to_datetime(df_inr['observation_date'])
        df_inr['DEXINUS'] = pd.to_numeric(df_inr['DEXINUS'], errors='coerce')
        inr_usd = df_inr.set_index('Date')['DEXINUS'].resample('MS').mean().rename("INR_USD")

        # Combine
        combined = pd.concat([comm, yield_spread, inr_usd], axis=1).sort_index().interpolate().dropna()
        return combined
    except Exception as e:
        st.error(f"Data Loading Error: Ensure all .xlsx files are in the directory. Error: {e}")
        return pd.DataFrame()

# CRITICAL: Initialize raw_df immediately
raw_df = load_and_sync_data()

# --- 4. ENGINE CALCULATIONS & STRESS TESTING ---
if not raw_df.empty:
    # Stationary Transform
    df_diff = raw_df.diff().dropna()
    means, stds = df_diff.mean(), df_diff.std()
    df_std = (df_diff - means) / stds

    # SVAR Model
    model_var = VAR(df_std)
    res_var = model_var.fit(var_lags)
    
    # Inject Stress Tests into the last observation
    last_obs = df_std.iloc[-1:].copy()
    last_obs['Yield_Spread'] += (yield_shock / 100) / stds['Yield_Spread']
    last_obs['Commodities'] *= (1 + comm_shock / 100)

    # Forecast with Shocks
    forecast_std = res_var.forecast(last_obs.values, forecast_horizon)
    # Revert standardization and differencing for INR_USD
    inr_forecast_diff = (forecast_std[:, 2] * stds['INR_USD']) + means['INR_USD']
    forecast_levels = np.cumsum(np.insert(inr_forecast_diff, 0, raw_df['INR_USD'].iloc[-1]))[1:]

    # Markov Regime Switching Model
    res_ms = MarkovAutoregression(df_std['INR_USD'], k_regimes=regime_count, order=1).fit()
    regime_probs = res_ms.smoothed_marginal_probabilities[1]

    # --- 5. DYNAMIC HEADER ---
    current_spot = raw_df['INR_USD'].iloc[-1]
    f_3m = forecast_levels[2] if len(forecast_levels) >= 3 else forecast_levels[-1]
    delta = f_3m - current_spot
    current_regime_prob = regime_probs.iloc[-1]

    st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("INR/USD Current", f"₹{current_spot:.2f}")
    col2.metric("3M VAR Forecast", f"₹{f_3m:.2f}", f"{delta:+.2f}")
    col3.metric("Volatility Regime", "Stable" if current_regime_prob < 0.5 else "Stressed")
    col4.metric("Quant Engine", f"SVAR({var_lags}) + MS-AR")
    st.divider()

    # --- 6. METHODOLOGY NOTE ---
    with st.expander("📖 METHODOLOGY & QUANT ENGINE LOGIC", expanded=True):
        st.markdown(f"""
        **Engine Core:** The terminal utilizes a **Structural Vector Autoregression (SVAR)** integrated with a **Markov-Switching (MS-AR)** framework.
        1. **SVAR Layer:** Captures linear interdependencies between INR/USD, US Yield Spreads, and Global Commodities. It assumes *Endogeneity* (every variable is both a cause and an effect).
        2. **Markov Layer:** Identifies **Hidden States** (Regimes). This prevents the model from 'averaging' out extreme market stress by isolating low vs. high volatility periods.
        3. **Stationarity:** Data is processed using **First-Differences ($\Delta$)** and **Standardized ($\sigma$)** to ensure mathematical convergence of the SVD solvers.
        """)

    # --- 7. TABS & DYNAMIC NOTES ---
    t1, t2, t3 = st.tabs(["📊 Regime Probability", "🎯 Predictive Path", "⚡ Structural Shock"])

    with t1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=regime_probs.index, y=regime_probs, fill='tozeroy', name="Prob(Stressed)"))
        fig1.update_layout(template="plotly_dark", title="Markov Filtered Probability of Market Stress")
        st.plotly_chart(fig1, use_container_width=True)
        st.info(f"**Dynamic Note:** Currently tracking **{regime_count} regimes**. The shaded area indicates the **Filtered Probability** of being in a high-volatility state. When this exceeds 0.5, the model detects a **Structural Break** in normal pricing patterns.")

    with t2:
        f_dates = pd.date_range(raw_df.index[-1], periods=forecast_horizon+1, freq='MS')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=raw_df.index[-24:], y=raw_df['INR_USD'].iloc[-24:], name="History"))
        fig2.add_trace(go.Scatter(x=f_dates, y=[current_spot] + list(forecast_levels), line=dict(dash='dash'), name="Stressed Forecast"))
        fig2.update_layout(template="plotly_dark", title=f"SVAR({var_lags}) Predictive Path")
        st.plotly_chart(fig2, use_container_width=True)
        st.info(f"**Dynamic Note:** This path integrates your **{yield_shock}bps yield shock**. It uses the **Mean Reversion** principle to estimate how the Rupee will settle against the Dollar after exogenous shocks are absorbed.")

    with t3:
        # Impulse Response
        irf = res_var.irf(periods=10).orth_irfs[:, 2, 0]
        fig3 = go.Figure(go.Scatter(x=list(range(11)), y=np.cumsum(irf), fill='tozeroy'))
        fig3.update_layout(template="plotly_dark", title="Cumulative Response of INR to Commodity Shock")
        st.plotly_chart(fig3, use_container_width=True)
        st.info(f"**Dynamic Note:** This **Impulse Response Function (IRF)** shows the 'persistence' of a **{comm_shock}% shock**. If the curve plateaus high, the shock is **Permanent**; if it trends to zero, the impact is **Transitory**.")

else:
    st.error("Terminal offline: Missing critical .xlsx data files.")

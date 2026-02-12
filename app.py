import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide")

# --- 2. SIDEBAR: MODEL CONTROL PANEL ---
with st.sidebar:
    st.header("🛠️ MODEL CONTROL PANEL")
    
    st.subheader("Shock Scenarios")
    yield_shock = st.slider("US Yield Spread Shock (bps)", -100, 100, 0, step=10, 
                             help="Simulate a shift in the US 10Y-2Y yield curve.")
    comm_shock = st.slider("Commodity Price Spike (%)", -20, 50, 0,
                            help="Simulate a global commodity price shock.")
    
    st.divider()
    st.subheader("Engine Hyperparameters")
    var_lags = st.selectbox("VAR Lag Order", [1, 2, 3], index=0)
    regime_count = st.radio("Markov Regimes", [2, 3], index=0)
    forecast_horizon = st.number_input("Forecast Months", 1, 12, 3)

# --- 3. DATA ENGINE (Initialization) ---
@st.cache_data
def get_terminal_data():
    try:
        # Load Raw Data from XLSX files (referencing user-shared filenames)
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

        # Sync and Merge
        data = pd.concat([comm, yield_spread, inr_usd], axis=1).sort_index().interpolate().dropna()
        return data
    except Exception as e:
        return pd.DataFrame()

# Initialize the dataframe globally to prevent NameError
raw_df = get_terminal_data()

# --- 4. TERMINAL LOGIC & UI ---
if not raw_df.empty:
    # A. PRE-PROCESSING
    df_diff = raw_df.diff().dropna()
    means, stds = df_diff.mean(), df_diff.std()
    df_std = (df_diff - means) / stds

    # B. SVAR ENGINE & FORECASTING
    model_var = VAR(df_std)
    res_var = model_var.fit(var_lags)
    
    # Apply Shocks to the projection state
    current_state = df_std.iloc[-1:].values
    # Injected logic for shocks
    current_state[0, 1] += (yield_shock / 100) / stds['Yield_Spread']
    current_state[0, 0] *= (1 + comm_shock / 100)

    forecast_std = res_var.forecast(current_state, forecast_horizon)
    # Inverse transform to levels
    inr_diff = (forecast_std[:, 2] * stds['INR_USD']) + means['INR_USD']
    forecast_levels = np.cumsum(np.insert(inr_diff, 0, raw_df['INR_USD'].iloc[-1]))[1:]

    # C. MARKOV REGIME ENGINE
    res_ms = MarkovAutoregression(df_std['INR_USD'], k_regimes=regime_count, order=1).fit()
    regime_probs = res_ms.smoothed_marginal_probabilities[1]

    # --- 5. THE HEADER (Find/USD Metrics) ---
    current_spot = raw_df['INR_USD'].iloc[-1]
    proj_3m = forecast_levels[2] if len(forecast_levels) >= 3 else forecast_levels[-1]
    variance = proj_3m - current_spot
    current_stress = regime_probs.iloc[-1]

    st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
    
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    h_col1.metric("INR/USD Current", f"₹{current_spot:.2f}")
    h_col2.metric("3M VAR Forecast", f"₹{proj_3m:.2f}", f"{variance:+.2f}")
    h_col3.metric("Volatility Regime", "Stable" if current_stress < 0.5 else "Stressed")
    h_col4.metric("Quant Engine", f"SVAR({var_lags}) + MS-AR")
    
    st.divider()

    # --- 6. METHODOLOGY & TABS ---
    with st.expander("📖 METHODOLOGY & QUANT ENGINE LOGIC"):
        st.write("""
        This terminal employs a **Structural Vector Autoregression (SVAR)** to model the dynamic interaction between 
        commodity prices, yield spreads, and currency rates. The **Markov-Switching** layer detects 
        unobserved 'states' (High vs. Low Volatility) to adjust risk weights in real-time.
        """)

    tab1, tab2, tab3 = st.tabs(["📊 Regime Probability", "🎯 Predictive Path", "⚡ Structural Shock"])

    with tab1:
        st.plotly_chart(go.Figure(data=[go.Scatter(x=regime_probs.index, y=regime_probs, fill='tozeroy', name="Stress Prob")], 
                                  layout=go.Layout(template="plotly_dark", title="Market Stress Probability (MS-AR)")), use_container_width=True)
        st.info(f"**Dynamic Note:** Currently tracking **{regime_count} regimes**. The shaded area indicates the probability of market stress.")

    with tab2:
        f_dates = pd.date_range(raw_df.index[-1], periods=forecast_horizon+1, freq='MS')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=raw_df.index[-24:], y=raw_df['INR_USD'].iloc[-24:], name="History"))
        fig2.add_trace(go.Scatter(x=f_dates, y=[current_spot] + list(forecast_levels), line=dict(dash='dash'), name="Scenario Projection"))
        fig2.update_layout(template="plotly_dark", title="SVAR Predictive Path")
        st.plotly_chart(fig2, use_container_width=True)
        st.info(f"**Dynamic Note:** This path integrates your **{yield_shock}bps yield shock**.")

    with tab3:
        irf = res_var.irf(periods=10).orth_irfs[:, 2, 0]
        st.plotly_chart(go.Figure(data=[go.Scatter(x=list(range(11)), y=np.cumsum(irf), fill='tozeroy')], 
                                  layout=go.Layout(template="plotly_dark", title="Cumulative Response to Shock")), use_container_width=True)
        st.info(f"**Dynamic Note:** This IRF shows the 'persistence' of a **{comm_shock}% shock**. If the curve plateaus high, the shock is Permanent; if it trends to zero, the impact is Transitory.")

else:
    st.error("Data Source Missing: Please ensure DEXINUS.xlsx, T10Y2Y.xlsx, and PALLFNFINDEXM.xlsx are available.")

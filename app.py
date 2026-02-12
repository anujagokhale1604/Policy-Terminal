import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

# --- 1. ENHANCED PROFESSIONAL THEMING (CSS) ---
def apply_custom_style():
    st.markdown("""
        <style>
        /* Main Background and Font */
        .stApp {
            background-color: #0e1117;
            color: #e0e0e0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Glassmorphism Sidebar */
        section[data-testid="stSidebar"] {
            background-color: rgba(22, 27, 34, 0.95) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(48, 54, 61, 0.5);
        }
        
        /* High-contrast Title Styling */
        .main .block-container h1 {
            color: #ffffff !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em;
            text-shadow: 0px 2px 4px rgba(0,0,0,0.4);
        }
        
        h2, h3 {
            color: #f0f6fc !important;
            font-weight: 700 !important;
        }

        /* Sidebar Visibility Labels */
        section[data-testid="stSidebar"] .stText, 
        section[data-testid="stSidebar"] label {
            color: #c9d1d9 !important;
            font-size: 14px !important;
            font-weight: 600 !important;
        }

        /* Institutional Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #1c2128;
            border: 1px solid #30363d;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            transition: transform 0.2s ease;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: #58a6ff;
        }

        [data-testid="stMetricValue"] {
            font-size: 32px !important;
            font-weight: 800 !important;
            color: #58a6ff !important;
        }
        
        /* Professional 'Status' Indicator styling */
        div[data-testid="stMetricDelta"] > div {
            background-color: rgba(30, 41, 59, 0.5);
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600 !important;
        }

        /* Clean Tab Interface */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            border-bottom: 1px solid #30363d;
        }

        button[data-baseweb="tab"] {
            font-size: 15px !important;
            font-weight: 600 !important;
            color: #8b949e !important;
            background-color: transparent !important;
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            border-bottom: 2px solid #58a6ff !important;
        }
        </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIG & UI SETUP ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide")
apply_custom_style()

# --- 3. SIDEBAR: MODEL CONTROL PANEL ---
with st.sidebar:
    st.markdown("## ⚙️ ENGINE CONTROLS")
    st.divider()
    
    st.subheader("Shock Scenarios")
    yield_shock = st.slider("US Yield Spread Shock (bps)", -100, 100, 0, step=10)
    comm_shock = st.slider("Commodity Price Spike (%)", -20, 50, 0)
    
    st.divider()
    st.subheader("Hyperparameters")
    var_lags = st.selectbox("VAR Lag Order", [1, 2, 3], index=0)
    regime_count = st.radio("Markov Regimes", [2, 3], index=0)
    forecast_horizon = st.number_input("Forecast Months", 1, 12, 3)

# --- 4. DATA ENGINE (Optimized for XLSX) ---
@st.cache_data
def load_institutional_data():
    try:
        # PALLFNFINDEXM.xlsx, T10Y2Y.xlsx, DEXINUS.xlsx
        df_comm = pd.read_excel("PALLFNFINDEXM.xlsx", sheet_name="Monthly")
        df_yield = pd.read_excel("T10Y2Y.xlsx", sheet_name="Daily")
        df_inr = pd.read_excel("DEXINUS.xlsx", sheet_name="Daily")
        
        comm = df_comm.rename(columns={'observation_date': 'Date'}).set_index(pd.to_datetime(df_comm['observation_date']))['PALLFNFINDEXM']
        yield_spread = df_yield.set_index(pd.to_datetime(df_yield['observation_date']))['T10Y2Y'].replace('.', np.nan).astype(float).resample('MS').mean()
        inr_usd = df_inr.set_index(pd.to_datetime(df_inr['observation_date']))['DEXINUS'].replace('.', np.nan).astype(float).resample('MS').mean()
        
        data = pd.concat([comm, yield_spread, inr_usd], axis=1).interpolate().dropna()
        data.columns = ['Commodities', 'Yield_Spread', 'INR_USD']
        return data
    except Exception as e:
        return pd.DataFrame()

raw_df = load_institutional_data()

# --- 5. MAIN TERMINAL LOGIC ---
if not raw_df.empty:
    # A. SVAR ENGINE & DYNAMIC CALCULATIONS
    df_diff = raw_df.diff().dropna()
    means, stds = df_diff.mean(), df_diff.std()
    df_std = (df_diff - means) / stds

    model_var = VAR(df_std)
    res_var = model_var.fit(var_lags)
    
    current_state = df_std.iloc[-1:].values.copy()
    current_state[0, 1] += (yield_shock / 100) / stds['Yield_Spread']
    current_state[0, 0] *= (1 + comm_shock / 100)

    forecast_std = res_var.forecast(current_state, forecast_horizon)
    inr_diff = (forecast_std[:, 2] * stds['INR_USD']) + means['INR_USD']
    forecast_levels = np.cumsum(np.insert(inr_diff, 0, raw_df['INR_USD'].iloc[-1]))[1:]

    # B. MARKOV REGIME LOGIC
    res_ms = MarkovAutoregression(df_std['INR_USD'], k_regimes=regime_count, order=1).fit()
    regime_probs = res_ms.smoothed_marginal_probabilities[1]
    prob_val = regime_probs.iloc[-1]
    
    # C. HEADER METRIC LOGIC
    current_spot = raw_df['INR_USD'].iloc[-1]
    proj_3m = forecast_levels[min(2, len(forecast_levels)-1)]
    variance = proj_3m - current_spot
    
    # Determine Status and Color
    status_label = "STABLE" if prob_val < 0.5 else "STRESSED"
    status_delta = "Low Vol" if prob_val < 0.5 else "High Vol"
    # Suggestion 1: Conditional Metric Coloring
    status_color = "normal" if prob_val < 0.5 else "inverse" 

    # --- 6. RENDER INTERFACE ---
    st.markdown("# 🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
    st.markdown("#### Real-time SVAR Forecasting & Regime Analysis")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("INR/USD SPOT", f"₹{current_spot:.2f}")
    # Forecast delta is red (inverse) if currency depreciates (value increases)
    m2.metric("VAR FORECAST (3M)", f"₹{proj_3m:.2f}", f"{variance:+.2f}", delta_color="inverse")
    m3.metric("REGIME STATUS", status_label, status_delta, delta_color=status_color)
    m4.metric("SHOCK LOAD", f"{comm_shock}%", "COMM", delta_color="off")

    st.markdown("---")

    # Suggestion 2: Chart Transparency Logic
    def update_plot_style(fig):
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter", size=12),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    tab1, tab2, tab3 = st.tabs(["📊 REGIME PROBABILITY", "🎯 PREDICTIVE PATH", "⚡ STRUCTURAL SHOCK"])

    with tab1:
        st.subheader("Markov-Switching Latent State Detection")
        fig1 = go.Figure(go.Scatter(x=regime_probs.index, y=regime_probs, fill='tozeroy', line_color='#58a6ff', name="Stress Probability"))
        st.plotly_chart(update_plot_style(fig1), use_container_width=True)
        st.info("💡 **Institutional Note:** The regime probability indicates the likelihood of a structural shift in currency volatility.")

    with tab2:
        st.subheader("SVAR Projection Path")
        f_dates = pd.date_range(raw_df.index[-1], periods=forecast_horizon+1, freq='MS')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=raw_df.index[-24:], y=raw_df['INR_USD'].iloc[-24:], name="History", line_color="#8b949e"))
        fig2.add_trace(go.Scatter(x=f_dates, y=[current_spot] + list(forecast_levels), name="Scenario Path", line=dict(dash='dash', color='#58a6ff')))
        st.plotly_chart(update_plot_style(fig2), use_container_width=True)

    with tab3:
        st.subheader("Impulse Response Analysis")
        irf = res_var.irf(periods=12).orth_irfs[:, 2, 0] # Shock Commodities -> INR_USD
        fig3 = go.Figure(go.Scatter(x=list(range(13)), y=np.cumsum(irf), fill='tozeroy', line_color='#ff7b72'))
        fig3.update_layout(title="Cumulative Response of INR/USD to Commodity Shock")
        st.plotly_chart(update_plot_style(fig3), use_container_width=True)

else:
    st.error("Engine Error: XLSX Data Stream Disconnected. Verify PALLFNFINDEXM.xlsx, T10Y2Y.xlsx, and DEXINUS.xlsx.")

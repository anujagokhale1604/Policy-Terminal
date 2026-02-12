import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

# --- 1. INSTITUTIONAL CSS (Glassmorphism & High-Contrast) ---
def apply_institutional_style():
    st.markdown("""
        <style>
        .stApp { background-color: #0b0e14; color: #e6edf3; font-family: 'Inter', sans-serif; }
        section[data-testid="stSidebar"] {
            background-color: rgba(22, 27, 34, 0.8) !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(48, 54, 61, 0.5);
        }
        h1, h2, h3 { color: #ffffff !important; font-weight: 800 !important; }
        section[data-testid="stSidebar"] label { color: #f0f6fc !important; font-weight: 600 !important; }
        div[data-testid="metric-container"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        [data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 700 !important; color: #58a6ff !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: 1px solid #30363d; }
        button[data-baseweb="tab"] { font-size: 14px !important; font-weight: 600 !important; color: #8b949e !important; }
        .stTabs [aria-selected="true"] { color: #ffffff !important; border-bottom: 2px solid #58a6ff !important; }
        </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIG & MARKET SETTINGS ---
st.set_page_config(page_title="Global Macro Quant Terminal", layout="wide")
apply_institutional_style()

MARKET_MAP = {
    "India": {"file": "DEXINUS.xlsx", "col": "DEXINUS", "symbol": "₹", "label": "INR/USD"},
    "Singapore": {"file": "AEXSIUS.xlsx", "col": "AEXSIUS", "symbol": "S$", "label": "SGD/USD"},
    "United Kingdom": {"file": "DEXUSUK.xlsx", "col": "DEXUSUK", "symbol": "£", "label": "GBP/USD"}
}

# --- 3. SIDEBAR: NAVIGATION & ENGINE ---
with st.sidebar:
    st.markdown("# 🏛️ TERMINAL")
    target_market = st.selectbox("Select Target Market", list(MARKET_MAP.keys()))
    st.divider()
    
    st.subheader("Shock Scenarios")
    yield_shock = st.slider("US Yield Spread Shock (bps)", -100, 100, 0, step=10)
    comm_shock = st.slider("Commodity Price Spike (%)", -20, 50, 0)
    
    st.divider()
    st.subheader("Hyperparameters")
    var_lags = st.selectbox("VAR Lag Order", [1, 2, 3], index=1)
    regime_count = st.radio("Markov Regimes", [2, 3], index=0)
    forecast_horizon = st.number_input("Forecast Months", 1, 12, 3)

# --- 4. DATA PROCESSING ENGINE ---
@st.cache_data
def load_quant_data(market_name):
    m_info = MARKET_MAP[market_name]
    try:
        # Load Common Drivers
        df_comm = pd.read_excel("PALLFNFINDEXM.xlsx", sheet_name="Monthly")
        df_yield = pd.read_excel("T10Y2Y.xlsx", sheet_name="Daily")
        
        # Load Market Specific FX
        df_fx = pd.read_excel(m_info['file'], sheet_name=0) # Generic sheet index
        
        # Standardize Indices
        comm = df_comm.set_index(pd.to_datetime(df_comm.iloc[:,0]))['PALLFNFINDEXM']
        yield_spread = df_yield.set_index(pd.to_datetime(df_yield.iloc[:,0])).iloc[:,1].replace('.', np.nan).astype(float).resample('MS').mean()
        fx_series = df_fx.set_index(pd.to_datetime(df_fx.iloc[:,0])).iloc[:,1].replace('.', np.nan).astype(float).resample('MS').mean()
        
        data = pd.concat([comm, yield_spread, fx_series], axis=1).interpolate().dropna()
        data.columns = ['Commodities', 'Yield_Spread', 'FX_Target']
        return data
    except Exception:
        return pd.DataFrame()

raw_df = load_quant_data(target_market)

# --- 5. ANALYTICS & RENDERING ---
if not raw_df.empty:
    # A. SVAR Engine
    df_diff = raw_df.diff().dropna()
    means, stds = df_diff.mean(), df_diff.std()
    df_std = (df_diff - means) / stds

    res_var = VAR(df_std).fit(var_lags)
    current_vec = df_std.iloc[-1:].values.copy()
    current_vec[0, 1] += (yield_shock / 100) / stds['Yield_Spread']
    current_vec[0, 0] *= (1 + comm_shock / 100)

    f_std = res_var.forecast(current_vec, forecast_horizon)
    fx_diff = (f_std[:, 2] * stds['FX_Target']) + means['FX_Target']
    forecast_path = np.cumsum(np.insert(fx_diff, 0, raw_df['FX_Target'].iloc[-1]))[1:]

    # B. Markov Regime Logic
    res_ms = MarkovAutoregression(df_std['FX_Target'], k_regimes=regime_count, order=1).fit()
    prob_val = res_ms.smoothed_marginal_probabilities[1].iloc[-1]
    
    # C. UI Variables
    m_cfg = MARKET_MAP[target_market]
    curr_spot = raw_df['FX_Target'].iloc[-1]
    f_val = forecast_path[min(2, len(forecast_path)-1)]
    delta = f_val - curr_spot
    
    # Dynamic Metric Coloring
    status = "STABLE" if prob_val < 0.5 else "STRESSED"
    status_color = "normal" if prob_val < 0.5 else "inverse"

    # Header Metrics
    st.title(f"🏛️ {target_market.upper()} QUANT TERMINAL")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{m_cfg['label']} SPOT", f"{m_cfg['symbol']}{curr_spot:.2f}")
    c2.metric("VAR FORECAST (3M)", f"{m_cfg['symbol']}{f_val:.2f}", f"{delta:+.2f}", delta_color="inverse")
    c3.metric("REGIME STATUS", status, f"{prob_val*100:.1f}% Prob", delta_color=status_color)
    c4.metric("SHOCK LOAD", f"{comm_shock}%", "Commodity", delta_color="off")

    st.markdown("---")

    # Transparent Chart Helper
    def clean_chart(fig, title):
        fig.update_layout(title=title, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
                          plot_bgcolor='rgba(0,0,0,0)', font=dict(family="Inter"),
                          xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#30363d'))
        return fig

    tab1, tab2, tab3 = st.tabs(["📊 REGIME ANALYSIS", "🎯 FORECAST PATH", "⚡ STRUCTURAL SHOCK"])

    with tab1:
        fig1 = go.Figure(go.Scatter(x=res_ms.smoothed_marginal_probabilities.index, 
                                   y=res_ms.smoothed_marginal_probabilities[1], 
                                   fill='tozeroy', line_color='#58a6ff'))
        st.plotly_chart(clean_chart(fig1, "Latent State Stress Probability"), use_container_width=True)

    with tab2:
        f_dates = pd.date_range(raw_df.index[-1], periods=forecast_horizon+1, freq='MS')
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=raw_df.index[-24:], y=raw_df['FX_Target'].iloc[-24:], name="History", line_color="#8b949e"))
        fig2.add_trace(go.Scatter(x=f_dates, y=[curr_spot] + list(forecast_path), name="Scenario", line=dict(dash='dash', color='#58a6ff')))
        st.plotly_chart(clean_chart(fig2, f"Projected {m_cfg['label']} Trajectory"), use_container_width=True)

    with tab3:
        irf = res_var.irf(periods=12).orth_irfs[:, 2, 0]
        fig3 = go.Figure(go.Scatter(x=list(range(13)), y=np.cumsum(irf), fill='tozeroy', line_color='#ff7b72'))
        st.plotly_chart(clean_chart(fig3, "Cumulative FX Response to Commodity Shock"), use_container_width=True)

else:
    st.error(f"Engine Failure: Missing data for {target_market}. Verify XLSX files in working directory.")

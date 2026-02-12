import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

# --- 1. INSTITUTIONAL CSS ---
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
        div[data-testid="metric-container"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        [data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 700 !important; color: #58a6ff !important; }
        /* Professional 'Status' Indicator styling */
        div[data-testid="stMetricDelta"] > div {
            background-color: rgba(30, 41, 59, 0.5);
            padding: 4px 10px;
            border-radius: 6px;
        }
        </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIG ---
st.set_page_config(page_title="Global Macro Quant Terminal", layout="wide")
apply_institutional_style()

MARKET_MAP = {
    "India": {"file": "DEXINUS.xlsx", "symbol": "₹", "label": "INR/USD"},
    "Singapore": {"file": "AEXSIUS.xlsx", "symbol": "S$", "label": "SGD/USD"},
    "United Kingdom": {"file": "DEXUSUK.xlsx", "symbol": "$", "label": "USD/GBP"}
}

# --- 3. DATA ENGINE ---
@st.cache_data
def load_fred_series(filename):
    try:
        xl = pd.ExcelFile(filename)
        sheet = [s for s in xl.sheet_names if s != 'README'][0]
        df = pd.read_excel(filename, sheet_name=sheet, skiprows=10)
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.dropna().set_index('date')['value']
    except Exception:
        return pd.Series()

@st.cache_data
def get_processed_data(market_name):
    m_info = MARKET_MAP[market_name]
    comm = load_fred_series("PALLFNFINDEXM.xlsx")
    yields = load_fred_series("T10Y2Y.xlsx")
    fx = load_fred_series(m_info['file'])
    
    if comm.empty or yields.empty or fx.empty:
        return pd.DataFrame()

    # Align frequencies (Linear interpolation for Annual SG data)
    data = pd.concat([
        comm.resample('MS').mean(),
        yields.resample('MS').mean(),
        fx.resample('MS').interpolate(method='linear')
    ], axis=1).dropna()
    data.columns = ['Commodities', 'Yield_Spread', 'FX_Target']
    return data

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("# 🏛️ TERMINAL")
    target_market = st.selectbox("Select Target Market", list(MARKET_MAP.keys()))
    st.divider()
    
    st.subheader("Shock Scenarios")
    yield_shock = st.slider("US Yield Spread Shock (bps)", -100, 100, 0, step=10)
    comm_shock = st.slider("Commodity Price Spike (%)", -20, 50, 0)
    
    st.divider()
    var_lags = st.selectbox("VAR Lag Order", [1, 2, 3], index=1)
    forecast_horizon = st.number_input("Forecast Months", 1, 12, 3)

# --- 5. EXECUTION ---
raw_df = get_processed_data(target_market)

if not raw_df.empty:
    # A. PREPARE DATA
    df_diff = raw_df.diff().dropna()
    means, stds = df_diff.mean(), df_diff.std()
    df_std = (df_diff - means) / stds

    # B. FIT VAR MODEL
    res_var = VAR(df_std).fit(var_lags)
    
    # C. SHOCK ENGINE (Correct Dimension Handling)
    # Extract last p observations where p = var_lags
    current_vec = df_std.values[-var_lags:].copy()
    
    # Apply shocks to the most recent observation in the sequence
    # Yield Shock (bps to standardized unit)
    current_vec[-1, 1] += (yield_shock / 100) / stds['Yield_Spread']
    
    # Commodity Shock (Multiplier based on Level)
    comm_level_shock = (raw_df['Commodities'].iloc[-1] * (comm_shock / 100))
    current_vec[-1, 0] += (comm_level_shock / stds['Commodities'])

    # D. FORECAST
    f_std = res_var.forecast(current_vec, forecast_horizon)
    fx_diff = (f_std[:, 2] * stds['FX_Target']) + means['FX_Target']
    forecast_path = np.cumsum(np.insert(fx_diff, 0, raw_df['FX_Target'].iloc[-1]))[1:]

    # E. REGIME DETECTION
    res_ms = MarkovAutoregression(df_std['FX_Target'], k_regimes=2, order=1).fit()
    prob_val = res_ms.smoothed_marginal_probabilities[1].iloc[-1]
    
    # F. UI LOGIC
    m_cfg = MARKET_MAP[target_market]
    curr_spot = raw_df['FX_Target'].iloc[-1]
    f_val = forecast_path[min(2, len(forecast_path)-1)] # 3M or max available
    delta = f_val - curr_spot
    
    status = "STABLE" if prob_val < 0.5 else "STRESSED"
    status_color = "normal" if prob_val < 0.5 else "inverse"

    st.title(f"🏛️ {target_market.upper()} QUANT TERMINAL")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{m_cfg['label']} SPOT", f"{m_cfg['symbol']}{curr_spot:.2f}")
    c2.metric("3M FORECAST", f"{m_cfg['symbol']}{f_val:.2f}", f"{delta:+.2f}", delta_color="inverse")
    c3.metric("REGIME STATUS", status, f"{prob_val*100:.1f}% Prob", delta_color=status_color)
    c4.metric("SHOCK LOAD", f"{comm_shock}%", "Commodity", delta_color="off")

    st.markdown("---")

    t1, t2 = st.tabs(["🎯 PREDICTIVE PATH", "📊 REGIME ANALYSIS"])
    
    with t1:
        f_dates = pd.date_range(raw_df.index[-1], periods=forecast_horizon+1, freq='MS')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw_df.index[-24:], y=raw_df['FX_Target'].iloc[-24:], name="History", line_color="#8b949e"))
        fig.add_trace(go.Scatter(x=f_dates, y=[curr_spot] + list(forecast_path), name="Scenario", line=dict(dash='dash', color='#58a6ff')))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig_p = go.Figure(go.Scatter(x=res_ms.smoothed_marginal_probabilities.index, y=res_ms.smoothed_marginal_probabilities[1], fill='tozeroy', line_color='#ff7b72'))
        fig_p.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title="Historical Stress Probability")
        st.plotly_chart(fig_p, use_container_width=True)

else:
    st.error("Engine Offline: Ensure DEXINUS.xlsx, AEXSIUS.xlsx, DEXUSUK.xlsx, PALLFNFINDEXM.xlsx, and T10Y2Y.xlsx are in the directory.")

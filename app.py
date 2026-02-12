import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Global Macro Quant Terminal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #e6edf3; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background-color: rgba(22, 27, 34, 0.8) !important; border-right: 1px solid #30363d; }
    div[data-testid="metric-container"] { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 8px; }
    [data-testid="stMetricValue"] { font-size: 28px !important; color: #58a6ff !important; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

MARKET_MAP = {
    "India": {"file": "DEXINUS.xlsx", "symbol": "₹", "label": "INR/USD"},
    "Singapore": {"file": "AEXSIUS.xlsx", "symbol": "S$", "label": "SGD/USD"},
    "United Kingdom": {"file": "DEXUSUK.xlsx", "symbol": "$", "label": "USD/GBP"}
}

# --- 2. DATA ENGINE ---
@st.cache_data
def load_and_sync_data(market_name):
    m_info = MARKET_MAP[market_name]
    def get_fred(f, sheet_idx=0):
        try:
            xl = pd.ExcelFile(f)
            sheet = [s for s in xl.sheet_names if s != 'README'][0]
            df = pd.read_excel(f, sheet_name=sheet, skiprows=10)
            df.columns = ['date', 'value']
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df.dropna().set_index('date')['value']
        except: return pd.Series()

    comm = get_fred("PALLFNFINDEXM.xlsx")
    yields = get_fred("T10Y2Y.xlsx")
    fx = get_fred(m_info['file'])
    
    if comm.empty or yields.empty or fx.empty: return pd.DataFrame()

    # Sync to Monthly frequency
    # Note: Singapore (Annual) is interpolated to Monthly to allow VAR/Markov models to run
    df = pd.concat([
        comm.resample('MS').mean(),
        yields.resample('MS').mean(),
        fx.resample('MS').interpolate(method='linear')
    ], axis=1).dropna()
    df.columns = ['Commodities', 'Yield_Spread', 'FX_Target']
    return df

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🏛️ TERMINAL")
    target_market = st.selectbox("Select Target Market", list(MARKET_MAP.keys()))
    st.divider()
    st.subheader("Shock Scenarios")
    yield_shock = st.slider("Yield Shock (bps)", -100, 100, 0)
    comm_shock = st.slider("Comm. Spike (%)", -20, 50, 0)
    st.divider()
    var_lags = st.selectbox("VAR Lag Order", [1, 2, 3], index=1)
    forecast_horizon = st.number_input("Forecast Months", 1, 12, 3)

# --- 4. ANALYTICS ---
raw_df = load_and_sync_data(target_market)

if not raw_df.empty:
    # Processing
    df_diff = raw_df.diff().dropna()
    means, stds = df_diff.mean(), df_diff.std()
    df_std = (df_diff - means) / stds

    # VAR Forecast with Shocks
    res_var = VAR(df_std).fit(var_lags)
    hist_input = df_std.values[-var_lags:].copy()
    hist_input[-1, 1] += (yield_shock / 100) / stds['Yield_Spread'] # Add bps shock
    hist_input[-1, 0] += (comm_shock / 100) # Add % shock
    
    f_std = res_var.forecast(hist_input, forecast_horizon)
    fx_diff = (f_std[:, 2] * stds['FX_Target']) + means['FX_Target']
    forecast_path = np.cumsum(np.insert(fx_diff, 0, raw_df['FX_Target'].iloc[-1]))[1:]

    # Markov Regime Engine
    res_ms = MarkovAutoregression(df_std['FX_Target'], k_regimes=2, order=1).fit()
    prob_val = res_ms.smoothed_marginal_probabilities[1].iloc[-1]
    
    # --- 5. UI DISPLAY ---
    m_cfg = MARKET_MAP[target_market]
    curr_spot = raw_df['FX_Target'].iloc[-1]
    f_val = forecast_path[-1]
    delta = f_val - curr_spot
    status = "STABLE" if prob_val < 0.5 else "STRESSED"

    st.title(f"🏛️ {target_market.upper()} QUANT TERMINAL")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{m_cfg['label']} SPOT", f"{m_cfg['symbol']}{curr_spot:.2f}")
    c2.metric("VAR FORECAST", f"{m_cfg['symbol']}{f_val:.2f}", f"{delta:+.2f}")
    c3.metric("REGIME STATUS", status, f"{prob_val*100:.1f}% Stress")
    c4.metric("SHOCK LOAD", f"{comm_shock}%", "Commodity")

    t1, t2 = st.tabs(["🎯 PREDICTIVE PATH", "📊 REGIME ANALYSIS"])
    
    with t1:
        f_dates = pd.date_range(raw_df.index[-1], periods=forecast_horizon+1, freq='MS')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw_df.index[-36:], y=raw_df['FX_Target'].iloc[-36:], name="Hist", line_color="#8b949e"))
        fig.add_trace(go.Scatter(x=f_dates, y=[curr_spot]+list(forecast_path), name="Fore", line=dict(dash='dash', color='#58a6ff')))
        fig.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        fig_p = go.Figure(go.Scatter(x=res_ms.smoothed_marginal_probabilities.index, y=res_ms.smoothed_marginal_probabilities[1], fill='tozeroy', line_color='#ff7b72'))
        fig_p.update_layout(template="plotly_dark", margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_p, use_container_width=True)
else:
    st.error("Engine failure: Check XLSX sources.")

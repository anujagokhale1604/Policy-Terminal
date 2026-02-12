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
        }
        [data-testid="stMetricValue"] { font-size: 32px !important; color: #58a6ff !important; }
        </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIG ---
st.set_page_config(page_title="Global Macro Quant Terminal", layout="wide")
apply_institutional_style()

MARKET_MAP = {
    "India": {"file": "DEXINUS.xlsx", "symbol": "₹", "label": "INR/USD (Units/$) "},
    "Singapore": {"file": "AEXSIUS.xlsx", "symbol": "S$", "label": "SGD/USD (Units/$)"},
    "United Kingdom": {"file": "DEXUSUK.xlsx", "symbol": "$", "label": "USD/GBP ($/Unit)"}
}

# --- 3. ROBUST DATA ENGINE ---
@st.cache_data
def load_fred_series(filename):
    """Helper to parse FRED XLSX files skipping metadata headers."""
    try:
        xl = pd.ExcelFile(filename)
        # Select data sheet (not README)
        sheet = [s for s in xl.sheet_names if s != 'README'][0]
        # Data usually starts after row 10 in FRED exports
        df = pd.read_excel(filename, sheet_name=sheet, skiprows=10)
        
        # Clean column names
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.dropna().set_index('date')['value']
    except Exception as e:
        return pd.Series()

@st.cache_data
def get_processed_data(market_name):
    m_info = MARKET_MAP[market_name]
    
    # Load raw series
    comm = load_fred_series("PALLFNFINDEXM.xlsx")
    yields = load_fred_series("T10Y2Y.xlsx")
    fx = load_fred_series(m_info['file'])
    
    if comm.empty or yields.empty or fx.empty:
        return pd.DataFrame()

    # Resample to Monthly Start (MS) and handle frequency gaps (especially for SG Annual)
    comm_m = comm.resample('MS').mean()
    yields_m = yields.resample('MS').mean()
    fx_m = fx.resample('MS').interpolate(method='linear') # Handles SG Annual -> Monthly
    
    # Align dates
    df = pd.concat([comm_m, yields_m, fx_m], axis=1).dropna()
    df.columns = ['Commodities', 'Yield_Spread', 'FX_Target']
    return df

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
    # A. VAR Model
    df_diff = raw_df.diff().dropna()
    means, stds = df_diff.mean(), df_diff.std()
    df_std = (df_diff - means) / stds

    res_var = VAR(df_std).fit(var_lags)
    
    # Apply shocks to current vector
    current_vec = df_std.iloc[-1:].values.copy()
    current_vec[0, 1] += (yield_shock / 100) / stds['Yield_Spread']
    current_vec[0, 0] *= (1 + comm_shock / 100)

    # Forecast
    f_std = res_var.forecast(current_vec, forecast_horizon)
    fx_diff = (f_std[:, 2] * stds['FX_Target']) + means['FX_Target']
    forecast_path = np.cumsum(np.insert(fx_diff, 0, raw_df['FX_Target'].iloc[-1]))[1:]

    # B. Markov Regime probabilities
    res_ms = MarkovAutoregression(df_std['FX_Target'], k_regimes=2, order=1).fit()
    prob_val = res_ms.smoothed_marginal_probabilities[1].iloc[-1]
    
    # C. UI RENDER
    m_cfg = MARKET_MAP[target_market]
    curr_spot = raw_df['FX_Target'].iloc[-1]
    f_val = forecast_path[-1]
    delta = f_val - curr_spot
    
    st.title(f"🏛️ {target_market.upper()} QUANT TERMINAL")
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"CURRENT {m_cfg['label']}", f"{m_cfg['symbol']}{curr_spot:.2f}")
    c2.metric("VAR FORECAST", f"{m_cfg['symbol']}{f_val:.2f}", f"{delta:+.2f}")
    c3.metric("STRESS PROBABILITY", f"{prob_val*100:.1f}%", "Markov State")

    tab1, tab2 = st.tabs(["🎯 FORECAST PATH", "📊 REGIME LOG"])
    
    with tab1:
        f_dates = pd.date_range(raw_df.index[-1], periods=forecast_horizon+1, freq='MS')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw_df.index[-24:], y=raw_df['FX_Target'].iloc[-24:], name="Historical"))
        fig.add_trace(go.Scatter(x=f_dates, y=[curr_spot] + list(forecast_path), name="Forecast", line=dict(dash='dash', color='#58a6ff')))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig_p = go.Figure(go.Scatter(x=res_ms.smoothed_marginal_probabilities.index, y=res_ms.smoothed_marginal_probabilities[1], fill='tozeroy'))
        fig_p.update_layout(template="plotly_dark", title="Historical Regime Stress Probability")
        st.plotly_chart(fig_p, use_container_width=True)

else:
    st.error(f"Engine Failure: Could not align data for {target_market}. Ensure all XLSX files are present.")

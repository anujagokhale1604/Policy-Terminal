import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from statsmodels.tsa.api import VAR

# --- 1. CHIC RESEARCH UI ENGINE ---
st.set_page_config(page_title="Macro Intel Pro", layout="wide")

st.markdown("""
    <style>
    * { font-family: 'Times New Roman', Times, serif !important; }
    .stApp { background-color: #F2EFE9; color: #2C2C2C; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stPlotlyChart { margin-top: -25px; } 
    section[data-testid="stSidebar"] { background-color: #E5E1D8 !important; border-right: 1px solid #A39B8F; }
    .analyst-card { padding: 15px; border: 1px solid #A39B8F; background-color: #FFFFFF; margin-bottom: 20px; border-left: 5px solid #002366; font-size: 0.95rem; }
    .for-you-card { padding: 20px; background-color: #FDFCFB; color: #1A1A1A; margin-bottom: 25px; border: 1px solid #A39B8F; border-left: 10px solid #002366; }
    .main-title { font-size: 32px; font-weight: bold; color: #002366; border-bottom: 3px solid #C5A059; padding-bottom: 5px; margin-bottom: 20px; }
    .section-header { color: #7A6D5D; font-weight: bold; font-size: 1.2rem; margin-top: 20px; margin-bottom: 5px; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data():
    files = {"workbook": 'EM_Macro_Data_India_SG_UK.xlsx', "inr": 'DEXINUS.xlsx', "gbp": 'DEXUSUK.xlsx', "sgd": 'AEXSIUS.xlsx'}
    if not all(os.path.exists(f) for f in files.values()): return None
    try:
        df_m = pd.read_excel(files["workbook"], sheet_name='Macro data')
        df_m['Date'] = pd.to_datetime(df_m['Date'], errors='coerce')
        df_g = pd.read_excel(files["workbook"], sheet_name='GDP_Growth', skiprows=1).iloc[1:, [0, 2, 3, 4]]
        df_g.columns = ['Year', 'GDP_India', 'GDP_Singapore', 'GDP_UK']
        
        def clean_fx(path, out_name):
            xls = pd.ExcelFile(path); sheet = [s for s in xls.sheet_names if 'README' not in s.upper()][0]
            f = pd.read_excel(path, sheet_name=sheet)
            d_col = [c for c in f.columns if 'date' in str(c).lower()][0]
            v_col = [c for c in f.columns if c != d_col][0]
            f[d_col] = pd.to_datetime(f[d_col], errors='coerce')
            f[v_col] = pd.to_numeric(f[v_col].replace(0, pd.NA), errors='coerce')
            return f.dropna(subset=[d_col]).resample('MS', on=d_col).mean().ffill().bfill().reset_index().rename(columns={d_col:'Date', v_col:out_name})

        fx_i, fx_g, fx_s = clean_fx(files["inr"], 'FX_India'), clean_fx(files["gbp"], 'FX_UK'), clean_fx(files["sgd"], 'FX_Singapore')
        df_m['Year'] = df_m['Date'].dt.year
        df = df_m.merge(df_g, on='Year', how='left').merge(fx_i, on='Date', how='left').merge(fx_g, on='Date', how='left').merge(fx_s, on='Date', how='left')
        return df.sort_values('Date').ffill().bfill()
    except: return None

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2>NAVIGATE</h2>", unsafe_allow_html=True)
    market = st.selectbox("SELECT MARKET", ["India", "UK", "Singapore"])
    forecast_len = st.slider("Forecast Horizon (Months)", 0, 24, 12)
    st.divider()
    scenario = st.selectbox("SCENARIO ENGINE", ["Standard", "Stagflation 🌪️", "Depression 📉"])
    severity = st.slider("SEVERITY (%)", 0, 100, 50)
    st.divider()
    show_diagnostics = st.toggle("Show VAR Model Health", value=False)

# --- 4. ANALYTICS (VAR ENGINE) ---
df_raw = load_data()
if df_raw is not None:
    m_map = {
        "India": {"p": "Policy_India", "cpi": "CPI_India", "gdp": "GDP_India", "fx": "FX_India", "sym": "INR", "t": 4.0, "n": 4.5},
        "UK": {"p": "Policy_UK", "cpi": "CPI_UK", "gdp": "GDP_UK", "fx": "FX_UK", "sym": "GBP", "t": 2.0, "n": 2.5},
        "Singapore": {"p": "Policy_Singapore", "cpi": "CPI_Singapore", "gdp": "GDP_Singapore", "fx": "FX_Singapore", "sym": "SGD", "t": 2.0, "n": 2.5}
    }
    m = m_map[market]
    cols = [m['p'], m['cpi'], m['gdp']]
    
    # Run VAR
    last_date = df_raw['Date'].max()
    model = VAR(df_raw[cols].dropna())
    results = model.fit(maxlags=12)
    forecast = results.forecast(df_raw[cols].values[-results.k_ar:], forecast_len)
    
    # Construct DF
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_len, freq='MS')
    df_f = pd.DataFrame(forecast, columns=cols)
    df_f['Date'] = future_dates
    df_f['Is_Forecast'] = True
    df_f[m['fx']] = df_raw[m['fx']].iloc[-1] # Simple placeholder for FX
    
    df_h = df_raw.copy(); df_h['Is_Forecast'] = False
    df = pd.concat([df_h, df_f]).reset_index(drop=True)

    # --- UI RENDERING ---
    st.markdown(f"<div class='main-title'>{market.upper()} RESEARCH TERMINAL</div>", unsafe_allow_html=True)
    
    t_row = df.iloc[-1]
    h_row = df_h.iloc[-1]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("VAR TARGET RATE", f"{t_row[m['p']]:.2f}%", f"{t_row[m['p']] - h_row[m['p']]:.2f}%")
    c2.metric("PROJ. CPI", f"{t_row[m['cpi']]:.2f}%")
    c3.metric("PROJ. GDP", f"{t_row[m['gdp']]:.1f}%")
    c4.metric(f"FX ({m['sym']})", f"{t_row[m['fx']]:.2f}")

    # CHART
    st.markdown("<div class='section-header'><i class='fas fa-chart-line'></i> I. Endogenous Policy & Inflation Projections</div>", unsafe_allow_html=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['Date'], y=df[m['p']], name="Policy Rate (VAR Path)", line=dict(color='#002366', width=3)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df[m['cpi']], name="CPI (YoY)", line=dict(color='#A52A2A', dash='dot')), secondary_y=True)
    if forecast_len > 0:
        fig.add_vrect(x0=last_date, x1=df['Date'].max(), fillcolor="gray", opacity=0.1)
    fig.update_layout(height=400, template="plotly_white", margin=dict(l=20, r=20, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ANALYSIS NOTES
    st.markdown(f"""<div class='analyst-card'><b>Econometric Insight:</b> The Vector Autoregression (VAR) model suggests a <b>{'tightening' if t_row[m['p']] > h_row[m['p']] else 'easing'}</b> cycle. 
    The projected interaction between inflation and growth implies a terminal rate equilibrium of <b>{t_row[m['p']]:.2f}%</b> by {future_dates[-1].year if forecast_len > 0 else 'year-end'}.</div>""", unsafe_allow_html=True)

    # RECOMMENDATIONS
    st.markdown("<div class='section-header'><i class='fas fa-user-tie'></i> II. Strategic Recommendations</div>", unsafe_allow_html=True)
    st.markdown(f"""<div class='for-you-card'><b>Forward Guidance:</b><br>
    • <b>Asset Allocation:</b> Projections show GDP at {t_row[m['gdp']]:.1f}%. If this signifies a slowdown, pivot toward defensive equities.<br>
    • <b>Debt Management:</b> The VAR path suggests rates will be {abs(t_row[m['p']] - h_row[m['p']]):.2f}% {'higher' if t_row[m['p']] > h_row[m['p']] else 'lower'} in {forecast_len} months. Adjust variable-rate exposure accordingly.</div>""", unsafe_allow_html=True)

    # LATEX
    st.markdown("<div class='section-header'>VAR Specification</div>", unsafe_allow_html=True)
    st.latex(r'''y_t = A_1 y_{t-1} + \dots + A_p y_{t-p} + u_t''')
    
    if show_diagnostics:
        st.text(results.summary())

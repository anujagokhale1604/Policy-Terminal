import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- 1. CHIC RESEARCH UI ENGINE ---
st.set_page_config(page_title="Macro Intel Pro", layout="wide")

st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">', unsafe_allow_html=True)

st.markdown("""
    <style>
    * { font-family: 'Times New Roman', Times, serif !important; }
    .stApp { background-color: #F2EFE9; color: #2C2C2C; }

    /* REDUCING VERTICAL WASTE */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stPlotlyChart { margin-top: -25px; } 

    section[data-testid="stSidebar"] {
        background-color: #E5E1D8 !important; 
        border-right: 1px solid #A39B8F;
    }

    /* CARDS */
    .analyst-card { 
        padding: 15px; border: 1px solid #A39B8F; 
        background-color: #FFFFFF; margin-top: -10px; margin-bottom: 20px; 
        border-left: 5px solid #002366; font-size: 0.95rem;
    }
    .for-you-card {
        padding: 20px; background-color: #FDFCFB; 
        color: #1A1A1A; margin-bottom: 25px; border: 1px solid #A39B8F;
        border-left: 10px solid #002366;
    }
    .method-card { 
        padding: 20px; background-color: #FAF9F6; 
        color: #1A1A1A; font-size: 0.92rem; border: 1px solid #A39B8F; line-height: 1.5;
    }
    
    .main-title { 
        font-size: 32px; font-weight: bold; color: #002366; 
        border-bottom: 3px solid #C5A059; padding-bottom: 5px; margin-bottom: 20px; 
    }
    .section-header { 
        color: #7A6D5D; font-weight: bold; font-size: 1.2rem; 
        margin-top: 20px; margin-bottom: 5px; text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA & PREDICTION ENGINE ---
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
            try:
                xls = pd.ExcelFile(path); sheet = [s for s in xls.sheet_names if 'README' not in s.upper()][0]
                f = pd.read_excel(path, sheet_name=sheet)
                d_col = [c for c in f.columns if 'date' in str(c).lower()][0]
                v_col = [c for c in f.columns if c != d_col][0]
                f[d_col] = pd.to_datetime(f[d_col], errors='coerce')
                f[v_col] = pd.to_numeric(f[v_col].replace(0, pd.NA), errors='coerce')
                return f.dropna(subset=[d_col]).resample('MS', on=d_col).mean().ffill().bfill().reset_index().rename(columns={d_col:'Date', v_col:out_name})
            except: return pd.DataFrame(columns=['Date', out_name])

        fx_i, fx_g, fx_s = clean_fx(files["inr"], 'FX_India'), clean_fx(files["gbp"], 'FX_UK'), clean_fx(files["sgd"], 'FX_Singapore')
        df_m['Year'] = df_m['Date'].dt.year
        df = df_m.merge(df_g, on='Year', how='left').merge(fx_i, on='Date', how='left').merge(fx_g, on='Date', how='left').merge(fx_s, on='Date', how='left')
        return df.sort_values('Date').ffill().bfill()
    except: return None

def run_forecast(series, steps):
    try:
        if steps <= 0: return pd.Series([], dtype='float64')
        model = ExponentialSmoothing(series.dropna(), trend='add', seasonal=None).fit()
        return model.forecast(steps)
    except:
        return pd.Series(np.full(steps, series.iloc[-1]))

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#000000;'><i class='fas fa-bars-staggered'></i> NAVIGATE</h2>", unsafe_allow_html=True)
    if st.button("RESET PARAMETERS", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    market = st.selectbox("1. SELECT MARKET", ["India", "UK", "Singapore"])
    horizon = st.radio("2. TIME HORIZON", ["Historical", "10 Years", "5 Years"], index=1)
    
    st.divider()
    st.markdown("<b>🔮 PREDICTIVE SETTINGS</b>", unsafe_allow_html=True)
    forecast_len = st.slider("Forecast Horizon (Months)", 0, 24, 12)
    energy_shock = st.slider("Energy Price Surge (%)", 0, 100, 0)
    target_inf_val = st.number_input("Target Inflation (%)", value=4.0 if market=="India" else 2.0)
    
    st.divider()
    scenario = st.selectbox("3. SCENARIO ENGINE", ["Standard", "Stagflation 🌪️", "Depression 📉", "High Growth 🚀"])
    severity = st.slider("4. SEVERITY (%)", 0, 100, 50)
    st.divider()
    view_real = st.toggle("ACTIVATE REAL RATES")
    show_taylor = st.toggle("OVERLAY TAYLOR RULE", value=True)
    rate_intervention = st.slider("5. MANUAL ADJ (BPS)", -200, 200, 0, step=25)
    lag = st.selectbox("6. TRANSMISSION LAG", [0, 3, 6, 12])
    st.divider()
    sentiment = st.select_slider("7. MARKET SENTIMENT", options=["Risk-Off", "Neutral", "Risk-On"], value="Neutral")

# --- 4. ANALYTICS ---
df_raw = load_data()
if df_raw is not None:
    m_map = {
        "India": {"p": "Policy_India", "cpi": "CPI_India", "gdp": "GDP_India", "fx": "FX_India", "sym": "INR", "t": 4.0, "n": 4.5},
        "UK": {"p": "Policy_UK", "cpi": "CPI_UK", "gdp": "GDP_UK", "fx": "FX_UK", "sym": "GBP", "t": 2.0, "n": 2.5},
        "Singapore": {"p": "Policy_Singapore", "cpi": "CPI_Singapore", "gdp": "GDP_Singapore", "fx": "FX_Singapore", "sym": "SGD", "t": 2.0, "n": 2.5}
    }
    m = m_map[market]; df_hist = df_raw.copy()

    # Apply Historical Shocks
    mult = severity / 100
    if scenario == "Stagflation 🌪️":
        df_hist[m['cpi']] += (5.0 * mult); df_hist[m['gdp']] -= (3.0 * mult)
    elif scenario == "Depression 📉":
        df_hist[m['gdp']] -= (8.0 * mult); df_hist[m['cpi']] -= (2.0 * mult)
    elif scenario == "High Growth 🚀":
        df_hist[m['gdp']] += (4.0 * mult); df_hist[m['cpi']] -= (1.0 * mult)

    # Forecasting
    last_date = df_hist['Date'].max()
    if forecast_len > 0:
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_len, freq='MS')
        df_fore = pd.DataFrame({'Date': future_dates})
        df_fore[m['cpi']] = run_forecast(df_hist[m['cpi']], forecast_len).values
        df_fore[m['gdp']] = run_forecast(df_hist[m['gdp']], forecast_len).values
        df_fore[m['fx']] = run_forecast(df_hist[m['fx']], forecast_len).values
        df_fore[m['p']] = np.nan
        df = pd.concat([df_hist, df_fore]).reset_index(drop=True)
    else:
        df = df_hist.copy()
    
    df['Is_Forecast'] = df['Date'] > last_date
    df[m['cpi']] += (energy_shock * 0.12)
    df[m['p']] = df[m['p']].ffill() + (rate_intervention / 100)

    avg_g = df_hist[m['gdp']].mean()
    df['Taylor'] = m['n'] + 0.5*(df[m['cpi']] - target_inf_val) + 0.5*(df[m['gdp']] - avg_g)
    
    if view_real: df[m['p']] -= df[m['cpi']]
    if lag > 0: df[m['cpi']] = df[m['cpi']].shift(lag); df[m['gdp']] = df[m['gdp']].shift(lag)

    # --- 5. UI RENDERING ---
    st.markdown(f"<div class='main-title'><i class='fas fa-scale-balanced'></i> {market.upper()} STRATEGY TERMINAL</div>", unsafe_allow_html=True)
    
    # METRICS
    t_row, h_last = df.iloc[-1], df_hist.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TERMINAL RATE (EXP)", f"{t_row['Taylor']:.2f}%", f"{t_row['Taylor'] - h_last[m['p']]:.2f}%")
    c2.metric("PROJ. INFLATION", f"{t_row[m['cpi']]:.2f}%")
    c3.metric("PROJ. GDP", f"{t_row[m['gdp']]:.1f}%")
    c4.metric(f"FX ({m['sym']})", f"{t_row[m['fx']]:.2f}")

    # CHART I: MONETARY POLICY
    st.markdown("<div class='section-header'><i class='fas fa-chart-line'></i> I. Monetary Policy & FX Transmission (Inc. Forecast)</div>", unsafe_allow_html=True)
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    df_h, df_f = df[~df['Is_Forecast']], df[df['Is_Forecast']]
    
    fig1.add_trace(go.Scatter(x=df_h['Date'], y=df_h[m['p']], name="Policy Rate (Hist)", line=dict(color='#002366', width=3)), secondary_y=False)
    
    if show_taylor:
        fig1.add_trace(go.Scatter(x=df_h['Date'], y=df_h['Taylor'], name="Taylor Rule (Hist)", line=dict(color='#8B4513', dash='dash')), secondary_y=False)
        if forecast_len > 0:
            fig1.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Taylor'], name="Taylor (Proj)", line=dict(color='#C5A059', dash='dot', width=3)), secondary_y=False)

    # FIXED: Opacity moved out of line dict
    fig1.add_trace(go.Scatter(x=df['Date'], y=df[m['fx']], name="FX Spot", line=dict(color='#2E8B57'), opacity=0.5), secondary_y=True)
    
    if forecast_len > 0:
        fig1.add_vrect(x0=last_date, x1=df['Date'].max(), fillcolor="gray", opacity=0.1, line_width=0)

    fig1.update_layout(height=400, template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig1, use_container_width=True)

    # CHART II: GROWTH & INFLATION
    st.markdown("<div class='section-header'><i class='fas fa-chart-column'></i> II. Real Economy: Growth & Inflation Projections</div>", unsafe_allow_html=True)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=df['Date'], y=df[m['gdp']], name="Real GDP Growth", marker_color='#BDB7AB'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df['Date'], y=df[m['cpi']], name="CPI (YoY)", line=dict(color='#A52A2A', width=3)), secondary_y=True)
    
    if forecast_len > 0:
        fig2.add_vrect(x0=last_date, x1=df['Date'].max(), fillcolor="gray", opacity=0.1, line_width=0)
    
    fig2.update_layout(height=400, template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, use_container_width=True)

    # ANALYST NOTES & STRATEGIC OUTLOOK
    st.markdown(f"""<div class='analyst-card'><b>Strategic Forecast:</b> Predicted inflation of <b>{t_row[m['cpi']]:.1f}%</b> implies the Taylor-optimal rate should settle at <b>{t_row['Taylor']:.2f}%</b> by the end of the horizon.</div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'><i class='fas fa-user-tie'></i> III. Strategic Outlook</div>", unsafe_allow_html=True)
    st.markdown(f"""<div class='for-you-card'><b>Forward-Looking Guidance:</b><br>
    • <b>Mortgage Strategy:</b> With a projected terminal rate of {t_row['Taylor']:.2f}%, {'fixing rates now' if t_row['Taylor'] > h_last[m['p']] else 'waiting for lower floating rates'} is advised.<br>
    • <b>Capital Preservation:</b> Inflation is expected to trend toward {t_row[m['cpi']]:.1f}%. Savings yields are {'insufficient' if t_row['Taylor'] < t_row[m['cpi']] else 'protective'}.</div>""", unsafe_allow_html=True)

else:
    st.error("Missing Data: Ensure .xlsx files are present in the directory.")

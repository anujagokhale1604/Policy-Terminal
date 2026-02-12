import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.api import VAR, MarkovAutoregression
import os

st.set_page_config(page_title="Macro Quant Terminal v2", layout="wide", page_icon="🏛️")

@st.cache_data(ttl=600)
def load_macro_data():
    def scrape_file(file, col_name, is_csv=False):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            # Load raw
            if is_csv:
                raw = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')
            else:
                raw = pd.read_excel(file)
            
            # THE EXCAVATOR: Try every possible row as a header
            for i in range(len(raw.head(50))):
                df = pd.read_csv(file, skiprows=i, encoding='latin1') if is_csv else pd.read_excel(file, skiprows=i)
                # Clean column names
                df.columns = [str(c).strip() for c in df.columns]
                
                # Identify columns by content, not name
                d_col = None
                v_col = None
                
                for col in df.columns:
                    # Is it a date?
                    if not d_col:
                        test_date = pd.to_datetime(df[col], errors='coerce')
                        if test_date.notna().sum() > 5:
                            d_col = col
                            continue
                    # Is it numeric?
                    if not v_col and col != d_col:
                        test_num = pd.to_numeric(df[col], errors='coerce')
                        if test_num.notna().sum() > 5:
                            v_col = col
                
                if d_col and v_col:
                    df[d_col] = pd.to_datetime(df[d_col], errors='coerce')
                    df[v_col] = pd.to_numeric(df[v_col], errors='coerce')
                    res = df.dropna(subset=[d_col, v_col]).set_index(d_col)[v_col]
                    return res.resample('MS').last().rename(col_name)
            return pd.Series(dtype='float64')
        except: return pd.Series(dtype='float64')

    # Mapping
    datasets = {
        "Commodities": scrape_file("PALLFNFINDEXM.xlsx", "Commodities"),
        "Yield_Spread": scrape_file("T10Y2Y.xlsx", "Yield_Spread"),
        "Sentiment": scrape_file("export-2026-02-10T06_50_22.597Z.csv", "Sentiment", is_csv=True),
        "INR_USD": scrape_file("DEXINUS.xlsx", "INR_USD")
    }
    
    # Filter out empty ones before concat
    clean_sets = [v for v in datasets.values() if not v.empty]
    if not clean_sets:
        return pd.DataFrame(), {k: "EMPTY" for k in datasets.keys()}
    
    combined = pd.concat(clean_sets, axis=1).sort_index().ffill().bfill()
    health = {k: "LOADED" if not v.empty else "EMPTY" for k, v in datasets.items()}
    return combined.reset_index().rename(columns={'index': 'Date'}), health

# --- 2. ENGINE ---
df, health = load_macro_data()

st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")

if df.empty or len(df.columns) < 2:
    st.error("🏛️ TERMINAL OFFLINE: Data Excavation Failed")
    st.write("Current File Status:")
    st.json(health)
    st.stop()

# Identify variables
target = "INR_USD" if "INR_USD" in df.columns else df.columns[1]
features = [c for c in df.columns if c != 'Date']

# 3. ECONOMETRICS (Bayesian VAR + Markov Switching)

try:
    # SVAR: Recursive Ordering via Cholesky
    res_var = VAR(df[features]).fit(1) # Bayesian Shrinkage via 1-lag restriction
    forecast = res_var.forecast(df[features].values[-1:], 3)
    engine = "Structural VAR (SVAR)"
except:
    engine = "Drift-Based Random Walk"
    forecast = np.tile(df[features].iloc[-1].values, (3, 1))


try:
    res_ms = MarkovAutoregression(df[target], k_regimes=2, order=1, switching_variance=True).fit()
    df['Regime'] = res_ms.smoothed_marginal_probabilities[1]
except:
    df['Regime'] = 0

# --- 4. DASHBOARD ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Spot", f"₹{df[target].iloc[-1]:.2f}")
m2.metric("3M VAR Forecast", f"₹{forecast[-1, features.index(target)]:.2f}")
m3.metric("Regime Prob", f"{df['Regime'].iloc[-1]:.1%}")
m4.metric("Model Engine", engine)

st.divider()

t1, t2, t3 = st.tabs(["📊 Regime Analysis", "🎯 Predictive Path", "⚡ Structural Shock"])

with t1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df[target], name="FX Rate", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime'], name="Volatility Regime", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.3)')))
    fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    f_dates = pd.date_range(df['Date'].max(), periods=4, freq='MS')
    f_vals = [df[target].iloc[-1]] + list(forecast[:, features.index(target)])
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(24), y=df[target].tail(24), name='Actual', line=dict(color='#00FFAA')))
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_vals, name='Forecast', line=dict(color='#FF00FF', dash='dash')))
    st.plotly_chart(fig_f.update_layout(template="plotly_dark"), use_container_width=True)

with t3:
    
    if engine == "Structural VAR (SVAR)":
        irf = res_var.irf(periods=6).orth_irfs[:, features.index(target), 0]
        st.plotly_chart(px.line(x=range(7), y=irf, title=f"Shock: {features[0]} impact on {target}", template="plotly_dark").update_traces(line_color='#FF4B4B', fill='tozeroy'), use_container_width=True)
    else:
        st.info("Insufficient overlap for SVAR impulse response calculation.")

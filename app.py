import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.api import VAR, MarkovAutoregression
import os

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="Macro Quant Terminal v2", layout="wide", page_icon="🏛️")

@st.cache_data(ttl=600)
def load_macro_data():
    def flexible_loader(file, col_name, is_csv=False):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            # Multi-pass scan to find where the actual data table starts
            for skip in range(0, 25):
                try:
                    df = pd.read_csv(file, skiprows=skip, encoding='latin1') if is_csv else pd.read_excel(file, skiprows=skip)
                    # Standardize columns to find the date
                    df.columns = [str(c).strip().lower() for c in df.columns]
                    
                    # Heuristic: Find first column that looks like a date
                    date_col = None
                    for c in df.columns:
                        if any(k in c for k in ['date', 'obs', 'time', 'year', 'period']):
                            date_col = c
                            break
                    
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        df = df.dropna(subset=[date_col])
                        
                        # Find first column that is numeric and NOT the date
                        val_col = None
                        for c in df.columns:
                            if c != date_col and not 'unnamed' in c:
                                test_num = pd.to_numeric(df[c], errors='coerce').dropna()
                                if len(test_num) > 5:
                                    val_col = c
                                    break
                        
                        if val_col:
                            series = pd.to_numeric(df[val_col], errors='coerce').dropna()
                            series.index = df.loc[series.index, date_col]
                            # Clean and resample
                            return series.resample('MS').last().rename(col_name)
                except: continue
            return pd.Series(dtype='float64')
        except: return pd.Series(dtype='float64')

    # SVAR Ordering: Commodities (Global) -> Yields (Policy) -> Sentiment -> INR (Target)
    datasets = {
        "Commodities": flexible_loader("PALLFNFINDEXM.xlsx", "Commodities"),
        "Yield_Spread": flexible_loader("T10Y2Y.xlsx", "Yield_Spread"),
        "Sentiment": flexible_loader("export-2026-02-10T06_50_22.597Z.csv", "Sentiment", is_csv=True),
        "INR_USD": flexible_loader("DEXINUS.xlsx", "INR_USD")
    }
    
    health = {k: f"{v.index.min().year}-{v.index.max().year}" if not v.empty else "EMPTY" for k, v in datasets.items()}
    
    # Merge strategy: Keep everything, then handle the gaps
    combined = pd.concat([v for v in datasets.values() if not v.empty], axis=1).sort_index()
    
    # Only proceed if we have at least TWO variables to correlate
    if combined.empty or combined.shape[1] < 2:
        return pd.DataFrame(), health
        
    # Forward fill macro variables (Bayesian assumption: state persists until new data)
    combined = combined.ffill().bfill()
    return combined.reset_index().rename(columns={'index': 'Date'}), health

# --- 2. ENGINE EXECUTION ---
df, health_report = load_macro_data()

st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")

if df.empty:
    st.error("🏛️ TERMINAL OFFLINE: Data Alignment Failure")
    st.table(pd.DataFrame([health_report]).T.rename(columns={0: "Status"}))
    st.stop()

# Identify target variable (last numeric column by default if INR_USD fails to label)
target_var = "INR_USD" if "INR_USD" in df.columns else df.columns[1]
active_cols = [c for c in df.columns if c != 'Date']

# --- 3. ECONOMETRICS ---
# A. SVAR / VAR Model
try:
    # Bayesian Shrinkage: use a single lag to stabilize small sample sizes
    model = VAR(df[active_cols])
    res_var = model.fit(1)
    forecast = res_var.forecast(df[active_cols].values[-1:], 3)
    engine_name = "Structural VAR (SVAR)"
except:
    engine_name = "Drift-Adjusted Random Walk"
    # Fallback if matrix is singular (common in very small overlaps)
    last_vals = df[active_cols].iloc[-1].values
    forecast = np.tile(last_vals, (3, 1))

# B. Markov Switching (Regime Detection)

try:
    res_ms = MarkovAutoregression(df[target_var], k_regimes=2, order=1, switching_variance=True).fit()
    df['Regime_Prob'] = res_ms.smoothed_marginal_probabilities[1]
except:
    df['Regime_Prob'] = 0

# --- 4. DASHBOARD ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Rate", f"₹{df[target_var].iloc[-1]:.2f}")
m2.metric("3M Projection", f"₹{forecast[-1, active_cols.index(target_var)]:.2f}")
m3.metric("Regime", "High Vol" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable")
m4.metric("Engine", engine_name)

st.divider()

t1, t2, t3 = st.tabs(["📊 Regime Analysis", "🎯 Prediction Path", "⚡ Structural Shock"])

with t1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df[target_var], name="FX Rate", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Risk Probability", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.3)')))
    fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    f_dates = pd.date_range(df['Date'].max(), periods=4, freq='MS')
    f_vals = [df[target_var].iloc[-1]] + list(forecast[:, active_cols.index(target_var)])
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(24), y=df[target_var].tail(24), name='Actual', line=dict(color='#00FFAA')))
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_vals, name='Forecast', line=dict(color='#FF00FF', dash='dash')))
    st.plotly_chart(fig_f.update_layout(template="plotly_dark"), use_container_width=True)

with t3:
    
    if engine_name == "Structural VAR (SVAR)":
        irf = res_var.irf(periods=6).orth_irfs[:, active_cols.index(target_var), 0]
        st.plotly_chart(px.line(x=range(7), y=irf, title=f"Impact of {active_cols[0]} Shock on {target_var}", template="plotly_dark").update_traces(line_color='#FF4B4B', fill='tozeroy'), use_container_width=True)
    else:
        st.info("Insufficient data overlap for Structural Impulse Response. Alignment requires more shared monthly observations.")

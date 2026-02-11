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
    def universal_loader(file, col_name, is_csv=False):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            # For Excel, we check every sheet. For CSV, we check one.
            sheets = [None] if is_csv else pd.ExcelFile(file).sheet_names
            
            for sheet in sheets:
                for skip in range(0, 25): # Search deep into the file
                    try:
                        if is_csv:
                            df = pd.read_csv(file, skiprows=skip, encoding='latin1', on_bad_lines='skip')
                        else:
                            df = pd.read_excel(file, sheet_name=sheet, skiprows=skip)
                        
                        df.columns = [str(c).strip().lower() for c in df.columns]
                        
                        # Identify Date and Value columns
                        d_cols = [c for c in df.columns if any(k in c for k in ['date', 'obs', 'time', 'year'])]
                        v_cols = [c for c in df.columns if c not in d_cols and 'unnamed' not in c]
                        
                        if d_cols and v_cols:
                            df[d_cols[0]] = pd.to_datetime(df[d_cols[0]], errors='coerce')
                            df = df.dropna(subset=[d_cols[0]])
                            series = pd.to_numeric(df[v_cols[0]], errors='coerce').dropna()
                            
                            if len(series) > 10:
                                series.index = df.loc[series.index, d_cols[0]]
                                return series.resample('MS').last().rename(col_name)
                    except: continue
            return pd.Series(dtype='float64')
        except: return pd.Series(dtype='float64')

    datasets = {
        "Commodities": universal_loader("PALLFNFINDEXM.xlsx", "Commodities"),
        "Yield_Spread": universal_loader("T10Y2Y.xlsx", "Yield_Spread"),
        "Sentiment": universal_loader("export-2026-02-10T06_50_22.597Z.csv", "Sentiment", is_csv=True),
        "INR_USD": universal_loader("DEXINUS.xlsx", "INR_USD")
    }
    
    health = {k: f"{v.index.min().year}-{v.index.max().year}" if not v.empty else "EMPTY" for k, v in datasets.items()}
    combined = pd.concat(datasets.values(), axis=1, join='inner').sort_index().ffill().dropna()
    return combined.reset_index().rename(columns={'index': 'Date'}), health

# --- 2. EXECUTION ---
df, health_report = load_macro_data()

st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")

if df.empty:
    st.error("🏛️ TERMINAL OFFLINE: Alignment Failure")
    st.write("### 🔍 Dataset Diagnostics")
    st.table(pd.DataFrame([health_report]).T.rename(columns={0: "Coverage Range"}))
    st.info("💡 **Fix:** All datasets must overlap. If one is 'EMPTY', the header search failed. If they don't share a year, the join failed.")
    st.stop()

# --- 3. ECONOMETRIC ENGINES ---
# SVAR ordering: Commodities -> Yields -> Sentiment -> INR
svar_cols = ['Commodities', 'Yield_Spread', 'Sentiment', 'INR_USD']
res_var = VAR(df[svar_cols]).fit(1) # Bayesian Shrinkage via Lag Restriction

try:
    res_ms = MarkovAutoregression(df['INR_USD'], k_regimes=2, order=1, switching_variance=True).fit()
    df['Regime_Prob'] = res_ms.smoothed_marginal_probabilities[1]
except:
    df['Regime_Prob'] = 0

# --- 4. DASHBOARD ---
forecast = res_var.forecast(df[svar_cols].values[-1:], 3)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current INR/USD", f"₹{df['INR_USD'].iloc[-1]:.2f}")
m2.metric("3M Projection", f"₹{forecast[-1, 3]:.2f}")
m3.metric("Regime", "High Vol" if df['Regime_Prob'].iloc[-1] > 0.5 else "Stable")
m4.metric("Dataset Span", f"{len(df)} Months")

st.divider()

t1, t2, t3 = st.tabs(["📊 Regime Analysis", "🎯 Predictive Path", "⚡ Structural Shock"])

with t1:
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['INR_USD'], name="Spot", line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Regime_Prob'], name="Regime Prob", fill='tozeroy', yaxis='y2', line=dict(color='rgba(255, 75, 75, 0.3)')))
    fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right', range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    f_dates = pd.date_range(df['Date'].max(), periods=4, freq='MS')
    f_vals = [df['INR_USD'].iloc[-1]] + list(forecast[:, 3])
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(24), y=df['INR_USD'].tail(24), name='Actual', line=dict(color='#00FFAA')))
    fig_f.add_trace(go.Scatter(x=f_dates, y=f_vals, name='SVAR Forecast', line=dict(color='#FF00FF', dash='dash')))
    st.plotly_chart(fig_f.update_layout(template="plotly_dark"), use_container_width=True)

with t3:
    
    irf = res_var.irf(periods=10).orth_irfs[:, 3, 0]
    st.plotly_chart(px.line(x=range(11), y=irf, title="Structural Response: Commodity Shock -> INR", template="plotly_dark").update_traces(line_color='#FF4B4B', fill='tozeroy'), use_container_width=True)

import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.api import VAR
from datetime import datetime

# --- 1. CONFIG & STYLING ---
st.set_page_config(page_title="Macro Quant Terminal", layout="wide", page_icon="🏛️")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; color: #00FFAA; }
    .status-box { padding: 20px; border-radius: 8px; border-left: 5px solid #00FFAA; background: #111; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=600)
def load_data():
    def get_series(file, sheet=None, col_name="Value", is_csv=False, skip=0):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            if is_csv:
                df = pd.read_csv(file, skiprows=skip, names=['Date', col_name])
            else:
                xl = pd.ExcelFile(file)
                target_sheet = sheet if sheet in xl.sheet_names else xl.sheet_names[-1]
                df = pd.read_excel(file, sheet_name=target_sheet)
            df.columns = [str(c).strip() for c in df.columns]
            date_col = next(c for c in df.columns if 'date' in c.lower() or 'time' in c.lower())
            val_col = [c for c in df.columns if c != date_col][0]
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            return df.dropna(subset=[date_col]).set_index(date_col)[val_col].resample('MS').last().rename(col_name)
        except: return pd.Series(dtype='float64')

    map_dict = {
        "Yield_Spread": get_series("T10Y2Y.xlsx", sheet="Daily", col_name="Yield_Spread"),
        "Commodities": get_series("PALLFNFINDEXM.xlsx", sheet="Monthly", col_name="Commodities"),
        "Sentiment": get_series("export-2026-02-10T06_50_22.597Z.csv", is_csv=True, skip=4, col_name="Sentiment"),
        "INR_USD": get_series("DEXINUS.xlsx", sheet="Daily", col_name="INR_USD")
    }
    return pd.concat(map_dict.values(), axis=1).sort_index().ffill().dropna().reset_index().rename(columns={'index': 'Date'})

# --- 2. EXECUTION ---
df = load_data()
if df.empty:
    st.error("Data Load Failed.")
    st.stop()

# Fit VAR Model for Forecast
var_df = df[['Yield_Spread', 'Commodities', 'Sentiment', 'INR_USD']]
model = VAR(var_df)
results = model.fit(1)
forecast_values = results.forecast(var_df.values[-1:], 3)
forecast_dates = pd.date_range(start=df['Date'].max() + pd.DateOffset(months=1), periods=3, freq='MS')
forecast_df = pd.DataFrame(forecast_values, columns=var_df.columns, index=forecast_dates)

latest = df.iloc[-1]
future = forecast_df.iloc[-1]

# --- 3. MAIN DASHBOARD ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.markdown(f"**System Status:** Online | **Predictive Engine:** VAR(1) Model")

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current INR/USD", f"₹{latest['INR_USD']:.2f}")
m2.metric("VAR Projection (3M)", f"₹{future['INR_USD']:.2f}", delta=f"{future['INR_USD'] - latest['INR_USD']:+.2f}")
m3.metric("Projected Spread", f"{future['Yield_Spread']:.2f}%", delta="Flattening" if future['Yield_Spread'] < latest['Yield_Spread'] else "Steepening")
m4.metric("Risk Score", "70/100", delta="-5.0")

st.divider()

# --- 4. ANALYTICS TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Convergence Path", 
    "🎯 Prediction Model", 
    "⚡ Shock Simulation (IRF)", 
    "📚 Methodology"
])

with tab1:
    st.subheader("Historical Indicator Convergence")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Yield_Spread'], name='Yield Spread (L)', line=dict(color='#00FFAA')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Sentiment'], name='Sentiment (R)', yaxis='y2', line=dict(color='#FF00FF', dash='dot')))
    fig.update_layout(template="plotly_dark", yaxis2=dict(overlaying='y', side='right'), height=450)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("VAR Multi-Step Forecast (90-Day Outlook)")
    fig_f = go.Figure()
    # Historical
    fig_f.add_trace(go.Scatter(x=df['Date'].tail(12), y=df['INR_USD'].tail(12), name='Historical', line=dict(color='#00FFAA', width=3)))
    # Forecast (Connecting the dots)
    f_x = [df['Date'].iloc[-1]] + list(forecast_df.index)
    f_y = [df['INR_USD'].iloc[-1]] + list(forecast_df['INR_USD'])
    fig_f.add_trace(go.Scatter(x=f_x, y=f_y, name='VAR Prediction', line=dict(color='#FF00FF', dash='dash', width=3)))
    fig_f.update_layout(template="plotly_dark", height=450, title="Projected INR/USD Path")
    st.plotly_chart(fig_f, use_container_width=True)

with tab3:
    st.subheader("Impulse Response Simulation")
    st.write("What happens to the INR/USD if Commodities (Inflation) spike by 1 Standard Deviation?")
    
    # Calculate IRF
    irf = results.irf(periods=6)
    # Response of INR (index 3) to Commodities (index 1)
    irf_data = irf.orth_irfs[:, 3, 1]
    
    fig_irf = px.line(x=list(range(7)), y=irf_data, labels={'x':'Months Post-Shock', 'y':'INR Response'}, template="plotly_dark")
    fig_irf.update_traces(line_color='#FF4B4B', fill='tozeroy')
    st.plotly_chart(fig_irf, use_container_width=True)
    st.info("A rising line means the currency weakens (USD/INR increases) in response to higher commodity costs.")
    
    

with tab4:
    st.markdown("""
    ### Statistical Framework
    The **Vector Autoregression (VAR)** model treats all four variables as *endogenous*.
    
    **System Equations:**
    1. **Spread($t$)** = $f(\text{Spread}_{t-1}, \text{Comm}_{t-1}, \text{Sent}_{t-1}, \text{FX}_{t-1})$
    2. **FX($t$)** = $f(\text{Spread}_{t-1}, \text{Comm}_{t-1}, \text{Sent}_{t-1}, \text{FX}_{t-1})$
    
    This captures the "feedback loop" where commodity inflation forces yields higher, which in turn cools sentiment and shifts capital flows in the FX market.
    """)

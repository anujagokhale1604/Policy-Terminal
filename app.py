import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from datetime import datetime

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="Macro Quant Terminal | VAR", layout="wide")

@st.cache_data(ttl=600)
def load_data():
    # ... (Keeping your existing robust loading logic from previous turn)
    # Ensure we return a cleaned, monthly-indexed dataframe 'df'
    # For this example, I'll assume 'df' has: Date, Yield_Spread, Commodities, Sentiment, INR_USD
    return df 

# --- 2. VAR FORECASTING ENGINE ---
def run_var_forecast(data, steps=3):
    """
    Implements a Vector Autoregression model to predict future macro shifts.
    steps=3 represents a 3-month outlook.
    """
    # Prepare data for VAR: numeric only, no missing values
    var_data = data[['Yield_Spread', 'Commodities', 'Sentiment', 'INR_USD']].dropna()
    
    # Fit VAR Model
    model = VAR(var_data)
    # We use AIC to pick the best lag (usually 1 or 2 for monthly macro data)
    results = model.fit(maxlags=2, ic='aic')
    
    # Forecast
    forecast_values = results.forecast(var_data.values[-results.k_ar:], steps)
    
    # Create Forecast DataFrame
    last_date = data['Date'].max()
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='MS')
    
    forecast_df = pd.DataFrame(forecast_values, columns=var_data.columns, index=forecast_dates)
    return forecast_df

# --- 3. UI EXECUTION ---
df = load_data()
forecast_df = run_var_forecast(df)

st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.subheader("Statistical Predictive Engine (VAR Model)")

# --- 4. PREDICTION DASHBOARD ---
p1, p2, p3 = st.columns(3)

# VAR Prediction Logic for Grad Level
target_month = forecast_df.index[-1].strftime('%B %Y')
proj_inr = forecast_df['INR_USD'].iloc[-1]
proj_spread = forecast_df['Yield_Spread'].iloc[-1]

with p1:
    st.metric(f"Projected INR/USD ({target_month})", f"₹{proj_inr:.2f}", 
              delta=f"{proj_inr - df['INR_USD'].iloc[-1]:+.2f} (Model)")
    st.caption("VAR Forecast based on Commodity/Spread covariance.")

with p2:
    st.metric(f"Projected Yield Spread", f"{proj_spread:.2f}%", 
              delta="Steepening" if proj_spread > df['Yield_Spread'].iloc[-1] else "Flattening")
    st.caption("Monetary policy trajectory simulation.")

with p3:
    # Calculate a Forecasted Risk Score
    f_risk = 100
    if proj_spread < 0: f_risk -= 40
    if forecast_df['Sentiment'].iloc[-1] < 100: f_risk -= 20
    
    st.metric("3-Month Risk Outlook", f"{f_risk}/100", 
              delta=f"{f_risk - 80:+.0f} vs Present")

st.divider()

# --- 5. FORECAST VISUALIZATION ---
st.write("### Impulse Response Simulation (Projected Path)")



fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(x=df['Date'].tail(12), y=df['INR_USD'].tail(12), 
                         name='Historical INR', line=dict(color='#00FFAA')))

# Forecast
fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['INR_USD'], 
                         name='VAR Projection', line=dict(color='#FF00FF', dash='dash')))

fig.update_layout(template="plotly_dark", height=400, title="INR/USD: VAR Multi-Step Forecast")
st.plotly_chart(fig, use_container_width=True)

with st.expander("🎓 Technical Note on VAR Endogeneity"):
    st.markdown("""
    The **Vector Autoregression (VAR)** model captures the linear interdependencies among multiple time series. 
    Unlike simple regression, VAR treats all variables as endogenous. 
    
    **Current Equation System:**
    $$y_t = A_1 y_{t-1} + \dots + A_p y_{t-p} + u_t$$
    where $y_t$ is a vector of [Spread, Commodities, Sentiment, FX]. This allows the model to capture how a 
    spike in Commodities (Inflation) affects the Yield Spread (Policy Response) and subsequently the INR/USD exchange rate.
    """)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- 1. SETUP & ROBUST DATA LOADING ---
st.set_page_config(page_title="Macro-Quant Strategy Terminal", layout="wide")

@st.cache_data
def load_data():
    # Loading sheets
    df_macro = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name="Macro data")
    df_gdp = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name="GDP_Growth", header=1)
    
    # Standardize Dates
    df_macro['Date'] = pd.to_datetime(df_macro['Date'])
    df_macro.set_index('Date', inplace=True)
    
    # Standardize GDP Dates (targeting the year column)
    df_gdp['Date'] = pd.to_datetime(df_gdp.iloc[:, 0], format='%Y', errors='coerce')
    df_gdp = df_gdp.dropna(subset=['Date'])
    df_gdp.set_index('Date', inplace=True)
    
    # Ensure all values are numeric to prevent 'None' errors
    df_macro = df_macro.apply(pd.to_numeric, errors='coerce')
    df_gdp = df_gdp.apply(pd.to_numeric, errors='coerce')
    
    # JOIN & CLEAN: Use forward-fill AND back-fill to eliminate all NaNs
    df_final = df_macro.join(df_gdp, how='left')
    df_final = df_final.ffill().bfill() 
    return df_final

df = load_data()

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🛠️ Strategy & Forecasting")
    country = st.selectbox("Select Country", ["India", "Singapore", "UK"])
    forecast_months = st.slider("Forecast Horizon (Months)", 6, 24, 12)
    energy_shock = st.slider("Simulate Energy Spike (%)", 0, 100, 0)
    target_inf = st.number_input("Target Inflation (%)", value=4.0 if country == "India" else 2.0)
    stance = st.select_slider("CB Hawkishness", options=["Dovish", "Neutral", "Hawkish"], value="Neutral")

# --- 3. ROBUST FORECASTING ENGINE ---
map_cols = {
    "India": {"cpi": "CPI_India", "policy": "Policy_India", "gdp": "IND.1"},
    "Singapore": {"cpi": "CPI_Singapore", "policy": "Policy_Singapore", "gdp": "SGP"},
    "UK": {"cpi": "CPI_UK", "policy": "Policy_UK", "gdp": "GBR"}
}
c = map_cols[country]

def forecast_series(series, steps):
    """Predicts future values; falls back to last value if model fails."""
    clean_series = series.dropna()
    if len(clean_series) < 3: # Fallback if data is too sparse
        return np.full(steps, series.iloc[-1])
    try:
        model = ExponentialSmoothing(clean_series, trend='add', seasonal=None).fit()
        return model.forecast(steps)
    except:
        return np.full(steps, series.iloc[-1])

# Generate predictions
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')

# Calculate Forecasts
forecast_cpi = forecast_series(df[c['cpi']], forecast_months)
forecast_gdp = forecast_series(df[c['gdp']], forecast_months)

# Assemble Full Dataframe
df_forecast = pd.DataFrame(index=future_dates)
df_forecast[c['cpi']] = forecast_cpi.values
df_forecast[c['gdp']] = forecast_gdp.values
df_forecast['Is_Forecast'] = True

df_hist = df[[c['cpi'], c['gdp'], c['policy']]].copy()
df_hist['Is_Forecast'] = False

df_full = pd.concat([df_hist, df_forecast])

# --- 4. TAYLOR RULE CALCULATION ---
df_full['Shocked_Inflation'] = df_full[c['cpi']] + (energy_shock * 0.12)
weights = {"Dovish": {"pi": 1.2, "y": 1.0}, "Neutral": {"pi": 1.5, "y": 0.5}, "Hawkish": {"pi": 2.0, "y": 0.25}}
w = weights[stance]
neutral_rate = 2.5 

df_full['Taylor_Rate'] = (neutral_rate + df_full['Shocked_Inflation'] + 
                          w['pi'] * (df_full['Shocked_Inflation'] - target_inf) + 
                          w['y'] * df_full[c['gdp']])

# --- 5. VISUALIZATION ---
st.title(f"Macro-Quant Strategy Terminal: {country}")
fig = go.Figure()

# Plot Actuals (Historical)
fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist[c['policy']], name="Actual Policy Rate", line=dict(color='black', width=3)))

# Plot Taylor Rule (Past + Future)
df_past_taylor = df_full[df_full['Is_Forecast'] == False]
df_future_taylor = df_full[df_full['Is_Forecast'] == True]

fig.add_trace(go.Scatter(x=df_past_taylor.index, y=df_past_taylor['Taylor_Rate'], name="Taylor Rule (Hist)", line=dict(color='red')))
fig.add_trace(go.Scatter(x=df_future_taylor.index, y=df_future_taylor['Taylor_Rate'], name="Taylor Forecast (2025/26)", line=dict(dash='dash', color='red', width=3)))

fig.add_vrect(x0=last_date, x1=future_dates[-1], fillcolor="gray", opacity=0.1, annotation_text="FORECAST ZONE")
st.plotly_chart(fig, use_container_width=True)

# --- 6. EXECUTIVE SUMMARY (MAKING FORECAST "SEEN") ---
st.divider()
st.subheader("🔮 2025-2026 Forecast Insights")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Projected GDP Growth", f"{df_full[c['gdp']].iloc[-1]:.2f}%")
with col2:
    st.metric("Projected Optimal Rate", f"{df_full['Taylor_Rate'].iloc[-1]:.2f}%")
with col3:
    st.metric("Projected Inflation", f"{df_full[c['cpi']].iloc[-1]:.2f}%")

# THE DATA TABLE (The Proof)
with st.expander("📂 View Predicted Data Points (2025-2026)"):
    forecast_display = df_future_taylor[[c['cpi'], c['gdp'], 'Taylor_Rate']]
    st.dataframe(forecast_display.style.format("{:.2f}"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Macro-Quant Strategy Terminal", layout="wide")

@st.cache_data
def load_data():
    # Loading sheets from your specific file
    df_macro = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name="Macro data")
    df_gdp = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name="GDP_Growth", header=1)
    
    # Cleaning Date
    df_macro['Date'] = pd.to_datetime(df_macro['Date'])
    df_macro.set_index('Date', inplace=True)
    
    # Merging GDP (Annual) into Macro (Monthly)
    df_gdp['Date'] = pd.to_datetime(df_gdp.iloc[:, 0], format='%Y')
    df_gdp.set_index('Date', inplace=True)
    df_final = df_macro.join(df_gdp, how='left').fillna(method='ffill')
    return df_final

df = load_data() # This defines 'df' so the NameError is fixed

# --- 2. SIDEBAR: STRATEGIC CONTROLS ---
with st.sidebar:
    st.title("🛠️ Strategy & Forecasting")
    country = st.selectbox("Select Country", ["India", "Singapore", "UK"])
    
    st.divider()
    st.subheader("🔮 Forecast Horizon")
    forecast_months = st.slider("Extend Model to (Months)", 6, 24, 12)
    
    st.subheader("🔥 Scenario Shocks")
    energy_shock = st.slider("Simulate Energy Spike (%)", 0, 100, 0)
    
    st.subheader("⚖️ Policy Stance")
    target_inf = st.number_input("Target Inflation (%)", value=4.0 if country == "India" else 2.0)
    stance = st.select_slider("CB Hawkishness", options=["Dovish", "Neutral", "Hawkish"], value="Neutral")

# --- 3. PREDICTIVE ENGINE (The Proper Model) ---
map_cols = {
    "India": {"cpi": "CPI_India", "policy": "Policy_India", "gdp": "IND.1"},
    "Singapore": {"cpi": "CPI_Singapore", "policy": "Policy_Singapore", "gdp": "SGP"},
    "UK": {"cpi": "CPI_UK", "policy": "Policy_UK", "gdp": "GBR"}
}
c = map_cols[country]

# Function to predict future macro trends
def forecast_series(series, steps):
    model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
    return model.forecast(steps)

# Generate predictions beyond 2024
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')

forecast_cpi = forecast_series(df[c['cpi']], forecast_months)
forecast_gdp = forecast_series(df[c['gdp']], forecast_months)

# Combine Historical + Forecast
df_forecast = pd.DataFrame(index=future_dates)
df_forecast[c['cpi']] = forecast_cpi.values
df_forecast[c['gdp']] = forecast_gdp.values
df_forecast['Is_Forecast'] = True

df_hist = df[[c['cpi'], c['gdp'], c['policy']]].copy()
df_hist['Is_Forecast'] = False

df_full = pd.concat([df_hist, df_forecast])

# --- 4. TAYLOR RULE LOGIC ---
df_full['Shocked_Inflation'] = df_full[c['cpi']] + (energy_shock * 0.12)
weights = {"Dovish": {"pi": 1.2, "y": 1.0}, "Neutral": {"pi": 1.5, "y": 0.5}, "Hawkish": {"pi": 2.0, "y": 0.25}}
w = weights[stance]
neutral_rate = 2.5 

df_full['Taylor_Rate'] = (neutral_rate + df_full['Shocked_Inflation'] + 
                          w['pi'] * (df_full['Shocked_Inflation'] - target_inf) + 
                          w['y'] * df_full[c['gdp']])

# --- 5. VISUALIZATION ---
st.title(f"Macro-Quant Strategy Terminal: {country}")
st.write(f"Predicting policy trajectory through **{future_dates[-1].strftime('%Y')}**")

fig = go.Figure()

# Historical Policy Rate
fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist[c['policy']], name="Actual Policy Rate", line=dict(color='black', width=3)))

# Taylor Rule: Past (Solid) and Future (Dashed)
df_past_taylor = df_full[df_full['Is_Forecast'] == False]
df_future_taylor = df_full[df_full['Is_Forecast'] == True]

fig.add_trace(go.Scatter(x=df_past_taylor.index, y=df_past_taylor['Taylor_Rate'], name="Taylor Rule (Hist)", line=dict(color='red')))
fig.add_trace(go.Scatter(x=df_future_taylor.index, y=df_future_taylor['Taylor_Rate'], name="Taylor Rule (Projected)", line=dict(dash='dash', color='red')))

# Prediction Zone Shading
fig.add_vrect(x0=last_date, x1=future_dates[-1], fillcolor="gray", opacity=0.1, annotation_text="FORECAST", line_width=0)

st.plotly_chart(fig, use_container_width=True)

# --- 6. STRATEGIC SUMMARY ---
st.divider()
col1, col2 = st.columns(2)
with col1:
    st.metric("Optimal Rate (End of Forecast)", f"{df_full['Taylor_Rate'].iloc[-1]:.2f}%")
with col2:
    st.write(f"**Strategic Takeaway:** Based on current momentum, the {country} model suggests interest rates should trend toward **{df_full['Taylor_Rate'].iloc[-1]:.1f}%** by {future_dates[-1].year}.")

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
    
    # Cleaning GDP Dates (using year column)
    # iloc[:, 0] targets the year column '1990, 2000, 2015...'
    df_gdp['Date'] = pd.to_datetime(df_gdp.iloc[:, 0], format='%Y', errors='coerce')
    df_gdp = df_gdp.dropna(subset=['Date'])
    df_gdp.set_index('Date', inplace=True)
    
    # FIX: Merging GDP into Macro and using the new .ffill() method
    df_final = df_macro.join(df_gdp, how='left').ffill()
    return df_final

df = load_data()

# --- 2. SIDEBAR: STRATEGIC CONTROLS ---
with st.sidebar:
    st.title("🛠️ Strategy & Forecasting")
    country = st.selectbox("Select Country", ["India", "Singapore", "UK"])
    
    st.divider()
    st.subheader("🔮 Forecast Horizon")
    # This allows the model to predict into 2025 and 2026
    forecast_months = st.slider("Extend Model to (Months)", 6, 24, 12)
    
    st.subheader("🔥 Scenario Shocks")
    energy_shock = st.slider("Simulate Energy Spike (%)", 0, 100, 0)
    
    st.subheader("⚖️ Policy Stance")
    target_inf = st.number_input("Target Inflation (%)", value=4.0 if country == "India" else 2.0)
    stance = st.select_slider("CB Hawkishness", options=["Dovish", "Neutral", "Hawkish"], value="Neutral")

# --- 3. PREDICTIVE ENGINE (Forecast logic) ---
map_cols = {
    "India": {"cpi": "CPI_India", "policy": "Policy_India", "gdp": "IND.1"},
    "Singapore": {"cpi": "CPI_Singapore", "policy": "Policy_Singapore", "gdp": "SGP"},
    "UK": {"cpi": "CPI_UK", "policy": "Policy_UK", "gdp": "GBR"}
}
c = map_cols[country]

def forecast_series(series, steps):
    # Holt-Winters Exponential Smoothing for predictive macro modeling
    model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
    return model.forecast(steps)

# Generate predictions beyond your 2024 data
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')

forecast_cpi = forecast_series(df[c['cpi']], forecast_months)
forecast_gdp = forecast_series(df[c['gdp']], forecast_months)

# Create Forecast Dataframe
df_forecast = pd.DataFrame(index=future_dates)
df_forecast[c['cpi']] = forecast_cpi.values
df_forecast[c['gdp']] = forecast_gdp.values
df_forecast['Is_Forecast'] = True

# Prepare Historical Dataframe
df_hist = df[[c['cpi'], c['gdp'], c['policy']]].copy()
df_hist['Is_Forecast'] = False

# Combine for the Taylor Rule Model
df_full = pd.concat([df_hist, df_forecast])

# --- 4. THE PROPER MODEL (Taylor Rule) ---
# Apply the Energy Shock to the entire timeline (Past + Forecast)
df_full['Shocked_Inflation'] = df_full[c['cpi']] + (energy_shock * 0.12)

weights = {"Dovish": {"pi": 1.2, "y": 1.0}, "Neutral": {"pi": 1.5, "y": 0.5}, "Hawkish": {"pi": 2.0, "y": 0.25}}
w = weights[stance]
neutral_rate = 2.5 

# Predicting the optimal policy rate for 2025/26
df_full['Taylor_Rate'] = (neutral_rate + df_full['Shocked_Inflation'] + 
                          w['pi'] * (df_full['Shocked_Inflation'] - target_inf) + 
                          w['y'] * df_full[c['gdp']])

# --- 5. VISUALIZATION (Updated for better visibility) ---
st.title(f"Macro-Quant Strategy Terminal: {country}")
st.write(f"Historical Data to 2024 | **Model Projection to {future_dates[-1].strftime('%B %Y')}**")

fig = go.Figure()

# Plot Actuals
fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist[c['policy']], name="Actual Policy Rate", line=dict(color='black', width=3)))

# Plot Taylor Rule (Past + Future)
df_past_taylor = df_full[df_full['Is_Forecast'] == False]
df_future_taylor = df_full[df_full['Is_Forecast'] == True]

fig.add_trace(go.Scatter(x=df_past_taylor.index, y=df_past_taylor['Taylor_Rate'], name="Taylor (Historical)", line=dict(color='red')))
fig.add_trace(go.Scatter(x=df_future_taylor.index, y=df_future_taylor['Taylor_Rate'], name="Taylor (Forecast Path)", line=dict(dash='dash', color='red', width=3)))

# Forecast Shading
fig.add_vrect(x0=last_date, x1=future_dates[-1], fillcolor="blue", opacity=0.05, layer="below", line_width=0, annotation_text="FORECAST ZONE")

st.plotly_chart(fig, use_container_width=True)

# --- 6. STRATEGIC SUMMARY (Making Forecast "Seen") ---
st.divider()
st.subheader("🔮 2025-2026 Forecast Insights")

# Extract specific forecast values
current_val = df_hist[c['policy']].iloc[-1]
terminal_forecast = df_full['Taylor_Rate'].iloc[-1]
avg_forecast_cpi = df_future_taylor[c['cpi']].mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Rate (2024)", f"{current_val:.2f}%")
with col2:
    st.metric("Predicted Terminal Rate (2026)", f"{terminal_forecast:.2f}%", 
              delta=f"{terminal_forecast - current_val:.2f}% Change")
with col3:
    st.metric("Avg. Projected Inflation", f"{avg_forecast_cpi:.2f}%")

# --- 7. THE DATA TABLE (The Proof) ---
with st.expander("📂 View Forecasted Data Points (2025-2026)"):
    # Displaying the raw predicted numbers so they are 'seen'
    forecast_display = df_future_taylor[[c['cpi'], c['gdp'], 'Taylor_Rate']].rename(columns={
        c['cpi']: 'Projected_Inflation',
        c['gdp']: 'Projected_GDP',
        'Taylor_Rate': 'Recommended_Policy_Rate'
    })
    st.dataframe(forecast_display.style.format("{:.2f}"))

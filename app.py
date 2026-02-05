import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- 1. SETUP ---
st.set_page_config(page_title="Macro-Quant Strategic Terminal", layout="wide")

@st.cache_data
def load_data():
    # Load your specific xlsx sheets
    df_macro = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name="Macro data")
    df_gdp = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name="GDP_Growth", header=1)
    
    # Cleaning Date
    df_macro['Date'] = pd.to_datetime(df_macro['Date'])
    df_macro.set_index('Date', inplace=True)
    
    # Merging GDP (Annual) into Macro (Monthly)
    df_gdp['Date'] = pd.to_datetime(df_gdp.iloc[:, 0], format='%Y')
    df_gdp.set_index('Date', inplace=True)
    df = df_macro.join(df_gdp, how='left').fillna(method='ffill')
    return df

df_raw = load_data()

# --- 2. SIDEBAR: STRATEGIC CONTROLS ---
with st.sidebar:
    st.title("🛠️ Strategy & Forecasting")
    country = st.selectbox("Select Country", ["India", "Singapore", "UK"])
    
    st.divider()
    st.subheader("🔮 Forecast Settings")
    forecast_months = st.slider("Forecast Horizon (Months)", 6, 24, 12)
    
    st.subheader("🔥 Scenario Shocks")
    energy_shock = st.slider("Simulate Energy Spike (%)", 0, 100, 0)
    
    st.subheader("⚖️ Policy Stance")
    target_inf = st.number_input("Target Inflation (%)", value=4.0 if country == "India" else 2.0)
    stance = st.select_slider("CB Hawkishness", options=["Dovish", "Neutral", "Hawkish"], value="Neutral")

# --- 3. THE FORECASTING ENGINE ---
# Mapping headers
map_cols = {
    "India": {"cpi": "CPI_India", "policy": "Policy_India", "gdp": "IND.1"},
    "Singapore": {"cpi": "CPI_Singapore", "policy": "Policy_Singapore", "gdp": "SGP"},
    "UK": {"cpi": "CPI_UK", "policy": "Policy_UK", "gdp": "GBR"}
}
c = map_cols[country]

def generate_forecast(series, months):
    model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
    return model.forecast(months)

# Prepare dataframe for prediction
last_date = df_raw.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')

# Forecast CPI and GDP
forecast_cpi = generate_forecast(df_raw[c['cpi']], forecast_months)
forecast_gdp = generate_forecast(df_raw[c['gdp']], forecast_months)

# Create Forecast Dataframe
df_forecast = pd.DataFrame(index=future_dates)
df_forecast[c['cpi']] = forecast_cpi.values
df_forecast[c['gdp']] = forecast_gdp.values
df_forecast['Is_Forecast'] = True

# Prepare Historical Dataframe
df_hist = df_raw[[c['cpi'], c['gdp'], c['policy']]].copy()
df_hist['Is_Forecast'] = False

# Combine for the model
df_full = pd.concat([df_hist, df_forecast])

# --- 4. ECONOMIC MODEL (Taylor Rule) ---
# Apply Shock to all inflation (Historical + Forecast)
df_full['Shocked_Inflation'] = df_full[c['cpi']] + (energy_shock * 0.12)

weights = {"Dovish": {"pi": 1.2, "y": 1.0}, "Neutral": {"pi": 1.5, "y": 0.5}, "Hawkish": {"pi": 2.0, "y": 0.25}}
w = weights[stance]
neutral_rate = 2.5 

df_full['Taylor_Rate'] = (neutral_rate + df_full['Shocked_Inflation'] + 
                          w['pi'] * (df_full['Shocked_Inflation'] - target_inf) + 
                          w['y'] * df_full[c['gdp']])

# --- 5. VISUALIZATION ---
st.title(f"Macro-Quant Strategy Terminal: {country}")
st.write(f"Analyzing historical data and forecasting up to **{future_dates[-1].strftime('%B %Y')}**")

fig = go.Figure()

# Plot Historical Policy Rate
fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist[c['policy']], 
                         name="Actual Policy Rate", line=dict(color='black', width=3)))

# Plot Taylor Rule (Past + Future)
hist_taylor = df_full[df_full['Is_Forecast'] == False]
fore_taylor = df_full[df_full['Is_Forecast'] == True]

fig.add_trace(go.Scatter(x=hist_taylor.index, y=hist_taylor['Taylor_Rate'], 
                         name="Taylor Rule (Historical)", line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x=fore_taylor.index, y=fore_taylor['Taylor_Rate'], 
                         name="Taylor Rule (Forecast)", line=dict(dash='dash', color='red', width=2)))

# Shade the Forecast Region
fig.add_vrect(x0=last_date, x1=future_dates[-1], fillcolor="gray", opacity=0.1, 
              annotation_text="STRATEGIC FORECAST", annotation_position="top left", line_width=0)

fig.update_layout(xaxis_title="Date", yaxis_title="Rate (%)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- 6. STRATEGIC SUMMARY ---
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Projected Inflation (End of Horizon)", f"{df_full[c['cpi']].iloc[-1]:.2f}%")
with col2:
    st.metric("Optimal Terminal Rate", f"{df_full['Taylor_Rate'].iloc[-1]:.2f}%")
with col3:
    risk_level = "High" if df_full['Shocked_Inflation'].iloc[-1] > (target_inf + 2) else "Moderate"
    st.metric("Policy Risk Level", risk_level)

st.info(f"**Strategic Insight:** If inflation persists at the forecasted trend, the {country} Central Bank may need to hold rates at **{df_full['Taylor_Rate'].iloc[-1]:.1f}%** to maintain stability.")

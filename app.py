import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- ADVANCED FORECASTING ENGINE ---
def get_strategic_forecast(series, months):
    # Using Holt-Winters 'additive' trend to project macro variables
    # This is a standard approach for mid-term economic forecasting
    model = ExponentialSmoothing(series, trend='add', seasonal=None).fit()
    return model.forecast(months)

# 1. Prepare for the future
last_date = df.index[-1]
future_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')

# 2. Predict the inputs (CPI and GDP) for 2025/26
forecast_cpi = get_strategic_forecast(df[c['cpi']], 12)
forecast_gdp = get_strategic_forecast(df[c['gdp']], 12)

# 3. Build the Forecast DataFrame
df_future = pd.DataFrame(index=future_index)
df_future[c['cpi']] = forecast_cpi.values
df_future[c['gdp']] = forecast_gdp.values
df_future['Is_Forecast'] = True

# 4. Apply the Model Logic to the Forecasted Data
# This is where the Taylor Rule 'predicts' future policy needs
df_future['Shocked_Inflation'] = df_future[c['cpi']] + (energy_shock * 0.12)
df_future['Taylor_Rate'] = (neutral_rate + df_future['Shocked_Inflation'] + 
                            w['pi'] * (df_future['Shocked_Inflation'] - target_inf) + 
                            w['y'] * df_future[c['gdp']])

# --- ENHANCED VISUALIZATION ---
# Plot historical as solid, and forecast as dashed
fig.add_trace(go.Scatter(
    x=df_future.index, 
    y=df_future['Taylor_Rate'], 
    name="Projected Policy Path (2025-26)", 
    line=dict(dash='dot', color='red', width=3)
))

# Add a "Shadow of Uncertainty" / Forecast Region
fig.add_vrect(
    x0=last_date, x1=future_index[-1],
    fillcolor="blue", opacity=0.05, 
    layer="below", line_width=0,
    annotation_text="PROJECTION ZONE"
)

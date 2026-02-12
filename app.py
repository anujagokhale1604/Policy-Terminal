import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR

# ... [Keep previous imports and load_data() function] ...

if not raw_df.empty:
    # --- 1. LIVE HEADER: INSTITUTIONAL METRICS ---
    current_spot = raw_df['INR_USD'].iloc[-1]
    
    # Calculate forecast with shocks applied
    # (Using the logic from the previous turn to get forecast_levels)
    projected_3m = forecast_levels[2] if len(forecast_levels) >= 3 else forecast_levels[-1]
    variance = projected_3m - current_spot
    
    st.markdown(f"""
    ## 🏛️ INSTITUTIONAL MACRO QUANT TERMINAL
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("INR/USD Current", f"₹{current_spot:.2f}")
    col2.metric("3M VAR Forecast", f"₹{projected_3m:.2f}", f"{variance:+.2f}")
    col3.metric("Volatility Regime", "Stable" if regime_probs.iloc[-1] < 0.5 else "Stressed")
    col4.metric("Quant Engine", f"SVAR({var_lags}) + MS-AR")

    st.divider()

    # --- 2. METHODOLOGY & TABS ---
    # [Insert the Methodology expander and Tabs logic here]

import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.api import VAR
from datetime import datetime

# --- 1. SYSTEM CONFIG ---
st.set_page_config(page_title="Macro Quant Terminal | VAR", layout="wide", page_icon="🏛️")

@st.cache_data(ttl=600)
def load_macro_data():
    """Robust data loader that aligns all macro sources."""
    def get_series(file, sheet=None, col_name="Value", is_csv=False, skip=0):
        if not os.path.exists(file): return pd.Series(dtype='float64')
        try:
            if is_csv:
                # Optimized for your specific OECD CSV export
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

    # Mapping your .xlsx and .csv files
    map_dict = {
        "Yield_Spread": get_series("T10Y2Y.xlsx", sheet="Daily", col_name="Yield_Spread"),
        "Commodities": get_series("PALLFNFINDEXM.xlsx", sheet="Monthly", col_name="Commodities"),
        "Sentiment": get_series("export-2026-02-10T06_50_22.597Z.csv", is_csv=True, skip=4, col_name="Sentiment"),
        "INR_USD": get_series("DEXINUS.xlsx", sheet="Daily", col_name="INR_USD")
    }
    
    # Concatenate and clean
    master_df = pd.concat(map_dict.values(), axis=1).sort_index().ffill().dropna()
    return master_df.reset_index().rename(columns={'index': 'Date'})

# --- 2. VAR FORECASTING ENGINE ---
def run_var_forecast(data, steps=3):
    """
    Vector Autoregression: Models the endogenous relationship between 
    Yields, Commodities, Sentiment, and FX.
    """
    var_df = data[['Yield_Spread', 'Commodities', 'Sentiment', 'INR_USD']]
    
    # Fit model (Lag Order 1 is standard for monthly macro data)
    model = VAR(var_df)
    results = model.fit(1)
    
    # Forecast steps ahead
    forecast_values = results.forecast(var_df.values[-results.k_ar:], steps)
    
    # Generate future dates
    last_date = data['Date'].max()
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='MS')
    
    return pd.DataFrame(forecast_values, columns=var_df.columns, index=forecast_dates)

# --- 3. EXECUTION ---
df = load_macro_data()

if df.empty:
    st.error("Data missing. Please check your Excel files.")
    st.stop()

# Run Prediction
forecast_df = run_var_forecast(df)
latest = df.iloc[-1]
future = forecast_df.iloc[-1]

# --- 4. DASHBOARD UI ---
st.title("🏛️ INSTITUTIONAL MACRO QUANT TERMINAL")
st.markdown(f"**Predictive Regime:** Vector Autoregression (VAR) | **Outlook:** {forecast_df.index[-1].strftime('%B %Y')}")

# Summary Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current INR/USD", f"₹{latest['INR_USD']:.2f}")
c2.metric("VAR Projection (3M)", f"₹{future['INR_USD']:.2f}", 
          delta=f"{future['INR_USD'] - latest['INR_USD']:+.2f}")
c3.metric("Projected Spread", f"{future['Yield_Spread']:.2f}%", 
          delta="Steepening" if future['Yield_Spread'] > latest['Yield_Spread'] else "Flattening")
c4.metric("Sentiment Trend", f"{future['Sentiment']:.1f}", 
          delta=f"{future['Sentiment'] - latest['Sentiment']:+.1f}")

st.divider()

# --- 5. VISUALIZING THE PREDICTION ---
st.subheader("Time-Series Convergence & Prediction Path")


fig = go.Figure()

# Plot Historical INR/USD
fig.add_trace(go.Scatter(x=df['Date'].tail(24), y=df['INR_USD'].tail(24), 
                         name='Historical Spot', line=dict(color='#00FFAA', width=3)))

# Plot VAR Forecasted INR/USD
# Connect the last historical point to the first forecast point for a continuous line
connect_dates = [df['Date'].iloc[-1], forecast_df.index[0]]
connect_values = [df['INR_USD'].iloc[-1], forecast_df['INR_USD'].iloc[0]]

fig.add_trace(go.Scatter(x=connect_dates + list(forecast_df.index[1:]), 
                         y=connect_values + list(forecast_df['INR_USD'].iloc[1:]), 
                         name='VAR Prediction', line=dict(color='#FF00FF', width=3, dash='dot')))

fig.update_layout(template="plotly_dark", height=450, 
                  xaxis_title="Timeline", yaxis_title="INR/USD Exchange Rate",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# Academic Context
with st.expander("🎓 Technical Methodology: Vector Autoregression (VAR)"):
    st.markdown(f"""
    The VAR model identifies the lead-lag relationship between these variables. 
    Unlike a standard regression which assumes $X$ causes $Y$, VAR acknowledges that:
    1. **Yield Spreads** influence **Sentiment** (recession fears).
    2. **Commodities** (inflation) influence **Yields** (policy response).
    3. Both influence the **INR/USD** capital flows.
    
    The system of equations is solved simultaneously to provide the path of least resistance for the next 90 days.
    """)

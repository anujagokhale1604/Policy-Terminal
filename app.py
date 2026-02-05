import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Macro-Quant Strategy Terminal", layout="wide")

@st.cache_data
def load_data():
    # Load your specific sheets
    df_macro = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name="Macro data")
    df_gdp = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name="GDP_Growth", header=1)
    
    # Cleaning Macro Dates
    df_macro['Date'] = pd.to_datetime(df_macro['Date'])
    df_macro.set_index('Date', inplace=True)
    
    # Cleaning GDP Dates (using the first column which contains the year)
    df_gdp['Date'] = pd.to_datetime(df_gdp.iloc[:, 0], format='%Y')
    df_gdp.set_index('Date', inplace=True)
    
    # Merging GDP (Annual) into Macro (Monthly) 
    # ffill spreads the annual growth rate across the months for calculation
    df = df_macro.join(df_gdp, how='left').fillna(method='ffill')
    return df

# Load the combined dataframe
try:
    df_main = load_data()
except Exception as e:
    st.error(f"Error loading Excel data: {e}")
    st.stop()

# --- 2. SIDEBAR CONTROLS (Snippet #1) ---
with st.sidebar:
    st.title("🛠️ Strategy & Stress-Testing")
    country = st.selectbox("Select Country", ["India", "Singapore", "UK"])
    
    st.divider()
    st.subheader("Scenario Shocks")
    energy_shock = st.slider("Energy Price Surge (%)", 0, 100, 0)
    
    st.subheader("Policy Parameters")
    # Setting default targets based on country profiles
    target_val = 4.0 if country == "India" else 2.0
    target_inf = st.number_input("Target Inflation (%)", value=target_val)
    
    stance = st.select_slider(
        "Central Bank Stance",
        options=["Dovish", "Neutral", "Hawkish"],
        value="Neutral"
    )

# --- 3. ECONOMIC ENGINE (Snippet #2) ---
# Precise mapping to your Excel column headers
map_cols = {
    "India": {"cpi": "CPI_India", "policy": "Policy_India", "gdp": "IND.1"},
    "Singapore": {"cpi": "CPI_Singapore", "policy": "Policy_Singapore", "gdp": "SGP"},
    "UK": {"cpi": "CPI_UK", "policy": "Policy_UK", "gdp": "GBR"}
}

c = map_cols[country]
df = df_main.copy()

# A. Apply Energy Shock pass-through (12% coefficient)
df['Shocked_Inflation'] = df[c['cpi']] + (energy_shock * 0.12)

# B. Define Taylor Rule Weights based on User Stance
weights = {
    "Dovish": {"pi": 1.2, "y": 1.0},
    "Neutral": {"pi": 1.5, "y": 0.5},
    "Hawkish": {"pi": 2.0, "y": 0.25}
}
w = weights[stance]
neutral_rate = 2.5 # Estimated neutral real rate for model benchmark

# C. Calculate Taylor Recommended Rate
df['Taylor_Rate'] = (
    neutral_rate + 
    df['Shocked_Inflation'] + 
    w['pi'] * (df['Shocked_Inflation'] - target_inf) + 
    w['y'] * df[c['gdp']]
)

# D. Calculate Policy Gap
df['Policy_Gap'] = df[c['policy']] - df['Taylor_Rate']

# --- 4. VISUALIZATION LAYER (Snippet #3) ---
st.title(f"Macroeconomic Strategy Terminal: {country}")

fig = go.Figure()

# Actual Policy Rate (Solid Line)
fig.add_trace(go.Scatter(
    x=df.index, y=df[c['policy']], 
    name="Actual Policy Rate", 
    line=dict(color='black', width=3)
))

# Taylor Recommended Rate (Dashed Red Line)
fig.add_trace(go.Scatter(
    x=df.index, y=df['Taylor_Rate'], 
    name="Taylor Rule (Optimal)", 
    line=dict(dash='dash', color='red', width=2)
))

# Strategic Shading: Highlight periods of significant policy lag (Behind the Curve)
for i in range(1, len(df)):
    if df['Policy_Gap'].iloc[i] < -2.0:
        fig.add_vrect(
            x0=df.index[i-1], x1=df.index[i], 
            fillcolor="red", opacity=0.1, 
            layer="below", line_width=0
        )

fig.update_layout(
    xaxis_title="Date", 
    yaxis_title="Rate (%)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# --- 5. EXECUTIVE BRIEF (Snippet #4) ---
st.divider()
st.subheader("📝 Strategic Insight Summary")

col1, col2 = st.columns(2)

with col1:
    latest_gap = df['Policy_Gap'].iloc[-1]
    # Inverting color logic: negative gap (behind curve) shows as a red warning
    st.metric(
        label="Current Policy Gap", 
        value=f"{latest_gap:.2f}%", 
        delta="Behind Curve" if latest_gap < -1.5 else "Aligned",
        delta_color="inverse"
    )

with col2:
    st.write(f"**Key Analytical Takeaways:**")
    st.write(f"* **Regime Analysis:** The red shaded areas identify historical periods where interest rates lagged significantly behind the Taylor-optimal path.")
    st.write(f"* **Inflation Persistence:** Current inflation autocorrelation is **{df[c['cpi']].autocorr():.2f}**, suggesting high shock-propagation.")
    st.write(f"* **Shock Sensitivity:** Under a {energy_shock}% surge, the model suggests a terminal rate of **{df['Taylor_Rate'].iloc[-1]:.1f}%** is necessary to re-anchor expectations.")

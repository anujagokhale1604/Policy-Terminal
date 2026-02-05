import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- SETUP ---
st.set_page_config(page_title="Macro-Quant Strategy Terminal", layout="wide")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    # Loading your specific xlsx sheets
    df_macro = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name="Macro data")
    df_gdp = pd.read_excel("EM_Macro_Data_India_SG_UK.xlsx", sheet_name="GDP_Growth", header=1)
    
    # Cleaning Date
    df_macro['Date'] = pd.to_datetime(df_macro['Date'])
    df_macro.set_index('Date', inplace=True)
    
    # Merging GDP (Annual) into Macro (Monthly) for Taylor Rule calculation
    # We use 'ffill' to spread annual growth across months for the model
    df_gdp['Date'] = pd.to_datetime(df_gdp.iloc[:, 0], format='%Y')
    df_gdp.set_index('Date', inplace=True)
    df = df_macro.join(df_gdp, how='left').fillna(method='ffill')
    return df

df = load_data()

# --- 2. SIDEBAR CONTROLS (Snippet #1 Location) ---
st.sidebar.title("🛠️ Strategy & Stress-Testing")
country = st.sidebar.selectbox("Select Country", ["India", "Singapore", "UK"])

energy_shock = st.sidebar.slider("Energy Price Surge (%)", 0, 100, 0)
target_inf = st.sidebar.number_input("Target Inflation (%)", value=4.0 if country == "India" else 2.0)
stance = st.sidebar.select_slider("Central Bank Stance", options=["Dovish", "Neutral", "Hawkish"], value="Neutral")

# --- 3. ECONOMIC ENGINE (Snippet #2 Location) ---
# Map titles to your exact file headers
map_cols = {
    "India": {"cpi": "CPI_India", "policy": "Policy_India", "gdp": "IND.1"},
    "Singapore": {"cpi": "CPI_Singapore", "policy": "Policy_Singapore", "gdp": "SGP"},
    "UK": {"cpi": "CPI_UK", "policy": "Policy_UK", "gdp": "GBR"}
}
c = map_cols[country]

# Calculate logic
df['Shocked_Inflation'] = df[c['cpi']] + (energy_shock * 0.12)
weights = {"Dovish": {"pi": 1.2, "y": 1.0}, "Neutral": {"pi": 1.5, "y": 0.5}, "Hawkish": {"pi": 2.0, "y": 0.25}}
w = weights[stance]
neutral_rate = 2.5 

df['Taylor_Rate'] = (neutral_rate + df['Shocked_Inflation'] + 
                     w['pi'] * (df['Shocked_Inflation'] - target_inf) + 
                     w['y'] * df[c['gdp']])
df['Policy_Gap'] = df[c['policy']] - df['Taylor_Rate']

# --- 4. VISUALIZATION (Snippet #3 Location) ---
st.title(f"Macroeconomic Strategy Terminal: {country}")

fig = go.Figure()

# Actual Policy Rate Trace
fig.add_trace(go.Scatter(x=df.index, y=df[c['policy']], name="Actual Policy Rate", line=dict(color='black', width=3)))

# NEW: Taylor Rule Trace
fig.add_trace(go.Scatter(x=df.index, y=df['Taylor_Rate'], name="Taylor Rule (Optimal)", line=dict(dash='dash', color='red')))

# NEW: Strategic Shading for Policy Lags
for i in range(1, len(df)):
    if df['Policy_Gap'].iloc[i] < -2.0:
        fig.add_vrect(x0=df.index[i-1], x1=df.index[i], fillcolor="red", opacity=0.1, layer="below", line_width=0)

fig.update_layout(xaxis_title="Date", yaxis_title="Rate (%)", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- 5. STRATEGIC BRIEF (Snippet #4 Location) ---
st.divider()
st.subheader("📝 Strategic Insight Summary")
col1, col2 = st.columns(2)

with col1:
    latest_gap = df['Policy_Gap'].iloc[-1]
    st.metric("Current Policy Gap", f"{latest_gap:.2f}%", 
              delta="Behind Curve" if latest_gap < -1.5 else "Aligned", delta_color="inverse")

with col2:
    st.write(f"**Key Analytical Takeaways:**")
    st.write(f"* **Regime Analysis:** Shaded red areas indicate periods of significant policy lag.")
    st.write(f"* **Scenario:** A {energy_shock}% shock requires a {df['Taylor_Rate'].iloc[-1]:.1f}% terminal rate.")

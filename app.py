import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.api import VAR
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from datetime import datetime
import warnings
import time
import requests

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Global Macro Quant Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;700&display=swap');

.stApp{background-color:#0d1117;color:#c9d1d9;font-family:'Inter',sans-serif}
section[data-testid="stSidebar"]{background-color:#161b22 !important;border-right:1px solid #30363d}
div[data-testid="metric-container"]{background-color:#161b22;border:1px solid #30363d;padding:16px;border-radius:6px}
[data-testid="stMetricValue"]{color:#58a6ff !important;font-family:'JetBrains Mono',monospace;font-size:1.8rem !important}
[data-testid="stMetricLabel"]{color:#8b949e !important;font-size:0.75rem !important;font-family:'JetBrains Mono',monospace;letter-spacing:1px;text-transform:uppercase}
[data-testid="stMetricDelta"]{font-family:'JetBrains Mono',monospace;font-size:0.8rem !important}
.mbox{background-color:#161b22;border-left:4px solid #58a6ff;padding:14px 18px;font-size:0.85rem;margin:10px 0;color:#8b949e;border-radius:0 6px 6px 0}
.wbox{background-color:#161b22;border-left:4px solid #f0883e;padding:14px 18px;font-size:0.85rem;margin:10px 0;color:#8b949e;border-radius:0 6px 6px 0}
.sbox{background-color:#161b22;border-left:4px solid #56d364;padding:14px 18px;font-size:0.85rem;margin:10px 0;color:#8b949e;border-radius:0 6px 6px 0}
.terminal-title{font-family:'JetBrains Mono',monospace;font-size:1.1rem;color:#58a6ff;letter-spacing:2px;text-transform:uppercase}
.regime-chip{display:inline-block;padding:4px 12px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:700;letter-spacing:1px}
.chip-stable{background:#0f2a0f;color:#56d364;border:1px solid #56d364}
.chip-stress{background:#2a0f0f;color:#ff7b72;border:1px solid #ff7b72}
h1,h2,h3{color:#e6edf3 !important}
.stTabs [data-baseweb="tab"]{font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#8b949e}
.stTabs [aria-selected="true"]{color:#58a6ff !important}
hr{border-color:#30363d}
</style>
""", unsafe_allow_html=True)

MARKETS = {
    "Singapore": {
        "fx_series": "EXSIUS",
        "symbol": "S$",
        "label": "SGD/USD",
        "flag": "🇸🇬",
        "color": "#58a6ff",
    },
    "India": {
        "fx_series": "EXINUS",
        "symbol": "₹",
        "label": "INR/USD",
        "flag": "🇮🇳",
        "color": "#f0883e",
    },
    "United Kingdom": {
        "fx_series": "EXUSUK",
        "symbol": "$",
        "label": "USD/GBP",
        "flag": "🇬🇧",
        "color": "#56d364",
    },
}

FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"


class FredFetchError(Exception):
    pass


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fred(series_id):
    """Fetch data from the official FRED API with retries."""
    api_key = st.secrets.get("FRED_API_KEY", "")

    if not api_key:
        raise FredFetchError(
            "Missing FRED_API_KEY. Add it in Streamlit secrets before running the app."
        )

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "2000-01-01",
    }

    headers = {
        "User-Agent": "macro-terminal/1.0",
        "Accept": "application/json",
    }

    last_error = None

    for attempt in range(4):
        try:
            resp = requests.get(
                FRED_API_BASE,
                params=params,
                headers=headers,
                timeout=(10, 60),
            )
            resp.raise_for_status()

            payload = resp.json()
            observations = payload.get("observations", [])

            if not observations:
                raise FredFetchError(f"FRED returned no observations for {series_id}")

            df = pd.DataFrame(observations)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            df = df[["date", "value"]].dropna()
            df = df.set_index("date").sort_index()

            if df.empty:
                raise FredFetchError(f"FRED series {series_id} contains no numeric values")

            return df["value"].resample("MS").mean()

        except Exception as e:
            last_error = e
            time.sleep(2 ** attempt)

    raise FredFetchError(f"Could not fetch FRED series {series_id}: {last_error}")


def load_all_data(market_name):
    """Load FX, commodity, and yield data from FRED live."""
    fx_id = MARKETS[market_name]["fx_series"]

    series_ids = {
        "FX_Target": fx_id,
        "Commodities": "PALLFNFINDEXM",
        "Yield_Spread": "T10Y2Y",
    }

    try:
        with st.spinner(f"Fetching live FRED data for {market_name}..."):
            data = {
                name: fetch_fred(series_id)
                for name, series_id in series_ids.items()
            }
    except FredFetchError as e:
        return None, str(e)

    df = pd.concat(
        [data["Commodities"], data["Yield_Spread"], data["FX_Target"]],
        axis=1,
    ).dropna()

    df.columns = ["Commodities", "Yield_Spread", "FX_Target"]
    df = df[df.index >= "2010-01-01"]

    if len(df) < 36:
        return None, "Insufficient data for VAR estimation. Need at least 36 months."

    return df, None


def fit_var_and_forecast(df, lags, horizon, y_shock_bps, c_shock_pct):
    """Difference, standardise, fit VAR, apply shocks, forecast."""
    df_diff = df.diff().dropna()

    means = df_diff.mean()
    stds = df_diff.std()

    df_std = (df_diff - means) / stds

    model = VAR(df_std)
    result = model.fit(lags)

    stable = result.is_stable()

    last_obs = df_std.values[-lags:].copy()

    last_obs[-1, 1] += (y_shock_bps / 100) / stds["Yield_Spread"]

    commodity_delta = df["Commodities"].iloc[-1] * (c_shock_pct / 100)
    last_obs[-1, 0] += commodity_delta / stds["Commodities"]

    fc_std = result.forecast(last_obs, horizon)

    fx_changes = (fc_std[:, 2] * stds["FX_Target"]) + means["FX_Target"]

    current = df["FX_Target"].iloc[-1]
    path = current + np.cumsum(fx_changes)

    return result, df_std, path, stable


def fit_markov(series_std):
    """Fit Markov-switching AR(1) for regime detection."""
    try:
        ms = MarkovAutoregression(
            series_std,
            k_regimes=2,
            order=1,
            switching_ar=False
        ).fit(disp=False)

        prob_stressed = ms.smoothed_marginal_probabilities[1]
        current_prob = float(prob_stressed.iloc[-1])

        return prob_stressed, current_prob, True

    except Exception:
        return pd.Series(dtype=float), 0.0, False


def regime_chip(prob):
    if prob > 0.6:
        return '<span class="regime-chip chip-stress">⚠ STRESSED</span>'
    elif prob > 0.35:
        return '<span class="regime-chip chip-stress" style="border-color:#f0883e;color:#f0883e">~ TRANSITION</span>'
    else:
        return '<span class="regime-chip chip-stable">✓ STABLE</span>'


with st.sidebar:
    st.markdown('<div class="terminal-title">Terminal Controls</div>', unsafe_allow_html=True)
    st.markdown("---")

    target = st.selectbox(
        "Market Universe",
        list(MARKETS.keys()),
        format_func=lambda x: f"{MARKETS[x]['flag']} {x}"
    )

    st.markdown("---")
    st.markdown(
        '<div class="terminal-title" style="font-size:0.85rem">Shock Parameters</div>',
        unsafe_allow_html=True
    )

    y_shock = st.slider(
        "US Yield Curve Shift (bps)",
        -150,
        150,
        0,
        step=10,
        help="Simulated parallel shift in 10Y-2Y Treasury spread"
    )

    c_shock = st.slider(
        "Commodity Basket Shock (%)",
        -30,
        60,
        0,
        step=5,
        help="Exogenous commodity price index shock"
    )

    st.markdown("---")
    st.markdown(
        '<div class="terminal-title" style="font-size:0.85rem">Model Parameters</div>',
        unsafe_allow_html=True
    )

    lags = st.selectbox(
        "VAR Lag Order",
        [1, 2, 3],
        index=1,
        help="Akaike criterion typically selects 2 for monthly macro data"
    )

    horizon = st.slider(
        "Forecast Horizon (months)",
        3,
        24,
        12,
        help="Extended to 24 months — covers full projection to end-2027"
    )

    st.markdown("---")
    last_updated = datetime.now().strftime("%d %b %Y %H:%M")
    st.caption(f"Data: FRED live feed\nLast fetch: {last_updated}")
    st.caption("Gokhale (2026) · ssrn.com/abstract=6514338")


cfg = MARKETS[target]
df, err = load_all_data(target)

if err:
    st.error(f"Data error: {err}")
    st.info("If you recently updated the code, clear Streamlit cache and reboot the app.")
    st.stop()

try:
    result, df_std, path, stable = fit_var_and_forecast(
        df,
        lags,
        horizon,
        y_shock,
        c_shock
    )
except Exception as e:
    st.error(f"VAR estimation failed: {e}")
    st.stop()

prob_series, current_prob, ms_ok = fit_markov(df_std["FX_Target"])

current_spot = float(df["FX_Target"].iloc[-1])
forecast_end = float(path[-1])
pct_change = (forecast_end / current_spot - 1) * 100
last_date = df.index[-1]
forecast_dates = pd.date_range(last_date, periods=horizon + 1, freq="MS")[1:]
end_label = forecast_dates[-1].strftime("%b %Y")

st.markdown(
    f'<div class="terminal-title">{cfg["flag"]} {target.upper()} QUANTITATIVE MACRO TERMINAL</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Live FRED data · VAR({lags}) · {horizon}-month horizon · "
    f"Data through {last_date.strftime('%B %Y')}"
)

st.markdown("---")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "CURRENT SPOT",
    f"{cfg['symbol']}{current_spot:.4f}",
    f"as of {last_date.strftime('%b %Y')}"
)

c2.metric(
    f"FORECAST {end_label}",
    f"{cfg['symbol']}{forecast_end:.4f}",
    f"{pct_change:+.2f}%"
)

c3.metric(
    "REGIME PROB",
    f"{current_prob * 100:.1f}%",
    "stressed state" if current_prob > 0.5 else "stable state",
    delta_color="inverse"
)

c4.metric(
    "VAR STABILITY",
    "CONVERGED" if stable else "CHECK",
    f"Lag order {lags}"
)

c5.metric(
    "DATA POINTS",
    f"{len(df)}",
    f"months · 2010-{last_date.year}"
)

st.markdown("---")

t1, t2, t3, t4 = st.tabs([
    "📈 Scenario Projection",
    "🔄 Regime Analysis",
    "📊 Data Explorer",
    "📑 Methodology"
])

with t1:
    st.subheader(f"{cfg['label']} Trajectory — Baseline + Shock Scenario")

    fig = go.Figure()

    hist = df["FX_Target"].iloc[-60:]

    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist.values,
        name="Historical",
        line=dict(color="#8b949e", width=1.5),
        hovertemplate="%{x|%b %Y}: %{y:.4f}<extra></extra>"
    ))

    x_fc = [last_date] + list(forecast_dates)
    y_fc = [current_spot] + list(path)

    fig.add_trace(go.Scatter(
        x=x_fc,
        y=y_fc,
        name=f"{horizon}M Projection",
        line=dict(dash="dash", color=cfg["color"], width=2.5),
        hovertemplate="%{x|%b %Y}: %{y:.4f}<extra></extra>"
    ))

    resid_std = float(df["FX_Target"].diff().std())
    upper = [v + resid_std * np.sqrt(i + 1) for i, v in enumerate(path)]
    lower = [v - resid_std * np.sqrt(i + 1) for i, v in enumerate(path)]

    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates)[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ Band",
        showlegend=True,
        hoverinfo="skip"
    ))

    fig.add_vline(
        x=last_date,
        line_dash="dot",
        line_color="#30363d",
        line_width=1,
        annotation_text="Forecast start",
        annotation_font_color="#8b949e",
        annotation_font_size=11
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#21262d", title=""),
        yaxis=dict(gridcolor="#21262d", title=cfg["label"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hovermode="x unified",
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="JetBrains Mono")
    )

    st.plotly_chart(fig, use_container_width=True)

    shock_desc = []

    if y_shock != 0:
        shock_desc.append(f"US yield curve shift of {y_shock:+d}bps")

    if c_shock != 0:
        shock_desc.append(f"commodity basket shock of {c_shock:+d}%")

    shock_str = " and ".join(shock_desc) if shock_desc else "no exogenous shock applied (baseline)"

    st.markdown(f"""<div class="mbox">
    <b>Analyst Note:</b> Scenario assumes {shock_str}. VAR({lags}) estimated on
    first-differenced, standardised monthly data: Commodities index, US 10Y-2Y spread,
    and {cfg['label']}. Projection: {cfg['symbol']}{current_spot:.4f} →
    {cfg['symbol']}{forecast_end:.4f} ({pct_change:+.2f}%) by {end_label}.
    Bands show ±1σ propagation of historical FX volatility.
    </div>""", unsafe_allow_html=True)

with t2:
    st.subheader("Markov-Switching Regime Probabilities")

    if ms_ok and not prob_series.empty:
        st.markdown(
            f"Current regime: {regime_chip(current_prob)} &nbsp; "
            f"<span style='font-family:JetBrains Mono;font-size:0.85rem;color:#8b949e'>"
            f"Probability of stressed state: {current_prob * 100:.1f}%</span>",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        fig_r = go.Figure()

        fig_r.add_trace(go.Scatter(
            x=prob_series.index,
            y=prob_series.values,
            fill="tozeroy",
            line=dict(color="#ff7b72", width=1.5),
            fillcolor="rgba(255,123,114,0.15)",
            name="P(Stressed regime)",
            hovertemplate="%{x|%b %Y}: %{y:.3f}<extra></extra>"
        ))

        fig_r.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="#30363d",
            annotation_text="0.5 threshold",
            annotation_font_color="#8b949e",
            annotation_font_size=11
        )

        fig_r.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", title="P(Stressed)", range=[0, 1]),
            height=360,
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="JetBrains Mono"),
            hovermode="x unified"
        )

        st.plotly_chart(fig_r, use_container_width=True)

        st.markdown(f"""<div class="mbox">
        <b>Markov-Switching AR(1):</b> Two-regime model estimated on standardised
        {cfg['label']} monthly changes. Regime 1 = low-variance stable state;
        Regime 2 = high-variance stressed state. Current stressed-regime probability:
        <b>{current_prob * 100:.1f}%</b>.
        </div>""", unsafe_allow_html=True)

    else:
        st.warning("Markov-switching estimation did not converge. Try a different market or lag order.")

with t3:
    st.subheader("Raw Data Explorer")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**FX Rate History**")

        fig_fx = go.Figure(go.Scatter(
            x=df.index,
            y=df["FX_Target"],
            line=dict(color=cfg["color"], width=1.5),
            hovertemplate="%{x|%b %Y}: %{y:.4f}<extra></extra>"
        ))

        fig_fx.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", title=cfg["label"]),
            font=dict(family="JetBrains Mono", size=10)
        )

        st.plotly_chart(fig_fx, use_container_width=True)

    with col_b:
        st.markdown("**Commodity Index & Yield Spread**")

        fig_2 = make_subplots(specs=[[{"secondary_y": True}]])

        fig_2.add_trace(go.Scatter(
            x=df.index,
            y=df["Commodities"],
            name="Commodity Index",
            line=dict(color="#f0883e", width=1.5)
        ), secondary_y=False)

        fig_2.add_trace(go.Scatter(
            x=df.index,
            y=df["Yield_Spread"],
            name="10Y-2Y Spread",
            line=dict(color="#56d364", width=1.5, dash="dot")
        ), secondary_y=True)

        fig_2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            font=dict(family="JetBrains Mono", size=10)
        )

        st.plotly_chart(fig_2, use_container_width=True)

    st.markdown("**Summary Statistics**")

    st.dataframe(
        df.describe().round(4).style.background_gradient(cmap="Blues", axis=0),
        use_container_width=True
    )

with t4:
    st.subheader("Computational Framework")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1. Vector Autoregression (VAR)")
        st.latex(r"Y_t = \nu + A_1 Y_{t-1} + \cdots + A_p Y_{t-p} + u_t")

        st.markdown("""
        Where $Y_t = [\\text{Commodities}_t, \\text{Yield Spread}_t, \\text{FX}_t]'$.
        Estimated on first-differenced, standardised monthly data to improve stationarity.
        Lag order is selected by the user. Shock injection is applied to the last observed
        window before forecasting.
        """)

        st.markdown("#### 3. Stability Condition")
        st.latex(r"\max|\lambda_i(A)| < 1")
        st.markdown("Stability is checked using `statsmodels` VARResults.is_stable().")

    with col2:
        st.markdown("#### 2. Markov-Switching AR(1)")
        st.latex(
            r"y_t = \mu_{S_t} + \phi y_{t-1} + \varepsilon_t,"
            r"\quad \varepsilon_t \sim N(0,\sigma^2_{S_t})"
        )

        st.markdown("""
        Two-regime model where $S_t \\in \\{1,2\\}$ follows a first-order Markov chain.
        Regime 1: lower-volatility stable state. Regime 2: higher-volatility stressed state.
        """)

        st.markdown("#### 4. Data Sources")

        st.markdown("""
        | Series | Source | Frequency |
        |--------|--------|-----------|
        | SGD/USD, INR/USD, USD/GBP | FRED | Monthly |
        | IMF Commodity Price Index | FRED: PALLFNFINDEXM | Monthly |
        | US 10Y-2Y Treasury Spread | FRED: T10Y2Y | Monthly |
        """)

    st.markdown("---")

    st.markdown(f"""<div class="sbox">
    <b>Research basis:</b> This terminal operationalises the VAR framework from
    Gokhale (2026), "Cross-Country Macroeconomic Dynamics: Inflation, Growth,
    and Monetary Policy — India, Singapore, and the United Kingdom."
    SSRN Working Paper.
    <a href="https://ssrn.com/abstract=6514338" style="color:#58a6ff">
    ssrn.com/abstract=6514338</a><br><br>
    Key finding: India's CPI Granger-causes Singapore's with a two-month lag
    (p = 0.028). Singapore's CPI Granger-causes the UK's (p = 0.039).
    Reverse directions not significant.
    </div>""", unsafe_allow_html=True)

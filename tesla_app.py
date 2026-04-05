"""
Tesla Stock Price Prediction — Stunning Streamlit App
Run: streamlit run tesla_app.py
Requires: Tesla.csv in the same folder
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA · AI Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --red:     #E31937;
    --red-dim: #8B0F22;
    --bg:      #080A0C;
    --surface: #0F1318;
    --card:    #141920;
    --border:  #1E2530;
    --text:    #E8EDF5;
    --muted:   #5A6478;
    --green:   #00C896;
    --amber:   #F5A623;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
}

/* Kill default streamlit padding */
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }
header { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--red-dim); border-radius: 2px; }

/* ── HERO ── */
.hero {
    position: relative;
    padding: 2.8rem 3rem 2rem 3rem;
    background: linear-gradient(135deg, #0F1318 0%, #0A0D11 60%, #120508 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(227,25,55,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: 'TSLA';
    position: absolute;
    bottom: -20px; right: 30px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 9rem;
    color: rgba(227,25,55,0.04);
    line-height: 1;
    pointer-events: none;
}
.hero-ticker {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem;
    line-height: 1;
    letter-spacing: 4px;
    background: linear-gradient(90deg, #E31937 0%, #FF6B6B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-desc {
    font-size: 0.92rem;
    color: #8A95A8;
    max-width: 580px;
    line-height: 1.6;
}

/* ── KPI CARDS ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.8rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: var(--red-dim); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--red), transparent);
}
.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--text);
    line-height: 1;
}
.kpi-delta {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    margin-top: 0.3rem;
}
.delta-up { color: var(--green); }
.delta-down { color: var(--red); }

/* ── SECTION HEADERS ── */
.sec-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 3px;
    color: var(--text);
    margin: 1.8rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.sec-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── CARDS ── */
.glass-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── MODEL RESULT CARD ── */
.result-card {
    background: linear-gradient(135deg, var(--card), #0A0D11);
    border: 1px solid var(--red-dim);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(227,25,55,0.08), transparent 60%);
}
.result-signal {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    letter-spacing: 6px;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.signal-buy  { color: var(--green); }
.signal-sell { color: var(--red); }
.result-conf {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: var(--muted);
}

/* ── ACCURACY BADGE ── */
.acc-badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
}
.acc-high { background: rgba(0,200,150,0.12); color: var(--green); border: 1px solid rgba(0,200,150,0.25); }
.acc-mid  { background: rgba(245,166,35,0.12); color: var(--amber); border: 1px solid rgba(245,166,35,0.25); }

/* ── INSIGHT BOX ── */
.insight {
    background: rgba(227,25,55,0.05);
    border-left: 3px solid var(--red);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #9BA8BC;
    margin: 0.4rem 0;
    line-height: 1.5;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
.sidebar-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 4px;
    color: var(--red);
    margin-bottom: 0.2rem;
}
.sidebar-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

/* Streamlit widget overrides */
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
.stSlider > div > div > div { background: var(--red) !important; }
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label { color: var(--muted) !important; font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading & feature engineering ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_and_engineer():
    df = pd.read_csv(os.path.join(BASE_DIR, "Tesla.csv"))
    df = df.drop(columns=["Adj Close"], errors="ignore")

    splitted = df["Date"].str.split("/", expand=True)
    df["day"]   = splitted[1].astype(int)
    df["month"] = splitted[0].astype(int)
    df["year"]  = splitted[2].astype(int)
    df["Date"]  = pd.to_datetime(df["Date"])

    df["is_quarter_end"] = np.where(df["month"] % 3 == 0, 1, 0)
    df["open-close"]     = df["Open"] - df["Close"]
    df["low-high"]       = df["Low"]  - df["High"]
    df["target"]         = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df["pct_change"]     = df["Close"].pct_change() * 100
    df["ma20"]           = df["Close"].rolling(20).mean()
    df["ma50"]           = df["Close"].rolling(50).mean()
    df["volatility"]     = df["Close"].rolling(20).std()
    return df.dropna().reset_index(drop=True)

@st.cache_resource
def train_models(df):
    features = df[["open-close", "low-high", "is_quarter_end"]]
    target   = df["target"]
    scaler   = StandardScaler()
    X        = scaler.fit_transform(features)
    X_train, X_val, y_train, y_val = train_test_split(X, target, test_size=0.1, random_state=2022)

    model_defs = {
        "Logistic Regression": LogisticRegression(),
        "SVM (Poly Kernel)":   SVC(kernel="poly", probability=True),
        "XGBoost":             XGBClassifier(eval_metric="logloss", verbosity=0),
    }
    results = {}
    for name, m in model_defs.items():
        m.fit(X_train, y_train)
        train_auc = roc_auc_score(y_train, m.predict_proba(X_train)[:, 1])
        val_auc   = roc_auc_score(y_val,   m.predict_proba(X_val)[:, 1])
        val_preds = m.predict(X_val)
        cm        = confusion_matrix(y_val, val_preds)
        results[name] = {
            "model": m, "scaler": scaler,
            "train_auc": train_auc, "val_auc": val_auc,
            "cm": cm, "X_val": X_val, "y_val": y_val,
            "val_preds": val_preds,
        }
    return results

df = load_and_engineer()
model_results = train_models(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ TSLA·AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Stock Prediction Engine</div>', unsafe_allow_html=True)
    st.divider()

    page = st.radio("", ["🏠 Overview", "📈 Price Analysis", "🤖 Model Arena", "🔮 Live Predictor"], label_visibility="collapsed")
    st.divider()

    best_model_name = max(model_results, key=lambda k: model_results[k]["val_auc"])
    best_auc = model_results[best_model_name]["val_auc"]
    st.markdown(f"""
    <div style='font-family:DM Mono,monospace;font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem'>Best Model</div>
    <div style='font-size:.9rem;color:#E8EDF5;font-weight:600'>{best_model_name}</div>
    <div style='font-family:DM Mono,monospace;font-size:.75rem;color:#00C896'>AUC {best_auc:.4f}</div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown(f"""
    <div style='font-family:DM Mono,monospace;font-size:0.7rem;color:var(--muted);line-height:2'>
    RECORDS &nbsp; <span style='color:#E8EDF5'>{len(df):,}</span><br>
    DATE FROM &nbsp; <span style='color:#E8EDF5'>Jun 2010</span><br>
    DATE TO &nbsp; <span style='color:#E8EDF5'>Mar 2017</span><br>
    FEATURES &nbsp; <span style='color:#E8EDF5'>3 engineered</span><br>
    MODELS &nbsp; <span style='color:#E8EDF5'>3 trained</span>
    </div>
    """, unsafe_allow_html=True)

# ── Hero Banner ───────────────────────────────────────────────────────────────
latest   = df.iloc[-1]
prev     = df.iloc[-2]
chg      = latest["Close"] - prev["Close"]
chg_pct  = (chg / prev["Close"]) * 100
bull_pct = (df["target"] == 1).mean() * 100

st.markdown(f"""
<div class="hero">
  <div class="hero-ticker">TESLA · TSLA</div>
  <div class="hero-sub">NASDAQ · Stock Price Prediction · Machine Learning Intelligence</div>
  <div class="hero-desc">
    AI-powered directional prediction engine using Logistic Regression, SVM, and XGBoost —
    trained on 1,692 trading days of Tesla price data from IPO to 2017.
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
delta_cls = "delta-up" if chg >= 0 else "delta-down"
delta_sym = "▲" if chg >= 0 else "▼"

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Latest Close</div>
    <div class="kpi-value">${latest['Close']:.2f}</div>
    <div class="kpi-delta {delta_cls}">{delta_sym} ${abs(chg):.2f} ({abs(chg_pct):.2f}%)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">All-Time High</div>
    <div class="kpi-value">${df['High'].max():.2f}</div>
    <div class="kpi-delta" style="color:var(--muted)">In dataset period</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Bullish Days</div>
    <div class="kpi-value">{bull_pct:.1f}%</div>
    <div class="kpi-delta delta-up">▲ of all trading days</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Best Model AUC</div>
    <div class="kpi-value">{best_auc:.3f}</div>
    <div class="kpi-delta" style="color:var(--muted)">{best_model_name}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown('<div class="sec-header">Price History</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        fill="tozeroy",
        fillcolor="rgba(227,25,55,0.06)",
        line=dict(color="#E31937", width=1.8),
        name="Close Price",
        hovertemplate="<b>%{x|%b %d %Y}</b><br>Close: $%{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["ma20"],
        line=dict(color="#F5A623", width=1.2, dash="dot"),
        name="MA 20", opacity=0.8
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["ma50"],
        line=dict(color="#00C896", width=1.2, dash="dot"),
        name="MA 50", opacity=0.8
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, font=dict(family="DM Mono", color="#5A6478", size=11),
        xaxis=dict(gridcolor="#1E2530", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#1E2530", showgrid=True, zeroline=False, title="Price (USD)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8A95A8")),
        hovermode="x unified",
        margin=dict(t=20, b=20, l=0, r=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Volume + Volatility
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-header">Volume</div>', unsafe_allow_html=True)
        fig_vol = go.Figure(go.Bar(
            x=df["Date"], y=df["Volume"],
            marker_color=np.where(df["target"] == 1, "#00C896", "#E31937"),
            opacity=0.7,
            hovertemplate="<b>%{x|%b %d %Y}</b><br>Volume: %{y:,.0f}<extra></extra>"
        ))
        fig_vol.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, font=dict(family="DM Mono", color="#5A6478", size=10),
            xaxis=dict(gridcolor="#1E2530"), yaxis=dict(gridcolor="#1E2530"),
            margin=dict(t=10, b=10, l=0, r=0), showlegend=False
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-header">Rolling Volatility (20d)</div>', unsafe_allow_html=True)
        fig_vix = go.Figure(go.Scatter(
            x=df["Date"], y=df["volatility"],
            fill="tozeroy",
            fillcolor="rgba(245,166,35,0.07)",
            line=dict(color="#F5A623", width=1.5),
            hovertemplate="<b>%{x|%b %d %Y}</b><br>Volatility: %{y:.2f}<extra></extra>"
        ))
        fig_vix.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=260, font=dict(family="DM Mono", color="#5A6478", size=10),
            xaxis=dict(gridcolor="#1E2530"), yaxis=dict(gridcolor="#1E2530"),
            margin=dict(t=10, b=10, l=0, r=0)
        )
        st.plotly_chart(fig_vix, use_container_width=True)

    # Yearly avg + target balance
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="sec-header">Yearly Avg Close</div>', unsafe_allow_html=True)
        yearly = df.groupby("year")["Close"].mean().reset_index()
        fig_yr = go.Figure(go.Bar(
            x=yearly["year"].astype(str), y=yearly["Close"],
            marker=dict(
                color=yearly["Close"],
                colorscale=[[0, "#8B0F22"], [1, "#E31937"]],
                showscale=False
            ),
            text=yearly["Close"].round(1),
            textposition="outside",
            textfont=dict(color="#8A95A8", size=10)
        ))
        fig_yr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, font=dict(family="DM Mono", color="#5A6478", size=10),
            xaxis=dict(gridcolor="#1E2530"), yaxis=dict(gridcolor="#1E2530"),
            margin=dict(t=30, b=10, l=0, r=0)
        )
        st.plotly_chart(fig_yr, use_container_width=True)

    with col4:
        st.markdown('<div class="sec-header">Bull vs Bear Days</div>', unsafe_allow_html=True)
        tc = df["target"].value_counts()
        fig_tgt = go.Figure(go.Pie(
            values=tc.values, labels=["🐂 Bull (Up)", "🐻 Bear (Down)"],
            hole=0.6,
            marker=dict(colors=["#00C896", "#E31937"]),
            textfont=dict(family="DM Mono", size=11, color="white"),
            hovertemplate="%{label}: %{value} days (%{percent})<extra></extra>"
        ))
        fig_tgt.add_annotation(
            text=f"<b>{bull_pct:.1f}%</b><br>Bullish",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="Bebas Neue", size=22, color="#E8EDF5")
        )
        fig_tgt.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=280, showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8A95A8")),
            margin=dict(t=10, b=10, l=0, r=0)
        )
        st.plotly_chart(fig_tgt, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRICE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Price Analysis":

    st.markdown('<div class="sec-header">Candlestick Chart</div>', unsafe_allow_html=True)

    years = sorted(df["year"].unique())
    sel_years = st.select_slider("Select Year Range", options=years, value=(years[0], years[-1]))
    dff = df[(df["year"] >= sel_years[0]) & (df["year"] <= sel_years[1])]

    fig_candle = go.Figure()
    fig_candle.add_trace(go.Candlestick(
        x=dff["Date"], open=dff["Open"], high=dff["High"],
        low=dff["Low"], close=dff["Close"],
        increasing_line_color="#00C896", decreasing_line_color="#E31937",
        name="OHLC"
    ))
    fig_candle.add_trace(go.Scatter(
        x=dff["Date"], y=dff["ma20"],
        line=dict(color="#F5A623", width=1, dash="dot"),
        name="MA20", opacity=0.9
    ))
    fig_candle.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=440, font=dict(family="DM Mono", color="#5A6478", size=11),
        xaxis=dict(gridcolor="#1E2530", rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor="#1E2530", title="Price (USD)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8A95A8")),
        margin=dict(t=20, b=20, l=0, r=0)
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # Distribution plots
    st.markdown('<div class="sec-header">Feature Distributions</div>', unsafe_allow_html=True)
    cols_dist = ["Open", "High", "Low", "Close", "Volume"]
    fig_dist = make_subplots(rows=1, cols=5, subplot_titles=cols_dist)
    palette = ["#E31937", "#FF6B6B", "#F5A623", "#00C896", "#5A6478"]
    for i, col in enumerate(cols_dist):
        fig_dist.add_trace(
            go.Histogram(x=dff[col], marker_color=palette[i], opacity=0.85,
                         name=col, nbinsx=40, showlegend=False),
            row=1, col=i+1
        )
    fig_dist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, font=dict(family="DM Mono", color="#5A6478", size=10),
        margin=dict(t=30, b=10, l=0, r=0)
    )
    fig_dist.update_xaxes(gridcolor="#1E2530", zeroline=False)
    fig_dist.update_yaxes(gridcolor="#1E2530", zeroline=False)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="sec-header">Correlation Matrix</div>', unsafe_allow_html=True)
    corr_cols = ["Open", "High", "Low", "Close", "Volume", "open-close", "low-high", "is_quarter_end"]
    corr = df[corr_cols].corr().round(2)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, "#0A0D11"], [0.5, "#8B0F22"], [1, "#E31937"]],
        text=corr.values, texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        hovertemplate="%{x} × %{y}: %{z}<extra></extra>"
    ))
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, font=dict(family="DM Mono", color="#5A6478", size=10),
        margin=dict(t=10, b=10, l=0, r=0)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Quarter-end analysis
    st.markdown('<div class="sec-header">Quarter-End Effect</div>', unsafe_allow_html=True)
    qe = df.groupby("is_quarter_end")[["Open","High","Low","Close"]].mean().round(2)
    qe.index = ["Non-Quarter-End", "Quarter-End"]
    fig_qe = go.Figure()
    for col, color in zip(["Open","High","Low","Close"],["#5A6478","#E31937","#00C896","#F5A623"]):
        fig_qe.add_trace(go.Bar(name=col, x=qe.index, y=qe[col],
                                 marker_color=color, opacity=0.85))
    fig_qe.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, font=dict(family="DM Mono", color="#5A6478", size=11),
        xaxis=dict(gridcolor="#1E2530"), yaxis=dict(gridcolor="#1E2530", title="Avg Price (USD)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8A95A8")),
        margin=dict(t=10, b=10, l=0, r=0)
    )
    st.plotly_chart(fig_qe, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ARENA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Arena":

    st.markdown('<div class="sec-header">Model Performance Comparison</div>', unsafe_allow_html=True)

    # AUC bar chart
    names  = list(model_results.keys())
    t_aucs = [model_results[n]["train_auc"] for n in names]
    v_aucs = [model_results[n]["val_auc"]   for n in names]

    fig_auc = go.Figure()
    fig_auc.add_trace(go.Bar(name="Train AUC", x=names, y=t_aucs,
                              marker_color="#5A6478", opacity=0.8, text=[f"{v:.4f}" for v in t_aucs],
                              textposition="outside", textfont=dict(color="#8A95A8", size=10)))
    fig_auc.add_trace(go.Bar(name="Validation AUC", x=names, y=v_aucs,
                              marker_color="#E31937", opacity=0.9, text=[f"{v:.4f}" for v in v_aucs],
                              textposition="outside", textfont=dict(color="#E8EDF5", size=10)))
    fig_auc.add_hline(y=0.5, line_dash="dot", line_color="#F5A623",
                       annotation_text="Random baseline", annotation_font_color="#F5A623")
    fig_auc.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=320, font=dict(family="DM Mono", color="#5A6478", size=11),
        xaxis=dict(gridcolor="#1E2530"), yaxis=dict(gridcolor="#1E2530", title="ROC-AUC Score", range=[0.4, 1.0]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8A95A8")),
        margin=dict(t=30, b=10, l=0, r=0)
    )
    st.plotly_chart(fig_auc, use_container_width=True)

    # Per-model confusion matrices
    st.markdown('<div class="sec-header">Confusion Matrices</div>', unsafe_allow_html=True)
    cm_cols = st.columns(3)
    for idx, (name, res) in enumerate(model_results.items()):
        with cm_cols[idx]:
            cm = res["cm"]
            is_best = name == best_model_name
            border_col = "#E31937" if is_best else "#1E2530"
            acc_badge = f'<span class="acc-badge acc-high">BEST ⚡</span>' if is_best else ""

            st.markdown(f"""
            <div class="glass-card" style="border-color:{border_col}; text-align:center">
              <div style="font-family:DM Mono,monospace;font-size:0.72rem;color:var(--muted);
                          text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem">
                {name}
              </div>
              {acc_badge}
            </div>
            """, unsafe_allow_html=True)

            fig_cm = go.Figure(go.Heatmap(
                z=cm, x=["Pred Bear", "Pred Bull"], y=["True Bear", "True Bull"],
                colorscale=[[0, "#0A0D11"], [0.5, "#8B0F22"], [1, "#E31937"]],
                text=cm, texttemplate="%{text}",
                textfont=dict(size=18, color="white", family="Bebas Neue"),
                hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
                showscale=False
            ))
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=220, font=dict(family="DM Mono", color="#5A6478", size=10),
                margin=dict(t=10, b=10, l=0, r=0)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            val_auc = res["val_auc"]
            train_auc = res["train_auc"]
            overfit_gap = train_auc - val_auc
            st.markdown(f"""
            <div style="font-family:DM Mono,monospace;font-size:0.75rem;color:var(--muted);line-height:2;text-align:center">
            Train AUC &nbsp;<span style="color:#E8EDF5">{train_auc:.4f}</span><br>
            Val AUC &nbsp;&nbsp;&nbsp;<span style="color:#00C896">{val_auc:.4f}</span><br>
            Overfit Gap &nbsp;<span style="color:{'#F5A623' if overfit_gap > 0.05 else '#00C896'}">{overfit_gap:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

    # Model insights
    st.markdown('<div class="sec-header">Insights</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight">📌 <b>Logistic Regression</b> — Fast linear classifier. Works well when the decision boundary is approximately linear. Lower complexity, less prone to overfitting on small feature sets.</div>
    <div class="insight">📌 <b>SVM (Polynomial Kernel)</b> — Captures non-linear patterns via kernel trick. Can overfit if degree is too high; probability=True uses Platt scaling for AUC computation.</div>
    <div class="insight">📌 <b>XGBoost</b> — Gradient boosted trees. Typically strongest on tabular data; handles feature interactions natively. Check for overfitting if train/val gap is large.</div>
    <div class="insight">⚡ <b>Features used:</b> open-close (intraday momentum), low-high (daily range/volatility proxy), is_quarter_end (seasonality flag). Simple but effective directional signals.</div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Predictor":

    st.markdown('<div class="sec-header">Live Signal Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#5A6478;font-size:.88rem;margin-bottom:1.2rem">Enter today\'s OHLC values to get a directional signal — will the next close be higher or lower?</div>', unsafe_allow_html=True)

    col_inp, col_res = st.columns([1, 1])

    with col_inp:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        model_choice = st.selectbox("🤖 Choose Model", list(model_results.keys()),
                                     index=list(model_results.keys()).index(best_model_name))
        st.divider()

        open_price  = st.number_input("Open Price ($)", min_value=1.0, max_value=5000.0, value=float(round(latest["Open"], 2)), step=0.01)
        close_price = st.number_input("Close Price ($)", min_value=1.0, max_value=5000.0, value=float(round(latest["Close"], 2)), step=0.01)
        high_price  = st.number_input("High Price ($)", min_value=1.0, max_value=5000.0, value=float(round(latest["High"], 2)), step=0.01)
        low_price   = st.number_input("Low Price ($)", min_value=1.0, max_value=5000.0, value=float(round(latest["Low"], 2)), step=0.01)
        month_input = st.slider("Month", 1, 12, int(latest["month"]))
        is_qe       = 1 if month_input % 3 == 0 else 0

        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.72rem;color:var(--muted)">Quarter-End Month: <span style="color:{"#00C896" if is_qe else "#5A6478"}">{"YES" if is_qe else "NO"}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("⚡ GENERATE SIGNAL", type="primary", use_container_width=True)

    with col_res:
        if predict_btn:
            res        = model_results[model_choice]
            oc         = open_price - close_price
            lh         = low_price  - high_price
            inp        = np.array([[oc, lh, is_qe]])
            inp_scaled = res["scaler"].transform(inp)
            pred       = res["model"].predict(inp_scaled)[0]
            proba      = res["model"].predict_proba(inp_scaled)[0]
            conf       = proba[pred] * 100
            bull_prob  = proba[1] * 100
            bear_prob  = proba[0] * 100

            signal     = "BUY ▲" if pred == 1 else "SELL ▼"
            sig_class  = "signal-buy" if pred == 1 else "signal-sell"
            bg_glow    = "rgba(0,200,150,0.08)" if pred == 1 else "rgba(227,25,55,0.08)"
            border_c   = "#00C896" if pred == 1 else "#E31937"

            st.markdown(f"""
            <div class="result-card" style="border-color:{border_c};background:linear-gradient(135deg,#141920,#0A0D11);">
              <div style="background:{bg_glow};position:absolute;inset:0;border-radius:14px;"></div>
              <div style="position:relative">
                <div style="font-family:DM Mono,monospace;font-size:.72rem;color:var(--muted);
                            text-transform:uppercase;letter-spacing:.15em;margin-bottom:.8rem">
                  {model_choice} · Signal
                </div>
                <div class="result-signal {sig_class}">{signal}</div>
                <div style="font-family:DM Mono,monospace;font-size:.82rem;color:#8A95A8;margin:.5rem 0 1.2rem">
                  Confidence: <span style="color:#E8EDF5">{conf:.1f}%</span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bull_prob,
                number=dict(suffix="%", font=dict(family="Bebas Neue", size=36, color="#E8EDF5")),
                title=dict(text="Bull Probability", font=dict(family="DM Mono", size=12, color="#5A6478")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(color="#5A6478", size=9)),
                    bar=dict(color="#00C896" if bull_prob > 50 else "#E31937", thickness=0.25),
                    bgcolor="#141920",
                    bordercolor="#1E2530",
                    steps=[
                        dict(range=[0, 40],   color="#1A0508"),
                        dict(range=[40, 60],  color="#141920"),
                        dict(range=[60, 100], color="#071A12"),
                    ],
                    threshold=dict(line=dict(color="#F5A623", width=2), thickness=0.75, value=50)
                )
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=250, font=dict(color="#E8EDF5"),
                margin=dict(t=40, b=10, l=30, r=30)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Input summary
            st.markdown(f"""
            <div style="font-family:DM Mono,monospace;font-size:.75rem;color:var(--muted);
                        background:var(--card);border:1px solid var(--border);border-radius:10px;
                        padding:.9rem 1.1rem;line-height:2.2">
            Open-Close Spread &nbsp; <span style="color:#E8EDF5">{oc:+.2f}</span><br>
            Low-High Range &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#E8EDF5">{lh:.2f}</span><br>
            Quarter-End Flag &nbsp;&nbsp;<span style="color:{'#00C896' if is_qe else '#5A6478'}">{'1 (Yes)' if is_qe else '0 (No)'}</span><br>
            Bull Probability &nbsp;&nbsp;&nbsp;<span style="color:#00C896">{bull_prob:.2f}%</span><br>
            Bear Probability &nbsp;&nbsp;&nbsp;<span style="color:#E31937">{bear_prob:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="height:420px;display:flex;align-items:center;justify-content:center;
                        border:1px dashed #1E2530;border-radius:14px;
                        font-family:DM Mono,monospace;font-size:.8rem;color:#2A3040;
                        flex-direction:column;gap:1rem">
              <div style="font-size:3rem">⚡</div>
              <div>Configure inputs and click GENERATE SIGNAL</div>
            </div>
            """, unsafe_allow_html=True)

    # Historical signal replay
    st.markdown('<div class="sec-header">Historical Signal Replay</div>', unsafe_allow_html=True)
    res_best = model_results[best_model_name]
    features_all = df[["open-close", "low-high", "is_quarter_end"]]
    X_all        = res_best["scaler"].transform(features_all)
    df["pred"]   = res_best["model"].predict(X_all)
    df["pred_prob_bull"] = res_best["model"].predict_proba(X_all)[:, 1]

    fig_replay = go.Figure()
    fig_replay.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        line=dict(color="#2A3040", width=1.5), name="Close", showlegend=True
    ))
    bull_sig = df[df["pred"] == 1]
    bear_sig = df[df["pred"] == 0]
    fig_replay.add_trace(go.Scatter(
        x=bull_sig["Date"], y=bull_sig["Close"],
        mode="markers", marker=dict(color="#00C896", size=4, opacity=0.6),
        name="🐂 Bull Signal"
    ))
    fig_replay.add_trace(go.Scatter(
        x=bear_sig["Date"], y=bear_sig["Close"],
        mode="markers", marker=dict(color="#E31937", size=4, opacity=0.6),
        name="🐻 Bear Signal"
    ))
    fig_replay.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=340, font=dict(family="DM Mono", color="#5A6478", size=11),
        xaxis=dict(gridcolor="#1E2530"), yaxis=dict(gridcolor="#1E2530", title="Price (USD)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8A95A8")),
        margin=dict(t=10, b=10, l=0, r=0), hovermode="x unified"
    )
    st.plotly_chart(fig_replay, use_container_width=True)

    st.markdown("""
    <div class="insight">⚠️ <b>Disclaimer:</b> This tool is for educational and research purposes only. 
    Predictions are based on historical patterns and do not constitute financial advice. 
    Past performance does not guarantee future results.</div>
    """, unsafe_allow_html=True)

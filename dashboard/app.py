"""
Kronos Quant Pipeline Dashboard
Real-time monitoring for portfolio, signals, risk, and execution
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKTEST_RESULTS_PATH = ROOT_DIR / "experiments/backtest_results.json"
IC_HISTORY_PATH = ROOT_DIR / "research/ic_history.parquet"

from dotenv import load_dotenv
load_dotenv()

from dashboard.demo_views import generate_demo_equity_series, generate_sample_portfolio
from dashboard.live_data import (
    get_data_freshness_indicator,
    load_backtest_results,
    load_live_dashboard_state,
    load_live_news,
    load_signal_ic_history,
    load_trade_history,
)

# ============================================================================
# MAIN DASHBOARD SETUP
# ============================================================================

st.set_page_config(
    page_title="Kronos Quant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
}
.metric-value {
    font-size: 32px;
    font-weight: bold;
}
.metric-label {
    font-size: 14px;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("⚙️ Dashboard Controls")

# Data source selection
st.sidebar.header("Data Source")
data_mode = st.sidebar.radio(
    "Select data mode",
    ["Live Pipeline", "Backtest Results", "Research Analysis"]
)

# Date range
st.sidebar.header("Date Range")
days_back = st.sidebar.slider("Days to analyze", 7, 365, 30)
end_date = datetime.now()
start_date = end_date - timedelta(days=days_back)

# Load live data based on mode
live_state = None
live_portfolio = None
live_signals = None
live_snapshot = None
live_risk = None
live_broker = None
live_drift = None

if data_mode == "Live Pipeline":
    live_state = load_live_dashboard_state()
    live_portfolio = live_state.target_portfolio
    live_signals = live_state.signals
    live_snapshot = live_state.snapshot
    live_risk = live_state.risk
    live_broker = live_state.broker
    live_drift = live_state.drift

bt_metrics = None
bt_equity = None

if data_mode == "Backtest Results":
    bt_res = load_backtest_results()
    if bt_res:
        bt_metrics = bt_res.get("metrics")
        bt_equity = bt_res.get("equity_curve")

# M-15 Fix: Dynamic Universe Filter
st.sidebar.header("Universe Filter")
available_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH"]
default_tickers = available_tickers[:5]

if live_portfolio is not None and not live_portfolio.empty:
    if "Ticker" in live_portfolio.columns:
        available_tickers = live_portfolio["Ticker"].tolist()
    else:
        available_tickers = live_portfolio.index.tolist()
    default_tickers = available_tickers[:10]

show_tickers = st.sidebar.multiselect(
    "Select tickers to display",
    available_tickers,
    default=default_tickers
)

# Display data freshness in sidebar
st.sidebar.markdown("---")
st.sidebar.header("📡 Data Status")
if data_mode == "Live Pipeline" and live_state:
    for msg in live_state.messages:
        st.sidebar.caption(msg)
    if live_portfolio is not None:
        st.sidebar.text(f"Targets: {get_data_freshness_indicator(live_portfolio.attrs.get('timestamp'))}")
    if live_broker is not None:
        st.sidebar.text(f"Broker: {get_data_freshness_indicator(live_broker.timestamp)}")
    elif live_portfolio is None and live_broker is None:
        st.sidebar.warning("No live data — run pipeline and/or connect Alpaca")
else:
    st.sidebar.text("Demo / research mode")

# ============================================================================
# ALERTS & NOTIFICATIONS
# ============================================================================

def check_alerts(live_risk, live_signals, live_news, live_snapshot=None):
    """Generate alerts based on risk thresholds and anomalies."""
    alerts = []

    if live_snapshot and live_snapshot.get("should_halt"):
        alerts.append(("🔴", "CRITICAL", f"Trading halt: {live_snapshot.get('halt_reason', 'unknown')}"))

    if live_risk and live_risk.get("adjustments"):
        adj = live_risk["adjustments"]
        vix = adj.get("vix", 0)
        if vix > 35:
            alerts.append(("🔴", "CRITICAL", f"Elevated VIX: {vix:.1f}"))
        exposure = adj.get("exposure_multiplier", 1.0)
        if exposure < 0.5:
            alerts.append(("🟡", "WARNING", f"Low recommended exposure: {exposure:.0%}"))

    if live_signals is not None and len(live_signals) > 0:
        if "Predicted_Return" in live_signals.columns:
            spread = live_signals["Predicted_Return"].std()
            if spread and spread < 0.001:
                alerts.append(("🟡", "WARNING", "Signals have very low cross-sectional dispersion"))

    if live_news:
        for news in live_news:
            if abs(news.get('score', 0)) > 0.8:
                alerts.append(("🟡", "WARNING", f"Sentiment spike on {news['ticker']}: {news['score']:+.2f}"))

    return alerts

# Check and display alerts
if data_mode == "Live Pipeline":
    alerts = check_alerts(live_risk, live_signals, None, live_snapshot)
    if alerts:
        st.subheader("🚨 Active Alerts")
        for icon, level, message in alerts[:3]:  # Show max 3 alerts
            color = "#ff4444" if level == "CRITICAL" else "#ffaa00"
            st.markdown(f"""
            <div style="padding: 10px; border-left: 4px solid {color}; background: rgba(255,0,0,0.05); margin-bottom: 8px; border-radius: 4px;">
                <b>{icon} {level}</b>: {message}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ All systems operational - no active alerts")

# Top metrics row
st.header("📈 Portfolio Overview")
metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)

if data_mode == "Live Pipeline":
    if live_state:
        equity = None
        if live_broker and live_broker.account:
            equity = live_broker.account.get("equity") or live_broker.account.get("portfolio_value")
        gross = live_snapshot.get("gross_exposure", 0) if live_snapshot else 0
        net = live_snapshot.get("net_exposure", 0) if live_snapshot else 0
        n_pos = live_snapshot.get("num_positions", 0) if live_snapshot else 0
        regime = live_snapshot.get("macro_regime", "n/a") if live_snapshot else "n/a"

        with metrics_col1:
            st.metric("Portfolio Value", f"${equity:,.0f}" if equity else "—")
        with metrics_col2:
            st.metric("Gross Exposure", f"{gross * 100:.1f}%")
        with metrics_col3:
            st.metric("Net Exposure", f"{net * 100:.1f}%")
        with metrics_col4:
            st.metric("Target Positions", str(n_pos))
        with metrics_col5:
            st.metric("Macro Regime", str(regime))
    else:
        st.info("Awaiting live pipeline data.")
elif data_mode != "Live Pipeline" and bt_metrics:
    with metrics_col1:
        total_ret = bt_metrics.get("total_return", 0)
        ann_ret   = bt_metrics.get("ann_return", 0)
        st.metric(label="Total Return", value=f"{total_ret:.1%}", delta=f"Ann. {ann_ret:.1%}")
    with metrics_col2:
        st.metric(label="Sharpe Ratio", value=f"{bt_metrics.get('sharpe_ratio', 0):.2f}")
    with metrics_col3:
        md = bt_metrics.get("max_drawdown", 0)
        st.metric(label="Max Drawdown", value=f"{md:.1%}", delta_color="inverse")
    with metrics_col4:
        st.metric(label="Rebalances", value=str(bt_metrics.get("n_rebalances", "—")))
    with metrics_col5:
        st.metric(label="Calmar", value=f"{bt_metrics.get('calmar_ratio', 0):.2f}")
else:
    st.info("Awaiting backtest data.")

st.divider()

# Row 2: Equity Curve and Drawdown
st.header("📉 Performance Analytics")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Equity Curve with Drawdown")

    if bt_equity:
        bt_df = pd.DataFrame(bt_equity)
        bt_df["date"] = pd.to_datetime(bt_df["date"])
        # Normalise to 100
        bt_df["equity"] = bt_df["value"] / bt_df["value"].iloc[0] * 100
        bt_df["drawdown_pct"] = bt_df["drawdown"] * 100
        dates   = bt_df["date"]
        equity  = bt_df["equity"]
        drawdown = bt_df["drawdown_pct"]
        data_label = "Backtest (realised)"
    else:
        # Clearly labelled demo fallback
        st.caption("⚠\ufe0f Demo data — run `python simulator/backtest.py` to replace with real results")
        dates, equity, drawdown = generate_demo_equity_series(start_date, end_date)
        data_label = "Demo (random)"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.7, 0.3]
    )
    fig.add_trace(
        go.Scatter(x=dates, y=equity, name=data_label,
                   line=dict(color='#667eea', width=2),
                   fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.1)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=drawdown, name="Drawdown %",
                   line=dict(color='red', width=1),
                   fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)'),
        row=2, col=1
    )
    fig.update_layout(height=500, showlegend=True, hovermode='x unified', template='plotly_white')
    fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Risk Metrics")

    if bt_equity and len(bt_equity) >= 4:
        rets = pd.Series([r["net_ret"] for r in bt_equity])
        vals = pd.Series([r["value"]   for r in bt_equity])
        periods_per_year = 12
        ann_vol  = float(rets.std() * np.sqrt(periods_per_year))
        ann_ret  = float((vals.iloc[-1] / vals.iloc[0]) ** (periods_per_year / len(rets)) - 1)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0
        neg_rets = rets[rets < 0]
        sortino  = ann_ret / (neg_rets.std() * np.sqrt(periods_per_year)) if len(neg_rets) > 0 else 0.0
        var_95   = float(np.percentile(rets, 5))
        cvar_95  = float(rets[rets <= var_95].mean()) if len(rets[rets <= var_95]) > 0 else var_95
        peak     = vals.cummax()
        max_dd   = float(((vals - peak) / peak).min())

        def _flag(v, good, warn):
            return "🟢" if v >= good else ("🟡" if v >= warn else "🔴")

        risk_data = {
            "Metric": ["VaR (95%)", "CVaR (95%)", "Ann. Volatility", "Ann. Return", "Sharpe", "Sortino"],
            "Value":  [f"{var_95:.2%}", f"{cvar_95:.2%}", f"{ann_vol:.1%}",
                       f"{ann_ret:.1%}", f"{sharpe:.2f}", f"{sortino:.2f}"],
            "Status": [
                _flag(var_95, -0.05, -0.10),
                _flag(cvar_95, -0.07, -0.15),
                _flag(-ann_vol, -0.20, -0.35),
                _flag(ann_ret, 0.05, 0),
                _flag(sharpe, 1.0, 0.5),
                _flag(sortino, 1.2, 0.6),
            ],
        }
    else:
        st.info("Run backtest to view real risk metrics.")

    st.markdown("---")
    st.subheader("Factor Exposure")
    st.info("Factor exposure modeling is not yet available in the live pipeline.")

st.divider()

# Row 3: Signal Quality and IC
st.header("🎯 Signal Quality & Model Drift")
signal_col1, signal_col2 = st.columns([1, 1])

with signal_col1:
    st.subheader("Information Coefficient (IC) Decay")

    ic_history = load_signal_ic_history()
    horizons = ["1D", "5D", "10D", "20D", "60D"]

    if ic_history is not None and len(ic_history) > 0 and "kronos" in ic_history.columns:
        # Real IC from ic_history.parquet — compute decay by rolling tail averages
        k = ic_history["kronos"].dropna()
        ic_kronos = [
            float(k.iloc[-1])             if len(k) >= 1  else 0.0,
            float(k.tail(5).mean())       if len(k) >= 5  else float(k.mean()),
            float(k.tail(10).mean())      if len(k) >= 10 else float(k.mean()),
            float(k.tail(20).mean())      if len(k) >= 20 else float(k.mean()),
            float(k.tail(60).mean())      if len(k) >= 60 else float(k.mean()),
        ]
        # FinBERT / Macro ICs not yet computed — show placeholder
        ic_sentiment = [None] * 5
        ic_macro     = [None] * 5
        if "timestamp" in ic_history.columns:
            st.caption(f"Last IC settlement: {ic_history['timestamp'].iloc[-1]}")
        else:
            st.caption("Real IC data from research/ic_history.parquet")
    else:
        st.info("Awaiting 5-day settlement to compute real IC.")
        ic_kronos    = [0.0] * 5
        ic_sentiment = [None] * 5
        ic_macro     = [None] * 5

    fig_ic = go.Figure()
    fig_ic.add_trace(go.Scatter(
        x=horizons, y=ic_kronos, name="Kronos",
        mode='lines+markers', line=dict(color='#667eea', width=2), marker=dict(size=10)
    ))
    if any(v is not None for v in ic_sentiment):
        fig_ic.add_trace(go.Scatter(
            x=horizons, y=ic_sentiment, name="FinBERT",
            mode='lines+markers', line=dict(color='#f093fb', width=2), marker=dict(size=10)
        ))
    if any(v is not None for v in ic_macro):
        fig_ic.add_trace(go.Scatter(
            x=horizons, y=ic_macro, name="Macro",
            mode='lines+markers', line=dict(color='#4facfe', width=2), marker=dict(size=10)
        ))
    fig_ic.add_hline(y=0.02, line_dash="dash", line_color="red",
                     annotation_text="Significance threshold")
    fig_ic.update_layout(
        xaxis_title="Forecast Horizon", yaxis_title="IC (Rank Correlation)",
        height=350, template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_ic, use_container_width=True)

with signal_col2:
    st.subheader("Model Performance Drift")
    
    if live_signals is not None and len(live_signals) > 0:
        conf = live_signals.get("Confidence", pd.Series([0.0])).mean()
        st.metric("Kronos Mean Confidence", f"{conf:.1%}")
        
        # Regime detection indicator
        if live_risk and 'regime' in live_risk:
            regime = live_risk['regime']
            regime_colors = {
                'risk_on': 'green',
                'risk_off': 'red',
                'stagflation': 'orange',
                'recovery': 'blue'
            }
            regime_color = regime_colors.get(regime.get('regime', 'neutral'), 'gray')
            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background: {regime_color}20; border-left: 4px solid {regime_color};">
                <b>Current Regime:</b> {regime.get('regime', 'Unknown').replace('_', ' ').title()}<br>
                <small>Equity Bias: {regime.get('equity_bias', 0):+.2f}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Awaiting live signals.")

st.divider()

# Row 4: Portfolio Holdings
st.header("💼 Portfolio Holdings")

if data_mode == "Live Pipeline":
    tab_target, tab_actual, tab_drift = st.tabs(["Target (Pipeline)", "Actual (Broker)", "Drift"])

    with tab_target:
        if live_portfolio is not None and len(live_portfolio) > 0:
            display = live_portfolio.copy()
            if "Weight" in display.columns:
                display["Weight %"] = (display["Weight"] * 100).round(2)
            st.dataframe(display, use_container_width=True, height=350)
        else:
            st.info("No target portfolio. Run: `python3 scripts/run_production.py --once`")

    with tab_actual:
        if live_broker is not None and not live_broker.positions.empty:
            actual = live_broker.positions.copy()
            actual["Weight %"] = (actual["Weight"] * 100).round(2)
            st.dataframe(actual, use_container_width=True, height=350)
        else:
            st.info("Connect Alpaca (ALPACA_API_KEY / ALPACA_SECRET_KEY) for live holdings.")

    with tab_drift:
        if live_drift is not None and len(live_drift) > 0:
            drift = live_drift.copy()
            drift["Target %"] = (drift["Target_Weight"] * 100).round(2)
            drift["Actual %"] = (drift["Actual_Weight"] * 100).round(2)
            drift["Drift %"] = (drift["Drift"] * 100).round(2)
            st.dataframe(
                drift[["Ticker", "Target %", "Actual %", "Drift %"]].head(25),
                use_container_width=True,
                height=350,
            )
        else:
            st.info("Drift requires both pipeline targets and broker positions.")
else:
    st.info("Switch to Live Pipeline to view holdings, or run a Backtest.")

st.divider()

# Row 4.5: Trade Execution Panel
st.header("🔴 Live Trade Execution")
exec_col1, exec_col2, exec_col3 = st.columns([2, 1, 1])

with exec_col1:
    st.subheader("Broker Positions (Live)")

    if data_mode == "Live Pipeline" and live_broker is not None and not live_broker.positions.empty:
        st.dataframe(live_broker.positions, use_container_width=True, height=250)
    elif data_mode == "Live Pipeline":
        st.info("No broker positions — check Alpaca keys and KRONOS_MODE.")
    else:
        st.dataframe(generate_sample_portfolio(), use_container_width=True, height=250)

with exec_col2:
    st.subheader("Recent Fills")

    trade_history = live_state.trades if live_state else load_trade_history()
    if data_mode == "Live Pipeline" and trade_history is not None and len(trade_history) > 0:
        st.dataframe(trade_history.tail(10), use_container_width=True, height=200)
        st.metric("Total fills logged", len(trade_history))
    elif data_mode == "Live Pipeline":
        st.info("No fills yet — trades logged after paper/live rebalances.")
    else:
        st.info("Switch to Live Pipeline for execution data.")

with exec_col3:
    st.subheader("Portfolio Summary")

    if data_mode == "Live Pipeline" and live_broker is not None:
        acct = live_broker.account
        st.metric("Equity", f"${float(acct.get('equity', 0)):,.0f}")
        st.metric("Cash", f"${float(acct.get('cash', 0)):,.0f}")
        st.metric("Buying Power", f"${float(acct.get('buying_power', 0)):,.0f}")
        if live_broker.positions is not None and not live_broker.positions.empty:
            st.metric("Open Positions", len(live_broker.positions))
    elif data_mode == "Live Pipeline" and live_snapshot:
        st.metric("Target Gross", f"{live_snapshot.get('gross_exposure', 0) * 100:.1f}%")
        st.metric("Target Net", f"{live_snapshot.get('net_exposure', 0) * 100:.1f}%")
        st.metric("Positions", live_snapshot.get("num_positions", 0))
    elif data_mode == "Live Pipeline":
        st.warning("No live account data.")
    else:
        st.metric("Gross Exposure", "145.2%")
        st.metric("Net Exposure", "82.5%")
        st.metric("Long/Short", "12/3")

st.divider()

# Row 5: TCA and Execution
st.header("⚡ Transaction Cost Analysis")
tca_col1, tca_col2 = st.columns([1, 1])

with tca_col1:
    st.subheader("Market Impact Model (Almgren-Chriss)")
    
    trade_history = live_state.trades if (live_state and hasattr(live_state, 'trades')) else load_trade_history()
    
    if trade_history is not None and not trade_history.empty and 'expected_price' in trade_history.columns and 'filled_price' in trade_history.columns:
        trades = trade_history.copy()
        trades["Trade Size ($)"] = trades["qty"] * trades["filled_price"]
        trades["Actual Impact (bps)"] = abs(trades["filled_price"] - trades["expected_price"]) / trades["expected_price"] * 10000
        fig_impact = px.scatter(
            trades,
            x="Trade Size ($)",
            y="Actual Impact (bps)",
            hover_data=["symbol"],
            title="Realized Slippage by Trade Size"
        )
        fig_impact.update_layout(height=350, template='plotly_white')
        st.plotly_chart(fig_impact, use_container_width=True)
    else:
        st.info("Awaiting live trade execution data to compute market impact.")

with tca_col2:
    st.subheader("Execution Quality Summary")
    
    if trade_history is not None and not trade_history.empty:
        fill_rate = (trade_history['status'] == 'FILLED').mean() * 100 if 'status' in trade_history.columns else 100.0
        
        avg_slippage = 0.0
        if 'expected_price' in trade_history.columns and 'filled_price' in trade_history.columns:
            slippage = abs(trade_history["filled_price"] - trade_history["expected_price"]) / trade_history["expected_price"] * 10000
            avg_slippage = slippage.mean()
            
        exec_metrics = pd.DataFrame({
            "Metric": ["Avg Slippage (bps)", "Fill Rate (%)", "Total Fills"],
            "Value": [f"{avg_slippage:.1f}", f"{fill_rate:.1f}%", f"{len(trade_history)}"]
        })
        st.dataframe(exec_metrics, hide_index=True, use_container_width=True)
    else:
        st.info("No trades logged yet.")

st.divider()

# Row 6: Ensemble Signals
st.header("🎛️ Signal Ensemble")

if data_mode == "Live Pipeline" and live_signals is not None and len(live_signals) > 0:
    signal_cols = [c for c in ["Predicted_Return", "Confidence"] if c in live_signals.columns]
    st.dataframe(
        live_signals[signal_cols].head(20) if signal_cols else live_signals.head(20),
        use_container_width=True,
    )
elif data_mode == "Live Pipeline":
    st.info("No signals cached — run the production pipeline first.")
else:
    st.info("Awaiting historical signal data for backtest analysis.")

st.divider()

# Row 7: Live News Feed with Sentiment
st.header("📰 Live News & Sentiment Feed")

# Load live news
live_news = None
if data_mode == "Live Pipeline":
    tickers_for_news = show_tickers if show_tickers else None
    live_news = load_live_news(tickers=tickers_for_news, max_articles=10)

news_col1, news_col2 = st.columns([2, 1])

with news_col1:
    st.subheader("Recent Financial News")
    
    if live_news:
        for news in live_news:
            sentiment_color = "🟢" if news["sentiment"] == "positive" else "🔴" if news["sentiment"] == "negative" else "🟡"
            score_display = f"{news['score']:+.2f}"
            border_color = "green" if news["sentiment"] == "positive" else "red" if news["sentiment"] == "negative" else "orange"
            
            # Clickable headline with URL
            headline_html = f"<a href='{news['url']}' target='_blank' style='font-weight: 500; color: inherit; text-decoration: none;'>🔗 {news['headline']}</a>" if news.get('url') else f"<div style='font-weight: 500;'>{news['headline']}</div>"
            
            st.markdown(f"""
            <div style="padding: 10px; border-left: 4px solid {border_color}; margin-bottom: 10px; background: rgba(0,0,0,0.02); border-radius: 4px;">
                <div style="font-size: 12px; color: gray;">{news['time']} • {news['source']} • {news['ticker']}</div>
                {headline_html}
                <div style="font-size: 12px;">{sentiment_color} Sentiment: <b>{score_display}</b> • Impact: {news['impact']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Awaiting live news data.")

with news_col2:
    st.subheader("Sentiment Distribution")
    
    if live_news and len(live_news) > 0:
        sentiments = [n.get('sentiment', 'neutral') for n in live_news]
        sentiment_counts = pd.DataFrame(pd.Series(sentiments).value_counts()).reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        
        fig_sentiment = px.pie(
            sentiment_counts,
            values="Count",
            names="Sentiment",
            color="Sentiment",
            color_discrete_map={"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"}
        )
        fig_sentiment.update_layout(height=250, showlegend=True)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Top Mentioned Tickers")
        
        tickers = [n.get('ticker') for n in live_news if n.get('ticker')]
        if tickers:
            ticker_mentions = pd.DataFrame(pd.Series(tickers).value_counts().head(8)).reset_index()
            ticker_mentions.columns = ["Ticker", "Mentions"]
            
            fig_mentions = px.bar(
                ticker_mentions, x="Mentions", y="Ticker", orientation='h'
            )
            fig_mentions.update_layout(height=250, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_mentions, use_container_width=True)
    else:
        st.info("Awaiting live news data to compute sentiment.")

st.divider()

# Row 8: Macro Regime State Machine
st.header("🌍 Macro Regime Detection")

if live_risk and 'regime' in live_risk:
    regime_data = live_risk['regime']
    regime_name = regime_data.get('regime', 'neutral').replace('_', ' ').title()
    equity_bias = regime_data.get('equity_bias', 0.0)
    
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.metric("Current Regime", regime_name)
    with r_col2:
        st.metric("Equity Bias", f"{equity_bias:+.2f}")
else:
    st.info("Awaiting live risk data.")

st.divider()

# Row 11: Live Pipeline Status
st.header("🔴 Live Pipeline Monitor")

# Auto-refresh control
auto_refresh_monitor = st.toggle("Enable Auto-Refresh (30s)", value=False, key="monitor_refresh")
if auto_refresh_monitor:
    st.markdown('<meta http-equiv="refresh" content="30">', unsafe_allow_html=True)

# Pipeline logs
with st.expander("📋 Pipeline Execution Logs", expanded=True):
    log_path = ROOT_DIR / "logs/production.log"
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                logs_tail = f.read().splitlines()[-30:]
                st.code("\n".join(logs_tail), language='log')
        except Exception as e:
            st.error(f"Could not read logs: {e}")
    else:
        st.info("No production logs found.")

st.divider()

# Footer
st.markdown("---")
st.caption(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Kronos Quant Pipeline v2.0")

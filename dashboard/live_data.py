"""Live data loaders for the Streamlit dashboard."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

CACHE_DIR = Path("./data/cache")


@dataclass
class BrokerState:
    account: Dict
    positions: pd.DataFrame
    timestamp: datetime
    connected: bool = True


@dataclass
class LiveDashboardState:
    target_portfolio: Optional[pd.DataFrame] = None
    signals: Optional[pd.DataFrame] = None
    snapshot: Optional[Dict] = None
    broker: Optional[BrokerState] = None
    drift: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    risk: Optional[Dict] = None
    messages: List[str] = field(default_factory=list)


def get_data_freshness_indicator(timestamp: Optional[datetime]) -> str:
    if timestamp is None:
        return "No data"
    age_minutes = (datetime.now() - timestamp).total_seconds() / 60
    if age_minutes < 1:
        return "Live (<1 min ago)"
    if age_minutes < 5:
        return f"{int(age_minutes)} min ago"
    if age_minutes < 30:
        return f"{int(age_minutes)} min ago"
    return f"{int(age_minutes)} min ago - Stale"


def _read_parquet_with_ts(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.attrs["timestamp"] = datetime.fromtimestamp(path.stat().st_mtime)
    return df


def load_target_portfolio() -> Optional[pd.DataFrame]:
    df = _read_parquet_with_ts(CACHE_DIR / "latest_positions.parquet")
    if df is not None:
        return df
    if not CACHE_DIR.exists():
        return None
    files = list(CACHE_DIR.glob("positions_*.parquet"))
    if not files:
        return None
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return _read_parquet_with_ts(latest)


def load_latest_signals() -> Optional[pd.DataFrame]:
    df = _read_parquet_with_ts(CACHE_DIR / "latest_signals.parquet")
    if df is not None:
        return df
    if not CACHE_DIR.exists():
        return None
    files = list(CACHE_DIR.glob("signals_*.parquet"))
    if not files:
        return None
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return _read_parquet_with_ts(latest)


def load_pipeline_snapshot() -> Optional[Dict]:
    path = CACHE_DIR / "latest_snapshot.json"
    if not path.exists() and CACHE_DIR.exists():
        files = list(CACHE_DIR.glob("snapshot_*.json"))
        if files:
            path = max(files, key=lambda p: p.stat().st_mtime)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    data["file_timestamp"] = datetime.fromtimestamp(path.stat().st_mtime)
    return data


def load_trade_history() -> Optional[pd.DataFrame]:
    path = CACHE_DIR / "trades.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.attrs["timestamp"] = datetime.fromtimestamp(path.stat().st_mtime)
    return df


def load_signal_ic_history() -> Optional[pd.DataFrame]:
    path = Path("./research/ic_history.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_backtest_results() -> Optional[Dict]:
    path = Path("./experiments/backtest_results.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_risk_metrics() -> Optional[Dict]:
    try:
        from risk.regime_detector import UnifiedRegimeDetector

        detector = UnifiedRegimeDetector()
        metrics = detector.analyze_current_regime()
        adjustments = detector.get_risk_adjustment_factors(metrics)
        return {"metrics": metrics, "adjustments": adjustments}
    except Exception:
        return None


def load_broker_state(paper: Optional[bool] = None) -> Optional[BrokerState]:
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        return None

    if paper is None:
        mode = os.getenv("KRONOS_MODE", "dry").lower()
        paper = mode != "live"

    try:
        from execution.broker_connector import create_broker

        broker = create_broker("alpaca", paper=paper)
        if not broker.connect():
            return None

        account = broker.get_account_info()
        positions_raw = broker.get_positions()
        broker.disconnect()

        rows = []
        equity = float(account.get("equity") or account.get("portfolio_value") or 0)
        for symbol, pos in positions_raw.items():
            mv = pos.quantity * pos.current_price
            weight = mv / equity if equity > 0 else 0.0
            rows.append({
                "Ticker": symbol,
                "Qty": pos.quantity,
                "Avg_Entry": pos.avg_entry_price,
                "Current": pos.current_price,
                "Market_Value": mv,
                "Weight": weight,
                "Unrealized_PnL": pos.unrealized_pnl,
                "Unrealized_PnL_Pct": (
                    (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price * 100
                    if pos.avg_entry_price else 0.0
                ),
            })

        positions_df = pd.DataFrame(rows)
        if not positions_df.empty:
            positions_df = positions_df.set_index("Ticker")

        return BrokerState(
            account=account,
            positions=positions_df,
            timestamp=datetime.now(),
        )
    except Exception:
        return None


def _weight_series(portfolio: pd.DataFrame) -> pd.Series:
    if portfolio is None or portfolio.empty:
        return pd.Series(dtype=float)
    df = portfolio
    if "Weight" in df.columns:
        w = df["Weight"]
    elif "weight" in df.columns:
        w = df["weight"]
    else:
        return pd.Series(dtype=float)
    if w.index.name != "Ticker" and "Ticker" in df.columns:
        return df.set_index("Ticker")[w.name]
    return w


def compute_drift(
    target: Optional[pd.DataFrame],
    broker: Optional[BrokerState],
) -> Optional[pd.DataFrame]:
    if target is None or broker is None or broker.positions.empty:
        return None

    target_w = _weight_series(target)
    if target_w.index.name != "Ticker" and "Ticker" in target.columns:
        target_w = target.set_index("Ticker")["Weight"]

    actual_w = broker.positions["Weight"]
    tickers = sorted(set(target_w.index) | set(actual_w.index))
    rows = []
    for ticker in tickers:
        tw = float(target_w.get(ticker, 0.0))
        aw = float(actual_w.get(ticker, 0.0))
        rows.append({
            "Ticker": ticker,
            "Target_Weight": tw,
            "Actual_Weight": aw,
            "Drift": aw - tw,
            "Abs_Drift": abs(aw - tw),
        })

    return pd.DataFrame(rows).sort_values("Abs_Drift", ascending=False)


def load_live_news(tickers=None, max_articles=10) -> List[Dict]:
    try:
        from signals.finbert_sentiment import NewsSentimentFetcher

        fetcher = NewsSentimentFetcher()
        all_news = []

        if tickers:
            for ticker in tickers[:5]:
                articles = fetcher.fetch_news_headlines(ticker, days=3)
                for article in articles[:3]:
                    sentiment_score, _ = fetcher.analyzer.analyze_text(article["title"])
                    all_news.append(_format_news_article(article, ticker, sentiment_score))
        else:
            articles = fetcher.fetch_news_headlines("SPY", days=2)
            for article in articles[:max_articles]:
                sentiment_score, _ = fetcher.analyzer.analyze_text(article["title"])
                all_news.append(_format_news_article(article, "MARKET", sentiment_score))

        return all_news[:max_articles]
    except Exception:
        return []


def _format_news_article(article: Dict, ticker: str, sentiment_score: float) -> Dict:
    return {
        "time": article.get("published_at", "Recent"),
        "headline": article["title"],
        "source": article.get("source", "News"),
        "url": article.get("url", ""),
        "ticker": ticker,
        "sentiment": (
            "positive" if sentiment_score > 0.2
            else "negative" if sentiment_score < -0.2
            else "neutral"
        ),
        "score": sentiment_score,
        "impact": "High" if abs(sentiment_score) > 0.5 else "Medium",
    }


def load_live_dashboard_state() -> LiveDashboardState:
    state = LiveDashboardState()
    state.target_portfolio = load_target_portfolio()
    state.signals = load_latest_signals()
    state.snapshot = load_pipeline_snapshot()
    state.broker = load_broker_state()
    state.trades = load_trade_history()
    state.risk = load_risk_metrics()
    state.drift = compute_drift(state.target_portfolio, state.broker)

    if state.target_portfolio is None:
        state.messages.append("No pipeline targets — run `python3 scripts/run_production.py --once`")
    if state.broker is None:
        state.messages.append("Broker not connected — set ALPACA_API_KEY / ALPACA_SECRET_KEY for live holdings")
    else:
        state.messages.append("Broker connected")

    return state

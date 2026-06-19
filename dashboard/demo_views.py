"""Demo / placeholder views for non-live dashboard modes."""

from datetime import datetime

import numpy as np
import pandas as pd


def generate_sample_portfolio():
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH"]
    weights = np.random.uniform(-5, 15, len(tickers))
    weights = weights / np.sum(np.abs(weights)) * 100
    return pd.DataFrame({
        "Ticker": tickers,
        "Weight": weights,
        "Signal": np.random.uniform(-1, 1, len(tickers)),
        "Entry_Price": np.random.uniform(50, 500, len(tickers)),
        "Current_Price": np.random.uniform(50, 500, len(tickers)),
        "PnL_Pct": np.random.uniform(-10, 20, len(tickers)),
        "Sector": np.random.choice(["Tech", "Finance", "Health", "Consumer"], len(tickers)),
    })


def generate_sample_equity_curve(days=90):
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    returns = np.random.normal(0.0005, 0.015, days)
    equity = 100 * np.cumprod(1 + returns)
    benchmark = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.012, days))
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    return pd.DataFrame({
        "Date": dates,
        "Equity": equity,
        "Benchmark": benchmark,
        "Drawdown": drawdown,
    })


def generate_demo_equity_series(start_date, end_date):
    """Random walk equity + drawdown for dashboard fallback."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)
    returns = np.random.normal(0.0005, 0.015, n_days)
    equity = 100 * np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    return dates, equity, drawdown

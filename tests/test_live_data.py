"""Tests for dashboard live data loaders."""

import json
from pathlib import Path

import pandas as pd
import pytest

from dashboard.live_data import (
    compute_drift,
    load_target_portfolio,
    BrokerState,
)


def test_load_target_portfolio_prefers_latest(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr("dashboard.live_data.CACHE_DIR", cache)

    older = pd.DataFrame({"Weight": [0.1]}, index=["AAPL"])
    older.to_parquet(cache / "positions_20250101_120000.parquet")

    latest = pd.DataFrame({"Weight": [0.2]}, index=["MSFT"])
    latest.to_parquet(cache / "latest_positions.parquet")

    df = load_target_portfolio()
    assert df is not None
    assert "MSFT" in df.index
    assert df.loc["MSFT", "Weight"] == 0.2


def test_compute_drift():
    target = pd.DataFrame({"Weight": [0.10, -0.05]}, index=["AAPL", "MSFT"])
    broker = BrokerState(
        account={"equity": 100000},
        positions=pd.DataFrame({
            "Weight": [0.08, -0.03],
        }, index=["AAPL", "MSFT"]),
        timestamp=pd.Timestamp.now(),
    )
    drift = compute_drift(target, broker)
    assert drift is not None
    assert drift.loc[drift["Ticker"] == "AAPL", "Drift"].iloc[0] == pytest.approx(-0.02)

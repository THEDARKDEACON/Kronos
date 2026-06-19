"""Tests for PIT universe snapshot resolution."""

import pandas as pd
import pytest

from data import ingestion


def test_pit_universe_uses_nearest_prior_annual_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion, "CACHE_DIR", str(tmp_path))

    snap = pd.DataFrame({"Ticker": ["AAPL", "MSFT"], "Sector": ["Tech", "Tech"]})
    snap.to_parquet(tmp_path / "sp500_universe_2011-01-01.parquet")

    df = ingestion.get_universe(as_of_date="2011-11-01")
    assert len(df) == 2
    assert set(df["Ticker"]) == {"AAPL", "MSFT"}


def test_pit_universe_raises_when_no_prior_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestion, "CACHE_DIR", str(tmp_path))

    snap = pd.DataFrame({"Ticker": ["AAPL"], "Sector": ["Tech"]})
    snap.to_parquet(tmp_path / "sp500_universe_2012-01-01.parquet")

    with pytest.raises(ValueError, match="No universe snapshot on or before"):
        ingestion.get_universe(as_of_date="2011-11-01")

"""Tests for production runner helpers."""

import pandas as pd
import pytest

from pipeline.core import apply_long_only_portfolio


def test_apply_long_only_drops_shorts_and_renormalizes():
    portfolio = pd.DataFrame({
        "Weight": [0.5, -0.5, 0.2],
        "Sector": ["Tech", "Tech", "Tech"],
        "Position": ["LONG", "SHORT", "LONG"],
        "Predicted_Return": [0.1, -0.1, 0.05],
    }, index=["AAPL", "MSFT", "GOOGL"])

    out = apply_long_only_portfolio(portfolio)

    assert "MSFT" not in out.index
    assert len(out) == 2
    assert out["Weight"].sum() == pytest.approx(1.0)


def test_load_prev_weights_from_cache(tmp_path, monkeypatch):
    import scripts.run_production as prod

    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "latest_positions.parquet"
    pd.DataFrame({
        "Weight": [0.6, -0.4],
        "Sector": ["Tech", "Tech"],
    }, index=["AAPL", "MSFT"]).to_parquet(path)

    monkeypatch.setattr(prod, "CACHE_DIR", cache)
    monkeypatch.setattr(prod, "LATEST_POSITIONS", path)

    weights = prod._load_prev_weights()
    assert weights == {"AAPL": 0.6, "MSFT": -0.4}


def test_load_prev_weights_missing_returns_none(tmp_path, monkeypatch):
    import scripts.run_production as prod

    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(prod, "CACHE_DIR", cache)
    monkeypatch.setattr(prod, "LATEST_POSITIONS", cache / "missing.parquet")

    assert prod._load_prev_weights() is None

"""Smoke test: walk-forward backtest calls pipeline with PIT args."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pipeline.core import PipelineResult
from simulator.backtest import WalkForwardBacktest


def _minimal_ohlcv(tickers, n=400):
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    out = {}
    for i, t in enumerate(tickers):
        base = 100 + i * 10
        out[t] = pd.DataFrame({
            'open': np.linspace(base, base + 20, n),
            'high': np.linspace(base + 5, base + 25, n),
            'low': np.linspace(base - 5, base + 15, n),
            'close': np.linspace(base + 2, base + 22, n),
            'volume': np.full(n, 1e6),
            'amount': np.zeros(n),
            'timestamps': dates,
        })
    return out


@pytest.mark.slow
def test_backtest_step_passes_pit_and_prev_weights(monkeypatch):
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    ohlcv = _minimal_ohlcv(tickers)
    captured = []

    portfolio = pd.DataFrame({
        'Weight': [0.33, -0.33, 0.0],
        'Sector': ['Tech'] * 3,
        'Position': ['LONG', 'SHORT', 'FLAT'],
        'Predicted_Return': [0.01, -0.01, 0.0],
    }, index=tickers[:3])

    def fake_run_pipeline(**kwargs):
        captured.append(kwargs)
        return PipelineResult(
            portfolio=portfolio,
            signals=pd.DataFrame(),
            kronos_signals=pd.DataFrame(),
            fundamentals_df=pd.DataFrame(),
            ohlcv_dict=kwargs.get('ohlcv_override', {}),
            safe_tickers=tickers,
        )

    monkeypatch.setattr('simulator.backtest.run_pipeline', fake_run_pipeline)
    monkeypatch.setattr(
        'simulator.backtest.get_universe',
        lambda as_of_date=None: pd.DataFrame({'Ticker': tickers, 'Sector': ['Tech'] * 3}),
    )
    monkeypatch.setattr(
        'simulator.backtest.get_fundamentals',
        lambda universe, as_of_date=None: pd.DataFrame(
            {'Sector': ['Tech'] * 3, 'PE_Ratio': [20.0] * 3, 'Debt_To_Equity': [30.0] * 3},
            index=tickers,
        ),
    )
    monkeypatch.setattr('simulator.backtest.get_historical_ohlcv', lambda *a, **k: ohlcv)

    sim = WalkForwardBacktest(
        start_date='2024-01-01',
        end_date='2024-02-01',
        universe_limit=3,
        strict_pit=False,
        use_ensemble=False,
    )
    sim.ohlcv_dict = ohlcv
    sim.seed_tickers = tickers
    sim.previous_weights = pd.Series({'AAPL': 0.2, 'MSFT': -0.2})

    result = sim.step('2024-01-01')

    assert result is not None
    assert len(captured) == 1
    assert captured[0]['as_of_date'] == '2024-01-01'
    assert captured[0]['ohlcv_override'] is not None
    assert captured[0]['prev_weights'] == {'AAPL': 0.2, 'MSFT': -0.2}

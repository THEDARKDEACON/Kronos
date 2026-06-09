"""Integration tests for pipeline.core with mocked data layer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pipeline.core import portfolio_to_weights, run_pipeline


@pytest.fixture
def mock_pipeline_data():
    universe = pd.DataFrame({
        'Ticker': ['AAPL', 'MSFT'],
        'Sector': ['Tech', 'Tech'],
    })
    fundamentals = pd.DataFrame({
        'Sector': ['Tech', 'Tech'],
        'PE_Ratio': [20.0, 25.0],
        'Debt_To_Equity': [30.0, 40.0],
    }, index=['AAPL', 'MSFT'])

    dates = pd.date_range('2024-01-01', periods=120, freq='D')
    ohlcv = {}
    for ticker, base in [('AAPL', 100), ('MSFT', 200)]:
        ohlcv[ticker] = pd.DataFrame({
            'open': np.linspace(base, base + 10, 120),
            'high': np.linspace(base + 5, base + 15, 120),
            'low': np.linspace(base - 5, base + 5, 120),
            'close': np.linspace(base + 2, base + 12, 120),
            'volume': np.full(120, 1e6),
            'amount': np.zeros(120),
            'timestamps': dates,
        })

    kronos_signals = pd.DataFrame({
        'Predicted_Return': [0.05, -0.02],
    }, index=['AAPL', 'MSFT'])

    portfolio = pd.DataFrame({
        'Sector': ['Tech', 'Tech'],
        'Weight': [0.5, -0.5],
        'Position': ['LONG', 'SHORT'],
        'Predicted_Return': [0.05, -0.02],
    }, index=['AAPL', 'MSFT'])

    return universe, fundamentals, ohlcv, kronos_signals, portfolio


def test_run_pipeline_dry(mock_pipeline_data):
    universe, fundamentals, ohlcv, kronos_signals, portfolio = mock_pipeline_data
    mock_gen = MagicMock()
    mock_gen.generate_signals.return_value = kronos_signals

    with patch('pipeline.core.get_universe', return_value=universe), \
         patch('pipeline.core.get_fundamentals', return_value=fundamentals), \
         patch('pipeline.core.get_safe_universe', return_value=['AAPL', 'MSFT']), \
         patch('pipeline.core.get_historical_ohlcv', return_value=ohlcv), \
         patch('pipeline.core.KronosAlphaGenerator', return_value=mock_gen), \
         patch('pipeline.core.construct_portfolio', return_value=portfolio), \
         patch('pipeline.core.ENSEMBLE_AVAILABLE', False):

        result = run_pipeline(use_ensemble=False, verbose=False)

    assert len(result.portfolio) == 2
    assert result.portfolio['Weight'].abs().sum() == pytest.approx(1.0)
    mock_gen.generate_signals.assert_called_once()


def test_portfolio_to_weights(mock_pipeline_data):
    _, _, _, _, portfolio = mock_pipeline_data
    weights = portfolio_to_weights(portfolio)
    assert weights == {'AAPL': 0.5, 'MSFT': -0.5}

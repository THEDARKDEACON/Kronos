"""Tests for research-grade pipeline wiring (PIT, IC, prev_weights)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from pipeline.core import _ohlcv_to_returns, run_pipeline


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


def test_ohlcv_to_returns_truncates_at_as_of_date():
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    ohlcv = {
        'AAPL': pd.DataFrame({
            'close': np.linspace(100, 110, 60),
            'timestamps': dates,
        })
    }
    rets = _ohlcv_to_returns(ohlcv, as_of_date='2024-01-31')
    assert not rets.empty
    assert len(rets) <= 31


def test_run_pipeline_passes_as_of_date_to_kronos(mock_pipeline_data):
    universe, fundamentals, ohlcv, kronos_signals, portfolio = mock_pipeline_data
    mock_gen = MagicMock()
    mock_gen.generate_signals.return_value = kronos_signals

    with patch('pipeline.core.get_universe', return_value=universe), \
         patch('pipeline.core.get_fundamentals', return_value=fundamentals), \
         patch('pipeline.core.get_safe_universe', return_value=['AAPL', 'MSFT']), \
         patch('pipeline.core.KronosAlphaGenerator', return_value=mock_gen), \
         patch('pipeline.core.construct_portfolio', return_value=portfolio) as mock_opt, \
         patch('pipeline.core.ENSEMBLE_AVAILABLE', False), \
         patch('pipeline.core.settle_predictions'), \
         patch('pipeline.core.record_predictions'):

        run_pipeline(
            as_of_date='2024-06-01',
            ohlcv_override=ohlcv,
            use_ensemble=False,
            prev_weights={'AAPL': 0.1},
            verbose=False,
        )

    mock_gen.generate_signals.assert_called_once()
    assert mock_gen.generate_signals.call_args.kwargs.get('as_of_date') == '2024-06-01'
    assert mock_opt.call_args.kwargs.get('prev_weights') == {'AAPL': 0.1}


def test_ensemble_adapt_weights_receives_ic_history(mock_pipeline_data):
    universe, fundamentals, ohlcv, kronos_signals, portfolio = mock_pipeline_data
    mock_gen = MagicMock()
    mock_gen.generate_signals.return_value = kronos_signals
    mock_ensemble_cls = MagicMock()
    mock_ensemble = mock_ensemble_cls.return_value
    mock_ensemble.combine_signals.return_value = pd.DataFrame({
        'Raw_Predicted_Return': [0.03, -0.01],
        'Confidence': [0.8, 0.7],
    }, index=['AAPL', 'MSFT'])

    ic_hist = {'kronos': [0.05, 0.04], 'sentiment': [], 'macro': []}

    with patch('pipeline.core.get_universe', return_value=universe), \
         patch('pipeline.core.get_fundamentals', return_value=fundamentals), \
         patch('pipeline.core.get_safe_universe', return_value=['AAPL', 'MSFT']), \
         patch('pipeline.core.KronosAlphaGenerator', return_value=mock_gen), \
         patch('pipeline.core.construct_portfolio', return_value=portfolio), \
         patch('pipeline.core.AdaptiveEnsemble', mock_ensemble_cls), \
         patch('pipeline.core.NewsSentimentFetcher') as mock_news, \
         patch('pipeline.core.MacroRegimeDetector') as mock_macro, \
         patch('pipeline.core.UnifiedRegimeDetector') as mock_risk, \
         patch('pipeline.core.load_ic_history_for_ensemble', return_value=ic_hist), \
         patch('pipeline.core.ENSEMBLE_AVAILABLE', True), \
         patch('pipeline.core.IC_TRACKER_AVAILABLE', True), \
         patch('pipeline.core.settle_predictions'), \
         patch('pipeline.core.record_predictions'):

        mock_news.return_value.get_universe_sentiment.return_value = pd.DataFrame({
            'Sentiment_Score': [0.1, -0.1],
            'Confidence': [0.5, 0.5],
        }, index=['AAPL', 'MSFT'])
        mock_macro.return_value.get_current_regime.return_value.primary_regime.value = 'risk_on'
        mock_macro.return_value.get_regime_factor_tilts.return_value = {}
        mock_risk.return_value.analyze_current_regime.return_value = MagicMock()
        mock_risk.return_value.get_risk_adjustment_factors.return_value = {'exposure_multiplier': 1.0}
        mock_risk.return_value.should_halt_trading.return_value = (False, '')

        run_pipeline(as_of_date='2024-06-01', ohlcv_override=ohlcv, verbose=False)

    mock_ensemble.adapt_weights.assert_called_once_with(ic_hist)

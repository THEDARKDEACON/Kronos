"""Tests for KronosAlphaGenerator ticker alignment."""

import numpy as np
import pandas as pd
import pytest

from strategy.kronos_alpha import KronosAlphaGenerator, KRONOS_AVAILABLE


@pytest.fixture
def sample_ohlcv():
    dfs = {}
    for ticker in ['AAPL', 'MSFT', 'GOOGL']:
        dates = pd.date_range('2024-01-01', periods=120, freq='D')
        dfs[ticker] = pd.DataFrame({
            'open': np.linspace(100, 110, 120),
            'high': np.linspace(105, 115, 120),
            'low': np.linspace(95, 105, 120),
            'close': np.linspace(102, 112, 120),
            'volume': np.full(120, 1000.0),
            'amount': np.zeros(120),
            'timestamps': dates,
        })
    return dfs


def test_generate_signals_raises_without_model(sample_ohlcv):
    if KRONOS_AVAILABLE:
        pytest.skip("Kronos is installed; fail-closed test requires missing model")

    gen = KronosAlphaGenerator.__new__(KronosAlphaGenerator)
    gen.predictor = None

    with pytest.raises(RuntimeError, match="Kronos model not loaded"):
        KronosAlphaGenerator.generate_signals(gen, sample_ohlcv)


def test_valid_ticker_list_length_matches_predictions(monkeypatch, sample_ohlcv):
    """Signal count must equal valid ticker count (guards duplicate-ticker bug)."""
    gen = KronosAlphaGenerator.__new__(KronosAlphaGenerator)
    gen.predictor = object()
    gen.batch_size = 32
    gen.use_amp = False
    gen.device = 'cpu'

    captured = {}

    def fake_predict_batch(**kwargs):
        captured['n'] = len(kwargs['df_list'])
        return [
            pd.DataFrame({'close': [105.0, 106.0]})
            for _ in kwargs['df_list']
        ]

    gen.predictor = type('P', (), {'predict_batch': lambda self, **kw: fake_predict_batch(**kw)})()

    import strategy.kronos_alpha as mod
    monkeypatch.setattr(mod, 'torch', type('T', (), {
        'cuda': type('C', (), {'amp': type('A', (), {'autocast': staticmethod(lambda **k: __import__('contextlib').nullcontext())})()})()
    })())

    signals = KronosAlphaGenerator.generate_signals(gen, sample_ohlcv, pred_len=2)

    assert len(signals) == captured['n'] == len(sample_ohlcv)
    assert set(signals.index) == set(sample_ohlcv.keys())

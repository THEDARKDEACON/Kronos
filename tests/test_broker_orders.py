"""Tests for broker execution helpers."""

from unittest.mock import MagicMock, patch

import pytest

from execution.broker_connector import ExecutionEngine, OrderSide


def test_rebalance_skips_small_deltas():
    broker = MagicMock()
    broker.connect.return_value = True
    broker.get_account_info.return_value = {'equity': 100000.0}
    broker.get_positions.return_value = {}
    broker.place_order.return_value = (True, 'order-1')
    broker.get_order_status.return_value = {'status': 'filled'}

    engine = ExecutionEngine(broker, transaction_cost_bps=5.0)
    report = engine.rebalance_portfolio({'AAPL': 0.001})

    assert report['success'] is True
    broker.place_order.assert_not_called()


def test_rebalance_places_buy_order():
    broker = MagicMock()
    broker.connect.return_value = True
    broker.get_account_info.return_value = {'equity': 100000.0}
    broker.get_positions.return_value = {}
    broker.place_order.return_value = (True, 'order-1')
    broker.get_order_status.return_value = {'status': 'filled'}

    engine = ExecutionEngine(broker, transaction_cost_bps=5.0)
    with patch.object(engine, '_estimate_slippage', return_value=0.0):
        report = engine.rebalance_portfolio({'AAPL': 0.10})

    assert report['executed'] >= 1
    order = broker.place_order.call_args[0][0]
    assert order.side == OrderSide.BUY
    assert order.symbol == 'AAPL'

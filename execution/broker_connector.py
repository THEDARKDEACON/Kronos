"""
Live Trading Connector Module
Provides paper/live trading integration with Alpaca and Interactive Brokers.
Supports order routing, position tracking, and real-time P&L monitoring.
"""

import os
import time
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    VWAP = "vwap"
    TWAP = "twap"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    time_in_force: str = "day"  # day, gtc, ioc
    id: Optional[str] = None


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    sector: str = "Unknown"


@dataclass
class PortfolioState:
    cash: float
    positions: Dict[str, Position]
    total_equity: float
    day_trading_pnl: float
    open_orders: List[Order]


class BrokerInterface(ABC):
    """Abstract base class for broker implementations."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to broker API."""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """Fetch account balance and buying power."""
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Fetch current positions."""
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> Tuple[bool, str]:
        """Submit order to broker. Returns (success, order_id)."""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order."""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """Check order fill status."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close connection."""
        pass


class AlpacaBroker(BrokerInterface):
    """
    Alpaca Markets integration.
    Supports paper trading and live trading.
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None, 
                 paper: bool = True, rate_limit_per_second: int = 200):
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.paper = paper
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        self.rate_limit = rate_limit_per_second
        self.last_request_time = 0
        self.client = None
        
    def connect(self) -> bool:
        try:
            import alpaca_trade_api as tradeapi
            self.client = tradeapi.REST(
                self.api_key, 
                self.secret_key, 
                self.base_url,
                api_version='v2'
            )
            # Test connection
            account = self.client.get_account()
            print(f"[Alpaca] Connected to {'PAPER' if self.paper else 'LIVE'} trading")
            print(f"[Alpaca] Account Status: {account.status}")
            print(f"[Alpaca] Buying Power: ${float(account.buying_power):,.2f}")
            return True
        except Exception as e:
            print(f"[Alpaca] Connection failed: {e}")
            return False
    
    def _rate_limit(self):
        """Simple rate limiting."""
        min_interval = 1.0 / self.rate_limit
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()
    
    def get_account_info(self) -> Dict:
        self._rate_limit()
        account = self.client.get_account()
        return {
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'equity': float(account.equity),
            'portfolio_value': float(account.portfolio_value),
            'status': account.status
        }
    
    def get_positions(self) -> Dict[str, Position]:
        self._rate_limit()
        positions = {}
        for pos in self.client.list_positions():
            positions[pos.symbol] = Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                unrealized_pnl=float(pos.unrealized_pl),
                realized_pnl=float(getattr(pos, 'realized_pl', 0.0))
            )
        return positions
    
    def place_order(self, order: Order) -> Tuple[bool, str]:
        self._rate_limit()
        try:
            side = order.side.value
            qty = abs(order.quantity)  # Alpaca doesn't use signed quantities
            
            if order.order_type == OrderType.MARKET:
                alpaca_order = self.client.submit_order(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force=order.time_in_force
                )
            elif order.order_type == OrderType.LIMIT and order.limit_price:
                alpaca_order = self.client.submit_order(
                    symbol=order.symbol,
                    qty=qty,
                    side=side,
                    type='limit',
                    time_in_force=order.time_in_force,
                    limit_price=order.limit_price
                )
            else:
                return False, "Unsupported order type"
            
            order.id = alpaca_order.id
            print(f"[Alpaca] Order placed: {side.upper()} {qty} {order.symbol} @ {order.order_type.value}")
            return True, alpaca_order.id
            
        except Exception as e:
            print(f"[Alpaca] Order failed: {e}")
            return False, str(e)
    
    def cancel_order(self, order_id: str) -> bool:
        self._rate_limit()
        try:
            self.client.cancel_order(order_id)
            return True
        except Exception as e:
            print(f"[Alpaca] Cancel failed: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict:
        self._rate_limit()
        try:
            order = self.client.get_order(order_id)
            return {
                'id': order.id,
                'status': order.status,  # new, partially_filled, filled, cancelled
                'filled_qty': float(order.filled_qty),
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None
            }
        except Exception as e:
            return {'error': str(e)}
    
    def disconnect(self):
        print("[Alpaca] Disconnected")


class InteractiveBrokersConnector(BrokerInterface):
    """
    Interactive Brokers TWS/IB Gateway integration.
    Uses ib_insync for async communication.
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, 
                 client_id: int = 1, paper: bool = True):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.paper = paper
        self.ib = None
        
    def connect(self) -> bool:
        try:
            from ib_insync import IB
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            print(f"[IBKR] Connected to TWS @ {self.host}:{self.port}")
            print(f"[IBKR] {'PAPER' if self.paper else 'LIVE'} trading mode")
            return self.ib.isConnected()
        except Exception as e:
            print(f"[IBKR] Connection failed: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        if not self.ib or not self.ib.isConnected():
            return {}
        account = self.ib.accountValues()
        summary = {av.tag: av.value for av in account if av.currency == 'USD'}
        return {
            'cash': float(summary.get('CashBalance', 0)),
            'buying_power': float(summary.get('BuyingPower', 0)),
            'equity': float(summary.get('NetLiquidation', 0)),
            'available_funds': float(summary.get('AvailableFunds', 0))
        }
    
    def get_positions(self) -> Dict[str, Position]:
        if not self.ib or not self.ib.isConnected():
            return {}
        
        positions = {}
        for pos in self.ib.positions():
            symbol = pos.contract.symbol
            positions[symbol] = Position(
                symbol=symbol,
                quantity=float(pos.position),
                avg_entry_price=float(pos.avgCost) if pos.avgCost else 0.0,
                current_price=0.0,  # Would need market data subscription
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
        return positions
    
    def place_order(self, order: Order) -> Tuple[bool, str]:
        if not self.ib or not self.ib.isConnected():
            return False, "Not connected"
        
        from ib_insync import Stock, MarketOrder, LimitOrder
        
        contract = Stock(order.symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        
        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(order.side.value, abs(order.quantity))
        elif order.order_type == OrderType.LIMIT and order.limit_price:
            ib_order = LimitOrder(order.side.value, abs(order.quantity), order.limit_price)
        else:
            return False, "Unsupported order type"
        
        trade = self.ib.placeOrder(contract, ib_order)
        order.id = str(trade.order.orderId)
        
        print(f"[IBKR] Order placed: {order.side.value.upper()} {abs(order.quantity)} {order.symbol}")
        return True, order.id
    
    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError(
            "IBKR cancel_order is not implemented. Track the Trade object "
            "returned by placeOrder() and call ib.cancelOrder(trade.order) directly."
        )
    
    def get_order_status(self, order_id: str) -> Dict:
        if not self.ib or not self.ib.isConnected():
            return {}
        # Simplified - would need to track trades
        return {}
    
    def disconnect(self):
        if self.ib:
            self.ib.disconnect()
            print("[IBKR] Disconnected")


class ExecutionEngine:
    """
    High-level execution engine that manages portfolio transitions.
    Handles rebalancing from target weights to actual positions.
    """
    
    def __init__(self, broker: BrokerInterface, transaction_cost_bps: float = 5.0):
        self.broker = broker
        self.transaction_cost = transaction_cost_bps / 10000.0
        self.execution_history = []
        try:
            from execution.algorithmic_execution import ExecutionManagementSystem
            self.ems = ExecutionManagementSystem(self.broker)
            self.ems.start()
        except ImportError:
            self.ems = None
        
    def rebalance_portfolio(self, target_weights: Dict[str, float], 
                           sector_map: Dict[str, str] = None,
                           use_vwap: bool = False) -> Dict:
        """
        Execute portfolio rebalancing to match target weights.
        
        Args:
            target_weights: Dict of symbol -> target portfolio weight
            sector_map: Optional sector mapping for logging
            use_vwap: Use VWAP execution instead of market orders
        
        Returns:
            Execution report with fills, costs, and slippage
        """
        if not self.broker.connect():
            return {'success': False, 'error': 'Broker connection failed'}
        
        # Get current positions
        current_positions = self.broker.get_positions()
        account = self.broker.get_account_info()
        
        current_weights = {}
        total_equity = account.get('equity', 0)
        
        for symbol, pos in current_positions.items():
            position_value = pos.quantity * pos.current_price
            current_weights[symbol] = position_value / total_equity if total_equity > 0 else 0
        
        # Fetch real last-close prices for all symbols in one batch
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())
        live_prices: Dict[str, float] = {}

        # First pass: use prices already in current positions (free, no API call)
        for symbol, pos in current_positions.items():
            if pos.current_price and pos.current_price > 0:
                live_prices[symbol] = pos.current_price

        # Second pass: fetch missing prices from yfinance in one batch call
        missing = [s for s in all_symbols if s not in live_prices]
        if missing:
            try:
                import yfinance as yf
                raw = yf.download(
                    missing, period="2d", progress=False, group_by="ticker",
                    auto_adjust=True
                )
                for sym in missing:
                    try:
                        if len(missing) == 1:
                            price = float(raw["Close"].dropna().iloc[-1])
                        else:
                            price = float(raw[sym]["Close"].dropna().iloc[-1])
                        if price > 0:
                            live_prices[sym] = price
                    except Exception:
                        pass
            except Exception as e:
                print(f"[Execution] yfinance price fetch failed: {e}")

        # Calculate required trades
        trades = []
        trade_prices: Dict[str, float] = {}  # symbol -> price used for share sizing (C-3)
        for symbol in all_symbols:
            target = target_weights.get(symbol, 0)
            current = current_weights.get(symbol, 0)
            delta = target - current

            if abs(delta) > 0.001:  # 10 bps minimum trade threshold
                trade_value = delta * total_equity
                current_price = live_prices.get(symbol)  # C-2: no fallback
                if not current_price or current_price <= 0:
                    print(f"   [Execution] SKIPPING {symbol}: no reliable price (fetch failed)")
                    continue
                shares = int(trade_value / current_price)

                if shares != 0:
                    side = OrderSide.BUY if shares > 0 else OrderSide.SELL
                    order_type = OrderType.VWAP if use_vwap else OrderType.MARKET

                    trades.append(Order(
                        symbol=symbol,
                        side=side,
                        quantity=abs(shares),
                        order_type=order_type
                    ))
                    trade_prices[symbol] = current_price  # C-3: record for cost accounting
        
        # Execute trades
        executed = []
        failed = []
        fills = []
        total_cost = 0.0
        
        for order in trades:
            success = False
            order_id = None
            
            if use_vwap and self.ems:
                self.ems.submit_twap_order(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    side=order.side,
                    duration_minutes=60,
                    num_buckets=10
                )
                # Assume success for EMS submission
                success = True
                order_id = "EMS_PENDING"
            else:
                success, order_id = self.broker.place_order(order)
                
            if success:
                executed.append(order)
                trade_value = order.quantity * trade_prices.get(order.symbol, 0.0)  # C-3: use actual price
                cost = trade_value * self.transaction_cost
                total_cost += cost
                
                time.sleep(0.1)
                status = self.broker.get_order_status(order_id)
                print(f"   [Execution] {order.symbol} {order.side.value}: {status.get('status', 'pending')}")
                fills.append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'qty': float(order.quantity),
                    'order_type': order.order_type.value,
                    'status': status.get('status', 'unknown'),
                    'filled_avg_price': status.get('filled_avg_price'),
                    'order_id': order_id,
                    'slippage_bps': self._estimate_slippage([order]),
                })
            else:
                failed.append(order)
        
        # Generate execution report
        report = {
            'success': len(failed) == 0,
            'timestamp': datetime.now().isoformat(),
            'total_trades': len(trades),
            'executed': len(executed),
            'failed': len(failed),
            'fills': fills,
            'transaction_costs': total_cost,
            'estimated_slippage_bps': self._estimate_slippage(executed),
            'positions_before': len(current_positions),
            'positions_after': len(target_weights),
            'turnover': sum(abs(t.quantity) for t in trades) / total_equity if total_equity > 0 else 0
        }
        
        self.execution_history.append(report)
        return report
    
    def execute_vwap_order(self, order: Order, 
                          num_buckets: int = 10,
                          duration_minutes: int = 60) -> Tuple[bool, List[str]]:
        """
        Execute a time-sliced order to reduce market impact.

        NOTE: This is a TWAP-style implementation (equal time buckets), NOT true
        VWAP. True VWAP requires real-time volume data to size each bucket
        proportionally to historical intraday volume. Rename/upgrade if live
        volume participation tracking is added.

        Splits a large order into IOC limit-order chunks executed at fixed
        intervals to reduce market impact.

        Args:
            order: The order to execute
            num_buckets: Number of time buckets to split order into
            duration_minutes: Total duration for execution

        Returns:
            (success, list_of_order_ids)
        """
        if order.order_type != OrderType.VWAP:
            # Fallback to regular execution
            success, order_id = self.broker.place_order(order)
            return success, [order_id] if success else []
        
        print(f"   [VWAP] Executing {order.side.value} {order.quantity} {order.symbol} over {duration_minutes} minutes...")
        
        # Calculate chunk sizes
        base_chunk = order.quantity // num_buckets
        remainder = order.quantity % num_buckets
        
        chunks = []
        for i in range(num_buckets):
            chunk_size = base_chunk + (1 if i < remainder else 0)
            if chunk_size > 0:
                chunks.append(chunk_size)
        
        # Execute chunks
        order_ids = []
        interval_seconds = (duration_minutes * 60) // len(chunks)
        
        for i, chunk_size in enumerate(chunks):
            chunk_order = Order(
                symbol=order.symbol,
                side=order.side,
                quantity=chunk_size,
                order_type=OrderType.LIMIT,  # Use limit for VWAP chunks
                time_in_force='ioc',
                id=None
            )
            
            success, order_id = self.broker.place_order(chunk_order)
            if success:
                order_ids.append(order_id)
                print(f"   [VWAP] Bucket {i+1}/{len(chunks)}: {chunk_size} shares")
            else:
                print(f"   [VWAP] Bucket {i+1}/{len(chunks)}: FAILED")
            
            # Wait between buckets
            if i < len(chunks) - 1:
                time.sleep(interval_seconds)  # L-2: removed redundant inner import
        
        success = len(order_ids) == len(chunks)
        return success, order_ids
    
    def _estimate_slippage(self, orders: List[Order]) -> float:
        """Estimate slippage based on order size and market impact model."""
        # Simplified Almgren-Chriss style model
        total_slippage = 0.0
        for order in orders:
            # Larger orders = more slippage
            size_factor = np.log10(order.quantity + 1) / 10  # ~0.1 for 1000 shares
            volatility_factor = 0.02  # Assumed 2% daily vol
            total_slippage += size_factor * volatility_factor * 10000  # Convert to bps
        return total_slippage / len(orders) if orders else 0.0
    
    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio snapshot."""
        account = self.broker.get_account_info()
        positions = self.broker.get_positions()
        
        return PortfolioState(
            cash=account.get('cash', 0),
            positions=positions,
            total_equity=account.get('equity', 0),
            day_trading_pnl=0.0,  # Calculate from fills
            open_orders=[]  # Track pending orders
        )
    
    def emergency_liquidate(self, timeout_seconds: int = 60) -> bool:
        """Emergency close all positions at market.

        Args:
            timeout_seconds: Hard wall-clock deadline; positions not submitted
                             before this limit are skipped with a warning.

        Returns:
            True only if every position was submitted successfully.
        """
        print("[🚨 EMERGENCY LIQUIDATION INITIATED]")

        # Verify broker is reachable before firing orders
        if not self.broker.connect():
            print("[🚨 LIQUIDATION ABORTED] Broker connection failed")
            return False

        positions = self.broker.get_positions()
        if not positions:
            print("[🚨 LIQUIDATION] No open positions found")
            return True

        succeeded: List[str] = []
        failed: List[str] = []
        deadline = time.time() + timeout_seconds

        for symbol, pos in positions.items():
            if time.time() > deadline:
                remaining = len(positions) - len(succeeded) - len(failed)
                print(f"[🚨 LIQUIDATION TIMEOUT] {remaining} position(s) not submitted")
                break

            if pos.quantity == 0:
                continue

            side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
            order = Order(
                symbol=symbol,
                side=side,
                quantity=abs(int(pos.quantity)),
                order_type=OrderType.MARKET,
                time_in_force='ioc',
            )
            success, order_id = self.broker.place_order(order)
            if success:
                succeeded.append(symbol)
                print(f"   [Liquidate] ✓ {symbol}: {pos.quantity} shares → order {order_id}")
            else:
                failed.append(symbol)
                print(f"   [Liquidate] ✗ {symbol}: ORDER FAILED")

        print(f"[🚨 LIQUIDATION COMPLETE] Succeeded: {len(succeeded)}, Failed: {len(failed)}")
        if failed:
            print(f"   Failed positions: {failed}")
        return len(failed) == 0


# Convenience factory
def create_broker(broker_type: str = 'alpaca', **kwargs) -> BrokerInterface:
    """Factory function to create broker instances."""
    if broker_type.lower() == 'alpaca':
        return AlpacaBroker(**kwargs)
    elif broker_type.lower() == 'alpaca_fix':
        return AlpacaFixBroker(**kwargs)
    elif broker_type.lower() == 'ibkr':
        return InteractiveBrokersConnector(**kwargs)
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")

import threading
import sys

class AlpacaFixBroker(AlpacaBroker):
    """
    FIX-enabled Alpaca Broker.
    Uses REST for get_account and get_positions (since FIX is only for order routing),
    but uses the QuickFIX engine for place_order and execution reports.
    """
    def __init__(self, api_key: str = None, secret_key: str = None, 
                 paper: bool = True, fix_config_path: str = "config/alpaca_fix.cfg"):
        super().__init__(api_key, secret_key, paper, rate_limit_per_second=200)
        self.fix_config_path = fix_config_path
        self.fix_engine = None
        self.initiator = None
        self._fix_thread = None
        self.fix_password = os.getenv('ALPACA_FIX_PASSWORD', self.secret_key)
        
    def connect(self) -> bool:
        # First connect REST
        rest_connected = super().connect()
        if not rest_connected:
            return False
            
        try:
            import quickfix as fix
            from execution.fix_engine import AlpacaFixEngine
        except ImportError:
            print("[FIX] quickfix package not installed. Cannot use AlpacaFixBroker.")
            return False
            
        try:
            self.fix_engine = AlpacaFixEngine(
                password=self.fix_password,
                execution_callback=self._on_execution_report
            )
            settings = fix.SessionSettings(self.fix_config_path)
            storeFactory = fix.FileStoreFactory(settings)
            logFactory = fix.FileLogFactory(settings)
            
            self.initiator = fix.SocketInitiator(self.fix_engine, storeFactory, settings, logFactory)
            
            self._fix_thread = threading.Thread(target=self.initiator.start)
            self._fix_thread.daemon = True
            self._fix_thread.start()
            
            print(f"[FIX] Initiator thread started using config {self.fix_config_path}")
            return True
        except Exception as e:
            print(f"[FIX] Failed to start FIX engine: {e}")
            return False
            
    def _on_execution_report(self, report):
        print(f"[FIX] Execution Update: {report}")
        
    def place_order(self, order: Order) -> Tuple[bool, str]:
        if self.fix_engine and self.fix_engine.is_logged_in:
            try:
                clOrdID = self.fix_engine.send_order(
                    symbol=order.symbol,
                    qty=abs(order.quantity),
                    side=order.side.value.upper(),
                    order_type=order.order_type.value.upper(),
                    limit_price=order.limit_price
                )
                if clOrdID:
                    order.id = clOrdID
                    return True, clOrdID
            except Exception as e:
                print(f"[FIX] Exception sending order: {e}")
                
        print("[FIX] Falling back to REST API for order placement...")
        return super().place_order(order)
        
    def disconnect(self):
        if self.initiator:
            self.initiator.stop()
        super().disconnect()

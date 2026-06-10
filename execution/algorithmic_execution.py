"""
Execution Management System (EMS)
Provides Algorithmic Execution (TWAP) and Broker Reconciliation State Management.
"""

import json
import time
import os
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from execution.broker_connector import BrokerInterface, Order, OrderType, OrderSide

CACHE_DIR = "data/cache"
RECOVERY_FILE = os.path.join(CACHE_DIR, "recovery_state.json")

@dataclass
class TWAPSchedule:
    symbol: str
    side: str
    total_quantity: float
    executed_quantity: float
    remaining_quantity: float
    num_buckets: int
    executed_buckets: int
    bucket_size: float
    interval_seconds: float
    next_execution_time: float

class OrderRecoveryManager:
    """Persists TWAP state to disk to survive Raspberry Pi reboots."""
    
    @staticmethod
    def load_state() -> Dict[str, TWAPSchedule]:
        if not os.path.exists(RECOVERY_FILE):
            return {}
        try:
            with open(RECOVERY_FILE, 'r') as f:
                raw_data = json.load(f)
                state = {}
                for sym, data in raw_data.items():
                    state[sym] = TWAPSchedule(**data)
                return state
        except Exception as e:
            print(f"[EMS Recovery] Failed to load recovery state: {e}")
            return {}
            
    @staticmethod
    def save_state(state: Dict[str, TWAPSchedule]):
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            raw_data = {sym: data.__dict__ for sym, data in state.items()}
            with open(RECOVERY_FILE, 'w') as f:
                json.dump(raw_data, f, indent=2)
        except Exception as e:
            print(f"[EMS Recovery] Failed to save recovery state: {e}")

    @staticmethod
    def clear_state():
        if os.path.exists(RECOVERY_FILE):
            os.remove(RECOVERY_FILE)

class ExecutionManagementSystem:
    """
    High-Performance EMS for slicing large orders over time to reduce market impact.
    Operates via async background threads so the main pipeline doesn't block.
    """
    def __init__(self, broker: BrokerInterface):
        self.broker = broker
        self.active_schedules: Dict[str, TWAPSchedule] = {}
        self._stop_event = threading.Event()
        self._worker_thread = None

    def start(self):
        """Resume any interrupted schedules and start the background worker."""
        if not self.broker.connect():
            print("[EMS] Broker connection failed. EMS offline.")
            return False
            
        self.active_schedules = OrderRecoveryManager.load_state()
        if self.active_schedules:
            print(f"[EMS] Recovered {len(self.active_schedules)} active TWAP schedules.")
            
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._twap_worker)
        self._worker_thread.daemon = True
        self._worker_thread.start()
        return True

    def stop(self):
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

    def submit_twap_order(self, symbol: str, quantity: float, side: OrderSide, duration_minutes: int = 60, num_buckets: int = 10):
        """Slices an order into smaller chunks and queues them for execution over time."""
        if quantity <= 0:
            return
            
        bucket_size = quantity / num_buckets
        interval_seconds = (duration_minutes * 60) / num_buckets
        
        schedule = TWAPSchedule(
            symbol=symbol,
            side=side.value,
            total_quantity=quantity,
            executed_quantity=0,
            remaining_quantity=quantity,
            num_buckets=num_buckets,
            executed_buckets=0,
            bucket_size=bucket_size,
            interval_seconds=interval_seconds,
            next_execution_time=time.time()
        )
        
        self.active_schedules[symbol] = schedule
        OrderRecoveryManager.save_state(self.active_schedules)
        print(f"[EMS] Registered TWAP: {side.value.upper()} {quantity} {symbol} over {duration_minutes}m ({num_buckets} slices)")

    def _twap_worker(self):
        """Background thread that executes order slices when their time comes."""
        while not self._stop_event.is_set():
            now = time.time()
            completed = []
            
            for symbol, schedule in self.active_schedules.items():
                if now >= schedule.next_execution_time and schedule.remaining_quantity > 0:
                    # Time to execute a slice
                    slice_qty = int(schedule.bucket_size)
                    
                    # Handle rounding on the final bucket
                    if schedule.executed_buckets == schedule.num_buckets - 1:
                        slice_qty = int(schedule.remaining_quantity)
                        
                    if slice_qty > 0:
                        order = Order(
                            symbol=schedule.symbol,
                            side=OrderSide.BUY if schedule.side == 'buy' else OrderSide.SELL,
                            quantity=slice_qty,
                            order_type=OrderType.MARKET, # You can switch to LIMIT for passive filling
                            time_in_force='ioc'
                        )
                        success, order_id = self.broker.place_order(order)
                        
                        if success:
                            schedule.executed_quantity += slice_qty
                            schedule.remaining_quantity -= slice_qty
                            schedule.executed_buckets += 1
                            schedule.next_execution_time = now + schedule.interval_seconds
                            print(f"[EMS Worker] Executed {slice_qty} {symbol}. Remaining: {schedule.remaining_quantity}")
                            OrderRecoveryManager.save_state(self.active_schedules)
                        else:
                            print(f"[EMS Worker] Failed to execute slice for {symbol}. Will retry.")
                            schedule.next_execution_time = now + 10 # Retry in 10s
                            
                if schedule.remaining_quantity <= 0:
                    completed.append(symbol)
                    
            for sym in completed:
                print(f"[EMS] TWAP completed for {sym}")
                del self.active_schedules[sym]
                OrderRecoveryManager.save_state(self.active_schedules)
                
            time.sleep(1) # Prevent CPU spinning

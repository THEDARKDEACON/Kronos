import sys
import pandas as pd

# Path mappings assuming script is run from root
sys.path.append('.')

from data.ingestion import get_universe, get_fundamentals, get_historical_ohlcv
from strategy.fundamental import get_safe_universe
from strategy.kronos_alpha import KronosAlphaGenerator
from optim.portfolio import construct_portfolio

class BacktestSimulator:
    def __init__(self, target_dates=None, transaction_cost_bps=5.0, max_drawdown_pct=10.0):
        self.target_dates = target_dates if target_dates else ["2023-01-01", "2023-06-01"]
        self.universe = get_universe() # Fetch base universe (S&P 500)
        self.generator = KronosAlphaGenerator(model_size="small", max_context=512)
        self.transaction_cost = transaction_cost_bps / 10000.0  # Convert bps to decimal (e.g., 5 bps = 0.0005)
        self.previous_portfolio = None  # Track previous weights for turnover calculation
        self.max_drawdown = max_drawdown_pct / 100.0  # Convert % to decimal
        self.peak_value = 1.0  # Track peak portfolio value
        self.current_value = 1.0  # Current portfolio value
        self.circuit_breaker_triggered = False  # Emergency stop flag
        
    def step(self, as_of_date: str):
        print(f"\n{'='*60}")
        print(f"Executing Historical Simulation Step: {as_of_date}")
        print(f"{'='*60}")
        
        print("1. Fetching Point-In-Time Fundamentals (90-Day Lag Enforced)...")
        fundamentals_df = get_fundamentals(self.universe, as_of_date=as_of_date)
        
        print("2. Neutralizing Cross-Sectional Risk...")
        safe_tickers = get_safe_universe(fundamentals_df, drop_bottom_pct=0.25)
        
        print(f"3. Fetching Chronologically Safegaurded Price Action (OHLCV)...")
        # Hard limits prices to <= as_of_date
        ohlcv_dict = get_historical_ohlcv(safe_tickers, as_of_date=as_of_date, lookback_days=400)
        
        print("4. Generating AI Alpha...")
        signal_df = self.generator.generate_signals(ohlcv_dict, lookback=400, pred_len=5)
        
        print("5. Optimizing Capital (CVXPY MVO Solver)...")
        portfolio = construct_portfolio(
            signals_df=signal_df, 
            fundamentals_df=fundamentals_df, 
            ohlcv_dict=ohlcv_dict,
            risk_aversion=1.0,
            l2_penalty=0.5
        )
        
        # Calculate transaction costs based on portfolio turnover
        if self.previous_portfolio is not None and not portfolio.empty:
            turnover_cost = self._calculate_turnover_cost(portfolio)
            portfolio['Turnover_Cost'] = turnover_cost
            print(f"   [Slippage] Transaction Cost (5bps per trade side): {turnover_cost:.4f} ({turnover_cost*100:.2f}%)")
        
        self.previous_portfolio = portfolio.copy() if not portfolio.empty else None
        
        return portfolio

    def _calculate_turnover_cost(self, current_portfolio: pd.DataFrame) -> float:
        """
        Calculate total transaction cost based on portfolio turnover.
        Cost = 5bps per side = 10bps round trip for any weight change.
        """
        prev_weights = self.previous_portfolio['Weight'] if 'Weight' in self.previous_portfolio.columns else pd.Series()
        curr_weights = current_portfolio['Weight'] if 'Weight' in current_portfolio.columns else pd.Series()
        
        # Align indices (handle tickers entering/exiting)
        all_tickers = set(prev_weights.index) | set(curr_weights.index)
        
        total_turnover = 0.0
        for ticker in all_tickers:
            prev_w = prev_weights.get(ticker, 0.0)
            curr_w = curr_weights.get(ticker, 0.0)
            # Absolute change in weight = turnover for that ticker
            total_turnover += abs(curr_w - prev_w)
        
        # Transaction cost: 5bps per side (buy or sell)
        cost = total_turnover * self.transaction_cost
        return cost

    def run(self):
        history_ledger = {}
        for date in self.target_dates:
            # Check circuit breaker before running step
            if self.circuit_breaker_triggered:
                print(f"\n[⚠️ CIRCUIT BREAKER] Simulation halted at {date} due to maximum drawdown exceeded.")
                break
                
            port = self.step(date)
            history_ledger[date] = port
            
            # Calculate portfolio returns and drawdown
            if not port.empty and 'Predicted_Return' in port.columns:
                # Simplified return calculation: weighted average of predicted returns
                weighted_return = (port['Weight'] * port['Predicted_Return']).sum()
                # Subtract transaction costs
                if 'Turnover_Cost' in port.columns:
                    net_return = weighted_return - port['Turnover_Cost'].iloc[0] if len(port) > 0 else weighted_return
                else:
                    net_return = weighted_return
                    
                self.current_value *= (1 + net_return)
                
                # Update peak value
                if self.current_value > self.peak_value:
                    self.peak_value = self.current_value
                    
                # Calculate drawdown
                drawdown = (self.peak_value - self.current_value) / self.peak_value
                
                print(f"\n[Simulator] Target Portfolio Generated for {date}:")
                print(port.head(10))
                print(f"Gross Exposure: {port['Weight'].abs().sum():.2%}")
                print(f"Portfolio Value: ${self.current_value:.4f} (Peak: ${self.peak_value:.4f})")
                print(f"Current Drawdown: {drawdown:.2%} (Max Allowed: {self.max_drawdown:.2%})")
                
                # Trigger circuit breaker if drawdown exceeds limit
                if drawdown > self.max_drawdown:
                    self.circuit_breaker_triggered = True
                    print(f"\n[🚨 CIRCUIT BREAKER TRIGGERED] Drawdown {drawdown:.2%} exceeds limit {self.max_drawdown:.2%}")
                    print("[🚨 HALTING ALL TRADING OPERATIONS]")
            else:
                print(f"\n[Simulator] Target Portfolio Generated for {date}:")
                print(port.head(10))
            
        print("\nSimulation Sequence Complete.")
        if self.circuit_breaker_triggered:
            print(f"Final Portfolio Value: ${self.current_value:.4f}")
            print(f"Maximum Drawdown: {(self.peak_value - self.current_value) / self.peak_value:.2%}")
        return history_ledger

if __name__ == "__main__":
    print("Initializing Point-In-Time Backtest Simulator...")
    # Run a single historical snapshot to prove no leakage
    sim = BacktestSimulator(target_dates=["2023-06-01"])
    ledger = sim.run()

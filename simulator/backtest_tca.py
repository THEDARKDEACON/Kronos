"""
TCA-Aware Backtesting Module
Extends backtest simulator with realistic transaction cost analysis.
Includes market impact modeling, slippage estimation, and implementation shortfall tracking.
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

sys.path.append('.')

from simulator.backtest import BacktestSimulator
from execution.market_impact import MarketImpactCalculator, AlmgrenChrissModel, TransactionCostAnalyzer
from data.corporate_actions import adjust_universe_prices


@dataclass
class TCAMetrics:
    """Transaction cost analysis metrics for a backtest step."""
    gross_return: float
    net_return: float
    transaction_costs_bps: float
    market_impact_bps: float
    slippage_bps: float
    implementation_shortfall: float
    participation_rate: float
    paper_pnl: float
    real_pnl: float


class TCAwareBacktestSimulator(BacktestSimulator):
    """
    Backtest simulator with Transaction Cost Analysis.
    
    Extends base simulator with:
    - Almgren-Chriss market impact model
    - Corporate action adjustments
    - Implementation shortfall tracking
    - Realistic execution simulation
    """
    
    def __init__(self,
                 target_dates=None,
                 transaction_cost_bps=5.0,
                 max_drawdown_pct=10.0,
                 use_market_impact=True,
                 impact_urgency='normal',
                 adjust_for_corporate_actions=True):
        super().__init__(target_dates, transaction_cost_bps, max_drawdown_pct)
        
        self.use_market_impact = use_market_impact
        self.impact_urgency = impact_urgency
        self.adjust_for_corporate_actions = adjust_for_corporate_actions
        
        # Initialize impact calculator
        self.impact_calculator = MarketImpactCalculator()
        self.tca_analyzer = TransactionCostAnalyzer()
        
        # Track detailed cost breakdown
        self.tca_history: List[TCAMetrics] = []
        
    def step(self, as_of_date: str):
        """
        Execute backtest step with TCA.
        """
        print(f"\n{'='*60}")
        print(f"Executing TCA-Aware Backtest Step: {as_of_date}")
        print(f"{'='*60}")
        
        # Step 1-2: Same as base class (fundamentals, universe)
        print("1. Fetching Point-In-Time Fundamentals...")
        from data.ingestion import get_fundamentals
        fundamentals_df = get_fundamentals(self.universe, as_of_date=as_of_date)
        
        print("2. Neutralizing Cross-Sectional Risk...")
        from strategy.fundamental import get_safe_universe
        safe_tickers = get_safe_universe(fundamentals_df, drop_bottom_pct=0.25)
        
        # Step 3: Fetch OHLCV
        print(f"3. Fetching Historical Price Action...")
        from data.ingestion import get_historical_ohlcv
        ohlcv_dict = get_historical_ohlcv(safe_tickers, as_of_date=as_of_date, lookback_days=400)
        
        # Step 3b: Adjust for corporate actions if enabled
        if self.adjust_for_corporate_actions:
            print("   [TCA] Adjusting for corporate actions (splits, dividends)...")
            from data.corporate_actions import adjust_universe_prices
            ohlcv_dict = adjust_universe_prices(
                ohlcv_dict,
                start_date=datetime.strptime(as_of_date, '%Y-%m-%d') - timedelta(days=400),
                end_date=datetime.strptime(as_of_date, '%Y-%m-%d')
            )
        
        # Step 4: Generate signals
        print("4. Generating AI Alpha...")
        signal_df = self.generator.generate_signals(ohlcv_dict, lookback=400, pred_len=5)
        
        # Step 5: Optimize portfolio
        print("5. Optimizing Capital...")
        from optim.portfolio import construct_portfolio
        portfolio = construct_portfolio(
            signals_df=signal_df,
            fundamentals_df=fundamentals_df,
            ohlcv_dict=ohlcv_dict,
            risk_aversion=1.0,
            l2_penalty=0.5,
            prev_weights=self.previous_portfolio['Weight'].to_dict() if self.previous_portfolio is not None else None
        )
        
        # Step 6: Calculate TCA
        if self.previous_portfolio is not None and not portfolio.empty:
            print("6. Calculating Transaction Cost Analysis...")
            tca_metrics = self._calculate_tca(portfolio, ohlcv_dict, as_of_date)
            portfolio['TCA_Market_Impact_bps'] = tca_metrics.market_impact_bps
            portfolio['TCA_Total_Cost_bps'] = tca_metrics.transaction_costs_bps
            portfolio['TCA_Implementation_Shortfall'] = tca_metrics.implementation_shortfall
            
            print(f"   [TCA] Market Impact: {tca_metrics.market_impact_bps:.2f} bps")
            print(f"   [TCA] Total Costs: {tca_metrics.transaction_costs_bps:.2f} bps")
            print(f"   [TCA] Paper P&L: ${tca_metrics.paper_pnl:,.2f}")
            print(f"   [TCA] Real P&L: ${tca_metrics.real_pnl:,.2f}")
        
        self.previous_portfolio = portfolio.copy() if not portfolio.empty else None
        
        return portfolio
    
    def _calculate_tca(self,
                      current_portfolio: pd.DataFrame,
                      ohlcv_dict: Dict[str, pd.DataFrame],
                      as_of_date: str) -> TCAMetrics:
        """
        Calculate transaction cost analysis for portfolio transition.
        """
        prev_weights = self.previous_portfolio['Weight'] if 'Weight' in self.previous_portfolio.columns else pd.Series()
        curr_weights = current_portfolio['Weight'] if 'Weight' in current_portfolio.columns else pd.Series()
        
        # Get all tickers
        all_tickers = set(prev_weights.index) | set(curr_weights.index)
        
        total_market_impact_bps = 0.0
        total_trade_value = 0.0
        total_participation = 0.0
        
        trade_details = []
        
        for ticker in all_tickers:
            prev_w = prev_weights.get(ticker, 0.0)
            curr_w = curr_weights.get(ticker, 0.0)
            delta = curr_w - prev_w  # Weight change
            
            if abs(delta) < 0.0001:  # Skip negligible trades
                continue
            
            # Get ticker data
            if ticker not in ohlcv_dict or ohlcv_dict[ticker].empty:
                continue
            
            df = ohlcv_dict[ticker]
            
            # Calculate trade size in shares (assume $1M portfolio)
            portfolio_value = 1_000_000
            price = df['close'].iloc[-1]
            trade_shares = int(delta * portfolio_value / price)
            trade_value = abs(trade_shares) * price
            
            # Get ADV and volatility
            adv = int(df['volume'].tail(20).mean())
            returns = df['close'].pct_change().dropna()
            volatility = returns.tail(20).std()
            
            if self.use_market_impact and adv > 0:
                # Calculate market impact
                impact = self.impact_calculator.calculate_trade_impact(
                    ticker=ticker,
                    shares=trade_shares,
                    avg_daily_volume=adv,
                    price=price,
                    volatility=volatility,
                    execution_time_hours=1.0 if self.impact_urgency == 'normal' else 0.5
                )
                
                market_impact_bps = impact['total_impact_bps']
                participation_rate = impact['participation_rate']
                
                total_market_impact_bps += market_impact_bps * abs(delta)
                total_participation += participation_rate * abs(delta)
                
                trade_details.append({
                    'ticker': ticker,
                    'delta': delta,
                    'shares': trade_shares,
                    'impact_bps': market_impact_bps,
                    'participation': participation_rate
                })
            
            total_trade_value += trade_value
        
        # Calculate gross turnover
        gross_turnover = sum(abs(curr_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers)
        
        # Fixed transaction cost (5bps per side)
        fixed_cost_bps = gross_turnover * self.transaction_cost * 10000
        
        # Market impact cost
        impact_cost_bps = total_market_impact_bps if gross_turnover > 0 else 0
        
        # Total transaction costs
        total_costs_bps = fixed_cost_bps + impact_cost_bps
        
        # Calculate returns
        if 'Predicted_Return' in current_portfolio.columns:
            gross_return = (current_portfolio['Weight'] * current_portfolio['Predicted_Return']).sum()
        else:
            gross_return = 0.0
        
        net_return = gross_return - (total_costs_bps / 10000)
        
        # P&L calculation (assume $1M portfolio)
        paper_pnl = gross_return * 1_000_000
        real_pnl = net_return * 1_000_000
        
        # Implementation shortfall
        implementation_shortfall = paper_pnl - real_pnl
        
        metrics = TCAMetrics(
            gross_return=gross_return,
            net_return=net_return,
            transaction_costs_bps=total_costs_bps,
            market_impact_bps=impact_cost_bps,
            slippage_bps=fixed_cost_bps,
            implementation_shortfall=implementation_shortfall,
            participation_rate=total_participation / gross_turnover if gross_turnover > 0 else 0,
            paper_pnl=paper_pnl,
            real_pnl=real_pnl
        )
        
        self.tca_history.append(metrics)
        
        # Record trades for calibration
        for trade in trade_details:
            self.tca_analyzer.record_trade(
                ticker=trade['ticker'],
                shares=trade['shares'],
                avg_execution_price=0,  # Would be filled in live trading
                arrival_price=0,
                adv=int(ohlcv_dict[trade['ticker']]['volume'].tail(20).mean()),
                volatility=ohlcv_dict[trade['ticker']]['close'].pct_change().tail(20).std(),
                decision_time=as_of_date,
                execution_time=as_of_date
            )
        
        return metrics
    
    def run(self):
        """
        Run backtest with full TCA reporting.
        """
        print("\n" + "="*60)
        print("TCA-AWARE BACKTEST SIMULATION")
        print("="*60)
        print(f"Configuration:")
        print(f"  Market Impact Model: {'Enabled' if self.use_market_impact else 'Disabled'}")
        print(f"  Corporate Actions: {'Adjusted' if self.adjust_for_corporate_actions else 'Raw'}")
        print(f"  Execution Urgency: {self.impact_urgency}")
        print(f"  Fixed Costs: {self.transaction_cost * 10000:.0f} bps per side")
        print("="*60)
        
        # Run base simulation
        history_ledger = super().run()
        
        # Generate TCA summary
        if self.tca_history:
            self._generate_tca_report()
        
        return history_ledger
    
    def _generate_tca_report(self):
        """
        Generate comprehensive TCA report.
        """
        if not self.tca_history:
            print("\n[TCA] No transaction data available for analysis.")
            return
        
        # Aggregate metrics
        avg_impact = np.mean([m.market_impact_bps for m in self.tca_history])
        avg_costs = np.mean([m.transaction_costs_bps for m in self.tca_history])
        avg_shortfall = np.mean([m.implementation_shortfall for m in self.tca_history])
        total_paper_pnl = sum([m.paper_pnl for m in self.tca_history])
        total_real_pnl = sum([m.real_pnl for m in self.tca_history])
        
        print("\n" + "="*60)
        print("TRANSACTION COST ANALYSIS REPORT")
        print("="*60)
        print(f"\nAGGREGATE METRICS")
        print(f"  Average Market Impact: {avg_impact:.2f} bps")
        print(f"  Average Total Costs: {avg_costs:.2f} bps")
        print(f"  Average Implementation Shortfall: ${avg_shortfall:,.2f}")
        print(f"\nP&L COMPARISON")
        print(f"  Paper P&L (ignoring costs): ${total_paper_pnl:,.2f}")
        print(f"  Real P&L (after costs): ${total_real_pnl:,.2f}")
        print(f"  Cost Drag: ${total_paper_pnl - total_real_pnl:,.2f} ({(1 - total_real_pnl/total_paper_pnl)*100:.1f}%)")
        
        # Calibrate model if enough data
        if len(self.tca_analyzer.trade_history) > 10:
            print(f"\nMODEL CALIBRATION")
            calibrated = self.tca_analyzer.calibrate_impact_model()
            print(f"  Estimated eta (temporary): {calibrated['eta']:.3f}")
            print(f"  Estimated gamma (permanent): {calibrated['gamma']:.3f}")
        
        print("="*60)


# Convenience function
def run_tca_backtest(target_dates: List[str],
                    use_market_impact: bool = True,
                    adjust_corporate_actions: bool = True) -> Dict:
    """
    Run a TCA-aware backtest and return results.
    """
    sim = TCAwareBacktestSimulator(
        target_dates=target_dates,
        use_market_impact=use_market_impact,
        adjust_for_corporate_actions=adjust_corporate_actions
    )
    
    results = sim.run()
    
    return {
        'portfolio_history': results,
        'tca_metrics': sim.tca_history,
        'final_value': sim.current_value,
        'max_drawdown': (sim.peak_value - sim.current_value) / sim.peak_value if sim.peak_value > 0 else 0
    }


if __name__ == "__main__":
    # Example usage
    print("Running TCA-Aware Backtest Example...")
    
    sim = TCAwareBacktestSimulator(
        target_dates=["2023-06-01", "2023-09-01", "2023-12-01"],
        use_market_impact=True,
        adjust_for_corporate_actions=True
    )
    
    results = sim.run()

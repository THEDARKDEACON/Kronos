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

# C-4 FIX: The original code inherited from a non-existent `BacktestSimulator`.
# The correct class is `WalkForwardBacktest` from simulator/backtest.py.
from simulator.backtest import WalkForwardBacktest
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


class TCAwareBacktestSimulator(WalkForwardBacktest):
    """
    Backtest simulator with Transaction Cost Analysis.

    Extends WalkForwardBacktest with:
    - Almgren-Chriss market impact model
    - Corporate action adjustments
    - Implementation shortfall tracking
    - Realistic execution simulation
    """

    def __init__(
        self,
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        transaction_cost_bps: float = 5.0,
        max_drawdown_pct: float = 10.0,
        universe_limit: int = 20,
        seed: Optional[int] = None,
        use_market_impact: bool = True,
        impact_urgency: str = 'normal',
        adjust_for_corporate_actions: bool = True,
    ):
        # C-4 FIX: Pass correct arguments to WalkForwardBacktest.__init__
        super().__init__(
            start_date=start_date,
            end_date=end_date,
            transaction_cost_bps=transaction_cost_bps,
            max_drawdown_pct=max_drawdown_pct,
            universe_limit=universe_limit,
            seed=seed,
        )

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

        Delegates core signal/portfolio logic to WalkForwardBacktest.step(),
        then layers in TCA cost attribution.
        """
        print(f"\n{'='*60}")
        print(f"Executing TCA-Aware Backtest Step: {as_of_date}")
        print(f"{'='*60}")

        # Run the base walk-forward step (handles PIT fundamentals, signals, portfolio)
        portfolio = super().step(as_of_date)

        if portfolio is None or portfolio.empty:
            return portfolio

        # Layer TCA on top if we have a previous portfolio to compare against
        if self.previous_weights is not None and len(self.previous_weights) > 0:
            print("   Calculating Transaction Cost Analysis...")
            # Reconstruct the PIT ohlcv for cost calculations
            as_of_ts = pd.Timestamp(as_of_date)
            ohlcv_pit = {
                t: df[df["timestamps"] <= as_of_ts].reset_index(drop=True)
                for t, df in self.ohlcv_dict.items()
                if "timestamps" in df.columns
            }

            if self.adjust_for_corporate_actions and ohlcv_pit:
                print("   [TCA] Adjusting for corporate actions (splits, dividends)...")
                try:
                    ohlcv_pit = adjust_universe_prices(
                        ohlcv_pit,
                        start_date=datetime.strptime(as_of_date, '%Y-%m-%d') - timedelta(days=400),
                        end_date=datetime.strptime(as_of_date, '%Y-%m-%d'),
                    )
                except Exception as e:
                    print(f"   [TCA] Corporate action adjustment failed: {e}")

            tca_metrics = self._calculate_tca(portfolio, ohlcv_pit, as_of_date)
            portfolio['TCA_Market_Impact_bps'] = tca_metrics.market_impact_bps
            portfolio['TCA_Total_Cost_bps'] = tca_metrics.transaction_costs_bps
            portfolio['TCA_Implementation_Shortfall'] = tca_metrics.implementation_shortfall

            print(f"   [TCA] Market Impact: {tca_metrics.market_impact_bps:.2f} bps")
            print(f"   [TCA] Total Costs: {tca_metrics.transaction_costs_bps:.2f} bps")
            print(f"   [TCA] Paper P&L: ${tca_metrics.paper_pnl:,.2f}")
            print(f"   [TCA] Real P&L: ${tca_metrics.real_pnl:,.2f}")

        return portfolio

    def _calculate_tca(
        self,
        current_portfolio: pd.DataFrame,
        ohlcv_dict: Dict[str, pd.DataFrame],
        as_of_date: str,
    ) -> TCAMetrics:
        """
        Calculate transaction cost analysis for portfolio transition.
        """
        prev_weights = self.previous_weights if self.previous_weights is not None else pd.Series(dtype=float)
        curr_weights = current_portfolio['Weight'] if 'Weight' in current_portfolio.columns else pd.Series(dtype=float)

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
                    volatility_20d=volatility,
                    execution_time_hours=1.0 if self.impact_urgency == 'normal' else 0.5,
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
                    'participation': participation_rate,
                })

            total_trade_value += trade_value

        # Calculate gross turnover
        gross_turnover = sum(
            abs(curr_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
            for t in all_tickers
        )

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
            real_pnl=real_pnl,
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
                execution_time=as_of_date,
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
        results = super().run()

        # Generate TCA summary
        if self.tca_history:
            self._generate_tca_report()

        return results

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
        if total_paper_pnl != 0:
            print(f"  Cost Drag: ${total_paper_pnl - total_real_pnl:,.2f} "
                  f"({(1 - total_real_pnl/total_paper_pnl)*100:.1f}%)")

        # Calibrate model if enough data
        if len(self.tca_analyzer.trade_history) > 10:
            print(f"\nMODEL CALIBRATION")
            calibrated = self.tca_analyzer.calibrate_impact_model()
            print(f"  Estimated eta (temporary): {calibrated['eta']:.3f}")
            print(f"  Estimated gamma (permanent): {calibrated['gamma']:.3f}")

        print("="*60)


# Convenience function
def run_tca_backtest(
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None,
    universe_limit: int = 20,
    use_market_impact: bool = True,
    adjust_corporate_actions: bool = True,
) -> Dict:
    """
    Run a TCA-aware backtest and return results.
    """
    sim = TCAwareBacktestSimulator(
        start_date=start_date,
        end_date=end_date,
        universe_limit=universe_limit,
        use_market_impact=use_market_impact,
        adjust_for_corporate_actions=adjust_corporate_actions,
    )

    results = sim.run()

    return {
        'results': results,
        'tca_metrics': sim.tca_history,
        'final_value': sim.current_value,
        'max_drawdown': (sim.peak_value - sim.current_value) / sim.peak_value if sim.peak_value > 0 else 0,
    }


if __name__ == "__main__":
    # Example usage
    print("Running TCA-Aware Backtest Example...")

    sim = TCAwareBacktestSimulator(
        start_date="2023-06-01",
        end_date="2023-12-01",
        use_market_impact=True,
        adjust_for_corporate_actions=True,
    )

    results = sim.run()

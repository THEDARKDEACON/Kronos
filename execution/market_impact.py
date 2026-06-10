"""
Market Impact Model
Implements Almgren-Chriss model for transaction cost estimation.
Separates temporary and permanent impact components.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class ImpactParameters:
    """Almgren-Chriss model parameters."""
    # Temporary impact (immediate cost, recovers)
    eta: float = 0.142  # Temporary impact coefficient (typically 0.1-0.2)
    
    # Permanent impact (long-lasting effect)
    gamma: float = 0.314  # Permanent impact coefficient (typically 0.2-0.5)
    
    # Decay parameters
    beta: float = 0.6  # Power law for temporary impact
    
    # Market parameters
    sigma: float = 0.02  # Daily volatility (2% default)
    theta: float = 0.5  # Market resiliency (how fast temporary impact decays)


class AlmgrenChrissModel:
    """
    Almgren-Chriss market impact model.
    
    Total Cost = Temporary Impact + Permanent Impact
    
    Temporary Impact: Short-term price movement from order execution,
                      decays over time
    
    Permanent Impact: Long-term price change from information leakage
    """
    
    def __init__(self, params: Optional[ImpactParameters] = None):
        self.params = params or ImpactParameters()
    
    def estimate_temporary_impact(self,
                                  order_size: int,
                                  avg_daily_volume: int,
                                  volatility: float,
                                  execution_time_hours: float = 1.0) -> float:
        """
        Estimate temporary market impact.
        
        Formula: eta * sigma * (X/ADV)^beta * sqrt(T)
        
        Where:
        - X = order size (shares)
        - ADV = average daily volume
        - sigma = daily volatility
        - T = execution time in days
        - eta = temporary impact coefficient
        
        Returns:
            Price impact as percentage (e.g., 0.001 = 10 bps)
        """
        if avg_daily_volume == 0:
            return 0.0
        
        participation_rate = order_size / avg_daily_volume
        
        # Power law for temporary impact
        temp_impact = (
            self.params.eta *
            volatility *
            (participation_rate ** self.params.beta) *
            np.sqrt(execution_time_hours / 6.5)  # Convert to trading days
        )
        
        return temp_impact
    
    def estimate_permanent_impact(self,
                                   order_size: int,
                                   avg_daily_volume: int,
                                   volatility: float) -> float:
        """
        Estimate permanent market impact.
        
        Formula: gamma * sigma * (X/ADV)
        
        Where:
        - X = order size (shares)
        - ADV = average daily volume
        - sigma = daily volatility
        - gamma = permanent impact coefficient
        
        Returns:
            Price impact as percentage
        """
        if avg_daily_volume == 0:
            return 0.0
        
        participation_rate = order_size / avg_daily_volume
        
        # Linear for permanent impact
        perm_impact = (
            self.params.gamma *
            volatility *
            participation_rate
        )
        
        return perm_impact
    
    def estimate_total_impact(self,
                              order_size: int,
                              avg_daily_volume: int,
                              volatility: float,
                              execution_time_hours: float = 1.0) -> Tuple[float, float, float]:
        """
        Estimate total market impact.
        
        Returns:
            (temporary_impact, permanent_impact, total_impact)
        """
        temp = self.estimate_temporary_impact(
            order_size, avg_daily_volume, volatility, execution_time_hours
        )
        
        perm = self.estimate_permanent_impact(
            order_size, avg_daily_volume, volatility
        )
        
        total = temp + perm
        
        return temp, perm, total
    
    def estimate_implementation_shortfall(self,
                                          shares: int,
                                          avg_price: float,
                                          decision_price: float,
                                          adv: int,
                                          volatility: float) -> Dict:
        """
        Calculate implementation shortfall vs decision price.
        
        Args:
            shares: Number of shares traded
            avg_price: Average execution price
            decision_price: Price at decision time (arrival price)
            adv: Average daily volume
            volatility: Daily volatility
        
        Returns:
            Dict with cost breakdown
        """
        # Total shortfall
        shortfall = (avg_price / decision_price - 1) if decision_price > 0 else 0
        
        # Decompose into components
        temp, perm, total = self.estimate_total_impact(
            abs(shares), adv, volatility
        )
        
        # Estimated costs
        estimated_temp = temp if shares > 0 else -temp  # Sign depends on direction
        estimated_perm = perm if shares > 0 else -perm
        
        return {
            'actual_shortfall': shortfall,
            'estimated_temporary_impact': estimated_temp,
            'estimated_permanent_impact': estimated_perm,
            'estimated_total_impact': estimated_temp + estimated_perm,
            'participation_rate': abs(shares) / adv if adv > 0 else 0,
            'cost_bps': shortfall * 10000
        }


class MarketImpactCalculator:
    """
    Calculates market impact for portfolio trades.
    """
    
    def __init__(self, model: Optional[AlmgrenChrissModel] = None):
        self.model = model or AlmgrenChrissModel()
    
    def calculate_trade_impact(self,
                               ticker: str,
                               shares: int,
                               avg_daily_volume: int,
                               price: float,
                               volatility_20d: float,
                               execution_time_hours: float = 1.0) -> Dict:
        """
        Calculate impact for a single trade.
        
        Returns:
            Dict with impact metrics
        """
        temp, perm, total = self.model.estimate_total_impact(
            abs(shares),
            avg_daily_volume,
            volatility_20d,
            execution_time_hours
        )
        
        # Dollar impact
        trade_value = abs(shares) * price
        temp_dollar = trade_value * temp
        perm_dollar = trade_value * perm
        total_dollar = trade_value * total
        
        return {
            'ticker': ticker,
            'shares': shares,
            'direction': 'buy' if shares > 0 else 'sell',
            'adv': avg_daily_volume,
            'participation_rate': abs(shares) / avg_daily_volume if avg_daily_volume > 0 else 0,
            'temp_impact_bps': temp * 10000,
            'perm_impact_bps': perm * 10000,
            'total_impact_bps': total * 10000,
            'temp_impact_dollar': temp_dollar,
            'perm_impact_dollar': perm_dollar,
            'total_impact_dollar': total_dollar,
            'execution_time_hours': execution_time_hours
        }
    
    def calculate_portfolio_impact(self,
                                   trades: pd.DataFrame,
                                   ohlcv_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate impact for all trades in a portfolio.
        
        Args:
            trades: DataFrame with columns [ticker, shares, execution_time_hours]
            ohlcv_dict: Dict of ticker -> OHLCV DataFrame
        
        Returns:
            DataFrame with impact metrics per trade
        """
        results = []
        
        for _, trade in trades.iterrows():
            ticker = trade['ticker']
            shares = trade['shares']
            exec_time = trade.get('execution_time_hours', 1.0)
            
            if ticker not in ohlcv_dict or ohlcv_dict[ticker].empty:
                continue
            
            df = ohlcv_dict[ticker]
            
            # Calculate ADV
            adv = int(df['volume'].tail(20).mean())
            
            # Calculate volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.tail(20).std()
            
            # Get current price
            price = df['close'].iloc[-1]
            
            impact = self.calculate_trade_impact(
                ticker, shares, adv, price, volatility, exec_time
            )
            
            results.append(impact)
        
        return pd.DataFrame(results)
    
    def optimize_execution_schedule(self,
                                    total_shares: int,
                                    avg_daily_volume: int,
                                    volatility: float,
                                    urgency: str = 'normal') -> pd.DataFrame:
        """
        Optimize execution schedule to minimize market impact.
        
        Args:
            total_shares: Total shares to trade
            avg_daily_volume: ADV
            volatility: Daily volatility
            urgency: 'high', 'normal', or 'low'
        
        Returns:
            DataFrame with recommended execution schedule
        """
        # Determine time horizon based on urgency
        urgency_hours = {
            'high': 0.5,    # 30 minutes
            'normal': 2.0,  # 2 hours
            'low': 6.5      # Full day
        }
        
        total_hours = urgency_hours.get(urgency, 2.0)
        
        # Optimal number of buckets (Almgren-Chriss theory)
        # More buckets = lower temporary impact, higher risk of price drift
        participation = total_shares / avg_daily_volume
        
        if participation < 0.01:  # Less than 1% of ADV
            n_buckets = 1  # Can execute immediately
        elif participation < 0.05:  # Less than 5% of ADV
            n_buckets = 3
        elif participation < 0.10:  # Less than 10% of ADV
            n_buckets = 5
        else:
            n_buckets = 10  # Very large order
        
        # Distribute shares across buckets
        base_size = total_shares // n_buckets
        remainder = total_shares % n_buckets
        
        schedule = []
        interval_minutes = (total_hours * 60) // n_buckets
        
        for i in range(n_buckets):
            bucket_size = base_size + (1 if i < remainder else 0)
            schedule.append({
                'bucket': i + 1,
                'shares': bucket_size,
                'time_minutes': i * interval_minutes,
                'estimated_impact_bps': self.model.estimate_temporary_impact(
                    bucket_size, avg_daily_volume, volatility, interval_minutes / 60
                ) * 10000
            })
        
        return pd.DataFrame(schedule)


class TransactionCostAnalyzer:
    """
    Analyzes transaction costs from actual execution data.
    """
    
    def __init__(self):
        self.trade_history: pd.DataFrame = pd.DataFrame()
    
    def record_trade(self,
                    ticker: str,
                    shares: int,
                    avg_execution_price: float,
                    arrival_price: float,
                    adv: int,
                    volatility: float,
                    decision_time: str,
                    execution_time: str):
        """
        Record a trade for TCA analysis.
        """
        trade = {
            'ticker': ticker,
            'shares': shares,
            'avg_price': avg_execution_price,
            'arrival_price': arrival_price,
            'shortfall': (avg_execution_price / arrival_price - 1) if arrival_price > 0 else 0,
            'shortfall_bps': (avg_execution_price / arrival_price - 1) * 10000 if arrival_price > 0 else 0,
            'adv': adv,
            'participation_rate': abs(shares) / adv if adv > 0 else 0,
            'volatility': volatility,
            'decision_time': decision_time,
            'execution_time': execution_time
        }
        
        self.trade_history = pd.concat([
            self.trade_history,
            pd.DataFrame([trade])
        ], ignore_index=True)
    
    def calibrate_impact_model(self) -> Dict:
        """
        Calibrate Almgren-Chriss parameters based on actual execution data.
        
        Returns:
            Dict with calibrated parameters
        """
        if self.trade_history.empty or len(self.trade_history) < 10:
            return {'eta': 0.142, 'gamma': 0.314}  # Default parameters
        
        # Simple linear regression to fit parameters
        # shortfall = eta * volatility * (participation^beta) + gamma * volatility * participation
        
        df = self.trade_history.copy()
        df['X_temp'] = df['volatility'] * (df['participation_rate'] ** 0.6)
        df['X_perm'] = df['volatility'] * df['participation_rate']
        
        # Fit temporary impact
        y = df['shortfall'].values
        X = np.column_stack([df['X_temp'].values, df['X_perm'].values])
        
        # Add constant
        X = np.column_stack([np.ones(len(X)), X])
        
        # OLS regression
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            eta = beta[1]
            gamma = beta[2]
        except Exception:  # L-1: don't swallow KeyboardInterrupt / SystemExit
            eta, gamma = 0.142, 0.314
        
        return {
            'eta': max(0.05, min(0.5, eta)),  # Constrain to reasonable bounds
            'gamma': max(0.1, min(1.0, gamma))
        }
    
    def generate_tca_report(self) -> str:
        """
        Generate transaction cost analysis report.
        """
        if self.trade_history.empty:
            return "No trades to analyze."
        
        df = self.trade_history
        
        report = f"""
TRANSACTION COST ANALYSIS REPORT
================================
Total Trades: {len(df)}

SHORTFALL METRICS
-----------------
Average Shortfall: {df['shortfall_bps'].mean():.2f} bps
Median Shortfall: {df['shortfall_bps'].median():.2f} bps
Std Dev: {df['shortfall_bps'].std():.2f} bps
Worst Trade: {df['shortfall_bps'].min():.2f} bps
Best Trade: {df['shortfall_bps'].max():.2f} bps

PARTICIPATION ANALYSIS
----------------------
Avg Participation Rate: {df['participation_rate'].mean():.2%}
Correlation (Participation vs Shortfall): {df['participation_rate'].corr(df['shortfall']):.3f}

TICKER BREAKDOWN
----------------
{df.groupby('ticker')['shortfall_bps'].agg(['mean', 'count']).sort_values('mean')}

CALIBRATED PARAMETERS
---------------------
{self.calibrate_impact_model()}
"""
        return report


# Convenience functions
def estimate_trade_cost(ticker: str,
                       shares: int,
                       avg_daily_volume: int,
                       price: float,
                       volatility: float,
                       execution_time_hours: float = 1.0) -> Dict:
    """
    Quick function to estimate trade cost.
    """
    calculator = MarketImpactCalculator()
    return calculator.calculate_trade_impact(
        ticker, shares, avg_daily_volume, price, volatility, execution_time_hours
    )


def get_optimal_execution_schedule(total_shares: int,
                                   avg_daily_volume: int,
                                   volatility: float,
                                   urgency: str = 'normal') -> pd.DataFrame:
    """
    Get optimal VWAP-like execution schedule.
    """
    calculator = MarketImpactCalculator()
    return calculator.optimize_execution_schedule(
        total_shares, avg_daily_volume, volatility, urgency
    )

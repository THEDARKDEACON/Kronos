"""
Kelly Criterion Position Sizing Module
Implements optimal bet sizing based on edge and variance.
Includes fractional Kelly for risk-adjusted position sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings


class KellyVariant(Enum):
    """Different Kelly Criterion implementations."""
    FULL_KELLY = "full"           # Pure Kelly (high variance)
    HALF_KELLY = "half"           # More conservative
    QUARTER_KELLY = "quarter"     # Most conservative (recommended)
    DUAL_KELLY = "dual"           # Separate long/short Kelly fractions
    LEVERAGED_KELLY = "leveraged" # Incorporates leverage constraints


@dataclass
class KellyParameters:
    """Parameters for Kelly Criterion calculation."""
    win_probability: float        # P(win)
    avg_win: float                # Average return on win
    avg_loss: float               # Average return on loss (positive value)
    fraction: float = 0.25        # Kelly fraction (default quarter-Kelly)
    max_position: float = 0.20      # Maximum single position size
    min_position: float = 0.001     # Minimum position threshold


class KellyCriterionCalculator:
    """
    Kelly Criterion position sizing calculator.
    
    The Kelly Criterion maximizes long-term geometric growth:
    f* = (p*b - q) / b
    
    Where:
    - f* = optimal fraction of capital to bet
    - p = probability of win
    - q = probability of loss (1-p)
    - b = win/loss ratio (net odds received)
    """
    
    def __init__(self, variant: KellyVariant = KellyVariant.QUARTER_KELLY,
                 transaction_costs: float = 0.001):  # 10 bps per side
        self.variant = variant
        self.tc = transaction_costs
        
    def calculate_simple_kelly(self, p: float, b: float, q: Optional[float] = None) -> float:
        """
        Calculate basic Kelly fraction.
        
        Args:
            p: Probability of win
            b: Average win / Average loss (odds)
            q: Probability of loss (defaults to 1-p)
        
        Returns:
            Optimal Kelly fraction
        """
        if q is None:
            q = 1 - p
        
        if p <= 0 or p >= 1 or b <= 0:
            return 0.0
        
        kelly = (p * b - q) / b
        return max(0.0, kelly)  # Kelly can't be negative
    
    def calculate_edge_kelly(self, params: KellyParameters) -> float:
        """
        Calculate Kelly fraction using edge and variance.
        
        f* = edge / variance
        
        Where:
        - edge = expected return
        - variance = return variance
        """
        p = params.win_probability
        win = params.avg_win
        loss = params.avg_loss
        
        # Expected value (edge)
        edge = p * win - (1 - p) * loss - 2 * self.tc  # Account for round-trip costs
        
        # Variance of Bernoulli outcome
        variance = p * (1 - p) * (win + loss) ** 2
        
        if variance == 0:
            return 0.0
        
        kelly = edge / variance
        
        # Apply Kelly fraction
        kelly *= params.fraction
        
        # Apply bounds
        kelly = np.clip(kelly, params.min_position, params.max_position)
        
        return kelly
    
    def calculate_from_returns(self, historical_returns: pd.Series,
                              fraction: float = 0.25,
                              window: int = 63) -> float:
        """
        Calculate Kelly fraction from historical return series.
        
        Uses moment-based Kelly: f* = mean / variance
        """
        if len(historical_returns) < window:
            return 0.0
        
        returns = historical_returns.tail(window)
        
        mean_return = returns.mean()
        variance = returns.var()
        
        if variance == 0 or variance < 1e-10:
            return 0.0
        
        # Adjust for transaction costs
        mean_return_adj = mean_return - 2 * self.tc / window
        
        # Kelly fraction
        kelly = mean_return_adj / variance
        
        # Apply fractional Kelly
        kelly *= fraction
        
        return np.clip(kelly, 0.0, 1.0)
    
    def calculate_portfolio_kelly(self, 
                                  signals: Dict[str, float],  # Expected returns
                                  cov_matrix: pd.DataFrame,
                                  fraction: float = 0.25) -> Dict[str, float]:
        """
        Calculate Kelly-optimal portfolio weights.
        
        Solves: max_w (w'μ - 0.5 * w'Σw)
        
        Subject to: sum(|w|) = 1 (fully invested)
        """
        symbols = list(signals.keys())
        
        # Align covariance matrix
        valid_symbols = [s for s in symbols if s in cov_matrix.columns]
        
        if len(valid_symbols) < 2:
            return {s: 1.0 / len(signals) for s in signals}
        
        mu = np.array([signals[s] for s in valid_symbols])
        sigma = cov_matrix.loc[valid_symbols, valid_symbols].values
        
        try:
            # Kelly optimal: w = Σ^(-1) * μ
            sigma_inv = np.linalg.inv(sigma)
            raw_weights = sigma_inv @ mu
            
            # Normalize to 100% gross exposure
            gross = np.sum(np.abs(raw_weights))
            if gross > 0:
                kelly_weights = raw_weights / gross * fraction
            else:
                kelly_weights = np.zeros(len(valid_symbols))
            
            return {s: kelly_weights[i] for i, s in enumerate(valid_symbols)}
            
        except np.linalg.LinAlgError:
            warnings.warn("Covariance matrix is singular, using equal weights")
            return {s: 1.0 / len(valid_symbols) for s in valid_symbols}
    
    def adjust_for_correlation(self, base_kelly: float,
                              correlations: List[float],
                              method: str = 'average') -> float:
        """
        Adjust Kelly fraction for portfolio correlation effects.
        
        Higher correlation = lower Kelly (diversification benefit reduced)
        """
        if not correlations:
            return base_kelly
        
        avg_corr = np.mean(correlations)
        
        if method == 'average':
            # Reduce Kelly as correlation increases
            adjustment = 1 - avg_corr
        elif method == 'conservative':
            adjustment = max(0, 1 - 2 * avg_corr)
        else:
            adjustment = 1.0
        
        return base_kelly * adjustment
    
    def simulate_kelly_growth(self, 
                             kelly_fraction: float,
                             returns: pd.Series,
                             initial_capital: float = 1.0,
                             n_simulations: int = 1000) -> pd.DataFrame:
        """
        Monte Carlo simulation of Kelly strategy performance.
        """
        results = []
        
        for _ in range(n_simulations):
            capital = initial_capital
            capital_history = [capital]
            
            for ret in returns:
                # Bet Kelly fraction of current capital
                bet_size = capital * kelly_fraction
                pnl = bet_size * ret
                capital += pnl
                capital_history.append(capital)
                
                if capital <= 0:
                    break
            
            results.append(capital_history)
        
        # Convert to DataFrame
        max_len = max(len(r) for r in results)
        padded = [r + [r[-1]] * (max_len - len(r)) for r in results]
        
        return pd.DataFrame(padded).T
    
    def find_optimal_kelly_fraction(self, 
                                   returns: pd.Series,
                                   fractions: List[float] = None,
                                   criterion: str = 'median_final') -> Tuple[float, pd.DataFrame]:
        """
        Find optimal Kelly fraction through simulation.
        
        Args:
            returns: Historical return series
            fractions: List of Kelly fractions to test
            criterion: 'median_final', 'sharpe', or 'max_drawdown'
        
        Returns:
            (optimal_fraction, simulation_stats)
        """
        if fractions is None:
            fractions = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        
        stats = []
        
        for frac in fractions:
            sim_results = self.simulate_kelly_growth(frac, returns, n_simulations=100)
            
            final_values = sim_results.iloc[-1]
            
            stat = {
                'kelly_fraction': frac,
                'median_final': final_values.median(),
                'mean_final': final_values.mean(),
                'std_final': final_values.std(),
                'worst_case': final_values.min(),
                'prob_ruin': (final_values <= 0.1).mean(),  # 90% capital loss
                'sharpe': final_values.mean() / final_values.std() if final_values.std() > 0 else 0
            }
            stats.append(stat)
        
        stats_df = pd.DataFrame(stats)
        
        # Select optimal based on criterion
        if criterion == 'median_final':
            optimal = stats_df.loc[stats_df['median_final'].idxmax(), 'kelly_fraction']
        elif criterion == 'sharpe':
            optimal = stats_df.loc[stats_df['sharpe'].idxmax(), 'kelly_fraction']
        elif criterion == 'max_drawdown':
            # Minimize probability of ruin
            optimal = stats_df.loc[stats_df['prob_ruin'].idxmin(), 'kelly_fraction']
        else:
            optimal = 0.25  # Default to quarter-Kelly
        
        return optimal, stats_df


class KellyPositionSizer:
    """
    Practical position sizer using Kelly Criterion with safety constraints.
    """
    
    def __init__(self,
                 kelly_fraction: float = 0.25,
                 max_position: float = 0.20,
                 min_position: float = 0.001,
                 max_correlation: float = 0.70):
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_position = min_position
        self.max_correlation = max_correlation
        self.calculator = KellyCriterionCalculator(KellyVariant.QUARTER_KELLY)
    
    def size_positions(self,
                      signals: Dict[str, float],      # Expected returns
                      variances: Dict[str, float],    # Return variances
                      correlations: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Generate position sizes using Kelly Criterion.
        
        Args:
            signals: Dict of symbol -> expected return
            variances: Dict of symbol -> return variance
            correlations: Optional correlation matrix
        
        Returns:
            Dict of symbol -> position weight
        """
        weights = {}
        
        for symbol, expected_return in signals.items():
            variance = variances.get(symbol, 0.01)  # Default 10% vol
            
            if variance == 0:
                weights[symbol] = 0.0
                continue
            
            # Kelly fraction for this position
            kelly = expected_return / variance * self.kelly_fraction
            
            # Apply bounds
            kelly = np.clip(kelly, -self.max_position, self.max_position)
            
            # Apply minimum threshold
            if abs(kelly) < self.min_position:
                kelly = 0.0
            
            weights[symbol] = kelly
        
        # Normalize gross exposure to 100%
        gross = sum(abs(w) for w in weights.values())
        if gross > 0:
            weights = {k: v / gross for k, v in weights.items()}
        
        return weights
    
    def apply_correlation_adjustment(self,
                                     weights: Dict[str, float],
                                     correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Reduce position sizes for highly correlated assets.
        """
        adjusted = weights.copy()
        
        symbols = list(weights.keys())
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if sym1 in correlation_matrix.columns and sym2 in correlation_matrix.columns:
                    corr = correlation_matrix.loc[sym1, sym2]
                    
                    if abs(corr) > self.max_correlation:
                        # Reduce both positions
                        factor = self.max_correlation / abs(corr)
                        adjusted[sym1] *= factor
                        adjusted[sym2] *= factor
        
        # Renormalize
        gross = sum(abs(w) for w in adjusted.values())
        if gross > 0:
            adjusted = {k: v / gross for k, v in adjusted.items()}
        
        return adjusted


# Integration with existing portfolio optimizer
def integrate_kelly_with_cvxpy(signals: Dict[str, float],
                                 variances: Dict[str, float],
                                 cov_matrix: pd.DataFrame,
                                 kelly_fraction: float = 0.25) -> Dict[str, float]:
    """
    Generate Kelly-informed initial weights for CVXPY optimization.
    
    This provides a smart starting point for the convex optimizer.
    """
    calculator = KellyCriterionCalculator(KellyVariant.QUARTER_KELLY)
    
    # Get Kelly-optimal portfolio weights
    kelly_weights = calculator.calculate_portfolio_kelly(
        signals, cov_matrix, fraction=kelly_fraction
    )
    
    return kelly_weights


# Risk of Ruin calculation
def calculate_ruin_probability(win_prob: float, 
                                win_size: float, 
                                loss_size: float,
                                bankroll: float = 1.0,
                                target_wealth: float = 2.0) -> float:
    """
    Calculate probability of ruin before reaching target wealth.
    
    Based on Gambler's Ruin theorem for biased random walk.
    """
    if win_prob >= 0.5:
        return 0.0  # No risk of ruin with positive edge
    
    if win_prob == 0:
        return 1.0  # Certain ruin
    
    # Odds ratio
    odds = (win_prob * win_size) / ((1 - win_prob) * loss_size)
    
    if odds == 1:
        return 1 - bankroll / target_wealth
    
    ruin_prob = ((1 / odds) ** bankroll - (1 / odds) ** target_wealth) / (1 - (1 / odds) ** target_wealth)
    
    return max(0.0, min(1.0, ruin_prob))

"""
Barra-Style Factor Risk Model
Implements multi-factor risk decomposition for portfolio construction and monitoring.
Factors: Market, Size, Value, Momentum, Quality, Volatility, Growth
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


@dataclass
class FactorExposure:
    """Factor exposure for a single asset or portfolio."""
    market_beta: float = 0.0
    size: float = 0.0          # Small minus Big
    value: float = 0.0         # High minus Low (Book/Market)
    momentum: float = 0.0      # 12-1 month returns
    quality: float = 0.0       # Profitability + Low leverage
    volatility: float = 0.0    # Low volatility
    growth: float = 0.0        # Sales/earnings growth
    liquidity: float = 0.0     # Turnover/Amihud
    
    def to_series(self) -> pd.Series:
        return pd.Series({
            'Market': self.market_beta,
            'Size': self.size,
            'Value': self.value,
            'Momentum': self.momentum,
            'Quality': self.quality,
            'Volatility': self.volatility,
            'Growth': self.growth,
            'Liquidity': self.liquidity
        })


class FactorRiskModel:
    """
    Multi-factor risk model for portfolio risk decomposition.
    Calculates factor exposures, covariance matrix, and risk contributions.
    """
    
    # Factor return correlations (simplified Barra USE4 style)
    DEFAULT_FACTOR_CORR = np.array([
        [1.00, -0.15, -0.20,  0.00,  0.05, -0.70,  0.20,  0.10],  # Market
        [-0.15, 1.00,  0.15, -0.10, -0.20,  0.25, -0.05, -0.15],  # Size
        [-0.20, 0.15,  1.00,  0.05,  0.30,  0.10, -0.30,  0.05],  # Value
        [0.00, -0.10,  0.05,  1.00,  0.15, -0.20,  0.40,  0.00],  # Momentum
        [0.05, -0.20,  0.30,  0.15,  1.00, -0.10,  0.20,  0.00],  # Quality
        [-0.70, 0.25,  0.10, -0.20, -0.10,  1.00, -0.15,  0.15],  # Volatility
        [0.20, -0.05, -0.30,  0.40,  0.20, -0.15,  1.00, -0.10],  # Growth
        [0.10, -0.15,  0.05,  0.00,  0.00,  0.15, -0.10,  1.00]   # Liquidity
    ])
    
    FACTOR_NAMES = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Volatility', 'Growth', 'Liquidity']
    
    def __init__(self, lookback_days: int = 252, factor_volatility_annual: Optional[np.ndarray] = None):
        self.lookback = lookback_days
        
        # Default annualized factor volatilities (Barra-style)
        if factor_volatility_annual is None:
            self.factor_vol = np.array([0.15, 0.08, 0.06, 0.08, 0.05, 0.10, 0.07, 0.04])
        else:
            self.factor_vol = factor_volatility_annual
        
        # Build factor covariance matrix
        factor_corr = self.DEFAULT_FACTOR_CORR
        self.factor_cov = np.outer(self.factor_vol, self.factor_vol) * factor_corr
        
        self.exposures_cache = {}
        
    def calculate_factor_exposures(self, ticker: str, ohlcv_df: pd.DataFrame, 
                                   fundamentals: Optional[Dict] = None) -> FactorExposure:
        """
        Calculate factor exposures for a single stock.
        
        Args:
            ticker: Stock symbol
            ohlcv_df: OHLCV price data
            fundamentals: Optional fundamental data (PE, Debt/Equity, etc.)
        
        Returns:
            FactorExposure object with 8 factor loadings
        """
        if len(ohlcv_df) < 60:
            return FactorExposure()  # Insufficient data
        
        df = ohlcv_df.copy()
        df['returns'] = df['close'].pct_change()
        
        # Market Beta (vs S&P 500 proxy - would use actual index in production)
        market_beta = self._estimate_beta(df['returns'])
        
        # Size (log market cap proxy - using price as proxy, would use actual market cap)
        latest_price = df['close'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        size_proxy = np.log(latest_price * avg_volume)  # Proxy for market cap
        size = (size_proxy - 15) / 3  # Normalize around typical large cap
        
        # Value (Book-to-market proxy using PE ratio inversion)
        if fundamentals and 'PE_Ratio' in fundamentals:
            pe = fundamentals['PE_Ratio']
            if pe and pe > 0:
                value = (20 - pe) / 10  # Low PE = high value
            else:
                value = 0.0
        else:
            value = 0.0
        
        # Momentum (12-1 month returns)
        if len(df) >= 252:
            mom_12m = df['close'].iloc[-1] / df['close'].iloc[-252] - 1
            mom_1m = df['close'].iloc[-1] / df['close'].iloc[-21] - 1
            momentum = mom_12m - mom_1m
        else:
            momentum = 0.0
        
        # Quality (composite: profitability, low leverage, stability)
        volatility = df['returns'].tail(60).std() * np.sqrt(252) if len(df) >= 60 else 0.2
        quality_score = 0.0
        if fundamentals:
            # Low debt-to-equity = higher quality
            de = fundamentals.get('Debt_To_Equity', 100)
            if de is not None:
                quality_score += (100 - min(de, 200)) / 200
        quality = quality_score - volatility  # Lower vol = higher quality
        
        # Volatility factor (low vol anomaly)
        vol_factor = -volatility  # Low volatility stocks outperform
        
        # Growth (sales/earnings growth proxy)
        if len(df) >= 63:
            price_growth = df['close'].iloc[-1] / df['close'].iloc[-63] - 1
            growth = price_growth * 2  # Proxy for earnings growth
        else:
            growth = 0.0
        
        # Liquidity (turnover)
        avg_turnover = (df['volume'] / avg_volume).tail(20).mean()
        liquidity = -np.log(avg_turnover) if avg_turnover > 0 else 0.0
        
        return FactorExposure(
            market_beta=market_beta,
            size=np.clip(size, -3, 3),
            value=np.clip(value, -3, 3),
            momentum=np.clip(momentum, -3, 3),
            quality=np.clip(quality, -3, 3),
            volatility=np.clip(vol_factor, -3, 3),
            growth=np.clip(growth, -3, 3),
            liquidity=np.clip(liquidity, -3, 3)
        )
    
    def _estimate_beta(self, returns: pd.Series, market_returns: Optional[pd.Series] = None) -> float:
        """Estimate market beta using rolling regression."""
        if market_returns is None:
            # Assume market return is cross-sectional average (proxy)
            market_returns = returns.rolling(20).mean()
        
        # Use last 1 year of data
        valid_data = pd.DataFrame({'stock': returns, 'market': market_returns}).dropna()
        
        if len(valid_data) < 30:
            return 1.0  # Default to market beta
        
        # Simple beta calculation: cov(stock, market) / var(market)
        cov = valid_data['stock'].cov(valid_data['market'])
        market_var = valid_data['market'].var()
        
        if market_var > 0:
            beta = cov / market_var
            return np.clip(beta, -2, 3)  # Reasonable bounds
        return 1.0
    
    def calculate_portfolio_factor_exposure(self, weights: Dict[str, float],
                                           exposures: Dict[str, FactorExposure]) -> FactorExposure:
        """
        Calculate weighted factor exposure for entire portfolio.
        
        Args:
            weights: Dict of symbol -> portfolio weight
            exposures: Dict of symbol -> FactorExposure
        
        Returns:
            Portfolio FactorExposure
        """
        if not weights or not exposures:
            return FactorExposure()
        
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight == 0:
            return FactorExposure()
        
        # Weighted average of factor exposures
        portfolio_exposure = FactorExposure()
        
        for symbol, weight in weights.items():
            if symbol in exposures:
                exp = exposures[symbol]
                normalized_weight = weight / total_weight  # Normalize to 100%
                
                portfolio_exposure.market_beta += normalized_weight * exp.market_beta
                portfolio_exposure.size += normalized_weight * exp.size
                portfolio_exposure.value += normalized_weight * exp.value
                portfolio_exposure.momentum += normalized_weight * exp.momentum
                portfolio_exposure.quality += normalized_weight * exp.quality
                portfolio_exposure.volatility += normalized_weight * exp.volatility
                portfolio_exposure.growth += normalized_weight * exp.growth
                portfolio_exposure.liquidity += normalized_weight * exp.liquidity
        
        return portfolio_exposure
    
    def calculate_factor_risk_contribution(self, portfolio_exposure: FactorExposure) -> pd.DataFrame:
        """
        Calculate risk contribution from each factor.
        
        Returns DataFrame with factor risk metrics.
        """
        exp_array = portfolio_exposure.to_series().values
        
        # Marginal risk contribution (factor covariance * exposure)
        marginal_risk = self.factor_cov @ exp_array
        
        # Risk contribution (exposure * marginal risk)
        risk_contrib = exp_array * marginal_risk
        
        # Percentage contribution
        total_risk = np.sum(risk_contrib)
        risk_pct = risk_contrib / total_risk if total_risk > 0 else np.zeros(len(risk_contrib))
        
        return pd.DataFrame({
            'Exposure': exp_array,
            'Marginal_Risk': marginal_risk,
            'Risk_Contribution': risk_contrib,
            'Risk_Percentage': risk_pct * 100
        }, index=self.FACTOR_NAMES)
    
    def estimate_portfolio_volatility(self, portfolio_exposure: FactorExposure) -> float:
        """
        Estimate annualized portfolio volatility from factor exposures.
        
        sigma_p = sqrt(exposures' * Cov * exposures)
        """
        exp_array = portfolio_exposure.to_series().values
        variance = exp_array @ self.factor_cov @ exp_array
        return np.sqrt(max(variance, 0))
    
    def check_factor_neutrality(self, portfolio_exposure: FactorExposure,
                                max_exposure: float = 0.5) -> Dict[str, bool]:
        """
        Check if portfolio is factor-neutral (within tolerance).
        
        Returns dict of factor -> is_neutral status.
        """
        exposures = portfolio_exposure.to_series()
        return {
            factor: abs(exposures[factor]) <= max_exposure
            for factor in self.FACTOR_NAMES
        }
    
    def build_factor_neutral_constraint(self, target_factor: str, 
                                       tolerance: float = 0.1) -> Dict:
        """
        Build constraint for CVXPY optimization to achieve factor neutrality.
        
        Returns constraint specification for portfolio optimizer.
        """
        return {
            'type': 'factor_neutral',
            'target_factor': target_factor,
            'tolerance': tolerance
        }
    
    def calculate_tracking_error(self, portfolio_exposure: FactorExposure,
                                benchmark_exposure: FactorExposure) -> float:
        """
        Calculate active risk (tracking error) vs benchmark.
        """
        active_exp = portfolio_exposure.to_series() - benchmark_exposure.to_series()
        variance = active_exp.values @ self.factor_cov @ active_exp.values
        return np.sqrt(max(variance, 0))
    
    def generate_risk_report(self, weights: Dict[str, float],
                            exposures: Dict[str, FactorExposure],
                            portfolio_value: float = 1.0) -> Dict:
        """
        Generate comprehensive risk report for portfolio.
        """
        portfolio_exp = self.calculate_portfolio_factor_exposure(weights, exposures)
        risk_contrib = self.calculate_factor_risk_contribution(portfolio_exp)
        portfolio_vol = self.estimate_portfolio_volatility(portfolio_exp)
        neutrality = self.check_factor_neutrality(portfolio_exp)
        
        # Value at Risk (parametric, 95% confidence)
        var_95 = 1.645 * portfolio_vol * portfolio_value
        
        # Expected Shortfall (CVaR)
        cvar_95 = 2.063 * portfolio_vol * portfolio_value  # For normal distribution
        
        return {
            'portfolio_exposure': portfolio_exp.to_series().to_dict(),
            'risk_contribution': risk_contrib.to_dict(),
            'annualized_volatility': portfolio_vol,
            'var_95_1day': var_95 / np.sqrt(252),  # Daily VaR
            'var_95_annual': var_95,
            'cvar_95': cvar_95,
            'factor_neutrality': neutrality,
            'concentration_risk': self._calculate_concentration(weights),
            'diversification_ratio': self._calculate_diversification_ratio(weights, exposures)
        }
    
    def _calculate_concentration(self, weights: Dict[str, float]) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration."""
        abs_weights = np.array([abs(w) for w in weights.values()])
        total = np.sum(abs_weights)
        if total == 0:
            return 0.0
        normalized = abs_weights / total
        return np.sum(normalized ** 2)
    
    def _calculate_diversification_ratio(self, weights: Dict[str, float],
                                        exposures: Dict[str, FactorExposure]) -> float:
        """Calculate diversification ratio (weighted avg vol / portfolio vol)."""
        # Simplified calculation
        portfolio_exp = self.calculate_portfolio_factor_exposure(weights, exposures)
        portfolio_vol = self.estimate_portfolio_volatility(portfolio_exp)
        
        # Weighted average individual volatility (assume 30% for all)
        weighted_avg_vol = 0.30 * sum(abs(w) for w in weights.values())
        
        return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0


class FactorNeutralOptimizer:
    """
    Extension to portfolio optimizer that enforces factor neutrality constraints.
    """
    
    def __init__(self, risk_model: FactorRiskModel):
        self.risk_model = risk_model
    
    def neutralize_factor(self, portfolio_weights: Dict[str, float],
                         portfolio_exposures: Dict[str, 'FactorExposure'],
                         target_factor: str = 'market_beta',
                         max_deviation: float = 0.1) -> Dict[str, float]:
        """
        Adjust portfolio weights to achieve factor neutrality.

        Args:
            portfolio_weights:   symbol -> weight (float)
            portfolio_exposures: symbol -> FactorExposure (pre-computed)
            target_factor:       attribute name on FactorExposure to neutralize
            max_deviation:       tolerance; skip adjustment if already within bounds

        This is a simplified proportional implementation — in production this
        should be integrated directly into the CVXPY optimization as a linear
        equality/inequality constraint.
        """
        current_exp = self.risk_model.calculate_portfolio_factor_exposure(
            portfolio_weights, portfolio_exposures
        )

        target_exposure = getattr(current_exp, target_factor, 0.0)

        if abs(target_exposure) <= max_deviation:
            return portfolio_weights  # Already neutral

        # Simple proportional adjustment (in production, use proper optimization)
        adjusted_weights = {}
        for symbol, weight in portfolio_weights.items():
            if symbol in portfolio_exposures:
                exp = portfolio_exposures[symbol]
                factor_loading = getattr(exp, target_factor, 0.0)
                # Reduce weights in same direction as exposure
                adjustment = 1.0 - (target_exposure * factor_loading * 0.1)
                adjusted_weights[symbol] = weight * adjustment
            else:
                adjusted_weights[symbol] = weight  # No exposure data — keep as-is

        # Renormalize
        total = sum(abs(w) for w in adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        return adjusted_weights

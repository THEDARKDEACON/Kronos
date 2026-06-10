"""
Market Regime Detection Module
Detects volatility and correlation regimes for dynamic risk management.
Implements VIX-based regime switching and cross-sectional correlation monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import yfinance as yf
import warnings


class VolatilityRegime(Enum):
    """Volatility-based market regimes."""
    LOW = "low_vol"
    NORMAL = "normal_vol"
    ELEVATED = "elevated_vol"
    HIGH = "high_vol"
    CRISIS = "crisis_vol"


class CorrelationRegime(Enum):
    """Correlation-based market regimes."""
    LOW_DISPERSION = "low_dispersion"  # Correlations low, stock picking works
    NORMAL_DISPERSION = "normal_dispersion"
    HIGH_DISPERSION = "high_dispersion"  # Correlations high, macro dominates


@dataclass
class RegimeMetrics:
    """Current regime metrics."""
    vol_regime: VolatilityRegime
    corr_regime: CorrelationRegime
    vix_level: float
    vix_percentile: float
    avg_correlation: float
    correlation_percentile: float
    realized_vol_20d: float
    realized_vol_60d: float
    timestamp: datetime


class VolatilityRegimeDetector:
    """
    Detects volatility regimes using VIX and realized volatility.
    """
    
    # VIX thresholds for regime classification
    VIX_THRESHOLDS = {
        'low': 15,
        'normal': 20,
        'elevated': 25,
        'high': 30
    }
    
    def __init__(self, lookback_days: int = 252):
        self.lookback = lookback_days
        self.vix_history: List[float] = []
        self.regime_history: List[VolatilityRegime] = []
        
    def fetch_vix_data(self, as_of_date: Optional[datetime] = None) -> pd.Series:
        """Fetch VIX data from Yahoo Finance."""
        if as_of_date is None:
            as_of_date = datetime.now()
        
        end_date = as_of_date.strftime('%Y-%m-%d')
        start_date = (as_of_date - timedelta(days=self.lookback)).strftime('%Y-%m-%d')
        
        try:
            vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            if not vix_data.empty:
                return vix_data['Close'].squeeze()
        except Exception as e:
            warnings.warn(f"Failed to fetch VIX data: {e}")
        
        return pd.Series()
    
    def calculate_realized_volatility(self, 
                                     returns: pd.Series, 
                                     window: int = 20) -> float:
        """
        Calculate annualized realized volatility.
        """
        if len(returns) < window:
            return 0.20  # Default 20% vol
        
        vol = returns.tail(window).std() * np.sqrt(252)
        return vol
    
    def classify_vix_regime(self, vix_current: float, vix_percentile: float) -> VolatilityRegime:
        """
        Classify VIX into regime.
        
        Uses both absolute level and percentile for robustness.
        """
        # Primary classification by level
        if vix_current < self.VIX_THRESHOLDS['low']:
            regime = VolatilityRegime.LOW
        elif vix_current < self.VIX_THRESHOLDS['normal']:
            regime = VolatilityRegime.NORMAL
        elif vix_current < self.VIX_THRESHOLDS['elevated']:
            regime = VolatilityRegime.ELEVATED
        elif vix_current < self.VIX_THRESHOLDS['high']:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.CRISIS
        
        # Override with percentile if extreme
        if vix_percentile > 0.95:
            return VolatilityRegime.CRISIS
        elif vix_percentile < 0.05:
            return VolatilityRegime.LOW
        
        return regime
    
    def get_regime_exposure_multiplier(self, regime: VolatilityRegime) -> float:
        """
        Get position sizing multiplier for regime.
        
        Returns: multiplier for gross exposure (1.0 = normal)
        """
        multipliers = {
            VolatilityRegime.LOW: 1.3,      # Add risk in low vol
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.ELEVATED: 0.7, # Reduce risk
            VolatilityRegime.HIGH: 0.4,     # Significant reduction
            VolatilityRegime.CRISIS: 0.2    # Crisis mode
        }
        return multipliers.get(regime, 1.0)
    
    def get_regime_stop_loss(self, regime: VolatilityRegime) -> float:
        """
        Get stop loss level for regime.
        
        Returns: stop loss percentage (e.g., -0.05 = -5%)
        """
        stops = {
            VolatilityRegime.LOW: -0.05,
            VolatilityRegime.NORMAL: -0.07,
            VolatilityRegime.ELEVATED: -0.04,
            VolatilityRegime.HIGH: -0.03,
            VolatilityRegime.CRISIS: -0.02
        }
        return stops.get(regime, -0.05)
    
    def detect_current_regime(self, as_of_date: Optional[datetime] = None) -> Tuple[VolatilityRegime, Dict]:
        """
        Detect current volatility regime.
        
        Returns: (regime, metadata_dict)
        """
        vix_series = self.fetch_vix_data(as_of_date)
        
        if vix_series.empty:
            return VolatilityRegime.NORMAL, {'vix': 20.0, 'error': 'No data'}
        
        vix_current = float(vix_series.iloc[-1])
        vix_percentile = (vix_series < vix_current).mean()
        
        regime = self.classify_vix_regime(vix_current, vix_percentile)
        
        metadata = {
            'vix_current': vix_current,
            'vix_ma20': float(vix_series.tail(20).mean()),
            'vix_percentile': vix_percentile,
            'vix_regime': regime.value,
            'exposure_multiplier': self.get_regime_exposure_multiplier(regime)
        }
        
        self.regime_history.append(regime)
        self.vix_history.append(vix_current)
        
        return regime, metadata


class CrossSectionalCorrelationMonitor:
    """
    Monitors cross-sectional correlation of stock returns.
    High correlation = low dispersion = macro dominates.
    """
    
    def __init__(self, lookback: int = 20, correlation_threshold: float = 0.7):
        self.lookback = lookback
        self.threshold = correlation_threshold
        self.correlation_history: List[float] = []
        
    def calculate_average_correlation(self, returns_df: pd.DataFrame) -> float:
        """
        Calculate average pairwise correlation of returns.
        
        Args:
            returns_df: DataFrame with tickers as columns, dates as index
        
        Returns:
            Average correlation (0 to 1)
        """
        if returns_df.shape[1] < 2 or len(returns_df) < self.lookback:
            return 0.5  # Default moderate correlation
        
        # Calculate correlation matrix
        corr_matrix = returns_df.tail(self.lookback).corr()

        # Extract upper triangle using a boolean mask (avoids dropping valid 0-correlations)
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        correlations = corr_matrix.values[mask]
        
        if len(correlations) == 0:
            return 0.5
        
        avg_corr = np.mean(correlations)
        
        self.correlation_history.append(avg_corr)
        if len(self.correlation_history) > 63:
            self.correlation_history = self.correlation_history[-63:]
        
        return avg_corr
    
    def classify_correlation_regime(self, avg_correlation: float) -> CorrelationRegime:
        """
        Classify correlation regime.
        """
        if avg_correlation < 0.3:
            return CorrelationRegime.LOW_DISPERSION
        elif avg_correlation < 0.6:
            return CorrelationRegime.NORMAL_DISPERSION
        else:
            return CorrelationRegime.HIGH_DISPERSION
    
    def should_reduce_stock_picking(self, avg_correlation: float) -> bool:
        """
        Determine if stock picking alpha is likely reduced.
        
        When correlations are high, stock-specific signals have less impact.
        """
        return avg_correlation > self.threshold
    
    def get_dispersion_score(self) -> float:
        """
        Get current dispersion score (inverse of correlation).
        
        Returns: 0 to 1, higher = more dispersion = better for stock picking
        """
        if not self.correlation_history:
            return 0.5
        
        current_corr = self.correlation_history[-1]
        return 1 - current_corr


class UnifiedRegimeDetector:
    """
    Unified detector combining volatility and correlation regimes.
    """
    
    def __init__(self):
        self.vol_detector = VolatilityRegimeDetector()
        self.corr_monitor = CrossSectionalCorrelationMonitor()
        self.current_metrics: Optional[RegimeMetrics] = None
        
    def analyze_current_regime(self,
                                  returns_df: Optional[pd.DataFrame] = None,
                                  as_of_date: Optional[datetime] = None) -> RegimeMetrics:
        """
        Analyze current market regime.
        
        Args:
            returns_df: DataFrame of stock returns (optional)
            as_of_date: Analysis date (optional)
        
        Returns:
            RegimeMetrics with full regime assessment
        """
        # Get volatility regime
        vol_regime, vol_meta = self.vol_detector.detect_current_regime(as_of_date)
        
        # Get correlation regime
        if returns_df is not None and not returns_df.empty:
            avg_corr = self.corr_monitor.calculate_average_correlation(returns_df)
            corr_regime = self.corr_monitor.classify_correlation_regime(avg_corr)
            corr_percentile = 0.5  # Would need historical context
        else:
            avg_corr = 0.5
            corr_regime = CorrelationRegime.NORMAL_DISPERSION
            corr_percentile = 0.5
        
        metrics = RegimeMetrics(
            vol_regime=vol_regime,
            corr_regime=corr_regime,
            vix_level=vol_meta.get('vix_current', 20.0),
            vix_percentile=vol_meta.get('vix_percentile', 0.5),
            avg_correlation=avg_corr,
            correlation_percentile=corr_percentile,
            realized_vol_20d=vol_meta.get('vix_current', 20.0) / 100,  # Proxy
            realized_vol_60d=vol_meta.get('vix_ma20', 20.0) / 100,
            timestamp=as_of_date or datetime.now()
        )
        
        self.current_metrics = metrics
        return metrics
    
    def get_risk_adjustment_factors(self, metrics: Optional[RegimeMetrics] = None) -> Dict[str, float]:
        """
        Get risk adjustment factors for current regime.
        
        Returns:
            Dict with position sizing, leverage, and stop-loss recommendations
        """
        if metrics is None:
            metrics = self.current_metrics
        
        if metrics is None:
            return {
                'exposure_multiplier': 1.0,
                'position_size_factor': 1.0,
                'stop_loss': -0.05,
                'leverage': 1.0
            }
        
        # Volatility-based adjustments
        vol_mult = self.vol_detector.get_regime_exposure_multiplier(metrics.vol_regime)
        vol_stop = self.vol_detector.get_regime_stop_loss(metrics.vol_regime)
        
        # Correlation-based adjustments
        if metrics.corr_regime == CorrelationRegime.HIGH_DISPERSION:
            corr_mult = 0.7  # Reduce stock picking
        elif metrics.corr_regime == CorrelationRegime.LOW_DISPERSION:
            corr_mult = 1.2  # Increase stock picking
        else:
            corr_mult = 1.0
        
        # Combined adjustments
        combined_mult = vol_mult * corr_mult
        
        return {
            'exposure_multiplier': combined_mult,
            'position_size_factor': combined_mult,
            'stop_loss': vol_stop,
            'leverage': min(combined_mult * 1.0, 1.5),
            'vol_regime': metrics.vol_regime.value,
            'corr_regime': metrics.corr_regime.value,
            'vix': metrics.vix_level,
            'avg_correlation': metrics.avg_correlation
        }
    
    def should_halt_trading(self, metrics: Optional[RegimeMetrics] = None) -> Tuple[bool, str]:
        """
        Determine if trading should be halted due to extreme conditions.
        
        Returns: (should_halt, reason)
        """
        if metrics is None:
            metrics = self.current_metrics
        
        if metrics is None:
            return False, "No metrics available"
        
        # Crisis-level VIX
        if metrics.vix_level > 40:
            return True, f"VIX crisis level: {metrics.vix_level:.1f}"
        
        # Extreme correlation (all stocks moving together)
        if metrics.avg_correlation > 0.9:
            return True, f"Extreme correlation: {metrics.avg_correlation:.2f}"
        
        # VIX spike (>50% increase in 5 days)
        if len(self.vol_detector.vix_history) >= 5:
            vix_5d_ago = self.vol_detector.vix_history[-5]
            if vix_5d_ago > 0 and metrics.vix_level / vix_5d_ago > 1.5:
                return True, f"VIX spike: {vix_5d_ago:.1f} -> {metrics.vix_level:.1f}"
        
        return False, "Normal conditions"


class DynamicPositionSizer:
    """
    Dynamic position sizing based on market regime.
    """
    
    def __init__(self, regime_detector: UnifiedRegimeDetector):
        self.detector = regime_detector
        
    def size_positions(self,
                      base_weights: Dict[str, float],
                      signals: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Adjust position sizes based on current regime.
        
        Args:
            base_weights: Initial position weights
            signals: Signal confidence scores (optional)
        
        Returns:
            Adjusted position weights
        """
        # Get regime adjustments
        adjustments = self.detector.get_risk_adjustment_factors()
        
        # Apply exposure multiplier
        mult = adjustments['position_size_factor']
        adjusted = {k: v * mult for k, v in base_weights.items()}
        
        # Apply signal-based sizing if available
        if signals is not None and 'Confidence' in signals.columns:
            for ticker in adjusted:
                if ticker in signals.index:
                    conf = signals.loc[ticker, 'Confidence']
                    adjusted[ticker] *= (0.5 + 0.5 * conf)  # Scale by confidence
        
        # Renormalize to maintain gross exposure
        gross = sum(abs(v) for v in adjusted.values())
        if gross > 0:
            adjusted = {k: v / gross for k, v in adjusted.items()}
        
        return adjusted
    
    def get_portfolio_constraints(self) -> Dict[str, float]:
        """
        Get portfolio constraints based on current regime.
        
        Returns:
            Dict with max position size, max sector exposure, etc.
        """
        adjustments = self.detector.get_risk_adjustment_factors()
        
        base_constraints = {
            'max_position': 0.20,
            'max_sector_exposure': 0.25,
            'max_drawdown': -0.10
        }
        
        # Adjust based on regime
        vol_regime = self.detector.current_metrics.vol_regime if self.detector.current_metrics else VolatilityRegime.NORMAL
        
        if vol_regime in [VolatilityRegime.HIGH, VolatilityRegime.CRISIS]:
            return {
                'max_position': 0.10,  # Reduce concentration
                'max_sector_exposure': 0.15,
                'max_drawdown': -0.05
            }
        elif vol_regime == VolatilityRegime.ELEVATED:
            return {
                'max_position': 0.15,
                'max_sector_exposure': 0.20,
                'max_drawdown': -0.07
            }
        
        return base_constraints


# Convenience functions
def get_current_regime_metrics() -> RegimeMetrics:
    """Quick function to get current regime metrics."""
    detector = UnifiedRegimeDetector()
    return detector.analyze_current_regime()


def get_regime_adjusted_portfolio(base_weights: Dict[str, float],
                                  returns_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Get regime-adjusted portfolio weights.
    """
    detector = UnifiedRegimeDetector()
    detector.analyze_current_regime(returns_df)
    
    sizer = DynamicPositionSizer(detector)
    return sizer.size_positions(base_weights)

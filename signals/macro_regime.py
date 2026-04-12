"""
Macro Regime Detection Module
Classifies market regimes based on VIX, yield curve, and macro indicators.
Provides regime-specific factor exposures and risk adjustments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import yfinance as yf


class MacroRegime(Enum):
    """Macro market regimes."""
    RISK_ON = "risk_on"              # Bull market, low vol
    RISK_OFF = "risk_off"            # Bear market, high vol
    GROWTH = "growth"                # Growth stocks outperform
    VALUE = "value"                  # Value stocks outperform
    INFLATION = "inflation"          # High inflation regime
    DEFLATION = "deflation"          # Low inflation/deflation
    RECESSION = "recession"          # Economic contraction
    RECOVERY = "recovery"            # Economic expansion


@dataclass
class RegimeState:
    """Current macro regime state."""
    primary_regime: MacroRegime
    secondary_regime: Optional[MacroRegime]
    vix_level: float
    yield_curve_slope: float
    dxy_strength: float
    confidence: float
    timestamp: datetime


class MacroRegimeDetector:
    """
    Detects macro market regimes using VIX, yield curve, and macro indicators.
    """
    
    def __init__(self, lookback_days: int = 63):
        self.lookback = lookback_days
        self.current_state: Optional[RegimeState] = None
        self.history: List[RegimeState] = []
        
    def fetch_macro_data(self, as_of_date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Fetch macro indicators from Yahoo Finance.
        
        Returns:
            Dict with VIX, yield curve, DXY, oil prices
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        end_date = as_of_date.strftime('%Y-%m-%d')
        start_date = (as_of_date - timedelta(days=self.lookback)).strftime('%Y-%m-%d')
        
        indicators = {}
        
        try:
            # VIX - Volatility index
            vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            if not vix_data.empty:
                indicators['vix_current'] = float(vix_data['Close'].iloc[-1].iloc[0])
                indicators['vix_ma20'] = float(vix_data['Close'].tail(20).mean().iloc[0])
                indicators['vix_percentile'] = float(
                    (vix_data['Close'] < indicators['vix_current']).mean().iloc[0]
                )
            else:
                indicators['vix_current'] = 20.0
                indicators['vix_ma20'] = 20.0
                indicators['vix_percentile'] = 0.5
            
            # Yield Curve (10Y - 2Y)
            try:
                ten_year = yf.download('^TNX', start=start_date, end=end_date, progress=False)
                two_year = yf.download('^FVX', start=start_date, end=end_date, progress=False)  # 5Y as proxy
                
                if not ten_year.empty and not two_year.empty:
                    tnx_current = ten_year['Close'].iloc[-1]
                    fvx_current = two_year['Close'].iloc[-1]
                    indicators['yield_curve'] = float(tnx_current - fvx_current)
                    indicators['yield_curve_30d_avg'] = float(
                        ten_year['Close'].tail(30).mean() - two_year['Close'].tail(30).mean()
                    )
                else:
                    indicators['yield_curve'] = 0.5
                    indicators['yield_curve_30d_avg'] = 0.5
            except:
                indicators['yield_curve'] = 0.5
                indicators['yield_curve_30d_avg'] = 0.5
            
            # DXY - Dollar strength
            dxy_data = yf.download('DX-Y.NYB', start=start_date, end=end_date, progress=False)
            if not dxy_data.empty:
                indicators['dxy_current'] = float(dxy_data['Close'].iloc[-1].iloc[0])
                indicators['dxy_ma20'] = float(dxy_data['Close'].tail(20).mean().iloc[0])
            else:
                indicators['dxy_current'] = 100.0
                indicators['dxy_ma20'] = 100.0
            
            # Oil prices (inflation proxy)
            oil_data = yf.download('CL=F', start=start_date, end=end_date, progress=False)
            if not oil_data.empty:
                indicators['oil_current'] = float(oil_data['Close'].iloc[-1])
                indicators['oil_change_30d'] = float(
                    (oil_data['Close'].iloc[-1] / oil_data['Close'].iloc[-30] - 1) 
                    if len(oil_data) >= 30 else 0.0
                )
            else:
                indicators['oil_current'] = 75.0
                indicators['oil_change_30d'] = 0.0
            
            # Gold (safe haven proxy)
            gold_data = yf.download('GC=F', start=start_date, end=end_date, progress=False)
            if not gold_data.empty:
                indicators['gold_change_30d'] = float(
                    (gold_data['Close'].iloc[-1].iloc[0] / gold_data['Close'].iloc[-30].iloc[0] - 1)
                    if len(gold_data) >= 30 else 0.0
                )
            else:
                indicators['gold_change_30d'] = 0.0
            
        except Exception as e:
            print(f"[Macro] Error fetching data: {e}")
            # Provide defaults
            indicators = {
                'vix_current': 20.0,
                'vix_ma20': 20.0,
                'vix_percentile': 0.5,
                'yield_curve': 0.5,
                'yield_curve_30d_avg': 0.5,
                'dxy_current': 100.0,
                'dxy_ma20': 100.0,
                'oil_current': 75.0,
                'oil_change_30d': 0.0,
                'gold_change_30d': 0.0
            }
        
        return indicators
    
    def detect_regime(self, indicators: Dict[str, float]) -> RegimeState:
        """
        Detect current macro regime from indicators.
        """
        vix = indicators['vix_current']
        vix_ma = indicators['vix_ma20']
        vix_pct = indicators['vix_percentile']
        
        yield_curve = indicators['yield_curve']
        dxy = indicators['dxy_current']
        oil_change = indicators['oil_change_30d']
        gold_change = indicators['gold_change_30d']
        
        # Primary regime detection
        # VIX-based risk on/off
        if vix > 30 or vix_pct > 0.8:
            primary = MacroRegime.RISK_OFF
        elif vix < 15 and vix_pct < 0.3:
            primary = MacroRegime.RISK_ON
        else:
            primary = MacroRegime.RECOVERY  # Neutral/mixed
        
        # Secondary regime based on macro factors
        secondary = None
        
        # Yield curve inversion = recession risk
        if yield_curve < -0.2:
            secondary = MacroRegime.RECESSION
        elif yield_curve < 0:
            secondary = MacroRegime.DEFLATION
        
        # Oil/Gold for inflation/deflation
        if oil_change > 0.1 or gold_change > 0.05:
            if secondary == MacroRegime.DEFLATION:
                secondary = MacroRegime.INFLATION  # Override
            elif secondary is None:
                secondary = MacroRegime.INFLATION
        elif oil_change < -0.1:
            if secondary is None:
                secondary = MacroRegime.DEFLATION
        
        # DXY strength for global risk factors
        if dxy > 105:
            # Strong dollar = risk off pressure on EM
            if primary == MacroRegime.RISK_ON:
                primary = MacroRegime.RECOVERY  # Downgrade
        
        # Calculate confidence based on signal clarity
        signals_agree = 0
        total_signals = 3
        
        if vix > 25 or vix < 18:
            signals_agree += 1
        if abs(yield_curve) > 0.5:
            signals_agree += 1
        if abs(oil_change) > 0.05:
            signals_agree += 1
        
        confidence = signals_agree / total_signals
        
        state = RegimeState(
            primary_regime=primary,
            secondary_regime=secondary,
            vix_level=vix,
            yield_curve_slope=yield_curve,
            dxy_strength=dxy,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        self.current_state = state
        self.history.append(state)
        
        return state
    
    def get_current_regime(self) -> RegimeState:
        """Get current regime (fetch + detect)."""
        indicators = self.fetch_macro_data()
        return self.detect_regime(indicators)
    
    def get_regime_factor_tilts(self, regime: MacroRegime) -> Dict[str, float]:
        """
        Get factor tilts for a specific regime.
        
        Returns dict of factor -> tilt (positive = overweight).
        """
        factor_tilts = {
            # Default: neutral
            'Market': 0.0,
            'Size': 0.0,
            'Value': 0.0,
            'Momentum': 0.0,
            'Quality': 0.0,
            'Growth': 0.0,
            'Volatility': 0.0
        }
        
        if regime == MacroRegime.RISK_ON:
            factor_tilts.update({
                'Market': 0.2,
                'Size': 0.1,      # Small cap
                'Growth': 0.2,
                'Momentum': 0.15,
                'Quality': -0.1   # Less quality focus
            })
        
        elif regime == MacroRegime.RISK_OFF:
            factor_tilts.update({
                'Market': -0.3,
                'Size': -0.1,
                'Value': 0.2,
                'Quality': 0.3,   # High quality
                'Volatility': -0.3 # Low vol
            })
        
        elif regime == MacroRegime.GROWTH:
            factor_tilts.update({
                'Growth': 0.3,
                'Value': -0.3,
                'Momentum': 0.1
            })
        
        elif regime == MacroRegime.VALUE:
            factor_tilts.update({
                'Growth': -0.3,
                'Value': 0.3,
                'Quality': 0.1
            })
        
        elif regime == MacroRegime.INFLATION:
            factor_tilts.update({
                'Value': 0.2,
                'Size': 0.1,
                'Growth': -0.2
            })
        
        elif regime == MacroRegime.DEFLATION:
            factor_tilts.update({
                'Growth': 0.2,
                'Value': -0.1,
                'Quality': 0.2
            })
        
        elif regime == MacroRegime.RECESSION:
            factor_tilts.update({
                'Market': -0.4,
                'Quality': 0.4,
                'Volatility': -0.4
            })
        
        return factor_tilts
    
    def get_position_scaling(self, regime: MacroRegime) -> float:
        """
        Get gross exposure scaling factor for regime.
        
        Returns multiplier for portfolio exposure (1.0 = normal).
        """
        scaling = {
            MacroRegime.RISK_ON: 1.2,
            MacroRegime.RISK_OFF: 0.5,
            MacroRegime.RECESSION: 0.3,
            MacroRegime.RECOVERY: 1.0,
            MacroRegime.GROWTH: 1.1,
            MacroRegime.VALUE: 1.0,
            MacroRegime.INFLATION: 0.8,
            MacroRegime.DEFLATION: 0.9
        }
        
        return scaling.get(regime, 1.0)
    
    def should_reduce_exposure(self, regime: MacroRegime, portfolio_vol: float = 0.15) -> Tuple[bool, float]:
        """
        Determine if exposure should be reduced and by how much.
        
        Returns: (should_reduce, target_exposure)
        """
        if regime in [MacroRegime.RISK_OFF, MacroRegime.RECESSION]:
            return True, 0.3  # Reduce to 30% gross
        
        elif regime == MacroRegime.INFLATION and portfolio_vol > 0.20:
            return True, 0.6
        
        elif portfolio_vol > 0.25:
            return True, 0.5
        
        return False, 1.0


class RegimeTransitionMonitor:
    """
    Monitors regime transitions and generates alerts.
    """
    
    def __init__(self, detector: MacroRegimeDetector, lookback: int = 10):
        self.detector = detector
        self.lookback = lookback
        self.regime_history: List[MacroRegime] = []
        
    def update(self):
        """Update regime history."""
        state = self.detector.get_current_regime()
        self.regime_history.append(state.primary_regime)
        
        # Keep only last N regimes
        if len(self.regime_history) > self.lookback:
            self.regime_history = self.regime_history[-self.lookback:]
    
    def is_regime_changing(self) -> Tuple[bool, Optional[MacroRegime]]:
        """
        Check if regime is transitioning.
        
        Returns: (is_changing, new_regime)
        """
        if len(self.regime_history) < 3:
            return False, None
        
        recent = self.regime_history[-3:]
        
        # Check for consistent new regime
        if len(set(recent)) == 1:
            current = recent[-1]
            if len(self.regime_history) >= 4:
                previous = self.regime_history[-4]
                if previous != current:
                    return True, current
        
        return False, None
    
    def get_stability_score(self) -> float:
        """
        Get regime stability score (0-1, higher = more stable).
        """
        if len(self.regime_history) < 5:
            return 0.5
        
        # Calculate regime consistency
        unique_regimes = len(set(self.regime_history))
        total_regimes = len(self.regime_history)
        
        stability = 1 - (unique_regimes - 1) / (total_regimes - 1)
        return max(0.0, min(1.0, stability))


# Convenience functions
def get_current_macro_regime() -> RegimeState:
    """Quick function to get current macro regime."""
    detector = MacroRegimeDetector()
    return detector.get_current_regime()


def get_regime_adjusted_weights(base_weights: Dict[str, float], 
                                 regime: MacroRegime) -> Dict[str, float]:
    """
    Adjust portfolio weights based on macro regime.
    """
    detector = MacroRegimeDetector()
    scaling = detector.get_position_scaling(regime)
    
    # Apply scaling
    adjusted = {k: v * scaling for k, v in base_weights.items()}
    
    # Renormalize
    gross = sum(abs(v) for v in adjusted.values())
    if gross > 0:
        adjusted = {k: v / gross for k, v in adjusted.items()}
    
    return adjusted

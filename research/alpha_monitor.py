"""
Alpha Decay Monitoring System
Tracks signal performance over time and alerts when alpha degrades.
Prevents "silent death" of trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
from scipy.stats import spearmanr


@dataclass
class ICMetrics:
    """Information coefficient metrics for a signal."""
    date: datetime
    ic_1d: float
    ic_5d: float
    ic_20d: float
    rank_ic: float
    t_stat: float
    p_value: float
    sample_size: int


@dataclass
class SignalHealth:
    """Health status of a signal."""
    signal_name: str
    current_ic: float
    ic_trend: str  # 'improving', 'stable', 'decaying'
    rolling_avg_ic: float
    days_since_positive: int
    is_alive: bool
    alert_level: str  # 'none', 'warning', 'critical'


class AlphaDecayMonitor:
    """
    Monitors signal performance and detects alpha decay.
    
    Key metrics tracked:
    - Information Coefficient (IC): Correlation between signals and returns
    - Hit rate: % of times signal direction is correct
    - Rolling IC: Moving average to detect trends
    - IC decay rate: How fast alpha is fading
    """
    
    # Alert thresholds
    IC_WARNING = 0.02    # IC below this is concerning
    IC_CRITICAL = 0.0    # IC below this means signal is dead
    IC_DECAY_DAYS = 5   # Days of negative IC before alert
    
    def __init__(self, lookback_window: int = 63):
        self.lookback = lookback_window
        
        # IC history per signal
        self.ic_history: Dict[str, List[ICMetrics]] = defaultdict(list)
        
        # Signal predictions and actual returns
        self.signal_predictions: Dict[str, pd.DataFrame] = {}
        self.actual_returns: pd.DataFrame = pd.DataFrame()
        
        # Alerts
        self.alerts: List[Dict] = []
        
    def calculate_information_coefficient(self,
                                       predictions: pd.Series,
                                       returns: pd.Series,
                                       method: str = 'pearson') -> Tuple[float, float]:
        """
        Calculate IC between predictions and realized returns.
        
        Args:
            predictions: Signal values
            returns: Realized forward returns
            method: 'pearson' (linear) or 'spearman' (rank)
        
        Returns:
            (IC, p_value)
        """
        # Align series
        common_idx = predictions.index.intersection(returns.index)
        pred = predictions.loc[common_idx].dropna()
        ret = returns.loc[common_idx].dropna()
        
        if len(pred) < 10:
            return 0.0, 1.0
        
        if method == 'spearman':
            ic, pval = spearmanr(pred, ret)
        else:
            ic = np.corrcoef(pred, ret)[0, 1]
            # Approximate p-value
            from scipy.stats import pearsonr
            _, pval = pearsonr(pred, ret)
        
        return ic if not np.isnan(ic) else 0.0, pval if not np.isnan(pval) else 1.0
    
    def update_signal_performance(self,
                                 signal_name: str,
                                 predictions: pd.Series,
                                 returns_1d: pd.Series,
                                 returns_5d: Optional[pd.Series] = None,
                                 returns_20d: Optional[pd.Series] = None,
                                 date: Optional[datetime] = None):
        """
        Update signal performance tracking.
        
        Args:
            signal_name: Name of the signal (e.g., 'kronos', 'sentiment')
            predictions: Signal predictions for date
            returns_1d: 1-day forward returns
            returns_5d: 5-day forward returns (optional)
            returns_20d: 20-day forward returns (optional)
            date: Date of prediction (default: today)
        """
        if date is None:
            date = datetime.now()
        
        # Calculate ICs
        ic_1d, p_1d = self.calculate_information_coefficient(predictions, returns_1d)
        
        ic_5d = 0.0
        if returns_5d is not None:
            ic_5d, _ = self.calculate_information_coefficient(predictions, returns_5d)
        
        ic_20d = 0.0
        if returns_20d is not None:
            ic_20d, _ = self.calculate_information_coefficient(predictions, returns_20d)
        
        # Rank IC
        rank_ic, _ = self.calculate_information_coefficient(predictions, returns_1d, 'spearman')
        
        # Store metrics
        metrics = ICMetrics(
            date=date,
            ic_1d=ic_1d,
            ic_5d=ic_5d,
            ic_20d=ic_20d,
            rank_ic=rank_ic,
            t_stat=ic_1d * np.sqrt(len(predictions)) if len(predictions) > 0 else 0,
            p_value=p_1d,
            sample_size=len(predictions)
        )
        
        self.ic_history[signal_name].append(metrics)
        
        # Trim history
        cutoff = date - timedelta(days=self.lookback)
        self.ic_history[signal_name] = [
            m for m in self.ic_history[signal_name] if m.date > cutoff
        ]
    
    def get_signal_health(self, signal_name: str) -> SignalHealth:
        """
        Get health status of a signal.
        
        Returns:
            SignalHealth with current status
        """
        if signal_name not in self.ic_history or not self.ic_history[signal_name]:
            return SignalHealth(
                signal_name=signal_name,
                current_ic=0.0,
                ic_trend='unknown',
                rolling_avg_ic=0.0,
                days_since_positive=0,
                is_alive=False,
                alert_level='critical'
            )
        
        history = self.ic_history[signal_name]
        
        # Current IC (most recent)
        current_ic = history[-1].ic_1d
        
        # Rolling average (last 20 days)
        recent_ics = [m.ic_1d for m in history[-20:]]
        rolling_avg = np.mean(recent_ics) if recent_ics else 0.0
        
        # IC trend
        if len(recent_ics) >= 10:
            early = np.mean(recent_ics[:5])
            late = np.mean(recent_ics[-5:])
            
            if late > early + 0.01:
                trend = 'improving'
            elif late < early - 0.01:
                trend = 'decaying'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'
        
        # Days since positive IC
        days_since_positive = 0
        for m in reversed(history):
            if m.ic_1d <= 0:
                days_since_positive += 1
            else:
                break
        
        # Determine alert level
        if current_ic <= self.IC_CRITICAL or days_since_positive >= self.IC_DECAY_DAYS:
            alert = 'critical'
            alive = False
        elif current_ic < self.IC_WARNING or rolling_avg < self.IC_WARNING:
            alert = 'warning'
            alive = True
        else:
            alert = 'none'
            alive = True
        
        return SignalHealth(
            signal_name=signal_name,
            current_ic=current_ic,
            ic_trend=trend,
            rolling_avg_ic=rolling_avg,
            days_since_positive=days_since_positive,
            is_alive=alive,
            alert_level=alert
        )
    
    def check_all_signals(self) -> List[SignalHealth]:
        """
        Check health of all monitored signals.
        
        Returns:
            List of SignalHealth for all signals
        """
        results = []
        
        for signal_name in self.ic_history:
            health = self.get_signal_health(signal_name)
            results.append(health)
            
            # Generate alerts
            if health.alert_level != 'none':
                self.alerts.append({
                    'timestamp': datetime.now(),
                    'signal': signal_name,
                    'alert_level': health.alert_level,
                    'message': self._generate_alert_message(health),
                    'metrics': {
                        'current_ic': health.current_ic,
                        'rolling_ic': health.rolling_avg_ic,
                        'trend': health.ic_trend
                    }
                })
        
        return results
    
    def _generate_alert_message(self, health: SignalHealth) -> str:
        """Generate human-readable alert message."""
        if health.alert_level == 'critical':
            return (
                f"CRITICAL: Signal '{health.signal_name}' is DEAD. "
                f"IC={health.current_ic:.3f}, "
                f"{health.days_since_positive} days since positive. "
                f"STOP USING THIS SIGNAL."
            )
        elif health.alert_level == 'warning':
            return (
                f"WARNING: Signal '{health.signal_name}' is decaying. "
                f"IC={health.current_ic:.3f} (rolling avg: {health.rolling_avg_ic:.3f}), "
                f"trend: {health.ic_trend}. "
                f"Consider reducing weight or investigating."
            )
        else:
            return f"Signal '{health.signal_name}' is healthy. IC={health.current_ic:.3f}"
    
    def generate_health_report(self) -> str:
        """
        Generate comprehensive health report for all signals.
        """
        healths = self.check_all_signals()
        
        report = ["\nALPHA DECAY MONITORING REPORT", "=" * 60]
        
        # Summary
        alive = sum(1 for h in healths if h.is_alive)
        critical = sum(1 for h in healths if h.alert_level == 'critical')
        warning = sum(1 for h in healths if h.alert_level == 'warning')
        
        report.append(f"\nSUMMARY")
        report.append(f"  Signals monitored: {len(healths)}")
        report.append(f"  Healthy: {alive}")
        report.append(f"  Warning: {warning}")
        report.append(f"  Critical: {critical}")
        
        # Details
        report.append(f"\nSIGNAL DETAILS")
        for h in healths:
            status = "🟢" if h.alert_level == 'none' else ("🟡" if h.alert_level == 'warning' else "🔴")
            report.append(
                f"  {status} {h.signal_name:20s} | "
                f"IC: {h.current_ic:+.3f} | "
                f"Rolling: {h.rolling_avg_ic:+.3f} | "
                f"Trend: {h.ic_trend:12s} | "
                f"Days -ve: {h.days_since_positive}"
            )
        
        # Recent alerts
        if self.alerts:
            report.append(f"\nRECENT ALERTS")
            for alert in self.alerts[-5:]:
                report.append(f"  [{alert['timestamp'].strftime('%Y-%m-%d %H:%M')}] "
                            f"{alert['alert_level'].upper()}: {alert['message'][:80]}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def plot_ic_history(self, signal_name: str):
        """
        Plot IC history for visualization.
        (Would require matplotlib)
        """
        if signal_name not in self.ic_history:
            print(f"No history for signal '{signal_name}'")
            return
        
        history = self.ic_history[signal_name]
        
        dates = [m.date for m in history]
        ic_1d = [m.ic_1d for m in history]
        ic_5d = [m.ic_5d for m in history]
        
        # Calculate rolling average
        window = 20
        rolling = pd.Series(ic_1d).rolling(window=window, min_periods=5).mean()
        
        # Would plot here with matplotlib
        print(f"\nIC History for {signal_name}:")
        print(f"  Current: {ic_1d[-1]:+.3f}")
        print(f"  20-day rolling: {rolling.iloc[-1]:+.3f}")
        print(f"  Min: {min(ic_1d):+.3f}, Max: {max(ic_1d):+.3f}")


class FactorCrowdingDetector:
    """
    Detects when too many market participants are using similar signals.
    Crowded factors tend to underperform and have higher volatility.
    """
    
    def __init__(self):
        self.factor_correlations: Dict[str, pd.DataFrame] = {}
        self.crowding_scores: Dict[str, float] = {}
    
    def calculate_factor_correlation(self,
                                    factor_returns: pd.DataFrame,
                                    window: int = 63) -> pd.DataFrame:
        """
        Calculate correlation between different factor returns.
        
        High correlation = crowding (everyone trading similar factors)
        """
        if len(factor_returns) < window:
            return pd.DataFrame()
        
        # Rolling correlation
        corr = factor_returns.tail(window).corr()
        
        return corr
    
    def detect_crowding(self,
                       factor_name: str,
                       factor_returns: pd.Series,
                       market_returns: pd.Series,
                       threshold: float = 0.7) -> Dict:
        """
        Detect if a factor is crowded.
        
        Indicators of crowding:
        1. High correlation with other popular factors
        2. Declining Sharpe ratio
        3. Increasing volatility
        4. High beta to market (losing diversification)
        
        Returns:
            Dict with crowding metrics
        """
        if len(factor_returns) < 63:
            return {'is_crowded': False, 'confidence': 0}
        
        # Calculate metrics
        recent = factor_returns.tail(63)
        older = factor_returns.iloc[-126:-63] if len(factor_returns) >= 126 else factor_returns.iloc[:63]
        
        # Sharpe ratio comparison
        recent_sharpe = recent.mean() / recent.std() if recent.std() > 0 else 0
        older_sharpe = older.mean() / older.std() if older.std() > 0 else 0
        sharpe_decline = recent_sharpe < older_sharpe * 0.5
        
        # Volatility increase
        recent_vol = recent.std()
        older_vol = older.std()
        vol_increase = recent_vol > older_vol * 1.3
        
        # Market correlation (beta)
        aligned = pd.DataFrame({'factor': factor_returns, 'market': market_returns}).dropna()
        if len(aligned) > 20:
            beta = np.polyfit(aligned['market'], aligned['factor'], 1)[0]
            high_beta = abs(beta) > 1.5
        else:
            high_beta = False
        
        # Crowding score (0-1)
        score = sum([sharpe_decline, vol_increase, high_beta]) / 3
        
        return {
            'factor': factor_name,
            'is_crowded': score > threshold,
            'crowding_score': score,
            'sharpe_decline': sharpe_decline,
            'vol_increase': vol_increase,
            'high_beta': high_beta,
            'recent_sharpe': recent_sharpe,
            'recent_vol': recent_vol
        }


# Convenience functions
def monitor_signal(signal_name: str,
                  predictions: pd.Series,
                  returns: pd.Series) -> SignalHealth:
    """
    Quick function to check signal health.
    """
    monitor = AlphaDecayMonitor()
    monitor.update_signal_performance(signal_name, predictions, returns)
    return monitor.get_signal_health(signal_name)


def check_all_signals_health(signals_dict: Dict[str, pd.Series],
                             returns: pd.Series) -> pd.DataFrame:
    """
    Check health of multiple signals.
    
    Args:
        signals_dict: Dict of signal_name -> predictions
        returns: Forward returns
    
    Returns:
        DataFrame with health metrics
    """
    monitor = AlphaDecayMonitor()
    
    for name, preds in signals_dict.items():
        monitor.update_signal_performance(name, preds, returns)
    
    healths = monitor.check_all_signals()
    
    return pd.DataFrame([
        {
            'Signal': h.signal_name,
            'Current_IC': h.current_ic,
            'Rolling_IC': h.rolling_avg_ic,
            'Trend': h.ic_trend,
            'Status': 'Alive' if h.is_alive else 'DEAD',
            'Alert': h.alert_level
        }
        for h in healths
    ])

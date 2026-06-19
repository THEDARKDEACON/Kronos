"""
Multi-Model Signal Ensemble Module
Combines Kronos price predictions with FinBERT sentiment and macro regime signals.
Implements adaptive weighting based on recent information coefficient (IC).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings


@dataclass
class ModelPerformance:
    """Track model performance metrics."""
    model_name: str
    ic_1d: float  # Information coefficient 1-day
    ic_5d: float  # Information coefficient 5-day
    volatility: float
    sharpe: float
    hit_rate: float
    decay_factor: float  # Weight decay if performance degrades


@dataclass
class EnsembleSignal:
    """Combined signal from multiple models."""
    ticker: str
    kronos_signal: float
    sentiment_signal: float
    macro_tilt: float
    ensemble_score: float
    confidence: float
    raw_predicted_return: float


class AdaptiveEnsemble:
    """
    Adaptive ensemble that weights models by their recent performance.
    """
    
    def __init__(self, 
                 kronos_weight: float = 0.5,
                 sentiment_weight: float = 0.3,
                 macro_weight: float = 0.2,
                 adaptation_speed: float = 0.1):  # How fast to adapt weights
        self.base_weights = {
            'kronos': kronos_weight,
            'sentiment': sentiment_weight,
            'macro': macro_weight
        }
        self.current_weights = self.base_weights.copy()
        self.adaptation_speed = adaptation_speed
        self.performance_history: Dict[str, List[ModelPerformance]] = {
            'kronos': [],
            'sentiment': [],
            'macro': []
        }
        self.ic_history: Dict[str, List[float]] = {
            'kronos': [],
            'sentiment': [],
            'macro': []
        }
    
    def calculate_information_coefficient(self,
                                        signals: pd.Series,
                                        forward_returns: pd.Series,
                                        rank_ic: bool = True) -> float:
        """
        Calculate information coefficient (correlation between signals and returns).
        
        Args:
            signals: Model predictions/signals
            forward_returns: Actual realized returns
            rank_ic: Use Spearman correlation (rank-based) vs Pearson
        
        Returns:
            IC value (-1 to 1)
        """
        # Align series
        common_idx = signals.index.intersection(forward_returns.index)
        s = signals.loc[common_idx]
        r = forward_returns.loc[common_idx]
        
        if len(s) < 10:
            return 0.0
        
        if rank_ic:
            # Spearman rank correlation (more robust)
            from scipy.stats import spearmanr
            corr, _ = spearmanr(s, r)
        else:
            # Pearson correlation
            corr = np.corrcoef(s, r)[0, 1]
        
        return corr if not np.isnan(corr) else 0.0
    
    def update_model_performance(self,
                                  model_name: str,
                                  signals: pd.Series,
                                  returns_1d: pd.Series,
                                  returns_5d: pd.Series):
        """
        Update performance tracking for a model.
        """
        ic_1d = self.calculate_information_coefficient(signals, returns_1d)
        ic_5d = self.calculate_information_coefficient(signals, returns_5d)
        
        # Track IC history
        self.ic_history[model_name].append(ic_1d)
        if len(self.ic_history[model_name]) > 63:  # Keep 3 months
            self.ic_history[model_name] = self.ic_history[model_name][-63:]
        
        # Calculate rolling metrics
        recent_ics = self.ic_history[model_name][-20:]
        
        perf = ModelPerformance(
            model_name=model_name,
            ic_1d=ic_1d,
            ic_5d=ic_5d,
            volatility=np.std(recent_ics) if recent_ics else 0.1,
            sharpe=np.mean(recent_ics) / (np.std(recent_ics) + 1e-6) if recent_ics else 0.0,
            hit_rate=np.mean([ic > 0 for ic in recent_ics]) if recent_ics else 0.5,
            decay_factor=1.0
        )
        
        self.performance_history[model_name].append(perf)
        if len(self.performance_history[model_name]) > 63:
            self.performance_history[model_name] = self.performance_history[model_name][-63:]
    
    def adapt_weights(self, ic_history: Optional[Dict[str, List[float]]] = None):
        """
        Adapt ensemble weights based on recent IC performance.
        Uses exponential weighting of recent ICs.
        """
        if ic_history:
            for model_name, values in ic_history.items():
                if model_name in self.ic_history and values:
                    self.ic_history[model_name] = list(values)

        new_weights = {}
        
        for model_name in ['kronos', 'sentiment', 'macro']:
            if not self.ic_history[model_name]:
                new_weights[model_name] = self.base_weights[model_name]
                continue
            
            # Calculate exponentially weighted IC
            recent_ics = self.ic_history[model_name][-20:]
            weights = np.exp(np.linspace(-1, 0, len(recent_ics)))  # Exponential decay
            weighted_ic = np.average([max(0, ic) for ic in recent_ics], weights=weights)
            
            new_weights[model_name] = weighted_ic
        
        # Normalize to sum to 1
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}
        else:
            new_weights = self.base_weights.copy()
        
        # Smooth transition (don't change too fast)
        for model_name in new_weights:
            self.current_weights[model_name] = (
                (1 - self.adaptation_speed) * self.current_weights[model_name] +
                self.adaptation_speed * new_weights[model_name]
            )
        
        # Renormalize
        total = sum(self.current_weights.values())
        self.current_weights = {k: v / total for k, v in self.current_weights.items()}
    
    def combine_signals(self,
                       kronos_signals: pd.DataFrame,
                       sentiment_signals: pd.DataFrame,
                       macro_tilts: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Combine signals from multiple models.
        
        Args:
            kronos_signals: DataFrame with 'Predicted_Return' column
            sentiment_signals: DataFrame with 'Sentiment_Score' column
            macro_tilts: Dict of factor tilts from macro regime
        
        Returns:
            DataFrame with ensemble scores
        """
        # Iterate all Kronos tickers; use sentiment where available, fall back to
        # Kronos-only for tickers FinBERT did not cover (M-4).
        sentiment_available = set(sentiment_signals.index)

        results = []

        for ticker in kronos_signals.index:
            # Kronos signal (normalize to -1 to 1)
            kronos_raw = kronos_signals.loc[ticker, 'Predicted_Return']
            kronos_norm = np.clip(kronos_raw / 0.5, -1, 1)  # Assuming 0.5 is typical max

            # Sentiment signal — use if available, else 0 (neutral)
            if ticker in sentiment_available:
                sentiment = sentiment_signals.loc[ticker, 'Sentiment_Score']
                sentiment_conf = sentiment_signals.loc[ticker, 'Confidence']
            else:
                sentiment = 0.0
                sentiment_conf = 0.0  # zero confidence → effectively ignored

            # Macro tilt (if available)
            macro = 0.0
            if macro_tilts and 'Growth' in macro_tilts:
                macro = macro_tilts.get('Growth', 0.0) * 0.3

            # Weighted combination
            ensemble_score = (
                self.current_weights['kronos'] * kronos_norm +
                self.current_weights['sentiment'] * sentiment * sentiment_conf +
                self.current_weights['macro'] * macro
            )

            # Confidence based on signal agreement
            signals = [kronos_norm, sentiment, macro]
            signal_variance = np.var(signals)
            confidence = 1 - min(signal_variance * 2, 1.0)  # Lower variance = higher confidence

            # Raw predicted return (denormalized)
            raw_return = ensemble_score * 0.5  # Scale back to return space

            results.append({
                'Ticker': ticker,
                'Kronos_Signal': kronos_norm,
                'Sentiment_Signal': sentiment,
                'Macro_Tilt': macro,
                'Ensemble_Score': ensemble_score,
                'Confidence': confidence,
                'Raw_Predicted_Return': raw_return,
                'Kronos_Weight': self.current_weights['kronos'],
                'Sentiment_Weight': self.current_weights['sentiment'],
                'Macro_Weight': self.current_weights['macro'],
                'Has_Sentiment': ticker in sentiment_available,
            })

        df = pd.DataFrame(results).set_index('Ticker')
        return df.sort_values('Ensemble_Score', ascending=False)


class BayesianModelAveraging:
    """
    Bayesian model averaging for more robust ensemble weighting.
    """
    
    def __init__(self, models: List[str], prior_weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.prior = prior_weights or {m: 1.0 / len(models) for m in models}
        self.likelihoods: Dict[str, List[float]] = {m: [] for m in models}
    
    def update_likelihood(self, model_name: str, prediction_error: float):
        """
        Update likelihood based on prediction error.
        Lower error = higher likelihood.
        """
        # Gaussian likelihood
        likelihood = np.exp(-0.5 * prediction_error ** 2)
        self.likelihoods[model_name].append(likelihood)
    
    def get_posterior_weights(self) -> Dict[str, float]:
        """
        Calculate posterior model weights using Bayes' theorem.
        """
        posterior = {}
        
        for model in self.models:
            if not self.likelihoods[model]:
                posterior[model] = self.prior[model]
            else:
                # Average likelihood
                avg_likelihood = np.mean(self.likelihoods[model][-20:])
                posterior[model] = self.prior[model] * avg_likelihood
        
        # Normalize
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: v / total for k, v in posterior.items()}
        
        return posterior


class SignalDecorrelator:
    """
    Decorrelates signals to maximize independent alpha sources.
    """
    
    def __init__(self):
        self.correlation_matrix: Optional[pd.DataFrame] = None
    
    def calculate_signal_correlations(self,
                                     kronos_signals: pd.Series,
                                     sentiment_signals: pd.Series,
                                     returns: pd.Series) -> pd.DataFrame:
        """
        Calculate correlation matrix of signals.
        """
        # Align all series
        common_idx = kronos_signals.index.intersection(sentiment_signals.index).intersection(returns.index)
        
        data = pd.DataFrame({
            'kronos': kronos_signals.loc[common_idx],
            'sentiment': sentiment_signals.loc[common_idx],
            'returns': returns.loc[common_idx]
        })
        
        self.correlation_matrix = data.corr()
        return self.correlation_matrix
    
    def orthogonalize_signals(self,
                           kronos: pd.Series,
                           sentiment: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Make signals orthogonal (remove common component).
        
        Returns independent signal components.
        """
        # Align
        common_idx = kronos.index.intersection(sentiment.index)
        k = kronos.loc[common_idx].values
        s = sentiment.loc[common_idx].values
        
        if len(k) < 10:
            return kronos, sentiment
        
        # Remove sentiment component from kronos
        # k_orthogonal = k - beta * s
        beta = np.cov(k, s)[0, 1] / (np.var(s) + 1e-6)
        k_ortho = k - beta * s
        
        kronos_ortho = pd.Series(k_ortho, index=common_idx)
        
        return kronos_ortho, sentiment.loc[common_idx]


# Convenience functions
def create_ensemble_signals(kronos_df: pd.DataFrame,
                          sentiment_df: pd.DataFrame,
                          macro_regime=None,
                          adapt_weights: bool = True) -> pd.DataFrame:
    """
    Quick function to create ensemble signals.
    """
    ensemble = AdaptiveEnsemble()
    
    if adapt_weights and hasattr(ensemble, 'adapt_weights'):
        ensemble.adapt_weights()
    
    macro_tilts = None
    if macro_regime:
        from signals.macro_regime import MacroRegimeDetector
        detector = MacroRegimeDetector()
        macro_tilts = detector.get_regime_factor_tilts(macro_regime)
    
    return ensemble.combine_signals(kronos_df, sentiment_df, macro_tilts)


def get_top_ensemble_picks(ensemble_df: pd.DataFrame, 
                          n_long: int = 20,
                          n_short: int = 20,
                          min_confidence: float = 0.3) -> Tuple[List[str], List[str]]:
    """
    Get top long and short picks from ensemble.
    
    Args:
        ensemble_df: DataFrame with 'Ensemble_Score' and 'Confidence'
        n_long: Number of long positions
        n_short: Number of short positions
        min_confidence: Minimum confidence threshold
    
    Returns:
        (long_tickers, short_tickers)
    """
    # Filter by confidence
    qualified = ensemble_df[ensemble_df['Confidence'] >= min_confidence]
    
    # Sort by ensemble score
    sorted_df = qualified.sort_values('Ensemble_Score', ascending=False)
    
    # Get top longs and shorts
    longs = sorted_df.head(n_long).index.tolist()
    shorts = sorted_df.tail(n_short).index.tolist()
    
    return longs, shorts

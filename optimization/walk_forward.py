"""
Walk-Forward Optimization (WFO) Framework
Prevents overfitting by optimizing on in-sample data and validating on out-of-sample periods.
Implements anchored and rolling window approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
import warnings


@dataclass
class WFOConfig:
    """Configuration for walk-forward optimization."""
    train_size: int  # Days in training window
    test_size: int   # Days in test/out-of-sample window
    step_size: int   # Days to step forward each iteration
    anchored: bool = False  # If True, train window grows; if False, rolling window
    min_train_size: int = 252  # Minimum days for training (for anchored)


@dataclass
class WFOResult:
    """Results from a single walk-forward step."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimal_params: Dict
    train_sharpe: float
    test_sharpe: float
    train_drawdown: float
    test_drawdown: float
    parameter_stability: float  # How much params changed from previous fold


class WalkForwardOptimizer:
    """
    Walk-forward optimization engine for strategy parameter tuning.
    
    Unlike simple backtest optimization, WFO:
    1. Splits data into multiple train/test periods
    2. Optimizes parameters on training data
    3. Evaluates on unseen test data
    4. Checks parameter stability across folds
    5. Reports realistic out-of-sample performance
    """
    
    def __init__(self, config: WFOConfig):
        self.config = config
        self.results: List[WFOResult] = []
        self.parameter_history: List[Dict] = []
        
    def generate_windows(self, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate train/test window pairs for walk-forward analysis.
        
        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        windows = []
        # In anchored mode the first test starts at start_date + train_size;
        # in rolling mode current tracks the start of each train window.
        current = start_date + timedelta(days=self.config.train_size) if self.config.anchored else start_date

        while True:
            if self.config.anchored:
                # Anchored: training window always begins at start_date and
                # grows to 'current' (the test boundary).
                train_start = start_date
                train_end = current
            else:
                # Rolling: fixed-size training window slides forward
                train_start = current
                train_end = current + timedelta(days=self.config.train_size)

            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_size)

            if test_end > end_date:
                break

            windows.append((train_start, train_end, test_start, test_end))
            current += timedelta(days=self.config.step_size)

        return windows
    
    def optimize(self, 
                 param_grid: Dict[str, List],
                 backtest_fn: Callable,
                 date_ranges: List[Tuple[datetime, datetime, datetime, datetime]],
                 metric: str = 'sharpe_ratio') -> Dict:
        """
        Run walk-forward optimization across parameter grid.
        
        Args:
            param_grid: Dict of parameter name -> list of values to try
            backtest_fn: Function that takes (train_start, train_end, **params) and returns metrics dict
            date_ranges: List of (train_start, train_end, test_start, test_end) tuples
            metric: Metric to optimize ('sharpe_ratio', 'returns', 'calmar')
        
        Returns:
            Dict with optimal parameters and WFO statistics
        """
        print(f"[WFO] Running walk-forward optimization with {len(date_ranges)} folds")
        print(f"[WFO] Parameter grid: {len(list(product(*param_grid.values())))} combinations")
        
        all_fold_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(date_ranges):
            print(f"\n[WFO] Fold {i+1}/{len(date_ranges)}: Train {train_start.date()} to {train_end.date()}")
            
            # Grid search on training data
            best_params = None
            best_train_metric = -np.inf
            train_results = {}
            
            param_combinations = list(product(*param_grid.values()))
            param_names = list(param_grid.keys())
            
            for combo in param_combinations:
                params = dict(zip(param_names, combo))
                
                try:
                    train_metrics = backtest_fn(train_start, train_end, **params)
                    train_value = train_metrics.get(metric, -np.inf)
                    
                    if train_value > best_train_metric:
                        best_train_metric = train_value
                        best_params = params
                        train_results = train_metrics
                        
                except Exception as e:
                    warnings.warn(f"Backtest failed for params {params}: {e}")
                    continue
            
            if best_params is None:
                print(f"[WFO] Warning: No valid parameters found for fold {i+1}")
                continue
            
            # Evaluate on test data
            print(f"[WFO] Testing optimal params on {test_start.date()} to {test_end.date()}...")
            try:
                test_metrics = backtest_fn(test_start, test_end, **best_params)
                test_value = test_metrics.get(metric, -np.inf)
            except Exception as e:
                print(f"[WFO] Test backtest failed: {e}")
                test_metrics = {'sharpe_ratio': -np.inf, 'max_drawdown': 1.0}
                test_value = -np.inf
            
            # Calculate parameter stability
            param_stability = self._calculate_parameter_stability(best_params, i)
            
            result = WFOResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                optimal_params=best_params,
                train_sharpe=train_results.get('sharpe_ratio', 0),
                test_sharpe=test_metrics.get('sharpe_ratio', 0),
                train_drawdown=train_results.get('max_drawdown', 1.0),
                test_drawdown=test_metrics.get('max_drawdown', 1.0),
                parameter_stability=param_stability
            )
            
            all_fold_results.append(result)
            self.parameter_history.append(best_params)
            
            print(f"[WFO] Train Sharpe: {result.train_sharpe:.3f}, Test Sharpe: {result.test_sharpe:.3f}")
            print(f"[WFO] Parameter stability: {param_stability:.2%}")
        
        self.results = all_fold_results
        
        # Aggregate results
        return self._aggregate_results()
    
    def _calculate_parameter_stability(self, current_params: Dict, fold_idx: int) -> float:
        """Calculate how much parameters changed from previous fold."""
        if fold_idx == 0 or not self.parameter_history:
            return 1.0  # Perfect stability for first fold
        
        prev_params = self.parameter_history[-1]
        
        # Calculate normalized parameter changes
        changes = []
        for key in current_params:
            if key in prev_params:
                prev_val = prev_params[key]
                curr_val = current_params[key]
                
                # Normalize by parameter range (if numeric)
                if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                    if abs(prev_val) > 0:
                        rel_change = abs(curr_val - prev_val) / abs(prev_val)
                        changes.append(1 - min(rel_change, 1.0))  # Convert to stability score
                    else:
                        changes.append(1.0 if curr_val == prev_val else 0.0)
                else:
                    changes.append(1.0 if curr_val == prev_val else 0.0)
        
        return np.mean(changes) if changes else 0.0
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results across all folds."""
        if not self.results:
            return {'error': 'No valid results'}
        
        train_sharpes = [r.train_sharpe for r in self.results]
        test_sharpes = [r.test_sharpe for r in self.results]
        stabilities = [r.parameter_stability for r in self.results]
        
        # Find most stable and performant parameters
        param_scores = {}
        for result in self.results:
            params_key = tuple(sorted(result.optimal_params.items()))
            if params_key not in param_scores:
                param_scores[params_key] = {'count': 0, 'avg_test_sharpe': 0, 'stability': 0}
            param_scores[params_key]['count'] += 1
            param_scores[params_key]['avg_test_sharpe'] += result.test_sharpe
            param_scores[params_key]['stability'] += result.parameter_stability
        
        # Average the scores
        for key in param_scores:
            count = param_scores[key]['count']
            param_scores[key]['avg_test_sharpe'] /= count
            param_scores[key]['stability'] /= count
        
        # Select best parameters (highest frequency + good performance)
        best_key = max(param_scores.keys(), 
                        key=lambda k: (param_scores[k]['count'], param_scores[k]['avg_test_sharpe']))
        optimal_params = dict(best_key)
        
        return {
            'optimal_params': optimal_params,
            'n_folds': len(self.results),
            'train_sharpe_mean': np.mean(train_sharpes),
            'train_sharpe_std': np.std(train_sharpes),
            'test_sharpe_mean': np.mean(test_sharpes),
            'test_sharpe_std': np.std(test_sharpes),
            'sharpe_degradation': np.mean(train_sharpes) - np.mean(test_sharpes),
            'parameter_stability_mean': np.mean(stabilities),
            'parameter_consistency': max(p['count'] for p in param_scores.values()) / len(self.results),
            'fold_results': [
                {
                    'train_period': f"{r.train_start.date()} to {r.train_end.date()}",
                    'test_period': f"{r.test_start.date()} to {r.test_end.date()}",
                    'train_sharpe': r.train_sharpe,
                    'test_sharpe': r.test_sharpe,
                    'params': r.optimal_params
                }
                for r in self.results
            ]
        }
    
    def is_overfit(self, degradation_threshold: float = 0.5, stability_threshold: float = 0.5) -> Tuple[bool, str]:
        """
        Check if optimization results indicate overfitting.
        
        Returns:
            (is_overfit, diagnosis_message)
        """
        if not self.results:
            return True, "No results available"
        
        train_sharpes = [r.train_sharpe for r in self.results]
        test_sharpes = [r.test_sharpe for r in self.results]
        stabilities = [r.parameter_stability for r in self.results]
        
        degradation = np.mean(train_sharpes) - np.mean(test_sharpes)
        avg_stability = np.mean(stabilities)
        
        issues = []
        
        if degradation > degradation_threshold:
            issues.append(f"High in-sample/out-of-sample degradation ({degradation:.2f})")
        
        if avg_stability < stability_threshold:
            issues.append(f"Unstable parameters across folds ({avg_stability:.1%} stability)")
        
        if np.mean(test_sharpes) < 0:
            issues.append("Negative out-of-sample Sharpe ratio")
        
        if np.std(test_sharpes) > 1.0:
            issues.append("High variance in test performance")
        
        if issues:
            return True, "Overfitting detected: " + "; ".join(issues)
        
        return False, "No significant overfitting detected"


class TimeSeriesCrossValidator:
    """
    Time-series aware cross-validation for financial data.
    Prevents look-ahead bias in model validation.
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0):
        self.n_splits = n_splits
        self.gap = gap  # Days between train and test to prevent leakage
    
    def split(self, dates: pd.DatetimeIndex) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for time-series CV.
        
        Returns list of (train_indices, test_indices) tuples.
        """
        n_samples = len(dates)
        fold_size = n_samples // (self.n_splits + 1)
        
        indices = []
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end + self.gap
            test_end = test_start + fold_size
            
            if test_end > n_samples:
                break
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            indices.append((train_idx, test_idx))
        
        return indices


class PurgedKFold:
    """
    Purged K-Fold cross-validation for financial time series.
    Removes overlapping observations between train and test sets.
    Based on Lopez de Prado's advances in financial machine learning.
    """
    
    def __init__(self, n_splits: int = 5, purge_gap: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.purge_gap = purge_gap  # Days to purge around test set
        self.embargo_pct = embargo_pct  # % of test to embargo from train
    
    def split(self, X: pd.DataFrame, labels: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/test splits.
        
        Args:
            X: Feature dataframe with DatetimeIndex
            labels: Optional labels (for embargo calculation)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")
        
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        indices = []
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = test_start + fold_size
            
            # Embargo: remove last % of test set from train
            embargo_start = int(test_end - (test_end - test_start) * self.embargo_pct)
            
            # Purge: exclude observations near test boundaries
            train_idx_1 = np.arange(0, max(0, test_start - self.purge_gap))
            train_idx_2 = np.arange(embargo_start + self.purge_gap, n_samples)
            train_idx = np.concatenate([train_idx_1, train_idx_2])
            
            test_idx = np.arange(test_start, test_end)
            
            indices.append((train_idx, test_idx))
        
        return indices


# Example usage and helper functions
def example_parameter_grid():
    """Example parameter grid for Kronos-based strategy."""
    return {
        'risk_aversion': [0.5, 1.0, 2.0, 5.0],
        'l2_penalty': [0.1, 0.5, 1.0, 2.0],
        'max_sector_exposure': [0.15, 0.20, 0.25, 0.30],
        'lookback_days': [200, 400, 600],
        'prediction_horizon': [3, 5, 10, 20]
    }


def calculate_sharpe_from_returns(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 30 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculate maximum drawdown from equity curve."""
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

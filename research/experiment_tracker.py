"""
MLflow Experiment Tracking Module
Tracks all model training runs, hyperparameters, metrics, and artifacts.
Ensures reproducibility and enables model comparison.
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import warnings

# Try to import MLflow, but make it optional
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed. Using local experiment tracking only.")


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    experiment_name: str
    run_name: Optional[str] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.run_name is None:
            self.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@dataclass
class ModelMetrics:
    """Metrics from a model run."""
    ic_1d: float
    ic_5d: float
    ic_20d: float
    sharpe_ratio: float
    max_drawdown: float
    annualized_return: float
    volatility: float
    win_rate: float
    calmar_ratio: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class ExperimentTracker:
    """
    Tracks machine learning experiments.
    
    Supports both MLflow (if available) and local JSON-based tracking.
    """
    
    def __init__(self,
                 experiment_name: str = "kronos_alpha",
                 tracking_uri: Optional[str] = None,
                 local_dir: str = "./experiments"):
        self.experiment_name = experiment_name
        self.local_dir = local_dir
        self.use_mlflow = MLFLOW_AVAILABLE
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Initialize MLflow if available
        if self.use_mlflow:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                mlflow.set_tracking_uri(f"file://{os.path.abspath(local_dir)}")
            
            mlflow.set_experiment(experiment_name)
            print(f"[Experiment] MLflow tracking enabled: {experiment_name}")
        else:
            print(f"[Experiment] Using local tracking: {local_dir}")
        
        self.current_run_id: Optional[str] = None
        self.run_data: Dict[str, Any] = {}
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new experiment run.
        
        Returns:
            Run ID
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = run_name or f"run_{timestamp}"
        
        if self.use_mlflow:
            run = mlflow.start_run(run_name=run_name)
            self.current_run_id = run.info.run_id
            
            if tags:
                mlflow.set_tags(tags)
            
            # Set default tags
            mlflow.set_tag("start_time", datetime.now().isoformat())
            mlflow.set_tag("experiment_name", self.experiment_name)
        else:
            # Local tracking
            self.current_run_id = f"{self.experiment_name}_{run_name}"
            self.run_data = {
                'run_id': self.current_run_id,
                'run_name': run_name,
                'experiment_name': self.experiment_name,
                'start_time': datetime.now().isoformat(),
                'tags': tags or {},
                'params': {},
                'metrics': {},
                'artifacts': []
            }
        
        print(f"[Experiment] Started run: {run_name} (ID: {self.current_run_id})")
        return self.current_run_id
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if not self.current_run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        # Convert non-primitive types to strings
        params = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
                 for k, v in params.items()}
        
        if self.use_mlflow:
            mlflow.log_params(params)
        else:
            self.run_data['params'].update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not self.current_run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        else:
            if step is not None:
                # Store as time series
                for key, value in metrics.items():
                    if key not in self.run_data['metrics']:
                        self.run_data['metrics'][key] = []
                    self.run_data['metrics'][key].append({'step': step, 'value': value})
            else:
                self.run_data['metrics'].update(metrics)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact."""
        if not self.current_run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        if not os.path.exists(local_path):
            warnings.warn(f"Artifact not found: {local_path}")
            return
        
        if self.use_mlflow:
            mlflow.log_artifact(local_path, artifact_path)
        else:
            # Copy to local experiment directory
            import shutil
            
            run_dir = os.path.join(self.local_dir, self.current_run_id)
            os.makedirs(run_dir, exist_ok=True)
            
            dest = os.path.join(run_dir, os.path.basename(local_path))
            shutil.copy2(local_path, dest)
            
            self.run_data['artifacts'].append({
                'local_path': local_path,
                'stored_path': dest,
                'artifact_path': artifact_path
            })
    
    def log_model(self, model, model_name: str = "model"):
        """Log a trained model."""
        if not self.current_run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        if self.use_mlflow:
            # Try to infer model flavor
            try:
                import sklearn
                if isinstance(model, sklearn.base.BaseEstimator):
                    mlflow.sklearn.log_model(model, model_name)
                    return
            except ImportError:
                pass
            
            # Generic pickle logging
            mlflow.pyfunc.log_model(model_name, python_model=model)
        else:
            # Save locally
            import pickle
            
            run_dir = os.path.join(self.local_dir, self.current_run_id)
            os.makedirs(run_dir, exist_ok=True)
            
            model_path = os.path.join(run_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.run_data['artifacts'].append({
                'type': 'model',
                'model_name': model_name,
                'path': model_path
            })
    
    def log_signal_performance(self,
                              signal_name: str,
                              predictions: pd.Series,
                              returns: pd.Series,
                              date: datetime):
        """
        Log signal performance metrics.
        """
        from scipy.stats import spearmanr
        
        # Calculate IC
        common_idx = predictions.index.intersection(returns.index)
        if len(common_idx) < 10:
            return
        
        ic, pval = spearmanr(predictions.loc[common_idx], returns.loc[common_idx])
        
        # Log metrics
        metrics = {
            f'{signal_name}_ic': ic,
            f'{signal_name}_ic_pval': pval,
            f'{signal_name}_sample_size': len(common_idx)
        }
        
        self.log_metrics(metrics)
    
    def log_backtest_results(self,
                            portfolio_history: Dict[str, pd.DataFrame],
                            tca_metrics: Optional[List] = None):
        """
        Log backtest results.
        """
        # Calculate aggregate metrics
        if not portfolio_history:
            return
        
        # Extract portfolio values
        dates = sorted(portfolio_history.keys())
        
        # Calculate returns
        values = []
        for date in dates:
            port = portfolio_history[date]
            if not port.empty and 'Weight' in port.columns:
                gross_exposure = port['Weight'].abs().sum()
                values.append(gross_exposure)
        
        if len(values) < 2:
            return
        
        returns = pd.Series(values).pct_change().dropna()
        
        if len(returns) > 0:
            metrics = {
                'backtest_sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'backtest_volatility': returns.std() * np.sqrt(252),
                'backtest_cumulative_return': (returns + 1).prod() - 1,
                'backtest_max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min()
            }
            
            self.log_metrics(metrics)
        
        # Log TCA metrics if available
        if tca_metrics:
            avg_impact = np.mean([m.market_impact_bps for m in tca_metrics]) if tca_metrics else 0
            avg_costs = np.mean([m.transaction_costs_bps for m in tca_metrics]) if tca_metrics else 0
            
            self.log_metrics({
                'tca_avg_market_impact_bps': avg_impact,
                'tca_avg_total_costs_bps': avg_costs
            })
    
    def end_run(self):
        """End the current run."""
        if not self.current_run_id:
            return
        
        if self.use_mlflow:
            mlflow.end_run()
        else:
            # Save local run data
            run_file = os.path.join(self.local_dir, f"{self.current_run_id}.json")
            
            with open(run_file, 'w') as f:
                json.dump(self.run_data, f, indent=2, default=str)
            
            print(f"[Experiment] Saved run data to: {run_file}")
        
        print(f"[Experiment] Ended run: {self.current_run_id}")
        self.current_run_id = None
        self.run_data = {}
    
    def get_run_history(self) -> pd.DataFrame:
        """
        Get history of all runs.
        """
        if self.use_mlflow:
            # Query MLflow
            from mlflow.tracking import MlflowClient
            
            client = MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            
            if experiment:
                runs = client.search_runs(experiment.experiment_id)
                
                data = []
                for run in runs:
                    data.append({
                        'run_id': run.info.run_id,
                        'run_name': run.data.tags.get('mlflow.runName', 'unknown'),
                        'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                        'status': run.info.status,
                        'metrics': run.data.metrics,
                        'params': run.data.params
                    })
                
                return pd.DataFrame(data)
        else:
            # Load from local files
            run_files = [f for f in os.listdir(self.local_dir) if f.endswith('.json')]
            
            data = []
            for run_file in run_files:
                with open(os.path.join(self.local_dir, run_file), 'r') as f:
                    run_data = json.load(f)
                    data.append({
                        'run_id': run_data.get('run_id'),
                        'run_name': run_data.get('run_name'),
                        'start_time': run_data.get('start_time'),
                        'metrics': run_data.get('metrics', {}),
                        'params': run_data.get('params', {})
                    })
            
            return pd.DataFrame(data)
        
        return pd.DataFrame()
    
    def compare_runs(self, run_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple runs.
        """
        history = self.get_run_history()
        
        if run_ids:
            history = history[history['run_id'].isin(run_ids)]
        
        if history.empty:
            return pd.DataFrame()
        
        # Extract key metrics
        comparison = []
        for _, row in history.iterrows():
            metrics = row.get('metrics', {})
            
            comparison.append({
                'run_id': row['run_id'],
                'run_name': row['run_name'],
                'sharpe': metrics.get('backtest_sharpe', 0) if isinstance(metrics, dict) else 0,
                'volatility': metrics.get('backtest_volatility', 0) if isinstance(metrics, dict) else 0,
                'max_drawdown': metrics.get('backtest_max_drawdown', 0) if isinstance(metrics, dict) else 0,
                'ic': metrics.get('kronos_ic', 0) if isinstance(metrics, dict) else 0
            })
        
        return pd.DataFrame(comparison)
    
    def get_best_run(self, metric: str = 'backtest_sharpe') -> Optional[str]:
        """
        Get the best run by a specific metric.
        """
        history = self.get_run_history()
        
        if history.empty:
            return None
        
        best_score = -np.inf
        best_run = None
        
        for _, row in history.iterrows():
            metrics = row.get('metrics', {})
            if isinstance(metrics, dict) and metric in metrics:
                score = metrics[metric]
                if score > best_score:
                    best_score = score
                    best_run = row['run_id']
        
        return best_run
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current_run_id:
            self.end_run()


# Convenience functions
def log_experiment(experiment_name: str,
                  params: Dict[str, Any],
                  metrics: Dict[str, float],
                  artifacts: Optional[List[str]] = None) -> str:
    """
    Quick function to log a single experiment run.
    """
    tracker = ExperimentTracker(experiment_name=experiment_name)
    
    run_id = tracker.start_run()
    
    try:
        tracker.log_params(params)
        tracker.log_metrics(metrics)
        
        if artifacts:
            for artifact in artifacts:
                tracker.log_artifact(artifact)
    finally:
        tracker.end_run()
    
    return run_id


def compare_experiments(experiment_name: str) -> pd.DataFrame:
    """
    Compare all runs in an experiment.
    """
    tracker = ExperimentTracker(experiment_name=experiment_name)
    return tracker.compare_runs()

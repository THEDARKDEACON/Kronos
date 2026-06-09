"""
Kronos Health Check & Monitoring
Validates all pipeline components are functioning
"""

import os
import sys
import json
import time
import psutil
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HealthChecker:
    """System health monitoring"""
    
    def __init__(self):
        self.checks = {}
        self.status = "unknown"
    
    def run_all_checks(self) -> Dict:
        """Run all health checks"""
        self.checks = {
            'timestamp': datetime.now().isoformat(),
            'system': self._check_system(),
            'data': self._check_data_sources(),
            'models': self._check_models(),
            'broker': self._check_broker(),
            'database': self._check_database()
        }
        
        # Determine overall status
        all_healthy = all(
            check.get('healthy', False) 
            for check in self.checks.values() 
            if isinstance(check, dict)
        )
        
        self.status = "healthy" if all_healthy else "unhealthy"
        self.checks['overall_status'] = self.status
        
        return self.checks
    
    def _check_system(self) -> Dict:
        """Check system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            healthy = (
                cpu_percent < 90 and 
                memory.percent < 90 and 
                disk.percent < 90
            )
            
            return {
                'healthy': healthy,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent,
                'uptime': time.time() - psutil.boot_time()
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def _check_data_sources(self) -> Dict:
        """Check data sources connectivity"""
        results = {}
        
        # Check yFinance (via import test)
        try:
            import yfinance as yf
            # Try fetching a single ticker
            data = yf.Ticker("SPY").history(period="1d")
            results['yfinance'] = {
                'healthy': len(data) > 0,
                'last_bar': str(data.index[-1]) if len(data) > 0 else None
            }
        except Exception as e:
            results['yfinance'] = {'healthy': False, 'error': str(e)}
        
        # Check FRED (if API key available)
        fred_key = os.getenv('FRED_API_KEY')
        if fred_key:
            try:
                from fredapi import Fred
                fred = Fred(api_key=fred_key)
                data = fred.get_series('DGS10', limit=1)
                results['fred'] = {
                    'healthy': len(data) > 0,
                    'last_value': float(data.iloc[-1]) if len(data) > 0 else None
                }
            except Exception as e:
                results['fred'] = {'healthy': False, 'error': str(e)}
        else:
            results['fred'] = {'healthy': True, 'status': 'no_api_key'}
        
        # Check news API
        news_key = os.getenv('NEWSAPI_KEY')
        if news_key:
            try:
                import requests
                response = requests.get(
                    'https://newsapi.org/v2/everything',
                    params={
                        'q': 'AAPL',
                        'pageSize': 1,
                        'apiKey': news_key
                    },
                    timeout=10
                )
                results['newsapi'] = {
                    'healthy': response.status_code == 200,
                    'status_code': response.status_code
                }
            except Exception as e:
                results['newsapi'] = {'healthy': False, 'error': str(e)}
        else:
            results['newsapi'] = {'healthy': True, 'status': 'no_api_key'}
        
        # Overall data health
        all_healthy = all(r.get('healthy', False) for r in results.values())
        results['healthy'] = all_healthy
        
        return results
    
    def _check_models(self) -> Dict:
        """Check ML models are loadable"""
        results = {}
        
        # Check Kronos model
        try:
            from strategy.kronos_alpha import KronosAlphaGenerator
            model = KronosAlphaGenerator(model_size='small')
            results['kronos'] = {
                'healthy': model.predictor is not None,
                'model_size': 'small',
            }
        except Exception as e:
            results['kronos'] = {'healthy': False, 'error': str(e)}
        
        # Check FinBERT
        try:
            from signals.finbert_sentiment import FinBERTSentimentAnalyzer
            analyzer = FinBERTSentimentAnalyzer()
            results['finbert'] = {
                'healthy': analyzer.model is not None,
                'model_name': 'ProsusAI/finbert'
            }
        except Exception as e:
            results['finbert'] = {'healthy': False, 'error': str(e)}
        
        # Overall model health
        all_healthy = all(r.get('healthy', False) for r in results.values())
        results['healthy'] = all_healthy
        
        return results
    
    def _check_broker(self) -> Dict:
        """Check broker connectivity"""
        mode = os.getenv('KRONOS_MODE', 'dry').lower()

        if mode in ('dry', 'backtest'):
            return {'healthy': True, 'mode': mode, 'note': 'No broker needed'}

        try:
            from execution.broker_connector import create_broker

            broker = create_broker('alpaca', paper=(mode == 'paper'))
            if not broker.connect():
                return {'healthy': False, 'mode': mode, 'error': 'Broker connection failed'}

            account = broker.get_account_info()
            broker.disconnect()
            return {
                'healthy': True,
                'mode': mode,
                'buying_power': float(account.get('buying_power', 0)),
                'cash': float(account.get('cash', 0)),
            }
        except Exception as e:
            return {
                'healthy': False,
                'mode': mode,
                'error': str(e),
            }
    
    def _check_database(self) -> Dict:
        """Check cache/database connectivity"""
        try:
            import pandas as pd
            from pathlib import Path
            
            # Check cache directory is writable
            cache_dir = Path('./data/cache')
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write
            test_file = cache_dir / 'health_check_test.parquet'
            pd.DataFrame({'test': [1, 2, 3]}).to_parquet(test_file)
            test_file.unlink()
            
            # Check existing data files
            data_files = list(cache_dir.glob('*.parquet'))
            
            return {
                'healthy': True,
                'cache_dir': str(cache_dir),
                'cached_files': len(data_files),
                'latest_file': str(max(data_files, key=lambda p: p.stat().st_mtime)) if data_files else None
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def get_metrics(self) -> Dict:
        """Get system metrics for Prometheus/Grafana"""
        if not self.checks:
            self.run_all_checks()
        
        return {
            'kronos_health_status': 1 if self.status == 'healthy' else 0,
            'kronos_cpu_usage': self.checks.get('system', {}).get('cpu_percent', 0),
            'kronos_memory_usage': self.checks.get('system', {}).get('memory_percent', 0),
            'kronos_disk_usage': self.checks.get('system', {}).get('disk_percent', 0),
            'kronos_data_sources_up': sum(
                1 for v in self.checks.get('data', {}).values() 
                if isinstance(v, dict) and v.get('healthy', False)
            ),
            'kronos_models_loaded': sum(
                1 for v in self.checks.get('models', {}).values()
                if isinstance(v, dict) and v.get('healthy', False)
            )
        }


def main():
    """Run health check"""
    checker = HealthChecker()
    results = checker.run_all_checks()
    
    print(json.dumps(results, indent=2, default=str))
    
    # Exit with error code if unhealthy
    if results['overall_status'] != 'healthy':
        print("\n⚠️  System is unhealthy!")
        sys.exit(1)
    else:
        print("\n✅ System is healthy!")
        sys.exit(0)


if __name__ == '__main__':
    main()

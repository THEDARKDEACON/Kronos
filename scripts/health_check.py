"""
Kronos Health Check & Monitoring
Validates pipeline components. Use --light to skip heavy model loads.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HealthChecker:
    """System health monitoring."""

    def __init__(self, light: bool = False):
        self.light = light
        self.checks = {}
        self.status = "unknown"

    def run_all_checks(self) -> Dict:
        """Run all health checks."""
        self.checks = {
            "timestamp": datetime.now().isoformat(),
            "mode": "light" if self.light else "full",
            "system": self._check_system(),
            "data": self._check_data_sources(),
            "models": self._check_models(),
            "broker": self._check_broker(),
            "database": self._check_database(),
        }

        all_healthy = all(
            check.get("healthy", False)
            for check in self.checks.values()
            if isinstance(check, dict)
        )

        self.status = "healthy" if all_healthy else "unhealthy"
        self.checks["overall_status"] = self.status
        return self.checks

    def _check_system(self) -> Dict:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            healthy = (
                cpu_percent < 90
                and memory.percent < 90
                and disk.percent < (98 if self.light else 90)
            )

            return {
                "healthy": healthy,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "uptime": time.time() - psutil.boot_time(),
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def _check_data_sources(self) -> Dict:
        results = {}

        try:
            import yfinance as yf

            data = yf.Ticker("SPY").history(period="1d")
            results["yfinance"] = {
                "healthy": len(data) > 0,
                "last_bar": str(data.index[-1]) if len(data) > 0 else None,
            }
        except Exception as e:
            results["yfinance"] = {"healthy": False, "error": str(e)}

        fred_key = os.getenv("FRED_API_KEY")
        if fred_key:
            try:
                from fredapi import Fred

                fred = Fred(api_key=fred_key)
                data = fred.get_series("DGS10", limit=1)
                results["fred"] = {
                    "healthy": len(data) > 0,
                    "last_value": float(data.iloc[-1]) if len(data) > 0 else None,
                }
            except Exception as e:
                results["fred"] = {"healthy": False, "error": str(e)}
        else:
            results["fred"] = {"healthy": True, "status": "no_api_key"}

        news_key = os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")
        if news_key:
            try:
                import requests

                response = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={"q": "AAPL", "pageSize": 1, "apiKey": news_key},
                    timeout=10,
                )
                results["newsapi"] = {
                    "healthy": response.status_code == 200,
                    "status_code": response.status_code,
                }
            except Exception as e:
                results["newsapi"] = {"healthy": False, "error": str(e)}
        else:
            results["newsapi"] = {"healthy": True, "status": "no_api_key"}

        results["healthy"] = all(r.get("healthy", False) for r in results.values())
        return results

    def _check_models(self) -> Dict:
        if self.light:
            return {
                "healthy": True,
                "mode": "light",
                "note": "Model load skipped — use full check before trading",
            }

        results = {}

        try:
            from strategy.kronos_alpha import KronosAlphaGenerator

            model = KronosAlphaGenerator(model_size="small")
            results["kronos"] = {
                "healthy": model.predictor is not None,
                "model_size": "small",
            }
        except Exception as e:
            results["kronos"] = {"healthy": False, "error": str(e)}

        try:
            from signals.finbert_sentiment import FinBERTSentimentAnalyzer

            analyzer = FinBERTSentimentAnalyzer()
            onnx_ok = getattr(analyzer, "session", None) is not None or getattr(
                analyzer, "model", None
            ) is not None
            results["finbert"] = {
                "healthy": onnx_ok,
                "model_name": "ProsusAI/finbert",
            }
        except Exception as e:
            results["finbert"] = {"healthy": False, "error": str(e)}

        results["healthy"] = all(r.get("healthy", False) for r in results.values())
        return results

    def _check_broker(self) -> Dict:
        mode = os.getenv("KRONOS_MODE", "dry").lower()
        has_keys = bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))

        if mode in ("dry", "backtest") and not has_keys:
            return {"healthy": True, "mode": mode, "note": "No broker keys configured"}

        try:
            from execution.broker_connector import create_broker

            paper = mode != "live"
            broker = create_broker("alpaca", paper=paper)
            if not broker.connect():
                return {"healthy": False, "mode": mode, "error": "Broker connection failed"}

            account = broker.get_account_info()
            broker.disconnect()
            return {
                "healthy": True,
                "mode": mode,
                "paper": paper,
                "buying_power": float(account.get("buying_power", 0)),
                "cash": float(account.get("cash", 0)),
            }
        except Exception as e:
            return {"healthy": False, "mode": mode, "error": str(e)}

    def _check_database(self) -> Dict:
        try:
            import pandas as pd

            cache_dir = Path("./data/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)

            test_file = cache_dir / "health_check_test.parquet"
            pd.DataFrame({"test": [1, 2, 3]}).to_parquet(test_file)
            test_file.unlink()

            data_files = list(cache_dir.glob("*.parquet"))

            return {
                "healthy": True,
                "cache_dir": str(cache_dir),
                "cached_files": len(data_files),
                "latest_file": str(max(data_files, key=lambda p: p.stat().st_mtime))
                if data_files
                else None,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def get_metrics(self) -> Dict:
        if not self.checks:
            self.run_all_checks()

        return {
            "kronos_health_status": 1 if self.status == "healthy" else 0,
            "kronos_cpu_usage": self.checks.get("system", {}).get("cpu_percent", 0),
            "kronos_memory_usage": self.checks.get("system", {}).get("memory_percent", 0),
            "kronos_disk_usage": self.checks.get("system", {}).get("disk_percent", 0),
            "kronos_data_sources_up": sum(
                1
                for v in self.checks.get("data", {}).values()
                if isinstance(v, dict) and v.get("healthy", False)
            ),
            "kronos_models_loaded": 0
            if self.light
            else sum(
                1
                for v in self.checks.get("models", {}).values()
                if isinstance(v, dict) and v.get("healthy", False)
            ),
        }


def main():
    parser = argparse.ArgumentParser(description="Kronos system health check")
    parser.add_argument(
        "--light",
        action="store_true",
        help="Skip Kronos/FinBERT model load (fast CI / dev check)",
    )
    args = parser.parse_args()

    checker = HealthChecker(light=args.light)
    results = checker.run_all_checks()

    print(json.dumps(results, indent=2, default=str))

    if results["overall_status"] != "healthy":
        print("\n⚠️  System is unhealthy!")
        sys.exit(1)
    print("\n✅ System is healthy!")
    sys.exit(0)


if __name__ == "__main__":
    main()

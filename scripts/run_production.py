"""
Kronos production runner — scheduled pipeline with optional broker execution.

Modes (KRONOS_MODE env):
  dry   — run pipeline, save state, no orders (default)
  paper — Alpaca paper account
  live  — Alpaca live account
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import schedule
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LOG_DIR = ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / 'data' / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'production.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger('kronos.production')

from pipeline.core import portfolio_to_weights, run_pipeline
from execution.broker_connector import ExecutionEngine, create_broker


def _send_alert(message: str) -> None:
    webhook = os.getenv('SLACK_WEBHOOK_URL')
    if webhook:
        import requests
        try:
            requests.post(webhook, json={'text': f'Kronos Alert: {message}'}, timeout=10)
        except Exception as exc:
            logger.error("Failed to send Slack alert: %s", exc)
    logger.warning("ALERT: %s", message)


def _append_trades(execution_report) -> None:
    if not execution_report or not execution_report.get("fills"):
        return
    import pandas as pd

    fills_df = pd.DataFrame(execution_report["fills"])
    trades_path = CACHE_DIR / "trades.parquet"
    if trades_path.exists():
        existing = pd.read_parquet(trades_path)
        fills_df = pd.concat([existing, fills_df], ignore_index=True)
    fills_df.to_parquet(trades_path)


def _save_state(result, execution_report=None) -> None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result.portfolio.to_parquet(CACHE_DIR / f'positions_{timestamp}.parquet')
    result.signals.to_parquet(CACHE_DIR / f'signals_{timestamp}.parquet')
    result.portfolio.to_parquet(CACHE_DIR / 'latest_positions.parquet')
    result.signals.to_parquet(CACHE_DIR / 'latest_signals.parquet')

    gross = result.portfolio['Weight'].abs().sum()
    snapshot = {
        'timestamp': timestamp,
        'num_positions': len(result.portfolio),
        'gross_exposure': float(gross),
        'net_exposure': float(result.portfolio['Weight'].sum()),
        'macro_regime': result.macro_regime,
        'should_halt': result.should_halt,
        'halt_reason': result.halt_reason,
        'execution': execution_report,
    }
    with open(CACHE_DIR / f'snapshot_{timestamp}.json', 'w') as f:
        json.dump(snapshot, f, indent=2)
    with open(CACHE_DIR / 'latest_snapshot.json', 'w') as f:
        json.dump(snapshot, f, indent=2)

    if execution_report:
        _append_trades(execution_report)

    logger.info("Saved state at %s", timestamp)


def _execute_portfolio(result, mode: str):
    if mode == 'dry':
        logger.info("Dry run — skipping broker execution")
        return None

    if result.should_halt:
        logger.warning("Execution skipped: %s", result.halt_reason)
        _send_alert(f"Execution skipped: {result.halt_reason}")
        return None

    broker = create_broker('alpaca', paper=(mode != 'live'))
    engine = ExecutionEngine(broker)
    target_weights = portfolio_to_weights(result.portfolio)
    report = engine.rebalance_portfolio(target_weights)
    if not report.get('success'):
        _send_alert(f"Execution failed: {report.get('error', report)}")
    return report


def run_once() -> None:
    mode = os.getenv('KRONOS_MODE', 'dry').lower()
    use_ensemble = os.getenv('USE_ENSEMBLE', 'true').lower() in ('1', 'true', 'yes')
    use_kelly = os.getenv('USE_KELLY', 'true').lower() in ('1', 'true', 'yes')
    universe_limit = os.getenv('UNIVERSE_LIMIT')
    universe_limit = int(universe_limit) if universe_limit else None

    logger.info("Starting pipeline run (mode=%s)", mode)

    try:
        result = run_pipeline(
            use_ensemble=use_ensemble,
            use_kelly=use_kelly,
            universe_limit=universe_limit,
            verbose=False,
        )
        logger.info(
            "Pipeline complete: %d positions, gross=%.2f%%",
            len(result.portfolio),
            result.portfolio['Weight'].abs().sum() * 100,
        )

        execution_report = _execute_portfolio(result, mode)
        _save_state(result, execution_report)

        if result.should_halt:
            _send_alert(f"Trading halt: {result.halt_reason}")

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        _send_alert(f"Pipeline failed: {exc}")
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Kronos production runner")
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()

    if args.once:
        run_once()
        return

    freq = os.getenv('REBALANCE_FREQUENCY', 'weekly').lower()
    run_time = os.getenv('REBALANCE_TIME', '09:30')

    if freq == 'daily':
        schedule.every().day.at(run_time).do(run_once)
    elif freq == 'monthly':
        schedule.every().month.at(run_time).do(run_once)
    else:
        schedule.every().monday.at(run_time).do(run_once)

    logger.info("Scheduler started: %s at %s", freq, run_time)
    run_once()

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == '__main__':
    main()

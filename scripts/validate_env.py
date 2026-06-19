#!/usr/bin/env python3
"""Validate Kronos environment variables and external service connectivity."""

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def _status(ok: bool, detail: str) -> dict:
    return {"ok": ok, "detail": detail}


def check_alpaca() -> dict:
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return _status(True, "not configured (optional for dry mode)")

    try:
        from execution.broker_connector import create_broker

        broker = create_broker("alpaca", paper=True)
        if not broker.connect():
            return _status(False, "connection failed")
        acct = broker.get_account_info()
        broker.disconnect()
        equity = float(acct.get("equity", 0))
        return _status(True, f"paper account active, equity=${equity:,.2f}")
    except Exception as exc:
        return _status(False, str(exc))


def check_newsapi() -> dict:
    key = os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")
    if not key:
        return _status(True, "not configured (sentiment uses neutral fallback)")

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": "AAPL", "pageSize": 1, "apiKey": key},
            timeout=10,
        )
        if resp.status_code == 200:
            return _status(True, "API key valid")
        return _status(False, f"HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as exc:
        return _status(False, str(exc))


def check_slack() -> dict:
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return _status(True, "not configured (alerts disabled)")

    try:
        resp = requests.post(
            url,
            json={"text": "Kronos validate_env ping (safe to ignore)"},
            timeout=10,
        )
        if resp.status_code == 200:
            return _status(True, "webhook accepted")
        return _status(
            False,
            f"HTTP {resp.status_code} — regenerate webhook in Slack app settings",
        )
    except Exception as exc:
        return _status(False, str(exc))


def check_fred() -> dict:
    key = os.getenv("FRED_API_KEY")
    if not key:
        return _status(True, "not configured (macro signals skipped)")

    try:
        from fredapi import Fred

        data = Fred(api_key=key).get_series("DGS10", limit=1)
        if len(data) > 0:
            return _status(True, f"last DGS10={float(data.iloc[-1]):.2f}")
        return _status(False, "empty series")
    except Exception as exc:
        return _status(False, str(exc))


CHECKS = {
    "alpaca": check_alpaca,
    "newsapi": check_newsapi,
    "slack": check_slack,
    "fred": check_fred,
}


def main():
    parser = argparse.ArgumentParser(description="Validate Kronos .env configuration")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any configured service fails (ignore unconfigured)",
    )
    args = parser.parse_args()

    print("Kronos environment validation\n")
    failures = []
    for name, fn in CHECKS.items():
        result = fn()
        icon = "✅" if result["ok"] else "❌"
        print(f"  {icon} {name}: {result['detail']}")
        if not result["ok"]:
            failures.append(name)

    if failures and args.strict:
        print(f"\nFailed: {', '.join(failures)}")
        sys.exit(1)
    if failures:
        print(f"\n⚠️  {len(failures)} check(s) failed (run with --strict to exit non-zero)")
    else:
        print("\n✅ All checks passed or optional services unconfigured")
    sys.exit(0)


if __name__ == "__main__":
    main()

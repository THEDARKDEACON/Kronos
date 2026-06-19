#!/usr/bin/env python3
"""Pre-flight checks before paper/live runs (no model load)."""

import argparse
import os
import sys
from pathlib import Path

import psutil
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

CACHE_DIR = ROOT / "data" / "cache"
LOG_DIR = ROOT / "logs"


def _ok(msg: str) -> tuple[bool, str]:
    return True, msg


def _fail(msg: str) -> tuple[bool, str]:
    return False, msg


def check_mode() -> tuple[bool, str]:
    mode = os.getenv("KRONOS_MODE", "dry").lower()
    if mode not in ("dry", "paper", "live"):
        return _fail(f"Invalid KRONOS_MODE={mode}")
    if mode == "live":
        return _ok(f"mode=live (REAL MONEY — double-check settings)")
    return _ok(f"mode={mode}")


def check_alpaca_if_needed() -> tuple[bool, str]:
    mode = os.getenv("KRONOS_MODE", "dry").lower()
    if mode == "dry":
        return _ok("broker not required (dry)")
    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return _fail("ALPACA_API_KEY / ALPACA_SECRET_KEY required for paper/live")
    return _ok("Alpaca keys present")


def check_disk() -> tuple[bool, str]:
    disk = psutil.disk_usage("/")
    pct = disk.percent
    if pct >= 95:
        return _fail(f"disk {pct:.0f}% full — free space before running pipeline")
    if pct >= 90:
        return _ok(f"disk {pct:.0f}% full (warning)")
    return _ok(f"disk {pct:.0f}% used")


def check_cache_writable() -> tuple[bool, str]:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        test = CACHE_DIR / ".preflight_write"
        test.write_text("ok")
        test.unlink()
        return _ok(f"cache writable ({CACHE_DIR})")
    except OSError as exc:
        return _fail(f"cache not writable: {exc}")


def check_logs_writable() -> tuple[bool, str]:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return _ok(f"logs dir ok ({LOG_DIR})")
    except OSError as exc:
        return _fail(f"logs not writable: {exc}")


def check_prev_weights_cache() -> tuple[bool, str]:
    path = CACHE_DIR / "latest_positions.parquet"
    if path.exists():
        return _ok("latest_positions.parquet found (turnover penalty enabled)")
    return _ok("no prior positions — first run uses prev_weights=None")


def check_long_only_paper() -> tuple[bool, str]:
    mode = os.getenv("KRONOS_MODE", "dry").lower()
    long_only = os.getenv("LONG_ONLY", "").lower() in ("1", "true", "yes")
    if mode in ("paper", "live") and not long_only:
        return _ok("long/short book — ensure Alpaca supports your shorts")
    if long_only:
        return _ok("LONG_ONLY enabled")
    return _ok("long_only not set (dry default)")


CHECKS = [
    ("mode", check_mode),
    ("alpaca", check_alpaca_if_needed),
    ("disk", check_disk),
    ("cache", check_cache_writable),
    ("logs", check_logs_writable),
    ("prev_weights", check_prev_weights_cache),
    ("long_only", check_long_only_paper),
]


def main():
    parser = argparse.ArgumentParser(description="Kronos pre-flight (no model load)")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any failure")
    args = parser.parse_args()

    print("Kronos pre-flight\n")
    failures = []
    for name, fn in CHECKS:
        ok, detail = fn()
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}: {detail}")
        if not ok:
            failures.append(name)

    if failures:
        print(f"\n❌ Failed: {', '.join(failures)}")
        if args.strict:
            sys.exit(1)
    else:
        print("\n✅ Pre-flight passed")
    sys.exit(0)


if __name__ == "__main__":
    main()

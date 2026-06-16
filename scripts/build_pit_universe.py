"""
scripts/build_pit_universe.py
==============================
H-1 FIX: Point-In-Time S&P 500 Constituent Snapshot Builder

Creates annual parquet snapshots of the S&P 500 constituent list so that the
walk-forward backtest can load the universe that *actually existed* at each
rebalance date, eliminating survivorship bias (audit finding C-1 / H-1).

Usage:
    # Build snapshots from 2018 to today (one parquet per year):
    python scripts/build_pit_universe.py

    # Specify a custom year range:
    python scripts/build_pit_universe.py --start-year 2015 --end-year 2024

    # Preview without writing files:
    python scripts/build_pit_universe.py --dry-run

How it works:
    1. Fetches the current S&P 500 list from Wikipedia (Ticker + Sector).
    2. Fetches the historical changes table from Wikipedia (date_added, date_removed).
    3. For each year, reconstructs the constituent set by starting with today's
       list and rolling back additions/removals chronologically.
    4. Saves one parquet per year to data/cache/sp500_universe_<YYYY>-01-01.parquet
       (matching the filename pattern that data/ingestion.get_universe() reads).

Limitations:
    - Wikipedia's change log only goes back to ~2000 and may have gaps.
    - For production-grade PIT data use Compustat or Sharadar point-in-time datasets.
    - Run once per year (or before each backtest) to keep snapshots current.
"""

import argparse
import io
import os
import sys
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import requests

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {'User-Agent': 'KronosQuant/1.0 (research)'}


def fetch_wiki_tables() -> tuple:
    """Fetch the current members table and historical changes table from Wikipedia."""
    print("Fetching S&P 500 constituent data from Wikipedia...")
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    # Table 0: current members (Symbol, GICS Sector, ...)
    current = tables[0][['Symbol', 'GICS Sector']].rename(
        columns={'Symbol': 'Ticker', 'GICS Sector': 'Sector'}
    )
    current['Ticker'] = current['Ticker'].str.replace('.', '-', regex=False)

    # Table 1: historical changes (Date, Added Ticker, Removed Ticker, ...)
    if len(tables) < 2:
        print("[WARNING] No changes table found on Wikipedia. Only today's snapshot will be saved.")
        return current, pd.DataFrame()

    changes = tables[1].copy()
    # Normalize column names — Wikipedia occasionally changes them
    changes.columns = [str(c).strip() for c in changes.columns]

    # Try to identify date, added, and removed columns robustly
    date_col = next((c for c in changes.columns if 'date' in c.lower()), None)
    added_col = next((c for c in changes.columns if 'added' in c.lower() and 'ticker' in c.lower()), None)
    removed_col = next((c for c in changes.columns if 'removed' in c.lower() and 'ticker' in c.lower()), None)
    added_sector_col = next(
        (c for c in changes.columns if 'added' in c.lower() and 'sector' in c.lower()), None
    )

    if not date_col:
        print("[WARNING] Could not identify date column in changes table. Snapshot rollback disabled.")
        return current, pd.DataFrame()

    keep_cols = {date_col: 'Date'}
    if added_col:
        keep_cols[added_col] = 'Added'
    if removed_col:
        keep_cols[removed_col] = 'Removed'
    if added_sector_col:
        keep_cols[added_sector_col] = 'Added_Sector'

    changes = changes[list(keep_cols.keys())].rename(columns=keep_cols)
    changes['Date'] = pd.to_datetime(changes['Date'], errors='coerce')
    changes = changes.dropna(subset=['Date']).sort_values('Date', ascending=False)

    return current, changes


def reconstruct_universe_at_year(
    current_df: pd.DataFrame,
    changes_df: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:
    """
    Roll back the current constituent set to January 1st of target_year by
    reversing each recorded addition/removal since that date.

    Logic:
    - If a ticker was *added* after target_year, remove it (it wasn't there yet).
    - If a ticker was *removed* after target_year, add it back (it was still there).
    """
    target_date = pd.Timestamp(f"{target_year}-01-01")

    # Start from today's set
    members = set(current_df['Ticker'].tolist())
    sector_map = dict(zip(current_df['Ticker'], current_df['Sector']))

    if changes_df.empty:
        print(f"  [Year {target_year}] No change history — using today's universe as-is.")
        return current_df.copy()

    # Only consider changes that happened AFTER the target date
    future_changes = changes_df[changes_df['Date'] > target_date]

    for _, row in future_changes.iterrows():
        # Reverse additions: something added after target_date → was not in index then
        if 'Added' in row and pd.notna(row.get('Added', None)):
            ticker = str(row['Added']).replace('.', '-').strip()
            if ticker:
                members.discard(ticker)

        # Reverse removals: something removed after target_date → was still in index then
        if 'Removed' in row and pd.notna(row.get('Removed', None)):
            ticker = str(row['Removed']).replace('.', '-').strip()
            if ticker and ticker not in members:
                # Try to recover sector from changes row or from current map
                sector = row.get('Added_Sector', sector_map.get(ticker, 'Unknown'))
                members.add(ticker)
                if ticker not in sector_map:
                    sector_map[ticker] = sector if pd.notna(sector) else 'Unknown'

    result = pd.DataFrame([
        {'Ticker': t, 'Sector': sector_map.get(t, 'Unknown')}
        for t in sorted(members)
    ])
    return result


def build_snapshots(
    start_year: int = 2018,
    end_year: int = None,
    dry_run: bool = False,
) -> None:
    if end_year is None:
        end_year = date.today().year

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    current_df, changes_df = fetch_wiki_tables()
    print(f"Current S&P 500: {len(current_df)} members")
    print(f"Change records: {len(changes_df)} rows covering "
          f"{changes_df['Date'].min().date() if not changes_df.empty else 'N/A'} "
          f"to {changes_df['Date'].max().date() if not changes_df.empty else 'N/A'}")

    print(f"\nBuilding snapshots for {start_year}–{end_year}...")

    for year in range(start_year, end_year + 1):
        snapshot_path = CACHE_DIR / f"sp500_universe_{year}-01-01.parquet"

        if snapshot_path.exists():
            print(f"  [Year {year}] Already exists: {snapshot_path.name} — skipping.")
            continue

        universe_df = reconstruct_universe_at_year(current_df, changes_df, year)
        print(f"  [Year {year}] Reconstructed {len(universe_df)} tickers → {snapshot_path.name}")

        if not dry_run:
            universe_df.to_parquet(snapshot_path)
            print(f"             Saved.")
        else:
            print(f"             [DRY RUN] Not saved.")

    # Also save today's snapshot (live date)
    today_str = date.today().isoformat()
    today_path = CACHE_DIR / f"sp500_universe_{today_str}.parquet"
    if not today_path.exists():
        print(f"\n  [Today] Saving current universe → {today_path.name}")
        if not dry_run:
            current_df.to_parquet(today_path)

    print("\n✓ PIT universe snapshot build complete.")
    print(f"  Snapshots are in: {CACHE_DIR}")
    print("  Re-run annually (or before each backtest) to add new year snapshots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build point-in-time S&P 500 universe snapshots for survivorship-bias-free backtesting."
    )
    parser.add_argument("--start-year", type=int, default=2018, help="First year to build snapshot for")
    parser.add_argument("--end-year", type=int, default=None, help="Last year (default: current year)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files")
    args = parser.parse_args()

    build_snapshots(
        start_year=args.start_year,
        end_year=args.end_year,
        dry_run=args.dry_run,
    )

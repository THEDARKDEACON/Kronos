import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import requests
import os
from typing import List, Dict, Optional
from datetime import datetime

CACHE_DIR = "data/cache"

def get_universe(as_of_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch S&P 500 constituents from Wikipedia with daily caching.

    Enforces strict Point-In-Time (PIT) validation if as_of_date is provided.
    """
    import io
    from datetime import date as _date

    os.makedirs(CACHE_DIR, exist_ok=True)
    
    target_date = pd.Timestamp(as_of_date).date() if as_of_date else _date.today()
    cache_file = os.path.join(CACHE_DIR, f"sp500_universe_{target_date}.parquet")

    if os.path.exists(cache_file):
        print(f"Loading S&P 500 universe from cache for {target_date}...")
        return pd.read_parquet(cache_file)
        
    if as_of_date:
        raise ValueError(
            f"DataProvenanceError: Strict PIT enforcement failed. "
            f"Missing historical universe cache for {target_date}. "
            f"Refusing to fall back to future universe to prevent survivorship bias."
        )

    print("Scraping S&P 500 universe from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(
            url,
            headers={'User-Agent': 'KronosQuant/1.0 (research)'},
            timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0][['Symbol', 'GICS Sector']].rename(
            columns={'Symbol': 'Ticker', 'GICS Sector': 'Sector'}
        )
        df['Ticker'] = df['Ticker'].str.replace('.', '-', regex=False)
        df.to_parquet(cache_file)
        print(f"Universe cached: {len(df)} tickers")
        return df
    except Exception as exc:
        # Graceful fallback: use the most recently cached universe file
        candidates = sorted(
            [f for f in os.listdir(CACHE_DIR) if f.startswith('sp500_universe_')]
        )
        if candidates:
            fallback = os.path.join(CACHE_DIR, candidates[-1])
            print(f"[WARNING] Wikipedia scrape failed ({exc}). Using cached universe: {candidates[-1]}")
            return pd.read_parquet(fallback)
        raise RuntimeError(
            f"Cannot fetch S&P 500 universe and no cache available: {exc}"
        ) from exc

def fetch_pit_fundamental(ticker: str, sector: str, as_of_date: pd.Timestamp, max_retries: int = 3) -> dict:
    """Fetch point-in-time fundamentals with exponential-backoff retry."""
    import time as _time
    allowed_date = as_of_date - pd.Timedelta(days=90)
    result = {'Ticker': ticker, 'Sector': sector, 'PE_Ratio': None, 'Debt_To_Equity': None}
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            if as_of_date.date() == pd.Timestamp.now().date():
                info = stock.info
                result['PE_Ratio'] = info.get('trailingPE', None)
                result['Debt_To_Equity'] = info.get('debtToEquity', None)
                return result

            bs = stock.quarterly_balance_sheet
            inc = stock.quarterly_income_stmt
            if bs.empty or inc.empty:
                return result

            valid_cols = [col for col in bs.columns if isinstance(col, pd.Timestamp) and col <= allowed_date]
            if not valid_cols:
                return result

            target_qtr = max(valid_cols)
            try:
                total_debt = bs.loc['Total Debt', target_qtr] if 'Total Debt' in bs.index else 0
                equity = bs.loc['Stockholders Equity', target_qtr] if 'Stockholders Equity' in bs.index else None
                if equity and equity > 0:
                    result['Debt_To_Equity'] = (total_debt / equity) * 100
            except KeyError:
                pass
            try:
                eps = inc.loc['Basic EPS', target_qtr] if 'Basic EPS' in inc.index else None
                if eps and eps > 0:
                    hist_price = stock.history(start=as_of_date, end=as_of_date + pd.Timedelta(days=3))
                    if not hist_price.empty:
                        result['PE_Ratio'] = hist_price.iloc[0]['Close'] / (eps * 4)
            except KeyError:
                pass
            return result
        except Exception as exc:  # M-6: was bare `except:` — now catches only Exception
            if attempt < max_retries - 1:
                _time.sleep(2 ** attempt)  # M-5: exponential backoff
            else:
                print(f"   [Fundamentals] Failed {ticker} after {max_retries} attempts: {exc}")
    return result

def get_fundamentals(universe_df: pd.DataFrame, as_of_date: Optional[str] = None) -> pd.DataFrame:
    target_ts = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.now()
    cache_file = os.path.join(CACHE_DIR, f"fundamentals_{target_ts.date()}.parquet")

    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        if len(df) > 0:
            print(f"Loading Fundamentals from local Parquet Cache for [{target_ts.date()}]...")
            return df
        print(
            f"[WARNING] Empty fundamentals cache for [{target_ts.date()}] — refetching "
            f"({cache_file})"
        )

    print(f"API Fetching PiT fundamentals as of [{target_ts.date()}]...")
    data = []
    # M-5: reduced from 30 to 8 workers to avoid Yahoo Finance rate-limiting (HTTP 429)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(fetch_pit_fundamental, row['Ticker'], row['Sector'], target_ts)
            for _, row in universe_df.iterrows()
        ]
        for future in concurrent.futures.as_completed(futures):
            data.append(future.result())

    if not data:
        return pd.DataFrame(columns=['Sector', 'PE_Ratio', 'Debt_To_Equity'])

    df = pd.DataFrame(data).set_index('Ticker')
    # Keep all tickers — missing PE/debt filled in get_safe_universe; dropping rows
    # here produced empty caches when Yahoo rate-limits historical quarterly data.

    if len(df) > 0:
        df.to_parquet(cache_file)
    return df

def get_historical_ohlcv(tickers: List[str], as_of_date: Optional[str] = None, lookback_days: int = 400, incremental: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical OHLCV data with intelligent caching and incremental updates.
    
    Args:
        tickers: List of ticker symbols
        as_of_date: Target end date for data
        lookback_days: Historical days to fetch
        incremental: If True, only fetch missing data since last cache
    """
    if not tickers:
        print("Warning: Ticker list is empty. Skipping OHLCV fetch.")
        return {}
        
    target_ts = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.now()
    # M-11: include ticker count and lookback in key so a changed universe/depth invalidates the cache
    cache_file = os.path.join(CACHE_DIR, f"ohlcv_{target_ts.date()}_{len(tickers)}t_{lookback_days}d.parquet")
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        print(f"Loading OHLCV Matrix from local Parquet Cache for [{target_ts.date()}]...")
        master_df = pd.read_parquet(cache_file)
        ohlcv_dict = {ticker: group.drop(columns=['Ticker']).reset_index(drop=True) 
                     for ticker, group in master_df.groupby('Ticker')}
        
        # Check if we have all requested tickers
        missing_tickers = [t for t in tickers if t not in ohlcv_dict]
        if not missing_tickers:
            print(f"   [Cache Hit] All {len(tickers)} tickers loaded from cache")
            return ohlcv_dict
        else:
            print(f"   [Partial Cache] {len(tickers) - len(missing_tickers)}/{len(tickers)} from cache, fetching {len(missing_tickers)} missing...")
            tickers = missing_tickers  # Only fetch missing
    else:
        ohlcv_dict = {}
    
    # Calculate required date range
    if incremental and ohlcv_dict:
        # Find the earliest date in existing data to determine what we need
        existing_dates = [df['timestamps'].max() for df in ohlcv_dict.values() if 'timestamps' in df.columns and len(df) > 0]
        if existing_dates:
            last_cached_date = min(existing_dates)  # Conservative: use earliest max date
            start_ts = last_cached_date - pd.Timedelta(days=5)  # Small overlap for safety
            print(f"   [Incremental] Fetching from {start_ts.date()} to {target_ts.date()}...")
        else:
            start_ts = target_ts - pd.Timedelta(days=lookback_days + 150)
    else:
        start_ts = target_ts - pd.Timedelta(days=lookback_days + 150)
        print(f"API Downloading chronologically safe OHLCV ending [{target_ts.date()}]...")
    
    data = yf.download(tickers, start=start_ts.strftime('%Y-%m-%d'), 
                      end=(target_ts + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), 
                      group_by='ticker', progress=False)
    
    new_frames = []
    
    for ticker in tickers:
        if len(tickers) == 1:
            df = data
        else:
            try:
                if hasattr(data.columns, 'levels'):
                    df = data[ticker].dropna()
                else:
                    df = data.dropna()
            except KeyError:
                continue

        if len(df) == 0:
            continue
            
        df = df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df = df[df.index <= target_ts]
        df['timestamps'] = df.index
        df['amount'] = df['close'] * df['volume']  # L-6: was hardcoded 0.0
        
        reset_df = df.reset_index(drop=True)
        
        # Merge with existing data if incremental
        if incremental and ticker in ohlcv_dict:
            existing_df = ohlcv_dict[ticker]
            combined_df = pd.concat([existing_df, reset_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['timestamps'], keep='last')
            combined_df = combined_df.sort_values('timestamps').reset_index(drop=True)
            ohlcv_dict[ticker] = combined_df
            print(f"   [Merged] {ticker}: {len(existing_df)} + {len(reset_df)} = {len(combined_df)} rows")
        else:
            ohlcv_dict[ticker] = reset_df
        
        # Prepare for cache
        cache_df = ohlcv_dict[ticker].copy()
        cache_df['Ticker'] = ticker
        new_frames.append(cache_df)
        
    # Update cache with merged data
    if new_frames:
        if os.path.exists(cache_file) and incremental:
            # Load existing and merge
            existing_master = pd.read_parquet(cache_file)
            new_master = pd.concat(new_frames)
            combined_master = pd.concat([existing_master, new_master])
            combined_master = combined_master.drop_duplicates(subset=['Ticker', 'timestamps'], keep='last')
            combined_master.to_parquet(cache_file)
        else:
            pd.concat(new_frames).to_parquet(cache_file)
        print(f"   [Cache Updated] Saved {len(ohlcv_dict)} tickers to {cache_file}")
        
    return ohlcv_dict

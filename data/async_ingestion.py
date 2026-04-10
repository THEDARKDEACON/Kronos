"""
Async Data Ingestion Module
Provides high-performance async data fetching using asyncio and aiohttp
for significant speedup over ThreadPoolExecutor.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf
import os

CACHE_DIR = "cache"

async def fetch_fundamental_async(session: aiohttp.ClientSession, ticker: str, sector: str, as_of_date: pd.Timestamp) -> dict:
    """Async wrapper for fetching fundamental data."""
    # yfinance doesn't support async natively, so we run in executor
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_pit_fundamental_sync, ticker, sector, as_of_date)

def _fetch_pit_fundamental_sync(ticker: str, sector: str, as_of_date: pd.Timestamp) -> dict:
    """Synchronous fundamental fetch (runs in thread pool)."""
    allowed_date = as_of_date - pd.Timedelta(days=90)
    result = {'Ticker': ticker, 'Sector': sector, 'PE_Ratio': None, 'Debt_To_Equity': None}
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
    except Exception:
        pass
    return result

async def get_fundamentals_async(universe_df: pd.DataFrame, as_of_date: Optional[str] = None, max_concurrent: int = 50) -> pd.DataFrame:
    """
    High-performance async fundamental fetching.
    
    Args:
        universe_df: DataFrame with Ticker and Sector columns
        as_of_date: Target date for fundamentals
        max_concurrent: Maximum concurrent API calls (default 50)
    """
    target_ts = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.now()
    cache_file = os.path.join(CACHE_DIR, f"fundamentals_{target_ts.date()}.parquet")
    
    if os.path.exists(cache_file):
        print(f"Loading Fundamentals from local Parquet Cache for [{target_ts.date()}]...")
        return pd.read_parquet(cache_file)
    
    print(f"Async API Fetching PiT fundamentals as of [{target_ts.date()}]...")
    
    # Create semaphore to limit concurrent connections
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_limit(session, ticker, sector):
        async with semaphore:
            return await fetch_fundamental_async(session, ticker, sector, target_ts)
    
    # Fetch all fundamentals concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_with_limit(session, row['Ticker'], row['Sector'])
            for _, row in universe_df.iterrows()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    data = [r for r in results if not isinstance(r, Exception)]
    
    df = pd.DataFrame(data).set_index('Ticker')
    df = df.dropna(subset=['PE_Ratio', 'Debt_To_Equity'], how='all')
    
    # Commit to Parquet Lake
    df.to_parquet(cache_file)
    return df

async def fetch_ohlcv_batch(tickers: List[str], start_date: str, end_date: str, max_concurrent: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Async batch fetch for OHLCV data.
    Uses yfinance with controlled concurrency.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_single(ticker):
        async with semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: _fetch_ohlcv_sync(ticker, start_date, end_date)
            )
    
    tasks = [fetch_single(t) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    ohlcv_dict = {}
    for ticker, result in zip(tickers, results):
        if not isinstance(result, Exception) and result is not None:
            ohlcv_dict[ticker] = result
    
    return ohlcv_dict

def _fetch_ohlcv_sync(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Synchronous OHLCV fetch for a single ticker."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            return None
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df['timestamps'] = df.index
        df['amount'] = 0.0
        return df.reset_index(drop=True)
    except Exception:
        return None

# Convenience wrapper for non-async callers
def get_fundamentals_fast(universe_df: pd.DataFrame, as_of_date: Optional[str] = None) -> pd.DataFrame:
    """Synchronous wrapper for async fundamental fetching."""
    return asyncio.run(get_fundamentals_async(universe_df, as_of_date))

def get_ohlcv_fast(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Synchronous wrapper for async OHLCV fetching."""
    return asyncio.run(fetch_ohlcv_batch(tickers, start_date, end_date))

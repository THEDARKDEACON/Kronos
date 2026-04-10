import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import requests
import os
from typing import List, Dict, Optional
from datetime import datetime

CACHE_DIR = "cache"

def get_universe() -> pd.DataFrame:
    """Scrapes the real S&P 500 constituents from Wikipedia."""
    # We don't cache the universe mapping itself since it's an instant Wikipedia fetch
    print("Scraping real S&P 500 universe from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    import io
    html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
    tables = pd.read_html(io.StringIO(html))
    df = tables[0][['Symbol', 'GICS Sector']].rename(columns={'Symbol': 'Ticker', 'GICS Sector': 'Sector'})
    df['Ticker'] = df['Ticker'].str.replace('.', '-', regex=False)
    return df

def fetch_pit_fundamental(ticker: str, sector: str, as_of_date: pd.Timestamp) -> dict:
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
        if bs.empty or inc.empty: return result
            
        valid_cols = [col for col in bs.columns if isinstance(col, pd.Timestamp) and col <= allowed_date]
        if not valid_cols: return result
            
        target_qtr = max(valid_cols)
        try:
            total_debt = bs.loc['Total Debt', target_qtr] if 'Total Debt' in bs.index else 0
            equity = bs.loc['Stockholders Equity', target_qtr] if 'Stockholders Equity' in bs.index else None
            if equity and equity > 0:
                result['Debt_To_Equity'] = (total_debt / equity) * 100 
        except KeyError: pass
        try:
            eps = inc.loc['Basic EPS', target_qtr] if 'Basic EPS' in inc.index else None
            if eps and eps > 0:
                hist_price = stock.history(start=as_of_date, end=as_of_date + pd.Timedelta(days=3))
                if not hist_price.empty:
                    result['PE_Ratio'] = hist_price.iloc[0]['Close'] / (eps * 4)
        except KeyError: pass
    except Exception: pass
    return result

def get_fundamentals(universe_df: pd.DataFrame, as_of_date: Optional[str] = None) -> pd.DataFrame:
    target_ts = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.now()
    cache_file = os.path.join(CACHE_DIR, f"fundamentals_{target_ts.date()}.parquet")
    
    if os.path.exists(cache_file):
        print(f"Loading Fundamentals from local Parquet Cache for [{target_ts.date()}]...")
        return pd.read_parquet(cache_file)
        
    print(f"API Fetching PiT fundamentals as of [{target_ts.date()}]...")
    data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(fetch_pit_fundamental, row['Ticker'], row['Sector'], target_ts) for _, row in universe_df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            data.append(future.result())
            
    df = pd.DataFrame(data).set_index('Ticker')
    df = df.dropna(subset=['PE_Ratio', 'Debt_To_Equity'], how='all')
    
    # Commit to Parquet Lake
    df.to_parquet(cache_file)
    return df

def get_historical_ohlcv(tickers: List[str], as_of_date: Optional[str] = None, lookback_days: int = 400) -> Dict[str, pd.DataFrame]:
    if not tickers:
        print("Warning: Ticker list is empty. Skipping OHLCV fetch.")
        return {}
        
    target_ts = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.now()
    cache_file = os.path.join(CACHE_DIR, f"ohlcv_{target_ts.date()}.parquet")
    
    if os.path.exists(cache_file):
        print(f"Loading OHLCV Matrix from local Parquet Cache for [{target_ts.date()}]...")
        master_df = pd.read_parquet(cache_file)
        # Reconstruct Dictionary
        return {ticker: group.drop(columns=['Ticker']).reset_index(drop=True) for ticker, group in master_df.groupby('Ticker')}
    
    start_ts = target_ts - pd.Timedelta(days=lookback_days + 150) 
    print(f"API Downloading chronologically safe OHLCV ending [{target_ts.date()}]...")
    
    data = yf.download(tickers, start=start_ts.strftime('%Y-%m-%d'), end=(target_ts + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), group_by='ticker', progress=False)
    
    ohlcv_dict = {}
    master_frames = []
    
    for ticker in tickers:
        if ticker in tickers and len(tickers) == 1: df = data
        else:
            try:
                if hasattr(data.columns, 'levels'): df = data[ticker].dropna()
                else: df = data.dropna()
            except KeyError: continue

        if len(df) == 0: continue
        df = df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df = df[df.index <= target_ts]
        df['timestamps'] = df.index
        df['amount'] = 0.0 
        
        reset_df = df.reset_index(drop=True)
        ohlcv_dict[ticker] = reset_df
        
        # Prepare for massive master frame dump
        cache_df = reset_df.copy()
        cache_df['Ticker'] = ticker
        master_frames.append(cache_df)
        
    if master_frames:
        pd.concat(master_frames).to_parquet(cache_file)
        
    return ohlcv_dict

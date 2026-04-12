"""Yahoo Finance data loader module."""
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from .ingestion import get_universe, fetch_pit_fundamental

__all__ = ['DataLoader']


class DataLoader:
    """Data loading interface for Yahoo Finance data."""
    
    def __init__(self):
        self.universe = get_universe()
    
    def fetch_data(self, tickers=None, period="1y", interval="1d"):
        """Fetch market data for tickers."""
        if tickers is None:
            tickers = self.universe['Ticker'].tolist()[:50]
        
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period, interval=interval)
                if not hist.empty:
                    data[ticker] = hist
            except Exception as e:
                print(f"Failed to fetch {ticker}: {e}")
        
        return data
    
    def get_universe_df(self):
        """Return universe dataframe."""
        return self.universe
    
    def fetch_fundamentals(self, ticker: str, as_of_date: Optional[datetime] = None):
        """Fetch fundamental data for a ticker."""
        if as_of_date is None:
            as_of_date = pd.Timestamp.now()
        
        # Get sector from universe
        sector = self.universe[self.universe['Ticker'] == ticker]['Sector'].iloc[0] if ticker in self.universe['Ticker'].values else 'Unknown'
        
        return fetch_pit_fundamental(ticker, sector, as_of_date)

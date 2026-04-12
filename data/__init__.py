# Data module for Kronos Quant Pipeline
from .ingestion import get_universe, fetch_pit_fundamental

__all__ = ['get_universe', 'fetch_pit_fundamental', 'DataLoader']

# Simple DataLoader wrapper for production pipeline
class DataLoader:
    """Data loading interface for production pipeline."""
    
    def __init__(self):
        self.universe = get_universe()
    
    def fetch_data(self, tickers=None, period="1y"):
        """Fetch market data for tickers."""
        import yfinance as yf
        
        if tickers is None:
            tickers = self.universe['Ticker'].tolist()[:50]  # Default to top 50
        
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if not hist.empty:
                    data[ticker] = hist
            except Exception as e:
                print(f"Failed to fetch {ticker}: {e}")
        
        return data
    
    def get_universe_df(self):
        """Return universe dataframe."""
        return self.universe

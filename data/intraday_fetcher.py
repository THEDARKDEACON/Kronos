"""
Intraday Data Fetching Module
Fetches hourly/30-minute OHLCV data for intraday signal generation.
Supports Polygon.io, Alpaca, and Yahoo Finance for intraday data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, time
import requests
import os
import warnings
from dataclasses import dataclass
from enum import Enum
import yfinance as yf


class IntradayInterval(Enum):
    """Supported intraday intervals."""
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    HOUR_1 = "1hour"


@dataclass
class IntradayBar:
    """Single intraday OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trades: Optional[int] = None


class IntradayDataFetcher:
    """
    Fetches intraday market data from multiple sources.
    """
    
    def __init__(self, 
                 polygon_api_key: Optional[str] = None,
                 alpaca_api_key: Optional[str] = None,
                 alpaca_secret_key: Optional[str] = None):
        self.polygon_key = polygon_api_key or os.getenv('POLYGON_API_KEY')
        self.alpaca_key = alpaca_api_key or os.getenv('ALPACA_API_KEY')
        self.alpaca_secret = alpaca_secret_key or os.getenv('ALPACA_SECRET_KEY')
        
    def _is_market_hours(self, dt: datetime) -> bool:
        """Check if datetime is within US market hours (9:30-16:00 ET)."""
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        if dt.weekday() >= 5:  # Weekend
            return False
        
        return market_open <= dt.time() <= market_close
    
    def fetch_intraday_yfinance(self,
                                ticker: str,
                                interval: IntradayInterval = IntradayInterval.HOUR_1,
                                days_back: int = 7) -> pd.DataFrame:
        """
        Fetch intraday data from Yahoo Finance.
        
        Note: Yahoo Finance has limitations on intraday data history.
        - 1m: 7 days max
        - 2m, 5m, 15m, 30m, 60m: 60 days max
        """
        interval_str = interval.value.replace('min', 'm').replace('hour', 'h')
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval_str,
                progress=False
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # Rename columns to standard format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add timestamp column
            df['timestamp'] = df.index
            df['amount'] = 0.0  # Not provided by yfinance
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            warnings.warn(f"Failed to fetch intraday data for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_intraday_polygon(self,
                               ticker: str,
                               interval: IntradayInterval = IntradayInterval.HOUR_1,
                               days_back: int = 30) -> pd.DataFrame:
        """
        Fetch intraday data from Polygon.io API.
        
        Requires POLYGON_API_KEY environment variable.
        """
        if not self.polygon_key:
            warnings.warn("Polygon API key not provided")
            return pd.DataFrame()
        
        multiplier = 1
        timespan = interval.value.replace('1', '')
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        params = {
            'apiKey': self.polygon_key,
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'results' not in data:
                return pd.DataFrame()
            
            bars = []
            for result in data['results']:
                bars.append({
                    'timestamp': datetime.fromtimestamp(result['t'] / 1000),
                    'open': result['o'],
                    'high': result['h'],
                    'low': result['l'],
                    'close': result['c'],
                    'volume': result['v'],
                    'vwap': result.get('vw'),
                    'trades': result.get('n')
                })
            
            df = pd.DataFrame(bars)
            df['amount'] = df['vwap'] * df['volume'] if 'vwap' in df.columns else 0.0
            
            return df
            
        except Exception as e:
            warnings.warn(f"Polygon fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_intraday_alpaca(self,
                              ticker: str,
                              interval: IntradayInterval = IntradayInterval.HOUR_1,
                              days_back: int = 30) -> pd.DataFrame:
        """
        Fetch intraday data from Alpaca Markets API.
        
        Requires ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.
        """
        if not self.alpaca_key or not self.alpaca_secret:
            warnings.warn("Alpaca credentials not provided")
            return pd.DataFrame()
        
        timeframe_map = {
            IntradayInterval.MIN_1: '1Min',
            IntradayInterval.MIN_5: '5Min',
            IntradayInterval.MIN_15: '15Min',
            IntradayInterval.MIN_30: '30Min',
            IntradayInterval.HOUR_1: '1Hour'
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
        
        headers = {
            'APCA-API-KEY-ID': self.alpaca_key,
            'APCA-API-SECRET-KEY': self.alpaca_secret
        }
        
        params = {
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'timeframe': timeframe_map.get(interval, '1Hour'),
            'limit': 10000
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            data = response.json()
            
            if 'bars' not in data or not data['bars']:
                return pd.DataFrame()
            
            bars = []
            for bar in data['bars']:
                bars.append({
                    'timestamp': datetime.fromisoformat(bar['t'].replace('Z', '+00:00')),
                    'open': bar['o'],
                    'high': bar['h'],
                    'low': bar['l'],
                    'close': bar['c'],
                    'volume': bar['v'],
                    'vwap': bar.get('vw'),
                    'trades': bar.get('n')
                })
            
            df = pd.DataFrame(bars)
            df['amount'] = df['vwap'] * df['volume'] if 'vwap' in df.columns else 0.0
            
            return df
            
        except Exception as e:
            warnings.warn(f"Alpaca fetch failed for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_intraday(self,
                      ticker: str,
                      interval: IntradayInterval = IntradayInterval.HOUR_1,
                      days_back: int = 7,
                      source: str = 'auto') -> pd.DataFrame:
        """
        Fetch intraday data from best available source.
        
        Args:
            ticker: Stock symbol
            interval: Data interval
            days_back: Days of history to fetch
            source: 'auto', 'polygon', 'alpaca', or 'yfinance'
        
        Returns:
            DataFrame with intraday bars
        """
        if source == 'auto':
            # Try sources in order of quality
            if self.polygon_key:
                source = 'polygon'
            elif self.alpaca_key:
                source = 'alpaca'
            else:
                source = 'yfinance'
        
        if source == 'polygon':
            return self.fetch_intraday_polygon(ticker, interval, days_back)
        elif source == 'alpaca':
            return self.fetch_intraday_alpaca(ticker, interval, days_back)
        else:
            return self.fetch_intraday_yfinance(ticker, interval, days_back)
    
    def fetch_universe_intraday(self,
                                tickers: List[str],
                                interval: IntradayInterval = IntradayInterval.HOUR_1,
                                days_back: int = 7) -> Dict[str, pd.DataFrame]:
        """
        Fetch intraday data for multiple tickers.
        
        Returns:
            Dict of ticker -> DataFrame
        """
        results = {}
        
        for i, ticker in enumerate(tickers):
            print(f"   Fetching intraday data for {ticker} ({i+1}/{len(tickers)})...")
            df = self.fetch_intraday(ticker, interval, days_back)
            if not df.empty:
                results[ticker] = df
        
        print(f"   Successfully fetched {len(results)}/{len(tickers)} tickers")
        return results


class RealTimeSignalGenerator:
    """
    Generates signals from intraday data in real-time.
    """
    
    def __init__(self, model_predictor=None):
        self.predictor = model_predictor
        self.last_signals: Dict[str, float] = {}
        
    def calculate_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate intraday technical features.
        """
        if df.empty or len(df) < 10:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volume features
        features['volume_ma'] = df['volume'].rolling(10).mean()
        features['relative_volume'] = df['volume'] / features['volume_ma']
        
        # VWAP features
        if 'vwap' in df.columns:
            features['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Intraday momentum
        features['momentum_3'] = df['close'] / df['close'].shift(3) - 1
        features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        
        # Volatility
        features['volatility_10'] = features['returns'].rolling(10).std()
        
        # Time-of-day effects
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        features['is_morning'] = (df['hour'] < 11).astype(int)
        features['is_afternoon'] = (df['hour'] > 14).astype(int)
        
        return features.dropna()
    
    def generate_intraday_signals(self,
                                 intraday_dict: Dict[str, pd.DataFrame],
                                 lookback_bars: int = 30) -> pd.DataFrame:
        """
        Generate intraday trading signals.
        
        Returns:
            DataFrame with signals for each ticker
        """
        signals = []
        
        for ticker, df in intraday_dict.items():
            if len(df) < lookback_bars:
                continue
            
            features = self.calculate_intraday_features(df)
            if features.empty:
                continue
            
            # Simple mean-reversion signal
            vwap_dev = features['vwap_deviation'].iloc[-1] if 'vwap_deviation' in features.columns else 0
            momentum = features['momentum_5'].iloc[-1]
            rel_volume = features['relative_volume'].iloc[-1]
            
            # Combined signal (mean reversion + momentum confirmation)
            signal = -vwap_dev * 0.5 + momentum * 0.3 + (rel_volume - 1) * 0.2
            
            signals.append({
                'Ticker': ticker,
                'Intraday_Signal': signal,
                'VWAP_Deviation': vwap_dev,
                'Momentum_5': momentum,
                'Relative_Volume': rel_volume,
                'Volatility': features['volatility_10'].iloc[-1] if 'volatility_10' in features.columns else 0,
                'Last_Price': df['close'].iloc[-1],
                'Timestamp': df['timestamp'].iloc[-1]
            })
        
        return pd.DataFrame(signals).set_index('Ticker') if signals else pd.DataFrame()
    
    def combine_with_daily_signals(self,
                                    intraday_signals: pd.DataFrame,
                                    daily_signals: pd.DataFrame,
                                    intraday_weight: float = 0.3) -> pd.DataFrame:
        """
        Combine intraday and daily signals.
        
        Args:
            intraday_signals: Signals from intraday data
            daily_signals: Signals from daily data (e.g., Kronos)
            intraday_weight: Weight for intraday signals (0-1)
        
        Returns:
            Combined signals DataFrame
        """
        # Align tickers
        common_tickers = intraday_signals.index.intersection(daily_signals.index)
        
        combined = pd.DataFrame(index=common_tickers)
        
        for ticker in common_tickers:
            daily_score = daily_signals.loc[ticker, 'Predicted_Return'] if 'Predicted_Return' in daily_signals.columns else 0
            intraday_score = intraday_signals.loc[ticker, 'Intraday_Signal'] if 'Intraday_Signal' in intraday_signals.columns else 0
            
            combined.loc[ticker, 'Daily_Signal'] = daily_score
            combined.loc[ticker, 'Intraday_Signal'] = intraday_score
            combined.loc[ticker, 'Combined_Signal'] = (
                (1 - intraday_weight) * daily_score +
                intraday_weight * intraday_score
            )
        
        return combined.sort_values('Combined_Signal', ascending=False)


class IntradayScheduler:
    """
    Schedules signal generation and trading during market hours.
    """
    
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    
    def __init__(self, signal_generator: RealTimeSignalGenerator):
        self.generator = signal_generator
        self.is_running = False
        
    def is_market_open(self) -> bool:
        """Check if US equity markets are currently open."""
        now = datetime.now()
        
        # Check weekend
        if now.weekday() >= 5:
            return False
        
        # Check market hours (9:30 AM - 4:00 PM ET)
        current_time = now.time()
        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE
    
    def get_next_signal_time(self, interval_minutes: int = 30) -> datetime:
        """
        Get next signal generation time.
        
        Args:
            interval_minutes: Signal generation interval
        
        Returns:
            Next datetime for signal generation
        """
        now = datetime.now()
        
        # Round up to next interval
        minutes_since_hour = now.minute % interval_minutes
        minutes_to_add = interval_minutes - minutes_since_hour if minutes_since_hour != 0 else 0
        
        next_time = now + timedelta(minutes=minutes_to_add)
        next_time = next_time.replace(second=0, microsecond=0)
        
        return next_time
    
    def should_generate_signal(self, last_signal_time: Optional[datetime],
                              interval_minutes: int = 30) -> bool:
        """
        Check if enough time has passed to generate new signals.
        """
        if last_signal_time is None:
            return True
        
        elapsed = (datetime.now() - last_signal_time).total_seconds() / 60
        return elapsed >= interval_minutes


# Convenience functions
def fetch_intraday_data(tickers: List[str],
                       interval: str = "1hour",
                       days_back: int = 7) -> Dict[str, pd.DataFrame]:
    """
    Quick function to fetch intraday data for tickers.
    """
    interval_map = {
        "1min": IntradayInterval.MIN_1,
        "5min": IntradayInterval.MIN_5,
        "15min": IntradayInterval.MIN_15,
        "30min": IntradayInterval.MIN_30,
        "1hour": IntradayInterval.HOUR_1
    }
    
    fetcher = IntradayDataFetcher()
    interval_enum = interval_map.get(interval, IntradayInterval.HOUR_1)
    
    return fetcher.fetch_universe_intraday(tickers, interval_enum, days_back)


def generate_intraday_alpha(tickers: List[str],
                           daily_signals: pd.DataFrame,
                           intraday_weight: float = 0.3) -> pd.DataFrame:
    """
    Generate combined alpha from intraday and daily signals.
    """
    # Fetch intraday data
    intraday_dict = fetch_intraday_data(tickers, interval="1hour", days_back=7)
    
    # Generate intraday signals
    generator = RealTimeSignalGenerator()
    intraday_signals = generator.generate_intraday_signals(intraday_dict)
    
    # Combine with daily signals
    combined = generator.combine_with_daily_signals(intraday_signals, daily_signals, intraday_weight)
    
    return combined

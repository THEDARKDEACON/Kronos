"""
Corporate Action Adjustments Module
Handles stock splits, dividends, and spinoffs for accurate price history.
Prevents false signals from unadjusted prices.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from dataclasses import dataclass


@dataclass
class CorporateAction:
    """Single corporate action record."""
    ticker: str
    action_type: str  # 'split', 'dividend', 'spinoff'
    ex_date: datetime
    ratio: float      # Split ratio or dividend amount
    announcement_date: Optional[datetime] = None


class CorporateActionAdjuster:
    """
    Adjusts price history for corporate actions.
    Uses multiplicative adjustment factors applied backwards.
    """
    
    def __init__(self):
        self.action_cache: Dict[str, pd.DataFrame] = {}
        
    def fetch_splits(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch stock split history from Yahoo Finance.
        
        Returns DataFrame with columns: Date, Split Ratio
        """
        try:
            # Use yfinance ticker object for splits
            tk = yf.Ticker(ticker)
            splits = tk.splits
            
            if splits is None or splits.empty:
                return pd.DataFrame(columns=['Date', 'Split_Ratio'])
            
            # Filter by date range
            splits = splits.reset_index()
            splits.columns = ['Date', 'Split_Ratio']
            splits['Date'] = pd.to_datetime(splits['Date']).dt.tz_localize(None)
            
            mask = (splits['Date'] >= start_date) & (splits['Date'] <= end_date)
            return splits[mask].sort_values('Date', ascending=False)  # Most recent first
            
        except Exception as e:
            warnings.warn(f"Failed to fetch splits for {ticker}: {e}")
            return pd.DataFrame(columns=['Date', 'Split_Ratio'])
    
    def fetch_dividends(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch dividend history from Yahoo Finance.
        
        Returns DataFrame with columns: Date, Dividend
        """
        try:
            tk = yf.Ticker(ticker)
            dividends = tk.dividends
            
            if dividends is None or dividends.empty:
                return pd.DataFrame(columns=['Date', 'Dividend'])
            
            # Filter by date range
            dividends = dividends.reset_index()
            dividends.columns = ['Date', 'Dividend']
            dividends['Date'] = pd.to_datetime(dividends['Date']).dt.tz_localize(None)
            
            mask = (dividends['Date'] >= start_date) & (dividends['Date'] <= end_date)
            return dividends[mask].sort_values('Date', ascending=False)
            
        except Exception as e:
            warnings.warn(f"Failed to fetch dividends for {ticker}: {e}")
            return pd.DataFrame(columns=['Date', 'Dividend'])
    
    def calculate_adjustment_factors(self,
                                     price_df: pd.DataFrame,
                                     splits_df: pd.DataFrame,
                                     dividends_df: pd.DataFrame,
                                     adjustment_type: str = 'both') -> pd.DataFrame:
        """
        Calculate cumulative adjustment factors.
        
        Args:
            price_df: DataFrame with price history (must have 'close' column)
            splits_df: DataFrame with splits
            dividends_df: DataFrame with dividends
            adjustment_type: 'splits', 'dividends', or 'both'
        
        Returns:
            DataFrame with adjustment factors for each date
        """
        if price_df.empty:
            return price_df
        
        df = price_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Initialize adjustment factor (starts at 1.0, applied backwards)
        df['adj_factor'] = 1.0
        
        # Apply split adjustments (multiplicative, applied backwards)
        if adjustment_type in ['splits', 'both'] and not splits_df.empty:
            for _, split in splits_df.iterrows():
                split_date = split['Date']
                split_ratio = split['Split_Ratio']
                
                # For forward splits (ratio > 1), multiply prices before split date by ratio
                # For reverse splits (ratio < 1), divide prices before split date by ratio
                mask = df['timestamp'] < split_date
                if split_ratio > 1:  # Forward split (e.g., 2:1)
                    df.loc[mask, 'adj_factor'] *= split_ratio
                else:  # Reverse split (e.g., 1:5)
                    df.loc[mask, 'adj_factor'] *= (1 / split_ratio)
        
        # Apply dividend adjustments (multiplicative, applied backwards)
        if adjustment_type in ['dividends', 'both'] and not dividends_df.empty:
            for _, div in dividends_df.iterrows():
                div_date = div['Date']
                div_amount = div['Dividend']
                
                # Get closing price on dividend date
                div_day_prices = df[df['timestamp'] == div_date]
                if not div_day_prices.empty:
                    close_price = div_day_prices['close'].iloc[0]
                    # Adjustment factor = (close - div) / close
                    # Applied to all prices before dividend date
                    div_factor = (close_price - div_amount) / close_price
                    mask = df['timestamp'] < div_date
                    df.loc[mask, 'adj_factor'] *= div_factor
        
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def apply_adjustments(self,
                         price_df: pd.DataFrame,
                         adjustment_factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply adjustment factors to price data.
        
        Returns:
            DataFrame with adjusted OHLCV columns
        """
        if price_df.empty or adjustment_factors_df.empty:
            return price_df
        
        df = price_df.copy()
        factors = adjustment_factors_df.copy()
        
        # Merge on timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        factors['timestamp'] = pd.to_datetime(factors['timestamp'])
        
        merged = df.merge(factors[['timestamp', 'adj_factor']], on='timestamp', how='left')
        merged['adj_factor'] = merged['adj_factor'].fillna(1.0)
        
        # Apply adjustment factors to price columns
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in merged.columns:
                # Divide by adjustment factor (backwards adjustment)
                merged[f'{col}_adj'] = merged[col] / merged['adj_factor']
        
        # Volume is adjusted inversely
        if 'volume' in merged.columns:
            merged['volume_adj'] = merged['volume'] * merged['adj_factor']
        
        return merged
    
    def adjust_prices(self,
                     ticker: str,
                     price_df: pd.DataFrame,
                     start_date: datetime,
                     end_date: datetime,
                     adjustment_type: str = 'both') -> pd.DataFrame:
        """
        Main method: fetch corporate actions and apply adjustments.
        
        Args:
            ticker: Stock symbol
            price_df: DataFrame with OHLCV data
            start_date: Start of period to check for actions
            end_date: End of period
            adjustment_type: 'splits', 'dividends', or 'both'
        
        Returns:
            DataFrame with adjusted and unadjusted prices
        """
        if price_df.empty:
            return price_df
        
        print(f"[Corp Actions] Adjusting {ticker} for {adjustment_type}...")
        
        # Extend date range to catch actions before our data starts
        extended_start = start_date - timedelta(days=365)
        
        # Fetch corporate actions
        splits = self.fetch_splits(ticker, extended_start, end_date)
        dividends = self.fetch_dividends(ticker, extended_start, end_date)
        
        if not splits.empty:
            print(f"   Found {len(splits)} splits: {splits.to_dict('records')}")
        if not dividends.empty:
            print(f"   Found {len(dividends)} dividends")
        
        # Calculate adjustment factors
        factors_df = self.calculate_adjustment_factors(
            price_df, splits, dividends, adjustment_type
        )
        
        # Apply adjustments
        adjusted_df = self.apply_adjustments(price_df, factors_df)
        
        # Cache for future use
        self.action_cache[ticker] = {
            'splits': splits,
            'dividends': dividends,
            'factors': factors_df
        }
        
        return adjusted_df
    
    def get_total_return_factor(self, ticker: str, price_df: pd.DataFrame) -> float:
        """
        Calculate total return adjustment factor from start to end.
        
        Returns multiplier that would convert price return to total return.
        """
        if price_df.empty or ticker not in self.action_cache:
            return 1.0
        
        cache = self.action_cache[ticker]
        if 'factors' not in cache or cache['factors'].empty:
            return 1.0
        
        # First factor / last factor gives total adjustment
        factors = cache['factors'].sort_values('timestamp')
        if len(factors) < 2:
            return 1.0
        
        start_factor = factors['adj_factor'].iloc[0]
        end_factor = factors['adj_factor'].iloc[-1]
        
        return end_factor / start_factor


class SurvivorshipBiasHandler:
    """
    Handles survivorship bias by tracking historical constituents and delistings.
    """
    
    def __init__(self):
        self.historical_constituents: Dict[datetime, List[str]] = {}
        self.delisted_prices: Dict[str, Dict] = {}
    
    def fetch_historical_sp500_constituents(self, date: datetime) -> List[str]:
        """
        Fetch S&P 500 constituents as of a specific historical date.
        
        Note: This is a simplified implementation. In production, you would:
        - Use a proper data vendor (S&P Dow Jones Indices)
        - Or scrape from historical Wikipedia snapshots via Wayback Machine
        - Or use CRSP/CapitalIQ data
        """
        # For now, return current constituents (acknowledging this has survivorship bias)
        # In production, replace with historical data
        from data.ingestion import get_universe
        return get_universe()
    
    def estimate_delisting_return(self,
                                  ticker: str,
                                  last_price: float,
                                  delisting_date: datetime) -> float:
        """
        Estimate return for delisted stock.
        
        Common scenarios:
        - Acquisition: typically 20-50% premium
        - Bankruptcy: typically -80% to -100%
        - Merger: conversion to acquirer stock
        """
        # Simplified: assume acquisition at 30% premium as most common
        # In production, look up actual deal terms
        return 0.30
    
    def mark_delisted_stocks(self,
                            tickers: List[str],
                            ohlcv_dict: Dict[str, pd.DataFrame],
                            as_of_date: datetime) -> Dict[str, str]:
        """
        Identify which stocks in universe are delisted as of date.
        
        Returns:
            Dict of ticker -> delisting reason
        """
        delisted = {}
        
        for ticker in tickers:
            if ticker not in ohlcv_dict or ohlcv_dict[ticker].empty:
                # Check if ticker exists at all
                try:
                    tk = yf.Ticker(ticker)
                    info = tk.info
                    if not info or info.get('regularMarketPrice') is None:
                        delisted[ticker] = 'no_data'
                except:
                    delisted[ticker] = 'delisted'
        
        return delisted


class CorporateActionValidator:
    """
    Validates that corporate action adjustments are correct.
    """
    
    def __init__(self):
        self.validation_errors: List[str] = []
    
    def validate_adjustments(self,
                            original_df: pd.DataFrame,
                            adjusted_df: pd.DataFrame,
                            ticker: str) -> bool:
        """
        Validate that adjustments look reasonable.
        
        Checks:
        - Adjusted prices should be positive
        - Adjusted prices should be continuous (no gaps)
        - Volume adjustments should be reasonable
        """
        is_valid = True
        
        # Check for negative prices
        for col in ['open_adj', 'high_adj', 'low_adj', 'close_adj']:
            if col in adjusted_df.columns:
                if (adjusted_df[col] <= 0).any():
                    self.validation_errors.append(
                        f"{ticker}: Negative prices in {col}"
                    )
                    is_valid = False
        
        # Check for extreme price jumps (more than 50% day-over-day)
        if 'close_adj' in adjusted_df.columns:
            returns = adjusted_df['close_adj'].pct_change().abs()
            extreme_jumps = returns > 0.50
            if extreme_jumps.any():
                # This might be valid for splits, but flag it
                n_jumps = extreme_jumps.sum()
                if n_jumps > len(adjusted_df) * 0.01:  # More than 1% of days
                    self.validation_errors.append(
                        f"{ticker}: {n_jumps} extreme price jumps (>50%) - possible error"
                    )
                    is_valid = False
        
        # Check volume consistency
        if 'volume_adj' in adjusted_df.columns and 'volume' in adjusted_df.columns:
            # Adjusted volume should generally be lower than raw (due to splits)
            # But can be higher for reverse splits
            ratio = adjusted_df['volume_adj'] / adjusted_df['volume']
            if (ratio < 0.1).any() or (ratio > 10).any():
                self.validation_errors.append(
                    f"{ticker}: Extreme volume adjustment ratios"
                )
                is_valid = False
        
        return is_valid
    
    def get_validation_report(self) -> str:
        """Get validation errors as formatted string."""
        if not self.validation_errors:
            return "All corporate action adjustments validated successfully."
        
        return "\n".join(["VALIDATION ERRORS:"] + self.validation_errors)


# Convenience functions
def adjust_prices_for_corporate_actions(ticker: str,
                                       price_df: pd.DataFrame,
                                       start_date: datetime,
                                       end_date: datetime) -> pd.DataFrame:
    """
    Quick function to adjust prices for corporate actions.
    """
    adjuster = CorporateActionAdjuster()
    return adjuster.adjust_prices(ticker, price_df, start_date, end_date)


def adjust_universe_prices(ohlcv_dict: Dict[str, pd.DataFrame],
                          start_date: datetime,
                          end_date: datetime) -> Dict[str, pd.DataFrame]:
    """
    Adjust all tickers in universe for corporate actions.
    """
    adjuster = CorporateActionAdjuster()
    validator = CorporateActionValidator()
    
    adjusted_dict = {}
    
    for ticker, df in ohlcv_dict.items():
        if df.empty:
            continue
        
        adjusted = adjuster.adjust_prices(ticker, df, start_date, end_date)
        
        # Validate
        is_valid = validator.validate_adjustments(df, adjusted, ticker)
        if not is_valid:
            print(f"   [Warning] Validation issues for {ticker}")
        
        # Return with both adjusted and original columns
        adjusted_dict[ticker] = adjusted
    
    # Print validation report
    print(validator.get_validation_report())
    
    return adjusted_dict

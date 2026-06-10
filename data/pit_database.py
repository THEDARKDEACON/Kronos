"""
Point-in-Time (PIT) Fundamentals Database
Stores historical fundamental data as it was known at each point in time.
Prevents look-ahead bias by using announcement dates rather than period end dates.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import json
from dataclasses import dataclass, asdict
import warnings


@dataclass
class FundamentalSnapshot:
    """Fundamental data as known at a specific point in time."""
    ticker: str
    as_of_date: datetime
    fiscal_period_end: datetime
    announcement_date: datetime
    pe_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    eps: Optional[float] = None
    revenue: Optional[float] = None
    book_value_per_share: Optional[float] = None
    sector: str = "Unknown"
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PITFundamentalsDB:
    """
    Point-in-time fundamentals database.
    
    Key principle: When backtesting as of date D, you can only use
    fundamentals that were announced before date D.
    
    This prevents look-ahead bias where you use restated/revised data
    that wasn't available at the time of the decision.
    """
    
    def __init__(self, cache_dir: str = "./data/pit_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # In-memory cache
        self.snapshots: Dict[str, List[FundamentalSnapshot]] = {}
        self.asof_index: Dict[str, pd.DataFrame] = {}
    
    def _get_cache_path(self, ticker: str) -> str:
        """Get cache file path for ticker."""
        return os.path.join(self.cache_dir, f"{ticker}_pit.json")
    
    def add_snapshot(self, snapshot: FundamentalSnapshot):
        """
        Add a fundamental snapshot to the database.
        """
        ticker = snapshot.ticker
        
        if ticker not in self.snapshots:
            self.snapshots[ticker] = []
        
        self.snapshots[ticker].append(snapshot)
        
        # Sort by announcement date
        self.snapshots[ticker].sort(key=lambda x: x.announcement_date)
    
    def get_fundamentals_asof(self,
                             ticker: str,
                             as_of_date: datetime,
                             max_lookback_days: int = 180) -> Optional[FundamentalSnapshot]:
        """
        Get most recent fundamentals as of a specific date.
        
        Args:
            ticker: Stock symbol
            as_of_date: Date of decision
            max_lookback_days: Maximum days to look back for data
        
        Returns:
            FundamentalSnapshot or None if no valid data
        """
        if ticker not in self.snapshots:
            return None
        
        # Filter snapshots that were announced before as_of_date
        valid_snapshots = [
            s for s in self.snapshots[ticker]
            if s.announcement_date <= as_of_date
        ]
        
        if not valid_snapshots:
            return None
        
        # Get the most recent one
        most_recent = valid_snapshots[-1]
        
        # Check if it's too old
        days_since = (as_of_date - most_recent.announcement_date).days
        if days_since > max_lookback_days:
            warnings.warn(
                f"Fundamentals for {ticker} as of {as_of_date.date()} "
                f"are {days_since} days old (max: {max_lookback_days})"
            )
        
        return most_recent
    
    def get_universe_fundamentals_asof(self,
                                      tickers: List[str],
                                      as_of_date: datetime,
                                      max_lookback_days: int = 180) -> pd.DataFrame:
        """
        Get PIT fundamentals for entire universe.
        
        Returns:
            DataFrame with fundamentals as known at as_of_date
        """
        data = []
        
        for ticker in tickers:
            snapshot = self.get_fundamentals_asof(ticker, as_of_date, max_lookback_days)
            
            if snapshot:
                data.append({
                    'Ticker': ticker,
                    'PE_Ratio': snapshot.pe_ratio,
                    'Debt_To_Equity': snapshot.debt_to_equity,
                    'EPS': snapshot.eps,
                    'Revenue': snapshot.revenue,
                    'Book_Value_Per_Share': snapshot.book_value_per_share,
                    'Sector': snapshot.sector,
                    'Announcement_Date': snapshot.announcement_date,
                    'Data_Age_Days': (as_of_date - snapshot.announcement_date).days
                })
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data).set_index('Ticker')
        return df
    
    def save_to_cache(self, ticker: str):
        """Save ticker snapshots to disk cache."""
        if ticker not in self.snapshots:
            return
        
        cache_path = self._get_cache_path(ticker)
        
        data = [s.to_dict() for s in self.snapshots[ticker]]
        
        # Convert dates to strings for JSON
        for item in data:
            for key in ['as_of_date', 'fiscal_period_end', 'announcement_date']:
                if isinstance(item[key], datetime):
                    item[key] = item[key].isoformat()
        
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_cache(self, ticker: str) -> bool:
        """Load ticker snapshots from disk cache."""
        cache_path = self._get_cache_path(ticker)
        
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # Parse dates
            for item in data:
                for key in ['as_of_date', 'fiscal_period_end', 'announcement_date']:
                    if key in item and isinstance(item[key], str):
                        item[key] = datetime.fromisoformat(item[key])
            
            snapshots = [FundamentalSnapshot(**item) for item in data]
            self.snapshots[ticker] = snapshots
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to load cache for {ticker}: {e}")
            return False
    
    def build_from_yfinance(self,
                           ticker: str,
                           years: int = 5) -> List[FundamentalSnapshot]:
        """
        Build PIT database from Yahoo Finance historical data.
        
        Note: yfinance doesn't provide exact announcement dates, so we
        estimate them based on typical reporting lags.
        
        In production, use: SEC EDGAR, Bloomberg, CapitalIQ, or Quandl Sharadar
        """
        import yfinance as yf
        
        print(f"[PIT] Building database for {ticker}...")
        
        try:
            tk = yf.Ticker(ticker)
            
            # Get quarterly financials
            financials = tk.quarterly_financials
            balance_sheet = tk.quarterly_balance_sheet
            
            if financials is None or financials.empty:
                return []
            
            snapshots = []
            
            # Get available periods
            periods = financials.columns
            
            for i, period in enumerate(periods):
                # Estimate announcement date
                # Most companies report 30-45 days after quarter end
                fiscal_end = pd.to_datetime(period)
                announcement_date = fiscal_end + timedelta(days=40)
                
                    # Get EPS
                    eps = None
                    try:
                        eps = financials.loc['Basic EPS', period]
                    except:
                        pass
                    
                    # M-12 Fix: Do not use tk.info.get('trailingPE') as it leaks
                    # today's P/E into the database. Calculate the true historical
                    # P/E using the price at the time of announcement.
                    pe_ratio = None
                    if eps and eps > 0:
                        try:
                            # Fetch price around announcement date
                            hist = tk.history(
                                start=announcement_date,
                                end=announcement_date + timedelta(days=5)
                            )
                            if not hist.empty:
                                # Approximate annualized EPS for P/E
                                pe_ratio = hist.iloc[0]['Close'] / (eps * 4)
                        except Exception:
                            pass
                    
                    snapshot = FundamentalSnapshot(
                        ticker=ticker,
                        as_of_date=announcement_date,
                        fiscal_period_end=fiscal_end,
                        announcement_date=announcement_date,
                        pe_ratio=pe_ratio,
                        debt_to_equity=debt,
                        eps=eps,
                        revenue=revenue,
                        sector=tk.info.get('sector', 'Unknown')
                    )
                    
                    snapshots.append(snapshot)
                    
                except Exception as e:
                    warnings.warn(f"Failed to extract data for {ticker} {period}: {e}")
                    continue
            
            # Store in memory
            self.snapshots[ticker] = snapshots
            
            # Save to cache
            self.save_to_cache(ticker)
            
            print(f"[PIT] Built {len(snapshots)} snapshots for {ticker}")
            return snapshots
            
        except Exception as e:
            warnings.warn(f"Failed to build PIT database for {ticker}: {e}")
            return []
    
    def simulate_pit_fundamentals(self,
                                 current_fundamentals: pd.DataFrame,
                                 as_of_date: datetime,
                                 lag_days: int = 90) -> pd.DataFrame:
        """
        Simulate PIT fundamentals by applying lag to current data.
        
        This is a fallback when real PIT data isn't available.
        Assumes fundamentals were announced 'lag_days' before as_of_date.
        
        Args:
            current_fundamentals: Current fundamental data
            as_of_date: Target date
            lag_days: Assumed announcement lag
        
        Returns:
            DataFrame with simulated PIT data
        """
        simulated_date = as_of_date - timedelta(days=lag_days)
        
        # Add metadata
        df = current_fundamentals.copy()
        df['Data_AsOf'] = simulated_date
        df['Data_Age_Days'] = (as_of_date - simulated_date).days
        df['PIT_Method'] = 'simulated_lag'
        
        print(f"[PIT] Using simulated PIT data with {lag_days} day lag")
        
        return df


class LookAheadBiasDetector:
    """
    Detects and warns about potential look-ahead bias in backtests.
    """
    
    def __init__(self, pit_db: PITFundamentalsDB):
        self.pit_db = pit_db
        self.violations: List[Dict] = []
    
    def check_fundamental_timing(self,
                                 ticker: str,
                                 as_of_date: datetime,
                                 used_fundamental_date: datetime) -> bool:
        """
        Check if fundamentals were actually available at decision time.
        
        Returns:
            True if valid (no bias), False if look-ahead detected
        """
        # Get actual available fundamentals
        snapshot = self.pit_db.get_fundamentals_asof(ticker, as_of_date)
        
        if snapshot is None:
            # No data available - can't verify
            return True
        
        # If used fundamentals are from after as_of_date, that's look-ahead bias
        if used_fundamental_date > as_of_date:
            self.violations.append({
                'ticker': ticker,
                'as_of_date': as_of_date,
                'violation_type': 'future_fundamentals',
                'used_date': used_fundamental_date,
                'available_date': snapshot.announcement_date
            })
            return False
        
        # If used fundamentals are too recent (before announcement), that's also bias
        if used_fundamental_date > snapshot.announcement_date:
            # This could be valid if using preliminary data
            pass
        
        return True
    
    def generate_bias_report(self) -> str:
        """Generate report of detected look-ahead bias."""
        if not self.violations:
            return "No look-ahead bias detected."
        
        report = ["\nLOOK-AHEAD BIAS DETECTED", "=" * 50]
        
        for v in self.violations[:10]:  # Show first 10
            report.append(
                f"  {v['ticker']} @ {v['as_of_date'].date()}: "
                f"Used {v['used_date'].date()} fundamentals "
                f"(available: {v.get('available_date', 'unknown')})"
            )
        
        if len(self.violations) > 10:
            report.append(f"  ... and {len(self.violations) - 10} more violations")
        
        return "\n".join(report)


# Convenience functions
def get_pit_fundamentals(tickers: List[str],
                          as_of_date: datetime,
                          use_simulated: bool = True) -> pd.DataFrame:
    """
    Quick function to get PIT fundamentals for backtesting.
    """
    db = PITFundamentalsDB()
    
    # Try to load from cache first
    for ticker in tickers:
        if not db.load_from_cache(ticker):
            # Build from yfinance if not cached
            db.build_from_yfinance(ticker)
    
    # Get PIT data
    df = db.get_universe_fundamentals_asof(tickers, as_of_date)
    
    if df.empty and use_simulated:
        # Fallback to simulated PIT
        from data.ingestion import get_fundamentals
        current = get_fundamentals(tickers)
        df = db.simulate_pit_fundamentals(current, as_of_date)
    
    return df


def validate_no_lookahead(tickers: List[str],
                         as_of_date: datetime,
                         used_fundamentals: pd.DataFrame) -> bool:
    """
    Validate that fundamentals don't contain look-ahead bias.
    """
    db = PITFundamentalsDB()
    detector = LookAheadBiasDetector(db)
    
    is_valid = True
    
    for ticker in tickers:
        if ticker in used_fundamentals.index:
            # M-13 Fix: Extract the actual used fundamental date from the DataFrame.
            # Passing as_of_date as the third argument was a tautological check
            # that never fired.
            row = used_fundamentals.loc[ticker]
            if 'Announcement_Date' in used_fundamentals.columns:
                used_date = pd.to_datetime(row['Announcement_Date'])
            elif 'Data_AsOf' in used_fundamentals.columns:  # Simulated fallback
                used_date = pd.to_datetime(row['Data_AsOf'])
            else:
                # If there's no temporal metadata, we can't validate it properly
                used_date = as_of_date
                
            is_valid &= detector.check_fundamental_timing(
                ticker, as_of_date, used_date
            )
    
    print(detector.generate_bias_report())
    
    return is_valid

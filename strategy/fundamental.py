import pandas as pd
from typing import List

def get_safe_universe(fundamentals_df: pd.DataFrame, drop_bottom_pct: float = 0.25) -> List[str]:
    """
    Takes fundamental data and filters out the bottom quartile of companies
    based on their sector-relative Z-scores for PE and Debt-to-Equity.
    """
    df = fundamentals_df.copy()
    
    if df.empty:
        return []
        
    # We want LOW P/E and LOW Debt/Equity
    # Fill missing values with median of their respective sectors so they aren't auto-penalized
    df['PE_Ratio'] = df.groupby('Sector')['PE_Ratio'].transform(lambda x: x.fillna(x.median()))
    df['Debt_To_Equity'] = df.groupby('Sector')['Debt_To_Equity'].transform(lambda x: x.fillna(x.median()))
    
    # Calculate Z-Scores per sector
    # Z = (X - Mean) / Std
    df['PE_Z'] = df.groupby('Sector')['PE_Ratio'].transform(
        lambda x: (x - x.mean()) / (x.std() if len(x)>1 else 1)
    )
    df['Debt_Z'] = df.groupby('Sector')['Debt_To_Equity'].transform(
        lambda x: (x - x.mean()) / (x.std() if len(x)>1 else 1)
    )
    
    # Calculate a combined anomaly score. 
    # High combined Z-score = High PE + High Debt = Mathematically dangerous.
    # Handle NaNs from math errors (like 0 std dev) safely
    df['PE_Z'] = df['PE_Z'].fillna(0)
    df['Debt_Z'] = df['Debt_Z'].fillna(0)
    
    df['Risk_Score'] = df['PE_Z'] + df['Debt_Z']
    
    # We want to identify the safe ones -> keep top (1 - pct), meaning we drop highest Risk_Score
    def filter_safe(sector_group):
        # Determine the cutoff value for the *highest risk* (e.g. top 25% risk)
        cutoff_rank = int(len(sector_group) * (1 - drop_bottom_pct))
        # Sort ascending by Risk (Lowest risk first)
        sorted_group = sector_group.sort_values(by='Risk_Score', ascending=True)
        # Keep only the rows up to the cutoff
        # If there are fewer than 4 stocks in a sector, we might keep all or most, handle via max(1)
        safe_count = max(1, cutoff_rank)
        return sorted_group.head(safe_count)
        
    safe_df = df.groupby('Sector', group_keys=False).apply(filter_safe)
    
    return safe_df.index.tolist()

if __name__ == "__main__":
    # Test stub
    data = pd.DataFrame({
        'Ticker': ['A', 'B', 'C', 'D', 'E'],
        'Sector': ['Tech', 'Tech', 'Tech', 'Tech', 'Energy'],
        'PE_Ratio': [10.0, 50.0, 15.0, 100.0, 5.0],
        'Debt_To_Equity': [1.0, 2.0, 1.5, 5.0, 0.5]
    }).set_index('Ticker')
    
    print("Raw Fundamentals:")
    print(data)
    
    safe_tickers = get_safe_universe(data, drop_bottom_pct=0.25)
    print(f"\nSafe Universe (dropped bottom 25% highest risk/val): {safe_tickers}")

import sys
import pandas as pd

from data.ingestion import get_universe, get_fundamentals, get_historical_ohlcv
from strategy.fundamental import get_safe_universe
from strategy.kronos_alpha import KronosAlphaGenerator
from optim.portfolio import construct_portfolio

def main():
    print("=== Kronos Fundamental Factor Engine Pipeline Starting ===\n")
    
    print("1. Fetching Investment Universe...")
    universe = get_universe()
    print(f"Loaded {len(universe)} tickers.\n")
    
    print("2. Fetching Fundamental Risk Data...")
    fundamentals_df = get_fundamentals(universe)
    print("Fundamentals Sample:")
    print(fundamentals_df.head(3), "\n")
    
    print("3. Executing Cross-Sectional Factor Neutralization...")
    # Drop the bottom 25% highest-risk companies in each sector
    safe_tickers = get_safe_universe(fundamentals_df, drop_bottom_pct=0.25)
    print(f"Safe Universe contains {len(safe_tickers)} tickers (Filtered out high risk).\n")
    
    # We only fetch OHLCV for the safe tickers to save bandwidth
    print("\n4. Fetching Historical Price Action (OHLCV) for Safe Universe...")
    ohlcv_dict = get_historical_ohlcv(safe_tickers, lookback_days=400)
    print(f"Loaded {len(ohlcv_dict)} valid historical price matrices.\n")
    
    print("5. Generating Alpha Signals with Kronos...")
    # Initialize generator using small model for performance
    generator = KronosAlphaGenerator(model_size="small", max_context=512)
    signal_df = generator.generate_signals(ohlcv_dict, lookback=400, pred_len=5)
    
    print("\nTop 5 Kronos Alpha Signals:")
    print(signal_df.head(), "\n")
    
    print("6. Executing Capital Optimization (Mean-Variance & L2 Penalties)...")
    portfolio = construct_portfolio(
        signals_df=signal_df, 
        fundamentals_df=fundamentals_df, 
        ohlcv_dict=ohlcv_dict,
        risk_aversion=1.0,
        l2_penalty=0.5
    )
    
    print("\n" + "="*50)
    print("FINAL CONSTRUCTED PORTFOLIO (Target Allocations)")
    print("="*50)
    print(portfolio)
    print("="*50)
    
    total_capital = portfolio['Weight'].sum()
    print(f"Total Allocated Capital: {total_capital:.2%}")
    print(f"Cash Drag: {1.0 - total_capital:.2%}")

if __name__ == "__main__":
    main()

import sys
import pandas as pd

from data.ingestion import get_universe, get_fundamentals, get_historical_ohlcv
from strategy.fundamental import get_safe_universe
from strategy.kronos_alpha import KronosAlphaGenerator
try:
    from signals.finbert_sentiment import NewsSentimentFetcher
    from signals.macro_regime import MacroRegimeDetector, get_current_macro_regime
    from signals.ensemble import AdaptiveEnsemble, get_top_ensemble_picks
    from risk.regime_detector import UnifiedRegimeDetector
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    print("[Note] Ensemble modules not available. Using Kronos-only signals.")
from optim.portfolio import construct_portfolio

def main(use_ensemble=True, use_kelly=True):
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
    
    # === ENSEMBLE SIGNAL GENERATION ===
    final_signals = signal_df
    if use_ensemble and ENSEMBLE_AVAILABLE:
        print("6. Generating Ensemble Signals (Kronos + FinBERT + Macro)...")
        
        # Fetch sentiment signals
        print("   [Ensemble] Fetching FinBERT sentiment signals...")
        sentiment_fetcher = NewsSentimentFetcher()
        sentiment_df = sentiment_fetcher.get_universe_sentiment(safe_tickers, lookback_days=7)
        
        # Fetch macro regime
        print("   [Ensemble] Detecting macro regime...")
        macro_detector = MacroRegimeDetector()
        regime_state = macro_detector.get_current_regime()
        print(f"   [Ensemble] Current regime: {regime_state.primary_regime.value}")
        
        # Combine signals
        print("   [Ensemble] Combining signals with adaptive weighting...")
        ensemble = AdaptiveEnsemble(
            kronos_weight=0.5,
            sentiment_weight=0.3,
            macro_weight=0.2
        )
        macro_tilts = macro_detector.get_regime_factor_tilts(regime_state.primary_regime)
        ensemble_df = ensemble.combine_signals(signal_df, sentiment_df, macro_tilts)
        
        print("\nTop 5 Ensemble Signals:")
        print(ensemble_df.head(), "\n")
        
        # Convert ensemble scores to signal format for portfolio construction
        final_signals = pd.DataFrame({
            'Predicted_Return': ensemble_df['Raw_Predicted_Return'],
            'Confidence': ensemble_df['Confidence']
        })
    else:
        print("6. Skipping ensemble (using Kronos-only signals)...")
        if use_ensemble and not ENSEMBLE_AVAILABLE:
            print("   [Note] Install transformers: pip install transformers")
    
    print(f"7. Executing Capital Optimization ({'Kelly Criterion' if use_kelly else 'Mean-Variance'})...")
    portfolio = construct_portfolio(
        signals_df=final_signals, 
        fundamentals_df=fundamentals_df, 
        ohlcv_dict=ohlcv_dict,
        risk_aversion=1.0,
        l2_penalty=0.5,
        use_kelly_criterion=use_kelly,
        kelly_fraction=0.25
    )
    
    print("\n" + "="*50)
    print("FINAL CONSTRUCTED PORTFOLIO (Target Allocations)")
    print("="*50)
    print(portfolio)
    print("="*50)
    
    gross_exposure = portfolio['Weight'].abs().sum()
    net_exposure = portfolio['Weight'].sum()
    print(f"Gross Exposure (Allocated Capital): {gross_exposure:.2%}")
    print(f"Net Exposure (Market Neutrality): {net_exposure:.2%}")
    print(f"Cash Drag: {1.0 - gross_exposure:.2%}")
    
    # === REGIME-BASED RISK WARNINGS ===
    if ENSEMBLE_AVAILABLE:
        print("\n8. Risk Assessment...")
        try:
            regime_detector = UnifiedRegimeDetector()
            regime_metrics = regime_detector.analyze_current_regime()
            adjustments = regime_detector.get_risk_adjustment_factors(regime_metrics)
            
            print(f"   VIX: {adjustments['vix']:.1f} ({adjustments['vol_regime']})")
            print(f"   Avg Correlation: {adjustments['avg_correlation']:.2f}")
            print(f"   Recommended Exposure: {adjustments['exposure_multiplier']:.0%}")
            
            should_halt, reason = regime_detector.should_halt_trading()
            if should_halt:
                print(f"   [WARNING] Trading halt recommended: {reason}")
        except Exception as e:
            print(f"   [Note] Risk assessment unavailable: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-ensemble', action='store_true', help='Disable multi-model ensemble')
    parser.add_argument('--no-kelly', action='store_true', help='Use mean-variance instead of Kelly')
    args = parser.parse_args()
    
    main(use_ensemble=not args.no_ensemble, use_kelly=not args.no_kelly)

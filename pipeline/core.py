"""Shared Kronos trading pipeline — used by main.py and production runner."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from data.ingestion import get_fundamentals, get_historical_ohlcv, get_universe

try:
    from research.ic_tracker import record_predictions, settle_predictions
    IC_TRACKER_AVAILABLE = True
except ImportError:
    IC_TRACKER_AVAILABLE = False
from optim.portfolio import construct_portfolio
from strategy.fundamental import get_safe_universe
from strategy.kronos_alpha import KronosAlphaGenerator

try:
    from signals.ensemble import AdaptiveEnsemble
    from signals.finbert_sentiment import NewsSentimentFetcher
    from signals.macro_regime import MacroRegimeDetector
    from risk.regime_detector import UnifiedRegimeDetector

    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False


@dataclass
class PipelineResult:
    portfolio: pd.DataFrame
    signals: pd.DataFrame
    kronos_signals: pd.DataFrame
    fundamentals_df: pd.DataFrame
    ohlcv_dict: Dict[str, pd.DataFrame]
    safe_tickers: List[str]
    should_halt: bool = False
    halt_reason: str = ""
    risk_adjustments: Dict = field(default_factory=dict)
    macro_regime: Optional[str] = None


def run_pipeline(
    *,
    as_of_date: Optional[str] = None,
    use_ensemble: bool = True,
    use_kelly: bool = True,
    universe_limit: Optional[int] = None,
    lookback_days: int = 400,
    model_size: Optional[str] = None,
    verbose: bool = True,
) -> PipelineResult:
    """
    Run the full signal → portfolio pipeline.

    Args:
        as_of_date: Point-in-time date (YYYY-MM-DD) for backtest; None for live.
        use_ensemble: Combine Kronos + FinBERT + macro when available.
        use_kelly: Use Kelly criterion in portfolio construction.
        universe_limit: Cap tickers after fundamental filter (dev/GPU limits).
        lookback_days: OHLCV history length for Kronos inference.
        model_size: Kronos model size override (default from MODEL_SIZE env or 'small').
        verbose: Print progress to stdout.
    """
    model_size = model_size or os.getenv('MODEL_SIZE', 'small')

    if verbose:
        print("=== Kronos Pipeline ===\n")

    universe = get_universe()
    if verbose:
        print(f"1. Universe: {len(universe)} tickers")

    fundamentals_df = get_fundamentals(universe, as_of_date=as_of_date)
    safe_tickers = get_safe_universe(fundamentals_df, drop_bottom_pct=0.25)
    if universe_limit:
        safe_tickers = safe_tickers[:universe_limit]
    if verbose:
        print(f"2. Safe universe: {len(safe_tickers)} tickers")

    ohlcv_dict = get_historical_ohlcv(
        safe_tickers, as_of_date=as_of_date, lookback_days=lookback_days
    )
    if not ohlcv_dict:
        raise RuntimeError("No OHLCV data fetched — cannot run pipeline")
    if verbose:
        print(f"3. OHLCV loaded: {len(ohlcv_dict)} tickers")

    # Settle any pending IC predictions before generating new ones
    if IC_TRACKER_AVAILABLE:
        try:
            settle_predictions(ohlcv_dict)
        except Exception as exc:
            if verbose:
                print(f"   [IC Tracker] Settlement skipped: {exc}")

    generator = KronosAlphaGenerator(model_size=model_size, max_context=512)
    kronos_signals = generator.generate_signals(ohlcv_dict, lookback=lookback_days, pred_len=5)
    if kronos_signals.empty:
        raise RuntimeError("Kronos produced no signals")
    if verbose:
        print(f"4. Kronos signals: {len(kronos_signals)} tickers")

    final_signals = kronos_signals
    macro_regime = None

    if use_ensemble and ENSEMBLE_AVAILABLE:
        if verbose:
            print("5. Ensemble signals (Kronos + FinBERT + macro)...")
        sentiment_fetcher = NewsSentimentFetcher()
        sentiment_df = sentiment_fetcher.get_universe_sentiment(
            list(kronos_signals.index), lookback_days=7
        )
        macro_detector = MacroRegimeDetector()
        regime_state = macro_detector.get_current_regime()
        macro_regime = regime_state.primary_regime.value
        if verbose:
            print(f"   Macro regime: {macro_regime}")

        ensemble = AdaptiveEnsemble(kronos_weight=0.5, sentiment_weight=0.3, macro_weight=0.2)
        macro_tilts = macro_detector.get_regime_factor_tilts(regime_state.primary_regime)
        ensemble_df = ensemble.combine_signals(kronos_signals, sentiment_df, macro_tilts)
        final_signals = pd.DataFrame({
            'Predicted_Return': ensemble_df['Raw_Predicted_Return'],
            'Confidence': ensemble_df['Confidence'],
        })
    elif verbose:
        print("5. Skipping ensemble (Kronos-only)")

    if verbose:
        opt_label = 'Kelly' if use_kelly else 'Mean-Variance'
        print(f"6. Portfolio optimization ({opt_label})...")

    portfolio = construct_portfolio(
        signals_df=final_signals,
        fundamentals_df=fundamentals_df,
        ohlcv_dict=ohlcv_dict,
        risk_aversion=1.0,
        l2_penalty=0.5,
        use_kelly_criterion=use_kelly,
        kelly_fraction=0.25,
    )
    if portfolio.empty:
        raise RuntimeError("Portfolio optimization returned empty result")

    should_halt = False
    halt_reason = ""
    risk_adjustments: Dict = {}

    if ENSEMBLE_AVAILABLE:
        try:
            regime_detector = UnifiedRegimeDetector()
            regime_metrics = regime_detector.analyze_current_regime()
            risk_adjustments = regime_detector.get_risk_adjustment_factors(regime_metrics)
            should_halt, halt_reason = regime_detector.should_halt_trading(regime_metrics)
            if verbose and should_halt:
                print(f"7. Trading halt: {halt_reason}")
        except Exception as exc:
            if verbose:
                print(f"7. Risk assessment unavailable: {exc}")

    result = PipelineResult(
        portfolio=portfolio,
        signals=final_signals,
        kronos_signals=kronos_signals,
        fundamentals_df=fundamentals_df,
        ohlcv_dict=ohlcv_dict,
        safe_tickers=safe_tickers,
        should_halt=should_halt,
        halt_reason=halt_reason,
        risk_adjustments=risk_adjustments,
        macro_regime=macro_regime,
    )

    # Record predictions for future IC settlement
    if IC_TRACKER_AVAILABLE:
        try:
            ensemble_df = final_signals if use_ensemble and ENSEMBLE_AVAILABLE else None
            record_predictions(kronos_signals, ensemble_df, as_of_date=as_of_date)
        except Exception as exc:
            if verbose:
                print(f"   [IC Tracker] Recording skipped: {exc}")

    return result


def portfolio_to_weights(portfolio: pd.DataFrame) -> Dict[str, float]:
    """Convert portfolio DataFrame to symbol → weight dict for execution."""
    if portfolio.empty:
        return {}
    return portfolio['Weight'].to_dict()

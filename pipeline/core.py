"""Shared Kronos trading pipeline — used by main.py and production runner."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from data.ingestion import get_fundamentals, get_historical_ohlcv, get_universe

try:
    from research.ic_tracker import load_ic_history_for_ensemble, record_predictions, settle_predictions
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


def _ohlcv_to_returns(
    ohlcv_dict: Dict[str, pd.DataFrame],
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    """Build aligned daily return matrix for regime detection."""
    series = {}
    cutoff = pd.Timestamp(as_of_date) if as_of_date else None
    for ticker, df in ohlcv_dict.items():
        if "timestamps" not in df.columns or "close" not in df.columns:
            continue
        hist = df.sort_values("timestamps")
        if cutoff is not None:
            hist = hist[hist["timestamps"] <= cutoff]
        if len(hist) < 22:
            continue
        close = hist.set_index("timestamps")["close"]
        series[ticker] = close.pct_change().dropna()
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).ffill().dropna()


def run_pipeline(
    *,
    as_of_date: Optional[str] = None,
    use_ensemble: bool = True,
    use_kelly: bool = True,
    universe_limit: Optional[int] = None,
    lookback_days: int = 400,
    model_size: Optional[str] = None,
    verbose: bool = True,
    ohlcv_override: Optional[Dict[str, pd.DataFrame]] = None,
    allow_universe_fallback: bool = False,
    prev_weights: Optional[Dict[str, float]] = None,
    long_only: bool = False,
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
        ohlcv_override: Pre-fetched OHLCV (backtest). Skips network fetch when provided.
        allow_universe_fallback: Use today's universe if PIT snapshot missing.
        prev_weights: Prior portfolio weights for turnover penalty in optimizer.
        long_only: Drop shorts and renormalize longs (recommended for paper).
    """
    model_size = model_size or os.getenv('MODEL_SIZE', 'small')

    if verbose:
        print("=== Kronos Pipeline ===\n")

    try:
        universe = get_universe(as_of_date=as_of_date)
    except ValueError as exc:
        if allow_universe_fallback:
            if verbose:
                print(f"   [WARNING] PIT universe missing for {as_of_date}; using current constituents.")
            universe = get_universe()
        else:
            raise exc
    if verbose:
        print(f"1. Universe: {len(universe)} tickers")

    fundamentals_df = get_fundamentals(universe, as_of_date=as_of_date)
    safe_tickers = get_safe_universe(fundamentals_df, drop_bottom_pct=0.25)
    if universe_limit:
        safe_tickers = safe_tickers[:universe_limit]
    if verbose:
        print(f"2. Safe universe: {len(safe_tickers)} tickers")

    if ohlcv_override is not None:
        available = set(ohlcv_override.keys())
        safe_tickers = [t for t in safe_tickers if t in available]
        if not safe_tickers:
            safe_tickers = sorted(available)[:universe_limit] if universe_limit else sorted(available)
            if verbose:
                print(f"   [WARNING] Fundamental filter had no OHLCV overlap; using {len(safe_tickers)} injected tickers.")
        ohlcv_dict = {t: ohlcv_override[t] for t in safe_tickers}
        if verbose:
            print(f"3. OHLCV (injected): {len(ohlcv_dict)} tickers")
    else:
        ohlcv_dict = get_historical_ohlcv(
            safe_tickers, as_of_date=as_of_date, lookback_days=lookback_days
        )
        if verbose:
            print(f"3. OHLCV loaded: {len(ohlcv_dict)} tickers. Applying Corporate Actions...")
    if not ohlcv_dict:
        raise RuntimeError("No OHLCV data fetched — cannot run pipeline")

    if ohlcv_override is None:
        try:
            from data.corporate_actions import adjust_universe_prices
            from datetime import datetime, timedelta
            target_ts = pd.Timestamp(as_of_date).to_pydatetime() if as_of_date else datetime.now()
            start_ts = target_ts - timedelta(days=lookback_days)

            ohlcv_dict = adjust_universe_prices(ohlcv_dict, start_ts, target_ts)
            if verbose:
                print("   Corporate Actions (Splits/Dividends) applied successfully.")
        except Exception as e:
            if verbose:
                print(f"   [WARNING] Failed to apply corporate actions: {e}")
    elif verbose:
        print("   Corporate actions skipped (injected OHLCV from backtest pre-fetch).")

    # Settle any pending IC predictions before generating new ones
    if IC_TRACKER_AVAILABLE:
        try:
            settle_predictions(ohlcv_dict, reference_date=as_of_date)
        except Exception as exc:
            if verbose:
                print(f"   [IC Tracker] Settlement skipped: {exc}")

    generator = KronosAlphaGenerator(model_size=model_size, max_context=512)
    kronos_signals = generator.generate_signals(
        ohlcv_dict, lookback=lookback_days, pred_len=5, as_of_date=as_of_date
    )
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
            list(kronos_signals.index), lookback_days=7, as_of_date=as_of_date
        )
        macro_detector = MacroRegimeDetector()
        regime_state = macro_detector.get_current_regime(as_of_date=as_of_date)
        macro_regime = regime_state.primary_regime.value
        if verbose:
            print(f"   Macro regime: {macro_regime}")

        ensemble = AdaptiveEnsemble(kronos_weight=0.5, sentiment_weight=0.3, macro_weight=0.2)
        if IC_TRACKER_AVAILABLE:
            ensemble.adapt_weights(load_ic_history_for_ensemble())
        else:
            ensemble.adapt_weights()
        if verbose:
            print(f"   Ensemble weights: kronos={ensemble.current_weights['kronos']:.2f} "
                  f"sentiment={ensemble.current_weights['sentiment']:.2f} "
                  f"macro={ensemble.current_weights['macro']:.2f}")
        macro_tilts = macro_detector.get_regime_factor_tilts(regime_state.primary_regime)
        ensemble_df = ensemble.combine_signals(kronos_signals, sentiment_df, macro_tilts)
        final_signals = pd.DataFrame({
            'Predicted_Return': ensemble_df['Raw_Predicted_Return'],
            'Confidence': ensemble_df['Confidence'],
        })
    elif verbose:
        print("5. Skipping ensemble (Kronos-only)")

    should_halt = False
    halt_reason = ""
    risk_adjustments: Dict = {}
    exposure_multiplier = 1.0

    if ENSEMBLE_AVAILABLE:
        try:
            regime_detector = UnifiedRegimeDetector()
            as_of_dt = pd.Timestamp(as_of_date).to_pydatetime() if as_of_date else None
            returns_df = _ohlcv_to_returns(ohlcv_dict, as_of_date=as_of_date)
            regime_metrics = regime_detector.analyze_current_regime(
                returns_df=returns_df if not returns_df.empty else None,
                as_of_date=as_of_dt,
            )
            risk_adjustments = regime_detector.get_risk_adjustment_factors(regime_metrics)
            exposure_multiplier = float(risk_adjustments.get("exposure_multiplier", 1.0))
            should_halt, halt_reason = regime_detector.should_halt_trading(regime_metrics)
            if verbose:
                print(f"6. Risk regime: exposure={exposure_multiplier:.2f}")
            if verbose and should_halt:
                print(f"   Trading halt: {halt_reason}")
        except Exception as exc:
            if verbose:
                print(f"6. Risk assessment unavailable: {exc}")

    if verbose:
        opt_label = 'Kelly' if use_kelly else 'Mean-Variance'
        print(f"7. Portfolio optimization ({opt_label})...")

    portfolio = construct_portfolio(
        signals_df=final_signals,
        fundamentals_df=fundamentals_df,
        ohlcv_dict=ohlcv_dict,
        risk_aversion=1.0,
        l2_penalty=0.5,
        prev_weights=prev_weights,
        use_kelly_criterion=use_kelly,
        kelly_fraction=0.25,
        exposure_multiplier=exposure_multiplier,
    )
    if portfolio.empty:
        raise RuntimeError("Portfolio optimization returned empty result")

    if long_only:
        portfolio = apply_long_only_portfolio(portfolio)
        if portfolio.empty:
            raise RuntimeError("Long-only filter removed all positions")
        if verbose:
            print(f"   Long-only book: {len(portfolio)} positions")

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


def apply_long_only_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Keep long positions only and renormalize to 100% gross (paper-safe)."""
    if portfolio.empty or "Weight" not in portfolio.columns:
        return portfolio

    out = portfolio[portfolio["Weight"] > 0.001].copy()
    if out.empty:
        return out

    gross = out["Weight"].sum()
    if gross > 0:
        out["Weight"] = out["Weight"] / gross
    if "Position" in out.columns:
        out["Position"] = "LONG"
    return out

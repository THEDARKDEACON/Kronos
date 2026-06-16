"""
Kronos Walk-Forward Backtest Simulator
=======================================
Runs a monthly walk-forward simulation over a date range, computing realised
(close-to-close) returns — NOT predicted returns — so the performance metrics
are grounded in actual price data.

Results are written to experiments/backtest_results.json for the dashboard.

Usage:
    python simulator/backtest.py                     # default: 2023-01-01 to today
    python simulator/backtest.py --start 2022-01-01 --end 2024-01-01 --universe 30
    python simulator/backtest.py --seed 42           # reproducible inference
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ingestion import get_universe, get_fundamentals, get_historical_ohlcv
from strategy.fundamental import get_safe_universe
from strategy.kronos_alpha import KronosAlphaGenerator
from optim.portfolio import construct_portfolio

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _monthly_dates(start: str, end: str) -> List[str]:
    """Return first-of-month dates between start and end (inclusive)."""
    cur = pd.Timestamp(start).replace(day=1)
    stop = pd.Timestamp(end)
    dates = []
    while cur <= stop:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur = (cur + pd.offsets.MonthBegin(1))
    return dates


def _realised_return(
    ohlcv_dict: Dict[str, pd.DataFrame],
    weights: pd.Series,
    as_of_date: str,
    horizon_trading_days: int = 21,  # trading days (≈1 month)
):
    """
    Compute weighted realised return over the next `horizon_trading_days` **trading
    days** from `as_of_date`. Returns (gross_return, min_intra_period_return).
    """
    as_of_ts = pd.Timestamp(as_of_date)
    daily_portfolio_returns = np.zeros(horizon_trading_days)
    
    for ticker, weight in weights.items():
        if ticker not in ohlcv_dict:
            continue
        df = ohlcv_dict[ticker]
        if "timestamps" not in df.columns or "close" not in df.columns or "open" not in df.columns:
            continue
            
        df = df.sort_values("timestamps").reset_index(drop=True)
        future = df[df["timestamps"] > as_of_ts].reset_index(drop=True)
        
        if len(future) == 0:
            continue
            
        p_start = float(future.iloc[0]["open"]) # Issue 5: execution at open
        
        if p_start <= 0:
            continue
            
        # Issue 6: Delisting trap
        # If a stock's data drops out prematurely (less than the full horizon)
        # we assume it went bankrupt (100% loss) for the missing days.
        actual_days = min(horizon_trading_days, len(future))
        for i in range(actual_days):
            p_current = float(future.iloc[i]["close"])
            daily_portfolio_returns[i] += weight * ((p_current - p_start) / p_start)
            
        if actual_days < horizon_trading_days:
            for i in range(actual_days, horizon_trading_days):
                daily_portfolio_returns[i] += weight * -1.0  # Bankrupt assumption

    gross_ret = float(daily_portfolio_returns[-1])
    min_intra_ret = float(np.min(daily_portfolio_returns))
    
    return gross_ret, min_intra_ret


class WalkForwardBacktest:
    def __init__(
        self,
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        transaction_cost_bps: float = 5.0,
        max_drawdown_pct: float = 20.0,
        universe_limit: int = 20,  # Default small enough to run in minutes
        seed: Optional[int] = None,
        strict_pit: bool = True,
    ):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.transaction_cost = transaction_cost_bps / 10_000
        self.max_drawdown = max_drawdown_pct / 100
        self.universe_limit = universe_limit
        self.strict_pit = strict_pit

        # M-7: Set random seeds for reproducibility of GPU inference
        if seed is not None:
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    # Deterministic CUDA ops (small speed penalty, required for reproducibility)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            except ImportError:
                pass
            print(f"[Seed] Random seed set to {seed}")

        # C-1: We still fetch the full OHLCV history once (it covers the entire
        # date range and is sliced per-step).  Universe and fundamentals are now
        # fetched INSIDE step() with the correct as_of_date to eliminate
        # survivorship bias.  We keep a "today" fetch here only to derive the
        # safe-ticker list for the OHLCV pre-fetch (we need to know which tickers
        # to pull data for before we start stepping).
        #
        # ⚠️  SURVIVORSHIP BIAS NOTE (H-3): The OHLCV pre-fetch still uses today's
        # ticker list as the candidate set.  Tickers that have been *added* to the
        # S&P 500 since start_date are included, but were not investable then.
        # The per-step fundamental filter (now PIT-correct) partially mitigates
        # this.  Full mitigation requires historical constituent snapshots — see
        # scripts/build_pit_universe.py.
        print("Fetching today's universe for OHLCV pre-fetch candidate list...")
        seed_universe = get_universe()                       # today's list, live mode
        seed_fundamentals = get_fundamentals(seed_universe)  # today's cache
        seed_tickers = get_safe_universe(seed_fundamentals, drop_bottom_pct=0.25)
        if self.universe_limit:
            seed_tickers = seed_tickers[: self.universe_limit]

        self.seed_tickers = seed_tickers
        print(f"OHLCV candidate universe: {len(seed_tickers)} tickers")

        # Calculate dynamic lookback based on start_date
        start_ts = pd.Timestamp(self.start_date)
        today_ts = pd.Timestamp(datetime.now().date())
        days_ago = max(0, (today_ts - start_ts).days)
        # 400 trading days ≈ 580 calendar days; add 60-day forward buffer
        lookback_needed = days_ago + 580 + 60

        print(f"Fetching full OHLCV history for {len(seed_tickers)} tickers "
              f"(Dynamic lookback: {lookback_needed} days)...")
        self.ohlcv_dict = get_historical_ohlcv(
            seed_tickers,
            as_of_date=None,           # fetch up to today
            lookback_days=lookback_needed,
        )
        print(f"OHLCV loaded: {len(self.ohlcv_dict)} tickers")

        self.generator = KronosAlphaGenerator(model_size="small", max_context=512)

        # State
        self.equity_curve: List[Dict] = []
        self.positions_history: List[Dict] = []
        self.current_value = 1.0
        self.peak_value = 1.0
        self.previous_weights: Optional[pd.Series] = None
        self.circuit_breaker = False

    def step(self, as_of_date: str) -> Optional[pd.DataFrame]:
        """Run one rebalance step. Returns portfolio or None on failure."""
        print(f"\n{'='*60}")
        print(f"Walk-Forward Step: {as_of_date}")
        print(f"{'='*60}")

        as_of_ts = pd.Timestamp(as_of_date)

        # C-1 FIX: Fetch fundamentals PIT for this specific rebalance date.
        # get_fundamentals with as_of_date will use the quarterly financials
        # that were available 90 days before as_of_date (see fetch_pit_fundamental),
        # preventing look-ahead bias in fundamental signals.
        # Universe PIT: if a historical parquet snapshot exists (built by
        # scripts/build_pit_universe.py), it will be loaded automatically.
        # Otherwise we fall back to today's universe with a visible warning.
        try:
            pit_universe = get_universe(as_of_date=as_of_date)
        except ValueError as e:
            if self.strict_pit:
                raise ValueError(
                    f"Strict PIT mode enabled: No historical universe snapshot for {as_of_date}. "
                    "Run scripts/build_pit_universe.py to generate it, or run with --allow-fallback."
                ) from e
            else:
                print(f"  [WARNING] No PIT universe snapshot for {as_of_date}. "
                      "Falling back to current constituents. Run scripts/build_pit_universe.py "
                      "to eliminate survivorship bias.")
                pit_universe = get_universe()

        pit_fundamentals = get_fundamentals(pit_universe, as_of_date=as_of_date)
        safe_tickers = get_safe_universe(pit_fundamentals, drop_bottom_pct=0.25)
        if self.universe_limit:
            safe_tickers = safe_tickers[: self.universe_limit]

        # Check if we have enough future data for a full period (prevents annualization distortion)
        future_lengths = [
            len(df[df["timestamps"] > as_of_ts])
            for t, df in self.ohlcv_dict.items() if t in safe_tickers and "timestamps" in df.columns
        ]
        max_future_len = max(future_lengths) if future_lengths else 0
        if max_future_len < 21:
            print(f"  [End of Data] Skipping {as_of_date}: Only {max_future_len}/21 future days available.")
            return None

        # Slice OHLCV to as_of_date (enforce no look-ahead)
        ohlcv_pit = {
            t: df[df["timestamps"] <= as_of_ts].reset_index(drop=True)
            for t, df in self.ohlcv_dict.items()
            if t in safe_tickers
            and "timestamps" in df.columns
            and len(df[df["timestamps"] <= as_of_ts]) >= 50
        }

        if not ohlcv_pit:
            print(f"  Not enough history for {as_of_date}, skipping.")
            return None

        signal_df = self.generator.generate_signals(ohlcv_pit, lookback=400, pred_len=5)
        if signal_df.empty:
            return None

        portfolio = construct_portfolio(
            signals_df=signal_df,
            fundamentals_df=pit_fundamentals,
            ohlcv_dict=ohlcv_pit,
            risk_aversion=1.0,
            l2_penalty=0.5,
        )
        if portfolio.empty:
            return None

        weights = portfolio["Weight"]

        # --- Transaction cost ---
        turnover_cost = 0.0
        if self.previous_weights is not None:
            all_t = set(weights.index) | set(self.previous_weights.index)
            turnover = sum(
                abs(weights.get(t, 0.0) - self.previous_weights.get(t, 0.0))
                for t in all_t
            )
        else:
            turnover = sum(abs(weights.get(t, 0.0)) for t in weights.index)
            
        turnover_cost = turnover * self.transaction_cost

        # --- Realised return over next ~1 month (21 trading days) ---
        # Passes full ohlcv_dict (not PIT-sliced) to see post-rebalance prices.
        gross_ret, min_intra_ret = _realised_return(
            self.ohlcv_dict, weights, as_of_date, horizon_trading_days=21
        )
        net_ret = gross_ret - turnover_cost

        # --- Update equity ---
        prev_val = self.current_value
        self.current_value *= 1 + net_ret
        if self.current_value > self.peak_value:
            self.peak_value = self.current_value
            
        end_of_month_dd = (self.peak_value - self.current_value) / self.peak_value
        intra_low_val = prev_val * (1 + min_intra_ret - turnover_cost)
        intra_dd = (self.peak_value - intra_low_val) / self.peak_value if self.peak_value > 0 else 0.0
        
        drawdown = max(end_of_month_dd, intra_dd)

        self.equity_curve.append(
            {
                "date": as_of_date,
                "value": round(self.current_value, 6),
                "gross_ret": round(gross_ret, 6),
                "net_ret": round(net_ret, 6),
                "drawdown": round(drawdown, 6),
                "turnover_cost": round(turnover_cost, 6),
                "n_positions": len(portfolio),
            }
        )

        self.positions_history.append(
            {
                "date": as_of_date,
                "positions": weights.to_dict(),
            }
        )
        self.previous_weights = weights

        print(f"  Gross Return: {gross_ret:.2%}  Net: {net_ret:.2%}  Drawdown: {drawdown:.2%}")
        print(f"  Portfolio Value: {self.current_value:.4f}  (Peak: {self.peak_value:.4f})")

        if drawdown > self.max_drawdown:
            self.circuit_breaker = True
            print(f"\n[🚨 CIRCUIT BREAKER] Drawdown {drawdown:.2%} > limit {self.max_drawdown:.2%}")

        return portfolio

    # ------------------------------------------------------------------
    def _compute_metrics(self) -> Dict:
        """Compute Sharpe, Calmar, max drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return {}

        rets = pd.Series([r["net_ret"] for r in self.equity_curve])
        values = pd.Series([r["value"] for r in self.equity_curve])
        dates = [r["date"] for r in self.equity_curve]

        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        n_periods = len(rets)
        periods_per_year = 12  # monthly rebalance
        ann_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        ann_vol = rets.std() * np.sqrt(periods_per_year)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        peak = values.cummax()
        drawdowns = (values - peak) / peak
        max_dd = float(drawdowns.min())
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        return {
            "start_date": dates[0],
            "end_date": dates[-1],
            "n_rebalances": n_periods,
            "total_return": round(total_return, 4),
            "ann_return": round(ann_return, 4),
            "ann_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "calmar_ratio": round(calmar, 4),
            "final_value": round(float(values.iloc[-1]), 6),
        }

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        dates = _monthly_dates(self.start_date, self.end_date)
        print(f"\nWalk-forward backtest: {dates[0]} → {dates[-1]} ({len(dates)} steps)")

        for date in dates:
            if self.circuit_breaker:
                print(f"\n[⚠️ CIRCUIT BREAKER] Halted before {date}")
                break
            self.step(date)

        metrics = self._compute_metrics()

        results = {
            "metrics": metrics,
            "equity_curve": self.equity_curve,
            "positions_history": [
                {"date": p["date"], "n_positions": len(p["positions"])}
                for p in self.positions_history
            ],
            "run_timestamp": datetime.now().isoformat(),
        }

        out_path = EXPERIMENTS_DIR / "backtest_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")
        print("\n=== Backtest Summary ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        return results


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos Walk-Forward Backtest")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--universe", type=int, default=None, help="Cap universe size")
    parser.add_argument("--max-drawdown", type=float, default=20.0, help="Circuit breaker %")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--allow-fallback", action="store_true", help="Allow fallback to modern universe if PIT missing")
    args = parser.parse_args()

    print("Initializing Kronos Walk-Forward Backtest...")
    sim = WalkForwardBacktest(
        start_date=args.start,
        end_date=args.end,
        universe_limit=args.universe,
        max_drawdown_pct=args.max_drawdown,
        seed=args.seed,
        strict_pit=not args.allow_fallback,
    )
    sim.run()

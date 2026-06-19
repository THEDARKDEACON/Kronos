"""
Kronos Walk-Forward Backtest Simulator
=======================================
Monthly walk-forward using the same ``pipeline.core.run_pipeline`` path as
production.  Realised returns (not predicted) ground performance metrics.

Results → ``experiments/backtest_results.json`` for the dashboard.

Usage:
    python simulator/backtest.py
    python simulator/backtest.py --start 2024-01-01 --end 2024-06-01 --universe 10
    python simulator/backtest.py --allow-fallback --no-ensemble
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ingestion import get_universe, get_fundamentals, get_historical_ohlcv
from strategy.fundamental import get_safe_universe
from pipeline.core import run_pipeline

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _monthly_dates(start: str, end: str) -> List[str]:
    cur = pd.Timestamp(start).replace(day=1)
    stop = pd.Timestamp(end)
    dates = []
    while cur <= stop:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur = cur + pd.offsets.MonthBegin(1)
    return dates


def _realised_return(
    ohlcv_dict: Dict[str, pd.DataFrame],
    weights: pd.Series,
    as_of_date: str,
    horizon_trading_days: int = 21,
):
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

        p_start = float(future.iloc[0]["open"])
        if p_start <= 0:
            continue

        actual_days = min(horizon_trading_days, len(future))
        for i in range(actual_days):
            p_current = float(future.iloc[i]["close"])
            daily_portfolio_returns[i] += weight * ((p_current - p_start) / p_start)

        if actual_days < horizon_trading_days:
            for i in range(actual_days, horizon_trading_days):
                daily_portfolio_returns[i] += weight * -1.0

    return float(daily_portfolio_returns[-1]), float(np.min(daily_portfolio_returns))


class WalkForwardBacktest:
    def __init__(
        self,
        start_date: str = "2023-01-01",
        end_date: Optional[str] = None,
        transaction_cost_bps: float = 5.0,
        max_drawdown_pct: float = 20.0,
        universe_limit: int = 20,
        seed: Optional[int] = None,
        strict_pit: bool = True,
        use_ensemble: bool = True,
        use_kelly: bool = True,
    ):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.transaction_cost = transaction_cost_bps / 10_000
        self.max_drawdown = max_drawdown_pct / 100
        self.universe_limit = universe_limit
        self.strict_pit = strict_pit
        self.use_ensemble = use_ensemble
        self.use_kelly = use_kelly

        if seed is not None:
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
            except ImportError:
                pass
            print(f"[Seed] Random seed set to {seed}")

        print("Fetching OHLCV candidate universe for pre-fetch...")
        try:
            seed_universe = (
                get_universe(as_of_date=self.start_date)
                if self.strict_pit
                else get_universe()
            )
        except ValueError:
            if self.strict_pit:
                raise ValueError(
                    f"No PIT universe for {self.start_date}. "
                    "Run: python scripts/build_pit_universe.py"
                ) from None
            seed_universe = get_universe()
        seed_fundamentals = get_fundamentals(
            seed_universe.iloc[: max(self.universe_limit * 10, self.universe_limit or 50)]
            if self.universe_limit and len(seed_universe) > self.universe_limit * 10
            else seed_universe,
            as_of_date=self.start_date if self.strict_pit else None,
        )
        seed_tickers = get_safe_universe(seed_fundamentals, drop_bottom_pct=0.25)
        if not seed_tickers and len(seed_universe) > 0:
            cap = self.universe_limit or len(seed_universe)
            seed_tickers = seed_universe["Ticker"].tolist()[:cap]
            print(
                f"   [WARNING] Fundamental filter returned 0 tickers; "
                f"using {len(seed_tickers)} universe names for OHLCV prefetch"
            )
        if self.universe_limit:
            seed_tickers = seed_tickers[: self.universe_limit]
        self.seed_tickers = seed_tickers
        print(f"OHLCV candidate universe: {len(seed_tickers)} tickers")

        start_ts = pd.Timestamp(self.start_date)
        today_ts = pd.Timestamp(datetime.now().date())
        days_ago = max(0, (today_ts - start_ts).days)
        lookback_needed = days_ago + 580 + 60

        print(f"Pre-fetching OHLCV ({lookback_needed} calendar days)...")
        self.ohlcv_dict = get_historical_ohlcv(
            seed_tickers,
            as_of_date=None,
            lookback_days=lookback_needed,
        )
        print(f"OHLCV loaded: {len(self.ohlcv_dict)} tickers")

        self.equity_curve: List[Dict] = []
        self.positions_history: List[Dict] = []
        self.current_value = 1.0
        self.peak_value = 1.0
        self.previous_weights: Optional[pd.Series] = None
        self.circuit_breaker = False

    def step(self, as_of_date: str) -> Optional[pd.DataFrame]:
        print(f"\n{'='*60}")
        print(f"Walk-Forward Step: {as_of_date}")
        print(f"{'='*60}")

        as_of_ts = pd.Timestamp(as_of_date)

        future_lengths = [
            len(df[df["timestamps"] > as_of_ts])
            for t, df in self.ohlcv_dict.items()
            if t in self.seed_tickers and "timestamps" in df.columns
        ]
        if not future_lengths or max(future_lengths) < 21:
            print(f"  [End of Data] Skipping {as_of_date}: insufficient future bars.")
            return None

        ohlcv_pit = {
            t: df[df["timestamps"] <= as_of_ts].reset_index(drop=True)
            for t, df in self.ohlcv_dict.items()
            if t in self.seed_tickers
            and "timestamps" in df.columns
            and len(df[df["timestamps"] <= as_of_ts]) >= 50
        }
        if not ohlcv_pit:
            print(f"  Not enough history for {as_of_date}, skipping.")
            return None

        try:
            prev_w = None
            if self.previous_weights is not None:
                prev_w = self.previous_weights.to_dict()
            result = run_pipeline(
                as_of_date=as_of_date,
                use_ensemble=self.use_ensemble,
                use_kelly=self.use_kelly,
                universe_limit=self.universe_limit,
                ohlcv_override=ohlcv_pit,
                allow_universe_fallback=not self.strict_pit,
                prev_weights=prev_w,
                verbose=False,
            )
        except Exception as exc:
            print(f"  Pipeline failed for {as_of_date}: {exc}")
            return None

        portfolio = result.portfolio
        if portfolio.empty:
            return None

        weights = portfolio["Weight"]

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

        gross_ret, min_intra_ret = _realised_return(
            self.ohlcv_dict, weights, as_of_date, horizon_trading_days=21
        )
        net_ret = gross_ret - turnover_cost

        prev_val = self.current_value
        self.current_value *= 1 + net_ret
        if self.current_value > self.peak_value:
            self.peak_value = self.current_value

        end_of_month_dd = (self.peak_value - self.current_value) / self.peak_value
        intra_low_val = prev_val * (1 + min_intra_ret - turnover_cost)
        intra_dd = (self.peak_value - intra_low_val) / self.peak_value if self.peak_value > 0 else 0.0
        drawdown = max(end_of_month_dd, intra_dd)

        self.equity_curve.append({
            "date": as_of_date,
            "value": round(self.current_value, 6),
            "gross_ret": round(gross_ret, 6),
            "net_ret": round(net_ret, 6),
            "drawdown": round(drawdown, 6),
            "turnover_cost": round(turnover_cost, 6),
            "n_positions": len(portfolio),
            "should_halt": result.should_halt,
        })
        self.positions_history.append({"date": as_of_date, "positions": weights.to_dict()})
        self.previous_weights = weights

        print(f"  Gross Return: {gross_ret:.2%}  Net: {net_ret:.2%}  Drawdown: {drawdown:.2%}")
        print(f"  Portfolio Value: {self.current_value:.4f}  (Peak: {self.peak_value:.4f})")

        if drawdown > self.max_drawdown:
            self.circuit_breaker = True
            print(f"\n[CIRCUIT BREAKER] Drawdown {drawdown:.2%} > limit {self.max_drawdown:.2%}")

        return portfolio

    def _compute_metrics(self) -> Dict:
        if len(self.equity_curve) < 2:
            return {}

        rets = pd.Series([r["net_ret"] for r in self.equity_curve])
        values = pd.Series([r["value"] for r in self.equity_curve])
        dates = [r["date"] for r in self.equity_curve]

        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        n_periods = len(rets)
        periods_per_year = 12
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

    def run(self) -> Dict:
        dates = _monthly_dates(self.start_date, self.end_date)
        print(f"\nWalk-forward backtest: {dates[0]} → {dates[-1]} ({len(dates)} steps)")

        for date in dates:
            if self.circuit_breaker:
                print(f"\n[CIRCUIT BREAKER] Halted before {date}")
                break
            try:
                self.step(date)
            except ValueError as exc:
                if self.strict_pit:
                    raise
                print(f"  Skipping {date}: {exc}")

        metrics = self._compute_metrics()
        results = {
            "metrics": metrics,
            "equity_curve": self.equity_curve,
            "positions_history": [
                {"date": p["date"], "n_positions": len(p["positions"])}
                for p in self.positions_history
            ],
            "run_timestamp": datetime.now().isoformat(),
            "config": {
                "use_ensemble": self.use_ensemble,
                "use_kelly": self.use_kelly,
                "universe_limit": self.universe_limit,
                "strict_pit": self.strict_pit,
            },
        }

        out_path = EXPERIMENTS_DIR / "backtest_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")
        if metrics:
            print("\n=== Backtest Summary ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kronos Walk-Forward Backtest")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--universe", type=int, default=20)
    parser.add_argument("--max-drawdown", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--no-kelly", action="store_true")
    args = parser.parse_args()

    sim = WalkForwardBacktest(
        start_date=args.start,
        end_date=args.end,
        universe_limit=args.universe,
        max_drawdown_pct=args.max_drawdown,
        seed=args.seed,
        strict_pit=not args.allow_fallback,
        use_ensemble=not args.no_ensemble,
        use_kelly=not args.no_kelly,
    )
    sim.run()

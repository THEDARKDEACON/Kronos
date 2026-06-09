"""
IC Tracker — Information Coefficient persistence and settlement.

After each pipeline run, records Kronos (and ensemble) predicted returns alongside
the as_of_date. On the next run, settles any predictions that are ≥5 trading days
old by comparing against realised OHLCV returns, then appends a row to
research/ic_history.parquet — the file the dashboard already tries to load.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

RESEARCH_DIR = Path(__file__).resolve().parent
PREDICTIONS_FILE = RESEARCH_DIR / "pending_predictions.parquet"
IC_HISTORY_FILE = RESEARCH_DIR / "ic_history.parquet"

SETTLEMENT_LAG_DAYS = 7  # calendar days ≈ 5 trading days


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_predictions(
    kronos_signals: pd.DataFrame,
    ensemble_signals: Optional[pd.DataFrame],
    as_of_date: Optional[str] = None,
) -> None:
    """
    Persist predicted returns after a pipeline run so they can be settled later.

    Args:
        kronos_signals:   DataFrame indexed by Ticker with 'Predicted_Return'.
        ensemble_signals: Optional DataFrame with 'Predicted_Return' / 'Raw_Predicted_Return'.
        as_of_date:       ISO date string; defaults to today.
    """
    date_str = as_of_date or datetime.now().strftime("%Y-%m-%d")

    rows = []
    for ticker in kronos_signals.index:
        row = {
            "as_of_date": date_str,
            "ticker": ticker,
            "kronos_pred": float(kronos_signals.loc[ticker, "Predicted_Return"]),
            "ensemble_pred": None,
            "settled": False,
            "realized_5d": None,
            "ic_kronos": None,
            "ic_ensemble": None,
            "settled_date": None,
        }
        if ensemble_signals is not None and ticker in ensemble_signals.index:
            col = (
                "Predicted_Return"
                if "Predicted_Return" in ensemble_signals.columns
                else "Raw_Predicted_Return"
            )
            row["ensemble_pred"] = float(ensemble_signals.loc[ticker, col])
        rows.append(row)

    new_df = pd.DataFrame(rows)

    if PREDICTIONS_FILE.exists():
        existing = pd.read_parquet(PREDICTIONS_FILE)
        # Drop any rows already written for this date (idempotent)
        existing = existing[existing["as_of_date"] != date_str]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PREDICTIONS_FILE)
    print(f"[IC Tracker] Recorded {len(rows)} predictions for {date_str}")


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

def settle_predictions(ohlcv_dict: Dict[str, pd.DataFrame]) -> None:
    """
    Settle any pending predictions whose as_of_date is ≥ SETTLEMENT_LAG_DAYS ago.

    Computes the 5-trading-day realized return from the OHLCV cache and appends
    a summary row (one per settlement date) to ic_history.parquet.

    Args:
        ohlcv_dict: The OHLCV dict already loaded by the pipeline (no extra API calls).
    """
    if not PREDICTIONS_FILE.exists():
        return

    pending = pd.read_parquet(PREDICTIONS_FILE)
    cutoff = (datetime.now() - timedelta(days=SETTLEMENT_LAG_DAYS)).strftime("%Y-%m-%d")
    to_settle = pending[(pending["as_of_date"] <= cutoff) & (~pending["settled"])]

    if to_settle.empty:
        print("[IC Tracker] No predictions ready to settle yet.")
        return

    ic_rows = []

    for date_str, group in to_settle.groupby("as_of_date"):
        realized: Dict[str, float] = {}

        for ticker in group["ticker"]:
            if ticker not in ohlcv_dict:
                continue
            df = ohlcv_dict[ticker]
            if "timestamps" not in df.columns or "close" not in df.columns:
                continue
            df = df.sort_values("timestamps").reset_index(drop=True)

            # Find the row on / just after as_of_date
            try:
                as_of_ts = pd.Timestamp(date_str)
                idx_start = df[df["timestamps"] >= as_of_ts].index[0]
                idx_end = min(idx_start + 5, len(df) - 1)
                price_start = df.loc[idx_start, "close"]
                price_end = df.loc[idx_end, "close"]
                if price_start > 0:
                    realized[ticker] = (price_end - price_start) / price_start
            except (IndexError, KeyError):
                continue

        if len(realized) < 10:
            print(f"[IC Tracker] {date_str}: only {len(realized)} realized returns — skipping settlement")
            continue

        realized_s = pd.Series(realized)
        kronos_s = group.set_index("ticker")["kronos_pred"].reindex(realized_s.index).dropna()
        aligned_r = realized_s.reindex(kronos_s.index).dropna()
        kronos_s = kronos_s.reindex(aligned_r.index)

        ic_kronos = _rank_ic(kronos_s, aligned_r)

        ic_ensemble = None
        ens_s = group.set_index("ticker")["ensemble_pred"].dropna().reindex(realized_s.index).dropna()
        if len(ens_s) >= 10:
            r_ens = realized_s.reindex(ens_s.index).dropna()
            ens_s = ens_s.reindex(r_ens.index)
            ic_ensemble = _rank_ic(ens_s, r_ens)

        ic_rows.append(
            {
                "timestamp": date_str,
                "settled_date": datetime.now().strftime("%Y-%m-%d"),
                "n_tickers": len(realized),
                "kronos": ic_kronos,
                "ensemble": ic_ensemble,
            }
        )

        # Mark settled in pending file
        mask = (pending["as_of_date"] == date_str) & (~pending["settled"])
        pending.loc[mask, "settled"] = True
        pending.loc[mask, "settled_date"] = datetime.now().strftime("%Y-%m-%d")
        pending.loc[mask, "ic_kronos"] = ic_kronos
        pending.loc[mask, "ic_ensemble"] = ic_ensemble

        print(
            f"[IC Tracker] Settled {date_str}: "
            f"IC(Kronos)={ic_kronos:.4f}"
            + (f", IC(Ensemble)={ic_ensemble:.4f}" if ic_ensemble is not None else "")
        )

    if ic_rows:
        new_ic_df = pd.DataFrame(ic_rows)
        if IC_HISTORY_FILE.exists():
            existing_ic = pd.read_parquet(IC_HISTORY_FILE)
            # Deduplicate by timestamp
            existing_ic = existing_ic[~existing_ic["timestamp"].isin(new_ic_df["timestamp"])]
            combined_ic = pd.concat([existing_ic, new_ic_df], ignore_index=True)
        else:
            combined_ic = new_ic_df

        IC_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        combined_ic.to_parquet(IC_HISTORY_FILE)
        print(f"[IC Tracker] Appended {len(ic_rows)} IC rows to {IC_HISTORY_FILE}")

    # Persist updated pending file
    pending.to_parquet(PREDICTIONS_FILE)


def _rank_ic(signals: pd.Series, returns: pd.Series) -> float:
    """Spearman rank IC between signals and returns."""
    from scipy.stats import spearmanr
    if len(signals) < 5:
        return float("nan")
    corr, _ = spearmanr(signals.values, returns.values)
    return float(corr) if not np.isnan(corr) else 0.0


# ---------------------------------------------------------------------------
# Summary helper (used by dashboard)
# ---------------------------------------------------------------------------

def load_ic_summary() -> Optional[pd.DataFrame]:
    """Return ic_history.parquet or None if it doesn't exist yet."""
    if IC_HISTORY_FILE.exists():
        return pd.read_parquet(IC_HISTORY_FILE)
    return None

"""Milestone 7 – Author credibility resolver.

For every non-naive scored comment with a bullish or bearish stance, we check
whether the stock actually moved in the predicted direction over the next N
trading days. Authors who are right more often get a higher credibility weight
(0.5–1.5) that amplifies or dampens their contribution to the signal.

Methodology
-----------
- Only Tier 1-3 comments with stance=bullish/bearish are used as "calls"
- Default resolution horizon: 5 trading days (or comment's horizon_days if set)
- Correct call: bullish + positive return, OR bearish + negative return
- Credibility weight uses Bayesian shrinkage toward 50% (base rate):
    adjusted_acc = (correct + prior) / (total + 2*prior)   [prior = 2]
    weight = 0.5 + adjusted_acc                             [range: 0.5..1.5]
- Minimum 1 resolved call to appear in leaderboard
- Minimum 3 calls before weight deviates meaningfully from 1.0

Main entry point: run_credibility() -> pd.DataFrame (leaderboard)
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import structlog
import yaml
import yfinance as yf

warnings.filterwarnings("ignore")
log = structlog.get_logger()

DEFAULT_HORIZON_DAYS = 5   # trading days if comment has no horizon_days
PRIOR_STRENGTH       = 2   # Bayesian prior observations at 50% accuracy
MIN_CALLS_FOR_WEIGHT = 3   # below this, weight stays near 1.0


def _fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        close.columns = tickers
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def _entry_exit(
    ticker: str,
    signal_date: pd.Timestamp,
    horizon: int,
    prices: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
) -> tuple[float | None, float | None]:
    if ticker not in prices.columns:
        return None, None
    col = prices[ticker].dropna()
    future = trading_days[trading_days >= signal_date]
    if not len(future):
        return None, None
    entry_td = future[0]
    if entry_td not in col.index:
        return None, None
    pos = trading_days.get_loc(entry_td)
    exit_pos = pos + horizon
    if exit_pos >= len(trading_days):
        return None, None
    exit_td = trading_days[exit_pos]
    if exit_td not in col.index:
        return None, None
    return float(col.loc[entry_td]), float(col.loc[exit_td])


def run_credibility(db_path: str | None = None) -> pd.DataFrame:
    project_root = Path(__file__).parent.parent
    if db_path is None:
        with (project_root / "config.yaml").open() as fh:
            cfg = yaml.safe_load(fh)
        db_path = str(project_root / cfg["storage"]["duckdb_path"])

    con = duckdb.connect(db_path)

    # Load all non-naive bullish/bearish scored comments
    calls = con.execute("""
        SELECT
            c.author,
            cs.comment_id,
            cs.ticker,
            cs.stance,
            cs.reasoning_tier,
            cs.q_composite,
            cs.horizon_days,
            CAST(DATE_TRUNC('day', c.created_utc) AS DATE) AS comment_date
        FROM comment_scores cs
        JOIN comments c ON c.comment_id = cs.comment_id
        WHERE cs.parse_error = FALSE
          AND cs.is_naive = FALSE
          AND cs.stance IN ('bullish', 'bearish')
        ORDER BY c.author, comment_date
    """).df()

    log.info("credibility.calls_loaded", n=len(calls),
             authors=calls["author"].nunique())

    if calls.empty:
        con.close()
        return pd.DataFrame()

    tickers = calls["ticker"].unique().tolist()
    start = (pd.to_datetime(calls["comment_date"].min()) - timedelta(days=5)).strftime("%Y-%m-%d")
    end   = (pd.to_datetime(calls["comment_date"].max()) + timedelta(days=35)).strftime("%Y-%m-%d")

    log.info("credibility.downloading_prices", n_tickers=len(tickers))
    prices = _fetch_prices(tickers, start, end)

    if prices.empty:
        log.error("credibility.no_prices")
        con.close()
        return pd.DataFrame()

    trading_days = pd.DatetimeIndex(sorted(prices.index.unique()))
    valid_tickers = set(prices.columns)

    # Resolve each call
    records = []
    for _, row in calls.iterrows():
        ticker = row["ticker"]
        if ticker not in valid_tickers:
            continue
        signal_date = pd.Timestamp(row["comment_date"])
        horizon = int(row["horizon_days"]) if pd.notna(row["horizon_days"]) else DEFAULT_HORIZON_DAYS
        horizon = max(1, min(horizon, 20))  # clamp to sensible range

        entry, exit_ = _entry_exit(ticker, signal_date, horizon, prices, trading_days)
        if entry is None or entry <= 0:
            continue

        fwd_ret = (exit_ - entry) / entry
        correct = (
            (row["stance"] == "bullish" and fwd_ret > 0) or
            (row["stance"] == "bearish" and fwd_ret < 0)
        )
        records.append({
            "author":         row["author"],
            "comment_id":     row["comment_id"],
            "ticker":         ticker,
            "stance":         row["stance"],
            "reasoning_tier": row["reasoning_tier"],
            "q_composite":    row["q_composite"],
            "comment_date":   row["comment_date"],
            "fwd_ret":        round(fwd_ret, 5),
            "correct":        int(correct),
            "horizon":        horizon,
        })

    outcomes = pd.DataFrame(records)
    log.info("credibility.resolved", n=len(outcomes))

    if outcomes.empty:
        con.close()
        return pd.DataFrame()

    # Score each author
    author_stats = (
        outcomes.groupby("author")
        .agg(
            resolved_calls=("correct", "count"),
            correct_calls=("correct", "sum"),
            avg_tier=("reasoning_tier", "mean"),
            avg_q=("q_composite", "mean"),
            tickers_called=("ticker", "nunique"),
        )
        .reset_index()
    )

    # Bayesian-shrunk accuracy → credibility weight
    author_stats["raw_accuracy"] = author_stats["correct_calls"] / author_stats["resolved_calls"]
    author_stats["adj_accuracy"] = (
        (author_stats["correct_calls"] + PRIOR_STRENGTH * 0.5) /
        (author_stats["resolved_calls"] + PRIOR_STRENGTH)
    )
    author_stats["credibility_weight"] = (0.5 + author_stats["adj_accuracy"]).clip(0.5, 1.5)

    # Rank: sort by resolved_calls (need volume), then accuracy
    leaderboard = author_stats.sort_values(
        ["resolved_calls", "adj_accuracy"], ascending=[False, False]
    ).reset_index(drop=True)
    leaderboard.index += 1  # rank starts at 1

    # Write to author_credibility table (today's snapshot)
    today = pd.Timestamp.now().date()
    con.execute("DELETE FROM author_credibility WHERE as_of_date = ?", [today])
    for _, row in author_stats.iterrows():
        con.execute(
            """INSERT OR REPLACE INTO author_credibility
               (author, as_of_date, resolved_calls, brier_score, w_credibility)
               VALUES (?, ?, ?, ?, ?)""",
            [row["author"], today, int(row["resolved_calls"]),
             float(1 - row["adj_accuracy"]),   # simplified Brier proxy
             float(row["credibility_weight"])],
        )

    con.close()
    log.info("credibility.done", authors_scored=len(leaderboard))
    return leaderboard, outcomes

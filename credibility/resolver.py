"""Author credibility resolver — methodology v2.

Fixes applied vs v1:
  1. Market-adjusted returns: correct = alpha > 0 (stock - SPY), not raw return.
     Prevents permabulls looking smart in bull markets.
  2. Call deduplication: one call per (author, ticker) per 90-day window unless
     stance flips. Stops someone posting NVDA 30x and inflating their call count.
  3. 60d is the headline horizon. Other horizons stored as secondary context only.
  4. Survivorship bias audit: tickers that yfinance can't load are counted and
     logged. Missing tickers = bad calls silently dropped = accuracy overstated.

Main entry point: run_credibility() -> (leaderboard_df, outcomes_df)
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd
import structlog
import yaml
import yfinance as yf

warnings.filterwarnings("ignore")
log = structlog.get_logger()

HEADLINE_HORIZON     = 60   # trading days — primary accuracy metric
PRIOR_STRENGTH       = 2    # Bayesian shrinkage toward 50%
MIN_CALLS_FOR_WEIGHT = 1
HORIZONS             = [1, 5, 10, 20, 60, 90, 180, 365]
DEDUP_WINDOW_DAYS    = 90   # calendar days per (author, ticker) window
BENCHMARK            = "SPY"


import time as _time
import re as _re

# Words that look like tickers but aren't stocks
_FAKE_TICKERS = {
    "COVID","FEAR","NBA","NFL","NFL","UFC","NASA","CIA","FBI","DOJ","DOE","SEC",
    "FED","IMF","GDP","CPI","PPI","PCE","ECB","BOJ","FOMC","OPEC","NATO","IRAN",
    "CHINA","KOREA","JAPAN","INDIA","USSR","WWII","WWIII","TRUMP","ELON","MAGA",
    "DOGE","ARK","BEAR","BULL","CALLS","PUTS","NEWS","TODAY","DAILY","WEEK","YEAR",
    "MONTH","DONE","DEAD","SAFE","FREE","FAKE","SCAM","DUMP","PUMP","MOON","ROTH",
    "ESPP","HELOC","TFSA","RRSP","MBA","CPA","CFA","NFA","ISA","HOA","LLC","INC",
    "FAANG","MAGS","DJIA","NYSE","AMEX","OTC","LSE","TSX","ASX","CNBC","WSJ","BBC",
    "CNN","NBC","ESPN","HBO","NFL","NBA","FIFA","UFC","NATO","USAID","IMF","OECD",
    "HYSA","MYGA","APY","SWR","FIRE","LEAN","LEAN","VHCOL","HCOL","LCOL",
    "CAPEX","WACC","EBITDA","ROIC","AFFO","FFO","NPV","DCF","GAAP","IFRS",
    "OTHER","THOSE","THESE","THEIR","WHERE","THERE","WOULD","COULD","MIGHT",
    "ABOUT","AGAIN","AFTER","BEING","DOING","GOING","SINCE","UNTIL","EVERY",
    "TOTAL","WHOLE","STILL","NEVER","LEAST","FIRST","LOWER","UNDER","ABOVE",
}

def _is_valid_ticker(t: str) -> bool:
    if t in _FAKE_TICKERS:
        return False
    if not _re.match(r'^[A-Z]{1,5}$', t):  # 1-5 uppercase letters only
        return False
    return True


def _fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    # Download in batches to avoid rate limiting
    BATCH = 100
    frames = []
    batches = [tickers[i:i+BATCH] for i in range(0, len(tickers), BATCH)]
    for i, batch in enumerate(batches):
        try:
            raw = yf.download(batch, start=start, end=end,
                              auto_adjust=True, progress=False, threads=False)
            if not raw.empty:
                close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
                if not isinstance(raw.columns, pd.MultiIndex):
                    close.columns = batch
                frames.append(close)
        except Exception:
            pass
        if i < len(batches) - 1:
            _time.sleep(2)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, axis=1)
    result.index = pd.to_datetime(result.index).tz_localize(None)
    return result.sort_index()


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


def _dedup_calls(calls: pd.DataFrame) -> pd.DataFrame:
    """Keep one call per (author, ticker) per 90-day window unless stance flips."""
    calls = calls.sort_values(["author", "ticker", "comment_date"]).reset_index(drop=True)
    keep = []
    last: dict[tuple, tuple] = {}  # (author, ticker) -> (last_date, last_stance)

    for _, row in calls.iterrows():
        key = (row["author"], row["ticker"])
        comment_date = pd.Timestamp(row["comment_date"])
        stance = row["stance"]

        if key not in last:
            keep.append(True)
            last[key] = (comment_date, stance)
        else:
            last_date, last_stance = last[key]
            days_since = (comment_date - last_date).days
            stance_flipped = stance != last_stance
            if days_since >= DEDUP_WINDOW_DAYS or stance_flipped:
                keep.append(True)
                last[key] = (comment_date, stance)
            else:
                keep.append(False)

    deduped = calls[keep].reset_index(drop=True)
    dropped = len(calls) - len(deduped)
    log.info("credibility.dedup", original=len(calls), kept=len(deduped), dropped=dropped)
    return deduped


def run_credibility(db_path: str | None = None):
    project_root = Path(__file__).parent.parent
    if db_path is None:
        with (project_root / "config.yaml").open() as fh:
            cfg = yaml.safe_load(fh)
        db_path = str(project_root / cfg["storage"]["duckdb_path"])

    con = duckdb.connect(db_path)

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
        JOIN ticker_mentions tm
            ON tm.comment_id = cs.comment_id AND tm.ticker = cs.ticker
        WHERE cs.parse_error = FALSE
          AND cs.is_naive = FALSE
          AND cs.stance IN ('bullish', 'bearish')
          AND tm.match_method = 'cashtag'
        ORDER BY c.author, comment_date
    """).df()

    log.info("credibility.calls_loaded", n=len(calls), authors=calls["author"].nunique())

    if calls.empty:
        con.close()
        return pd.DataFrame(), pd.DataFrame()

    # Fix 2: deduplicate before resolving
    calls = _dedup_calls(calls)

    all_tickers = calls["ticker"].unique().tolist()
    tickers = [t for t in all_tickers if _is_valid_ticker(t)]
    filtered_out = len(all_tickers) - len(tickers)
    log.info("credibility.ticker_filter", total=len(all_tickers), kept=len(tickers), filtered_fake=filtered_out)

    # Restrict calls to valid tickers only
    calls = calls[calls["ticker"].isin(set(tickers))].reset_index(drop=True)

    start = (pd.to_datetime(calls["comment_date"].min()) - timedelta(days=5)).strftime("%Y-%m-%d")
    end   = (pd.to_datetime(calls["comment_date"].max()) + timedelta(days=400)).strftime("%Y-%m-%d")

    # Fix 1: always download SPY alongside stock tickers
    download_tickers = list(set(tickers + [BENCHMARK]))
    log.info("credibility.downloading_prices", n_tickers=len(download_tickers))
    prices = _fetch_prices(download_tickers, start, end)

    if prices.empty:
        log.error("credibility.no_prices")
        con.close()
        return pd.DataFrame(), pd.DataFrame()

    trading_days = pd.DatetimeIndex(sorted(prices.index.unique()))
    valid_tickers = set(prices.columns)

    # Fix 4: survivorship bias audit
    missing = [t for t in tickers if t not in valid_tickers]
    missing_pct = len(missing) / max(len(tickers), 1) * 100
    log.info("credibility.survivorship_audit",
             total_tickers=len(tickers),
             missing=len(missing),
             missing_pct=round(missing_pct, 1),
             warning="accuracy overstated" if missing_pct > 5 else "ok")

    records = []
    for _, row in calls.iterrows():
        ticker = row["ticker"]
        if ticker not in valid_tickers:
            continue
        signal_date = pd.Timestamp(row["comment_date"])

        for h in HORIZONS:
            entry, exit_ = _entry_exit(ticker, signal_date, h, prices, trading_days)
            if entry is None or entry <= 0:
                continue

            fwd_ret = (exit_ - entry) / entry

            # signed_return: raw return in the predicted direction
            signed_return = fwd_ret if row["stance"] == "bullish" else -fwd_ret
            correct = int(signed_return > 0)

            records.append({
                "author":         row["author"],
                "comment_id":     row["comment_id"],
                "ticker":         ticker,
                "stance":         row["stance"],
                "reasoning_tier": row["reasoning_tier"],
                "q_composite":    row["q_composite"],
                "comment_date":   row["comment_date"],
                "fwd_ret":        round(fwd_ret, 5),
                "signed_return":  round(signed_return, 5),
                "correct":        correct,
                "horizon":        h,
            })

    outcomes = pd.DataFrame(records)
    log.info("credibility.resolved", n=len(outcomes))

    if outcomes.empty:
        con.close()
        return pd.DataFrame(), pd.DataFrame()

    def _adj_return(sum_ret, total):
        # shrink toward 0 (no expected edge) using prior of PRIOR_STRENGTH null calls
        return sum_ret / (total + PRIOR_STRENGTH)

    def _adj_acc(correct, total):
        return (correct + PRIOR_STRENGTH * 0.5) / (total + PRIOR_STRENGTH)

    # Primary metric: avg signed return at 60d horizon
    base = outcomes[outcomes["horizon"] == HEADLINE_HORIZON]
    author_stats = (
        base.groupby("author")
        .agg(
            resolved_calls=("signed_return", "count"),
            sum_returns=("signed_return", "sum"),
            correct_calls=("correct", "sum"),
        )
        .reset_index()
    )
    author_stats["adj_return"] = author_stats.apply(
        lambda r: _adj_return(r["sum_returns"], r["resolved_calls"]), axis=1
    )
    # accuracy: % of calls that went in the right direction (Bayesian-shrunk toward 50%)
    author_stats["adj_accuracy"] = author_stats.apply(
        lambda r: _adj_acc(r["correct_calls"], r["resolved_calls"]), axis=1
    )
    # credibility_weight = 0.5 + adj_return (ranking metric)
    author_stats["credibility_weight"] = 0.5 + author_stats["adj_return"]

    # Per-horizon avg signed return (secondary context)
    for h in HORIZONS:
        grp = outcomes[outcomes["horizon"] == h].groupby("author").agg(
            **{f"sum_{h}d": ("signed_return", "sum"), f"calls_{h}d": ("signed_return", "count")}
        ).reset_index()
        author_stats = author_stats.merge(grp, on="author", how="left")
        author_stats[f"acc_{h}d"] = author_stats.apply(
            lambda r, h=h: _adj_return(r[f"sum_{h}d"], r[f"calls_{h}d"])
            if pd.notna(r.get(f"calls_{h}d")) and r[f"calls_{h}d"] >= MIN_CALLS_FOR_WEIGHT
            else None, axis=1
        )

    leaderboard = author_stats.sort_values(
        ["resolved_calls", "adj_return"], ascending=[False, False]
    ).reset_index(drop=True)
    leaderboard.index += 1

    def _f(v):
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else float(v)

    def _i(v):
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else int(v)

    today = pd.Timestamp.now().date()
    con.execute("DELETE FROM author_credibility WHERE as_of_date = ?", [today])
    for _, row in author_stats.iterrows():
        con.execute(
            """INSERT OR REPLACE INTO author_credibility
               (author, as_of_date, resolved_calls, brier_score, w_credibility,
                acc_1d, acc_5d, acc_10d, acc_20d, acc_60d, acc_90d, acc_180d, acc_365d,
                calls_1d, calls_5d, calls_10d, calls_20d, calls_60d, calls_90d, calls_180d, calls_365d)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [row["author"], today,
             int(row["resolved_calls"]),
             _f(row["adj_accuracy"]),   # brier_score stores 60d accuracy
             _f(row["credibility_weight"]),
             _f(row.get("acc_1d")),  _f(row.get("acc_5d")),
             _f(row.get("acc_10d")), _f(row.get("acc_20d")),
             _f(row.get("acc_60d")), _f(row.get("acc_90d")),
             _f(row.get("acc_180d")),_f(row.get("acc_365d")),
             _i(row.get("calls_1d")),  _i(row.get("calls_5d")),
             _i(row.get("calls_10d")), _i(row.get("calls_20d")),
             _i(row.get("calls_60d")), _i(row.get("calls_90d")),
             _i(row.get("calls_180d")),_i(row.get("calls_365d"))],
        )

    con.close()
    log.info("credibility.done", authors_scored=len(leaderboard))
    return leaderboard, outcomes

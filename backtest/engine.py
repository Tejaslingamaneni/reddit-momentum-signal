"""Milestone 6 – Backtest engine.

Aligns daily_signal to actual stock returns and computes Spearman IC at each
forward horizon. Compares four variants:
  signal        = conviction * attention_z         (our full signal)
  conviction    = raw conviction (no attention)
  attention_z   = raw attention z-score only
  naive_net     = (#bullish - #bearish) / n_total  (no quality weighting)

Main entry point: run_backtest() -> pd.DataFrame
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import scipy.stats as stats
import structlog
import yaml
import yfinance as yf

log = structlog.get_logger()
warnings.filterwarnings("ignore", category=FutureWarning)


def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Return a wide DataFrame of adjusted close prices, columns = ticker."""
    if not tickers:
        return pd.DataFrame()
    log.info("backtest.download_prices", n_tickers=len(tickers), start=start, end=end)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = tickers

    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def _next_trading_day(date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """First trading day >= date."""
    future = trading_days[trading_days >= date]
    return future[0] if len(future) > 0 else None


def _forward_return(
    ticker: str,
    entry_date: pd.Timestamp,
    horizon: int,
    prices: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
) -> float | None:
    if ticker not in prices.columns:
        return None
    col = prices[ticker].dropna()
    if col.empty:
        return None

    # Entry: next trading day on or after signal date
    entry_td = _next_trading_day(entry_date, trading_days)
    if entry_td is None or entry_td not in col.index:
        return None

    # Exit: horizon trading days after entry
    entry_pos = trading_days.get_loc(entry_td)
    exit_pos = entry_pos + horizon
    if exit_pos >= len(trading_days):
        return None
    exit_td = trading_days[exit_pos]
    if exit_td not in col.index:
        return None

    p_entry = col.loc[entry_td]
    p_exit = col.loc[exit_td]
    if p_entry <= 0 or np.isnan(p_entry) or np.isnan(p_exit):
        return None
    return float((p_exit - p_entry) / p_entry)


def _ic_row(label: str, signal_col: pd.Series, ret_col: pd.Series, horizon: int) -> dict:
    mask = signal_col.notna() & ret_col.notna()
    s = signal_col[mask]
    r = ret_col[mask]
    n = len(s)
    if n < 5:
        return {"signal": label, "horizon_days": horizon, "ic": np.nan, "t_stat": np.nan, "p_value": np.nan, "n": n}
    ic, pval = stats.spearmanr(s, r)
    t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-12))
    return {"signal": label, "horizon_days": horizon, "ic": round(float(ic), 4),
            "t_stat": round(float(t_stat), 3), "p_value": round(float(pval), 4), "n": n}


def run_backtest() -> pd.DataFrame:
    project_root = Path(__file__).parent.parent
    with (project_root / "config.yaml").open() as fh:
        cfg = yaml.safe_load(fh)

    horizons: list[int] = cfg["backtest"]["forward_horizons_days"]
    db_path = str(project_root / cfg["storage"]["duckdb_path"])

    con = duckdb.connect(db_path)

    # Load signals — only rows where at least conviction is defined
    sdf = con.execute(
        """
        SELECT ds.ticker, ds.signal_date,
               ds.conviction, ds.attention_z, ds.signal,
               ds.n_comments_total,
               -- naive signal: net bullish fraction from raw counts
               (SELECT (SUM(CASE WHEN cs.stance='bullish' THEN 1.0 ELSE 0.0 END)
                        - SUM(CASE WHEN cs.stance='bearish' THEN 1.0 ELSE 0.0 END))
                        / NULLIF(COUNT(*), 0)
                FROM comment_scores cs
                JOIN comments c ON c.comment_id = cs.comment_id
                WHERE cs.ticker = ds.ticker
                  AND CAST(DATE_TRUNC('day', c.created_utc) AS DATE) = ds.signal_date
                  AND cs.parse_error = FALSE) AS naive_net
        FROM daily_signal ds
        WHERE ds.conviction IS NOT NULL
        ORDER BY ds.signal_date, ds.ticker
        """
    ).df()
    con.close()

    log.info("backtest.signals_loaded", rows=len(sdf), tickers=sdf["ticker"].nunique())

    tickers = sdf["ticker"].unique().tolist()
    start_date = (sdf["signal_date"].min() - timedelta(days=5)).strftime("%Y-%m-%d")
    end_date   = (sdf["signal_date"].max() + timedelta(days=35)).strftime("%Y-%m-%d")

    prices = _download_prices(tickers, start_date, end_date)

    if prices.empty:
        log.error("backtest.no_price_data")
        return pd.DataFrame()

    trading_days = pd.DatetimeIndex(sorted(prices.index.unique()))
    valid_tickers = [t for t in tickers if t in prices.columns and not prices[t].dropna().empty]
    log.info("backtest.valid_tickers", n=len(valid_tickers), of=len(tickers))

    results: list[dict] = []

    for horizon in horizons:
        log.info("backtest.horizon", days=horizon)
        rows = []
        for _, row in sdf.iterrows():
            ticker = row["ticker"]
            if ticker not in valid_tickers:
                continue
            entry_date = pd.Timestamp(row["signal_date"])
            fwd_ret = _forward_return(ticker, entry_date, horizon, prices, trading_days)
            rows.append({
                "ticker": ticker,
                "signal_date": entry_date,
                "signal": row["signal"],
                "conviction": row["conviction"],
                "attention_z": row["attention_z"],
                "naive_net": row["naive_net"],
                "fwd_ret": fwd_ret,
            })

        panel = pd.DataFrame(rows).dropna(subset=["fwd_ret"])
        log.info("backtest.pairs", horizon=horizon, n=len(panel))

        for label, col in [
            ("full_signal",  panel["signal"]),
            ("conviction",   panel["conviction"]),
            ("attention_z",  panel["attention_z"]),
            ("naive_net",    panel["naive_net"]),
        ]:
            results.append(_ic_row(label, col, panel["fwd_ret"], horizon))

    report = pd.DataFrame(results)
    return report


if __name__ == "__main__":
    df = run_backtest()
    if df.empty:
        print("No results.")
    else:
        print(df.pivot_table(index="horizon_days", columns="signal", values=["ic", "t_stat", "n"]).to_string())

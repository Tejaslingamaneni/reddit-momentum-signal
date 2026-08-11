"""Milestone 5 – Daily signal construction.

Aggregates comment_scores into a per-(ticker, date) signal:
  1. Compute tier-weighted bull/bear mass per day
  2. Roll up over window_days
  3. Conviction = net_mass / total_mass  (bounded -1..1)
  4. Attention z-score = how unusual today's comment volume is vs rolling baseline
  5. Signal = conviction * attention_z  (null if insufficient data)

Main entry point: run_signal() -> dict
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import structlog
import yaml

log = structlog.get_logger()

TIER_WEIGHT = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}


def run_signal() -> dict[str, Any]:
    project_root = Path(__file__).parent.parent
    with (project_root / "config.yaml").open() as fh:
        cfg = yaml.safe_load(fh)

    sig_cfg = cfg["signal"]
    window_days: int = sig_cfg["window_days"]
    min_total_mass: float = sig_cfg["min_total_mass"]
    attention_rolling_days: int = sig_cfg["attention_rolling_days"]
    scorer_version: str = cfg["scoring"]["scorer_version"]

    db_path = str(project_root / cfg["storage"]["duckdb_path"])
    con = duckdb.connect(db_path)
    schema_ddl = (project_root / "db" / "schema.sql").read_text()
    con.execute(schema_ddl)

    log.info("signal.start", window_days=window_days, min_total_mass=min_total_mass)

    # ── Step 1: per-(ticker, day) aggregation ─────────────────────────────────
    # Each comment's mass is multiplied by the author's credibility weight.
    # Authors not yet in author_credibility default to weight=1.0.
    daily_raw = con.execute(
        """
        SELECT
            cs.ticker,
            CAST(DATE_TRUNC('day', c.created_utc) AS DATE) AS signal_date,
            SUM(CASE WHEN cs.stance = 'bullish' AND NOT cs.is_naive
                THEN CASE cs.reasoning_tier
                    WHEN 1 THEN 0.5 WHEN 2 THEN 1.0 WHEN 3 THEN 2.0 ELSE 0.0
                END * cs.q_composite
                  * COALESCE(ac.w_credibility, 1.0)
                ELSE 0.0 END) AS bull_mass,
            SUM(CASE WHEN cs.stance = 'bearish' AND NOT cs.is_naive
                THEN CASE cs.reasoning_tier
                    WHEN 1 THEN 0.5 WHEN 2 THEN 1.0 WHEN 3 THEN 2.0 ELSE 0.0
                END * cs.q_composite
                  * COALESCE(ac.w_credibility, 1.0)
                ELSE 0.0 END) AS bear_mass,
            COUNT(*)                                    AS n_total,
            SUM(CASE WHEN cs.is_naive     THEN 1 ELSE 0 END) AS n_naive,
            SUM(CASE WHEN NOT cs.is_naive THEN 1 ELSE 0 END) AS n_scored
        FROM comment_scores cs
        JOIN comments c ON c.comment_id = cs.comment_id
        LEFT JOIN (
            SELECT author, w_credibility
            FROM author_credibility
            WHERE as_of_date = (SELECT MAX(as_of_date) FROM author_credibility)
        ) ac ON ac.author = c.author
        WHERE cs.parse_error = FALSE
          AND cs.scorer_version = ?
        GROUP BY cs.ticker, DATE_TRUNC('day', c.created_utc)
        ORDER BY cs.ticker, signal_date
        """,
        [scorer_version],
    ).df()

    if daily_raw.empty:
        log.warning("signal.no_data")
        con.close()
        return {"rows_written": 0}

    log.info("signal.daily_raw", rows=len(daily_raw), tickers=daily_raw["ticker"].nunique())

    # ── Step 2: rolling window aggregation per ticker ─────────────────────────
    records: list[dict[str, Any]] = []

    for ticker, grp in daily_raw.groupby("ticker"):
        grp = grp.sort_values("signal_date").copy()

        bull = grp["bull_mass"].to_numpy(dtype=float)
        bear = grp["bear_mass"].to_numpy(dtype=float)
        n_tot = grp["n_total"].to_numpy(dtype=float)
        n_naive = grp["n_naive"].to_numpy(dtype=int)
        n_scored = grp["n_scored"].to_numpy(dtype=int)
        dates = grp["signal_date"].tolist()

        for i, date in enumerate(dates):
            lo = max(0, i - window_days + 1)
            w_bull = float(bull[lo : i + 1].sum())
            w_bear = float(bear[lo : i + 1].sum())
            w_n_total = int(n_tot[lo : i + 1].sum())

            net = w_bull - w_bear
            total_mass = w_bull + w_bear

            conviction: float | None = None
            if total_mass >= min_total_mass:
                conviction = net / (total_mass + 1e-9)
                conviction = max(-1.0, min(1.0, conviction))

            # Attention z-score: today vs rolling baseline (look-back only)
            att_lo = max(0, i - attention_rolling_days)
            baseline = n_tot[att_lo:i]  # exclude today
            attention_z: float | None = None
            if len(baseline) >= 5:
                mu = float(baseline.mean())
                sigma = max(float(baseline.std()), 0.5)  # floor avoids division by ~0
                raw_z = (n_tot[i] - mu) / sigma
                attention_z = float(np.clip(raw_z, -5.0, 5.0))

            signal: float | None = None
            if conviction is not None and attention_z is not None:
                signal = conviction * attention_z

            records.append(
                {
                    "ticker": ticker,
                    "signal_date": date,
                    "reasoned_bull_mass": w_bull,
                    "reasoned_bear_mass": w_bear,
                    "net_reasoning": net,
                    "conviction": conviction,
                    "attention_z": attention_z,
                    "signal": signal,
                    "n_comments_total": w_n_total,
                    "n_comments_naive": int(n_naive[lo : i + 1].sum()),
                    "n_comments_scored": int(n_scored[lo : i + 1].sum()),
                }
            )

    out = pd.DataFrame(records)
    log.info("signal.rows_built", rows=len(out))

    # ── Step 3: write to daily_signal ─────────────────────────────────────────
    # Only keep rows where we have enough mass to compute a conviction score.
    out = out[out["conviction"].notna()].copy()
    log.info("signal.rows_with_conviction", rows=len(out))

    con.execute("DELETE FROM daily_signal")
    con.execute(
        """
        INSERT INTO daily_signal
        SELECT
            ticker, signal_date,
            reasoned_bull_mass, reasoned_bear_mass, net_reasoning,
            conviction, attention_z, signal,
            n_comments_total, n_comments_naive, n_comments_scored
        FROM out
        """
    )

    rows_written = len(out)
    log.info("signal.done", rows_written=rows_written)
    con.close()
    return {"rows_written": rows_written}

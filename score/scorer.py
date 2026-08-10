"""Milestone 3 – LLM-based comment scorer (Gemini backend).

Calls Gemini via function calling to produce structured quality scores for each
(comment_id, ticker) pair in `ticker_mentions` that has not yet been scored.

Main entry point: run_score(limit=None) -> dict
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import time

import duckdb
import httpx
import structlog
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger()

# ── Tool / function declaration ────────────────────────────────────────────────

SCORE_FUNCTION: dict[str, Any] = {
    "name": "score_comment",
    "description": "Score a Reddit comment for investment reasoning quality and stance.",
    "parameters": {
        "type": "object",
        "properties": {
            "stance": {
                "type": "string",
                "enum": ["bullish", "bearish", "neutral", "unclear"],
                "description": "Directional view expressed in the comment",
            },
            "reasoning_tier": {
                "type": "integer",
                "description": "0=naive/spam, 1=generic opinion, 2=specific reasoning, 3=falsifiable+adversarial",
            },
            "q_specificity": {
                "type": "number",
                "description": "0-1: how specific is the claim?",
            },
            "q_catalyst": {
                "type": "number",
                "description": "0-1: is there a named catalyst?",
            },
            "q_evidence": {
                "type": "number",
                "description": "0-1: is there cited evidence?",
            },
            "q_falsifiability": {
                "type": "number",
                "description": "0-1: is there a testable prediction?",
            },
            "q_steelman": {
                "type": "number",
                "description": "0-1: does it engage with the opposing view?",
            },
            "horizon_days": {
                "type": "integer",
                "description": "Investment horizon in days if mentioned, else omit",
            },
            "catalyst_date": {
                "type": "string",
                "description": "ISO date YYYY-MM-DD if a specific catalyst date is named, else omit",
            },
            "rationale": {
                "type": "string",
                "description": "1-2 sentence explanation of the tier assignment",
            },
        },
        "required": [
            "stance", "reasoning_tier",
            "q_specificity", "q_catalyst", "q_evidence", "q_falsifiability", "q_steelman",
            "rationale",
        ],
    },
}

SYSTEM_PROMPT = (
    "You are a financial reasoning quality evaluator. Your job is to assess whether a Reddit "
    "comment about a specific stock ticker demonstrates genuine investment reasoning or is just "
    "noise/sentiment.\n\n"
    "Score strictly and symmetrically: a naive bullish comment and a naive bearish comment should "
    "both score Tier 0. A well-reasoned bearish thesis should score just as highly as a well-reasoned "
    "bullish thesis.\n\n"
    "Focus on the QUALITY of reasoning, not whether you agree with the conclusion."
)

USER_PROMPT_TEMPLATE = (
    "Ticker: {ticker}\n\n"
    "Reddit comment:\n"
    "---\n"
    "{body}\n"
    "---\n\n"
    "Score this comment using the score_comment function."
)


# ── Error-row factory ──────────────────────────────────────────────────────────

def _error_row(comment_id: str, ticker: str, scorer_version: str) -> dict[str, Any]:
    return {
        "comment_id": comment_id,
        "ticker": ticker,
        "stance": "unclear",
        "reasoning_tier": 0,
        "q_specificity": 0.0,
        "q_catalyst": 0.0,
        "q_evidence": 0.0,
        "q_falsifiability": 0.0,
        "q_steelman": 0.0,
        "q_composite": 0.0,
        "is_naive": True,
        "horizon_days": None,
        "catalyst_date": None,
        "rationale": None,
        "scorer_version": scorer_version,
        "scored_at": datetime.now(timezone.utc),
        "parse_error": True,
    }


# ── Core API call ──────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"


def _call_api(model_name: str, ticker: str, body: str) -> dict[str, Any]:
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(ticker=ticker, body=body)}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "function", "function": SCORE_FUNCTION}],
        "stream": False,
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Extract tool call result
    msg = data.get("message", {})
    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        return tool_calls[0]["function"]["arguments"]

    # Fallback: parse JSON from content if model didn't use tool call
    import json, re
    content = msg.get("content", "")
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"No tool call or JSON in Ollama response: {content[:200]}")


# ── Public scorer ──────────────────────────────────────────────────────────────

def score_one(
    model_name: str,
    comment_id: str,
    ticker: str,
    body: str,
    scorer_version: str,
) -> dict[str, Any]:

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _with_retry() -> dict[str, Any]:
        return _call_api(model_name, ticker, body)

    try:
        raw = _with_retry()
    except Exception as exc:
        log.error("scorer.error", comment_id=comment_id, ticker=ticker, error=str(exc))
        return _error_row(comment_id, ticker, scorer_version)

    q_scores = [
        float(raw.get("q_specificity", 0)),
        float(raw.get("q_catalyst", 0)),
        float(raw.get("q_evidence", 0)),
        float(raw.get("q_falsifiability", 0)),
        float(raw.get("q_steelman", 0)),
    ]
    q_composite = sum(q_scores) / len(q_scores)
    reasoning_tier = int(raw.get("reasoning_tier", 0))

    # Validate catalyst_date — local models sometimes return durations or freetext
    _cd = raw.get("catalyst_date") or None
    import re as _re
    catalyst_date: str | None = _cd if (_cd and _re.match(r"^\d{4}-\d{2}-\d{2}$", str(_cd))) else None

    horizon_days_raw = raw.get("horizon_days")
    try:
        horizon_days: int | None = int(horizon_days_raw) if horizon_days_raw is not None else None
    except (ValueError, TypeError):
        horizon_days = None

    return {
        "comment_id": comment_id,
        "ticker": ticker,
        "stance": raw.get("stance", "unclear"),
        "reasoning_tier": reasoning_tier,
        "q_specificity": q_scores[0],
        "q_catalyst": q_scores[1],
        "q_evidence": q_scores[2],
        "q_falsifiability": q_scores[3],
        "q_steelman": q_scores[4],
        "q_composite": q_composite,
        "is_naive": reasoning_tier == 0,
        "horizon_days": horizon_days,
        "catalyst_date": catalyst_date,
        "rationale": raw.get("rationale"),
        "scorer_version": scorer_version,
        "scored_at": datetime.now(timezone.utc),
        "parse_error": False,
    }


# ── Insert helper ──────────────────────────────────────────────────────────────

def _insert_score(con: duckdb.DuckDBPyConnection, row: dict[str, Any]) -> None:
    con.execute(
        """
        INSERT OR IGNORE INTO comment_scores (
            comment_id, ticker, stance, reasoning_tier,
            q_specificity, q_catalyst, q_evidence, q_falsifiability, q_steelman,
            q_composite, is_naive, horizon_days, catalyst_date,
            rationale, scorer_version, scored_at, parse_error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            row["comment_id"], row["ticker"], row["stance"], row["reasoning_tier"],
            row["q_specificity"], row["q_catalyst"], row["q_evidence"],
            row["q_falsifiability"], row["q_steelman"], row["q_composite"],
            row["is_naive"], row["horizon_days"], row["catalyst_date"],
            row["rationale"], row["scorer_version"], row["scored_at"], row["parse_error"],
        ],
    )


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_score(limit: int | None = None) -> dict[str, Any]:
    project_root = Path(__file__).parent.parent
    with (project_root / "config.yaml").open() as fh:
        cfg = yaml.safe_load(fh)

    scoring_cfg = cfg["scoring"]
    scorer_version: str = scoring_cfg["scorer_version"]
    model_name: str = scoring_cfg["model"]
    max_comment_tokens: int = scoring_cfg["max_comment_tokens"]
    batch_size: int = scoring_cfg["batch_size"]
    max_char_len: int = max_comment_tokens * 4

    db_path: str = cfg["storage"]["duckdb_path"]
    resolved_db = str(project_root / db_path)
    effective_limit = limit if limit is not None else 100

    log.info("scorer.start", model=model_name, limit=effective_limit, scorer_version=scorer_version)

    # Ollama runs locally — no API key needed

    con = duckdb.connect(resolved_db)
    schema_ddl = (project_root / "db" / "schema.sql").read_text()
    con.execute(schema_ddl)

    # ── Fetch unique comments to score ───────────────────────────────────────
    # One API call per unique comment (not per comment+ticker).
    # Filters: min upvotes, min length, exclude WSB (mostly memes).
    min_upvotes: int = scoring_cfg.get("min_upvotes", 10)
    min_body_chars: int = scoring_cfg.get("min_body_chars", 100)
    exclude_subs: list[str] = scoring_cfg.get("exclude_subreddits", ["wallstreetbets"])

    exclude_clause = ", ".join(f"'{s}'" for s in exclude_subs)

    unique_comments = con.execute(
        f"""
        SELECT DISTINCT c.comment_id, c.body, c.score AS upvotes
        FROM ticker_mentions tm
        JOIN comments c ON c.comment_id = tm.comment_id
        WHERE c.subreddit NOT IN ({exclude_clause})
          AND c.score >= ?
          AND length(c.body) >= ?
          AND length(c.body) <= ?
          AND NOT EXISTS (
              SELECT 1 FROM comment_scores cs
              WHERE cs.comment_id = c.comment_id
                AND cs.scorer_version = ?
          )
        ORDER BY c.score DESC
        LIMIT ?
        """,
        [min_upvotes, min_body_chars, max_char_len, scorer_version, effective_limit],
    ).fetchall()

    log.info("scorer.unique_comments_fetched", count=len(unique_comments))

    scored = 0
    errors = 0

    for i, (comment_id, body, upvotes) in enumerate(unique_comments, start=1):
        # Get all tickers for this comment
        tickers = [
            r[0] for r in con.execute(
                "SELECT ticker FROM ticker_mentions WHERE comment_id = ?",
                [comment_id],
            ).fetchall()
        ]

        # Score once using the first ticker as context
        primary_ticker = tickers[0] if tickers else "UNKNOWN"
        base_row = score_one(model_name, comment_id, primary_ticker, body, scorer_version)

        # Fan out: insert a score row for every ticker on this comment
        for ticker in tickers:
            row = {**base_row, "ticker": ticker}
            _insert_score(con, row)

        # No rate limiting needed — Ollama runs locally

        if base_row["parse_error"]:
            errors += 1
        else:
            scored += 1

        if i % batch_size == 0:
            log.info(
                "scorer.progress",
                processed=i,
                total=len(unique_comments),
                scored=scored,
                errors=errors,
            )

    con.close()
    totals: dict[str, Any] = {"scored": scored, "errors": errors}
    log.info("scorer.done", **totals)
    return totals

"""
Orchestrates a full ingest run across all configured subreddits.
Idempotent: duplicate comment_ids are silently dropped by the storage layer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import structlog
import yaml

from ingest.client import fetch_comments
from ingest.storage import finish_run, init_db, start_run, upsert_comments

log = structlog.get_logger()


def _config_hash(config_path: str = "config.yaml") -> str:
    raw = Path(config_path).read_bytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[return-value]


def run_ingest(config_path: str = "config.yaml") -> dict[str, int]:
    """
    Pull comments for every configured subreddit and write to DuckDB + NDJSON.
    Returns a summary dict: {subreddit: n_inserted}.
    """
    cfg = load_config(config_path)
    db_path = cfg["storage"]["duckdb_path"]
    raw_dir = cfg["storage"]["raw_dir"]

    # Ensure data directories and schema exist
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    init_db(db_path)

    run_id = start_run(db_path, stage="ingest", config_hash=_config_hash(config_path))

    reddit_cfg = cfg["reddit"]
    subreddits: list[str] = reddit_cfg["subreddits"]
    source: str = reddit_cfg["ingest_source"]
    posts_limit: int = reddit_cfg["posts_per_sub"]
    comment_depth: int = reddit_cfg["comment_depth"]
    history_days: int = reddit_cfg["history_days"]

    totals: dict[str, int] = {}

    for sub in subreddits:
        log.info("ingest.subreddit.start", subreddit=sub, source=source)
        buffer: list[dict[str, Any]] = []
        sub_total = 0

        for comment in fetch_comments(
            subreddit=sub,
            source=source,
            posts_limit=posts_limit,
            comment_depth=comment_depth,
            history_days=history_days,
        ):
            buffer.append(comment)
            if len(buffer) >= 500:  # flush in batches of 500
                sub_total += upsert_comments(db_path, run_id, buffer, raw_dir, sub)
                buffer.clear()

        if buffer:  # flush remainder
            sub_total += upsert_comments(db_path, run_id, buffer, raw_dir, sub)

        totals[sub] = sub_total
        log.info("ingest.subreddit.done", subreddit=sub, inserted=sub_total)

    finish_run(db_path, run_id)
    log.info("ingest.complete", run_id=run_id, totals=totals)
    return totals

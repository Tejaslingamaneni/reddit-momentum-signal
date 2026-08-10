"""Write ingested comments to DuckDB and raw NDJSON snapshots."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import structlog

log = structlog.get_logger()


def _db(path: str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(path)


def init_db(db_path: str, schema_path: str = "db/schema.sql") -> None:
    """Create all tables if they don't exist."""
    ddl = Path(schema_path).read_text()
    con = _db(db_path)
    con.executemany("", [])  # ensure connection is open
    con.execute(ddl)
    con.close()
    log.info("db.init", path=db_path)


def new_run_id() -> str:
    return str(uuid.uuid4())


def start_run(db_path: str, stage: str, config_hash: str) -> str:
    run_id = new_run_id()
    con = _db(db_path)
    con.execute(
        """
        INSERT INTO runs (run_id, started_at, stage, config_hash)
        VALUES (?, ?, ?, ?)
        """,
        [run_id, datetime.now(timezone.utc), stage, config_hash],
    )
    con.close()
    log.info("run.start", run_id=run_id, stage=stage)
    return run_id


def finish_run(db_path: str, run_id: str) -> None:
    con = _db(db_path)
    con.execute(
        "UPDATE runs SET completed_at = ? WHERE run_id = ?",
        [datetime.now(timezone.utc), run_id],
    )
    con.close()
    log.info("run.finish", run_id=run_id)


def upsert_comments(
    db_path: str,
    run_id: str,
    comments: list[dict[str, Any]],
    raw_dir: str,
    subreddit: str,
) -> int:
    """
    Write comments to DuckDB (ignore duplicates by comment_id) and append
    to the daily NDJSON snapshot. Returns count of net-new rows inserted.
    """
    if not comments:
        return 0

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    # ── Raw NDJSON snapshot (immutable, append-only) ─────────────────────────
    raw_path = Path(raw_dir) / date_str
    raw_path.mkdir(parents=True, exist_ok=True)
    ndjson_file = raw_path / f"{subreddit}_{run_id}.ndjson"
    with ndjson_file.open("a") as f:
        for c in comments:
            f.write(json.dumps(c["raw_json"]) + "\n")

    # ── DuckDB upsert (INSERT OR IGNORE) ─────────────────────────────────────
    con = _db(db_path)
    rows = [
        (
            c["comment_id"],
            c["post_id"],
            c["subreddit"],
            c["author"],
            c["body"],
            c["created_utc"],
            c["score"],
            c.get("parent_id"),
            now,
            run_id,
            json.dumps(c["raw_json"]),
        )
        for c in comments
    ]

    before = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]  # type: ignore[index]
    con.executemany(
        """
        INSERT OR IGNORE INTO comments
            (comment_id, post_id, subreddit, author, body,
             created_utc, score, parent_id, ingested_at, run_id, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    after = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]  # type: ignore[index]
    con.close()

    inserted = after - before
    log.info(
        "storage.upsert",
        subreddit=subreddit,
        candidates=len(comments),
        inserted=inserted,
        run_id=run_id,
    )
    return inserted

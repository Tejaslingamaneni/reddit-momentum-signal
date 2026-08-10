"""
Reddit data clients.

Two backends, selectable via config:
  - 'praw'         : Reddit's live API — good for ongoing collection, ~1000 posts/sub
  - 'arctic_shift' : Community Pushshift mirror — historical bootstrap, no auth needed
  - 'both'         : Arctic Shift for history, PRAW for the gap to now
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Generator

import httpx
import praw
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

log = structlog.get_logger()

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"


# ── Shared helpers ────────────────────────────────────────────────────────────

def _normalize_comment(raw: dict[str, Any], subreddit: str) -> dict[str, Any]:
    """Flatten a raw Reddit comment dict into our storage schema."""
    return {
        "comment_id": raw.get("id", ""),
        "post_id": raw.get("link_id", raw.get("post_id", "")).replace("t3_", ""),
        "subreddit": subreddit,
        "author": raw.get("author", "[deleted]"),
        "body": raw.get("body", ""),
        "created_utc": datetime.fromtimestamp(
            float(raw.get("created_utc", 0)), tz=timezone.utc
        ),
        "score": int(raw.get("score", 0)),
        "parent_id": raw.get("parent_id"),
        "raw_json": raw,
    }


def _skip_comment(c: dict[str, Any]) -> bool:
    body = c.get("body", "")
    return body in ("[deleted]", "[removed]", "") or c.get("author") == "AutoModerator"


# ── PRAW client ───────────────────────────────────────────────────────────────

def _praw_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
        username=os.environ.get("REDDIT_USERNAME"),
        password=os.environ.get("REDDIT_PASSWORD"),
        ratelimit_seconds=300,
    )


def fetch_praw(
    subreddit: str,
    posts_limit: int = 500,
    comment_depth: int = -1,
) -> Generator[dict[str, Any], None, None]:
    """
    Yield normalized comment dicts from PRAW (hot + new listings).
    comment_depth: -1 = full tree, 0 = top-level only.
    """
    reddit = _praw_client()
    sub = reddit.subreddit(subreddit)

    seen_posts: set[str] = set()

    for listing in [sub.hot(limit=posts_limit // 2), sub.new(limit=posts_limit // 2)]:
        for post in listing:
            if post.id in seen_posts:
                continue
            seen_posts.add(post.id)

            try:
                post.comments.replace_more(limit=None if comment_depth == -1 else 0)
                flat = (
                    post.comments.list()
                    if comment_depth == -1
                    else list(post.comments)
                )
            except Exception as exc:
                log.warning("praw.replace_more_failed", post=post.id, error=str(exc))
                continue

            for comment in flat:
                if not hasattr(comment, "body"):
                    continue
                raw = vars(comment)
                # praw objects aren't plain dicts — extract what we need
                raw_dict = {
                    "id": comment.id,
                    "link_id": comment.link_id,
                    "author": str(comment.author) if comment.author else "[deleted]",
                    "body": comment.body,
                    "created_utc": comment.created_utc,
                    "score": comment.score,
                    "parent_id": comment.parent_id,
                    "subreddit": subreddit,
                }
                if _skip_comment(raw_dict):
                    continue
                yield _normalize_comment(raw_dict, subreddit)

            time.sleep(0.1)  # be polite — stay well under 100 QPM


# ── Arctic Shift client ───────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _arctic_get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()  # type: ignore[return-value]


def fetch_arctic_shift(
    subreddit: str,
    history_days: int = 30,
    batch_size: int = 100,
) -> Generator[dict[str, Any], None, None]:
    """
    Yield normalized comment dicts from Arctic Shift for the past `history_days`.
    No auth required — free public API, reasonable rate limits.
    """
    after = int((datetime.now(timezone.utc) - timedelta(days=history_days)).timestamp())
    before = int(datetime.now(timezone.utc).timestamp())

    params: dict[str, Any] = {
        "subreddit": subreddit,
        "after": after,
        "before": before,
        "limit": batch_size,
        "sort": "asc",
        "fields": "id,link_id,author,body,created_utc,score,parent_id,subreddit",
    }

    fetched = 0
    while True:
        try:
            data = _arctic_get(f"{ARCTIC_BASE}/comments/search", params)
        except Exception as exc:
            log.error("arctic_shift.fetch_failed", subreddit=subreddit, error=str(exc))
            break

        items: list[dict[str, Any]] = data.get("data", [])
        if not items:
            break

        for raw in items:
            if _skip_comment(raw):
                continue
            yield _normalize_comment(raw, subreddit)

        fetched += len(items)
        log.info("arctic_shift.batch", subreddit=subreddit, fetched=fetched)

        # Paginate: move `after` to the last item's timestamp
        last_ts = items[-1].get("created_utc", before)
        if int(last_ts) >= before or len(items) < batch_size:
            break

        params["after"] = int(last_ts)
        time.sleep(1.0)  # Arctic Shift asks for politeness


# ── Unified fetch ─────────────────────────────────────────────────────────────

def fetch_comments(
    subreddit: str,
    source: str,
    posts_limit: int = 500,
    comment_depth: int = -1,
    history_days: int = 30,
) -> Generator[dict[str, Any], None, None]:
    """
    Dispatch to the right backend(s) based on config source setting.
    source: 'praw' | 'arctic_shift' | 'both'
    """
    if source in ("arctic_shift", "both"):
        log.info("fetch.arctic_shift.start", subreddit=subreddit, days=history_days)
        yield from fetch_arctic_shift(subreddit, history_days=history_days)

    if source in ("praw", "both"):
        log.info("fetch.praw.start", subreddit=subreddit)
        yield from fetch_praw(subreddit, posts_limit=posts_limit, comment_depth=comment_depth)

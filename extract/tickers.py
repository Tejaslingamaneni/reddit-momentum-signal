"""Milestone 2 – Ticker extraction from Reddit comments.

Three extraction methods:
  1. cashtag   – $TICKER regex, confidence=1.0
  2. name_match – company name fuzzy-word-boundary match, confidence=0.85

Results are upserted into the `ticker_mentions` table in DuckDB.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import structlog
import yaml

log = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")

# Common English words / abbreviations that collide with real tickers.
STOPLIST: set[str] = {
    # Common English words
    "A", "IT", "ALL", "ARE", "BE", "DO", "GO", "HI", "IN", "IS", "ME",
    "ON", "OR", "SO", "TO", "AT", "BY", "NO", "ANY", "CAN", "FOR", "NOW",
    "OUT", "SAY", "SEE", "THE", "TWO", "USE", "WAY", "WHO", "WHY", "YES",
    "YET", "YOU", "OWN", "OLD", "NEW", "GET", "GOT", "HAS", "HAD", "BUT",
    "NOT", "AND", "ITS", "WAS", "HER", "HIS", "OUR", "OFF", "TOO", "PUT",
    "LOT", "SET", "LET", "MAY", "PAY", "RUN", "TRY", "BUY", "CUT", "HIT",
    "LOW", "HIGH", "HOLD", "LONG", "SHORT", "SELL", "OPEN", "EDIT", "SAME",
    # Geography / country codes
    "US", "UK", "EU", "CA", "JP", "CN", "KR", "AU", "DE", "FR",
    # Time / units
    "AM", "PM", "YTD", "YOY", "QOQ", "MOM",
    # Reddit slang
    "OP", "OC", "TIL", "IMO", "IMHO", "AFAIK", "TLDR", "TL", "DR",
    "LMAO", "LMFAO", "LOL", "WTF", "OMG", "SMH", "FYI", "NVM", "IDK",
    "ELI", "AMA", "WSB", "DD", "YOLO", "FUD", "HODL", "FOMO",
    # Financial abbreviations (not tickers)
    "IPO", "ETF", "ETN",
    "PE", "PS", "PB", "EPS", "FCF", "DCF", "ROE", "ROI", "IRR",
    "ATH", "ATL", "DCA", "MRQ", "TTM", "GDP", "CPI", "FED",
    "SEC", "ICO", "NFT", "CEO", "CFO", "CTO", "COO", "VP",
    "AI", "ML", "API", "EV", "AVG", "MAX", "MIN",
    # Non-US markets / technology terms
    "KOSPI", "NIKKEI", "FTSE", "DAX", "DRAM", "NAND", "SK",
    # Misc uppercase noise / profanity / common abbreviations
    "EDIT", "TLDR", "AFAICT", "IANAL", "IYKYK",
    "THIS", "THAT", "THEM", "THEN", "THAN", "THUS", "THEY",
    "JUST", "ALSO", "EVEN", "WHEN", "WHAT", "WELL", "BEEN",
    "WILL", "WITH", "HAVE", "FROM", "LIKE", "SOME", "MOST",
    "OVER", "ONLY", "REAL", "VERY", "EACH", "BOTH", "MORE",
    "INTO", "UPON", "NEXT", "LAST", "ONCE", "MUCH", "MANY",
    "SUCH", "WERE", "DOES", "SAID", "MADE", "GIVE", "LOOK",
    "COME", "WENT", "KEEP", "KNOW", "TAKE", "MAKE", "FEEL",
    "GOOD", "BEST", "LESS", "NEED", "SHOW", "TURN", "MOVE",
    # Common acronyms not tickers
    "USA", "LLM", "AWS", "GCP", "AWS", "IRA", "LLC", "INC",
    "LTD", "CORP", "EBIT", "EBITDA", "CAGR", "NAV", "AUM",
    "SPX", "NDX", "VIX", "CAPE", "RAM", "ROM", "GPU", "CPU",
    "LBO", "MBO", "RTO", "WFH", "BTC", "ETH", "XRP",
    # Slang / profanity
    "FUCK", "SHIT", "DAMN", "HELL", "CRAP", "HOLY",
    "AH", "OH", "HA", "HM", "UH", "OK", "RIP", "GG",
    "MY", "OF", "AN", "IF", "AS", "UP",
}

# Popular WSB / meme tickers not always in S&P 500.
WSB_EXTRAS: dict[str, str] = {
    "GME":  "GameStop Corp",
    "AMC":  "AMC Entertainment Holdings",
    "BBBY": "Bed Bath & Beyond",
    "PLTR": "Palantir Technologies",
    "SOFI": "SoFi Technologies",
    "RIVN": "Rivian Automotive",
    "LCID": "Lucid Group",
    "HOOD": "Robinhood Markets",
    "CLOV": "Clover Health Investments",
    "SPCE": "Virgin Galactic Holdings",
    "WKHS": "Workhorse Group",
    "NAKD": "Naked Brand Group",
    "SNDL": "Sundial Growers",
    "EXPR": "Express Inc",
    "BB":   "BlackBerry Ltd",
    "NOK":  "Nokia Corp",
    "WISH": "ContextLogic Inc",
    "CPRX": "Catalyst Biosciences",
    "MVIS": "MicroVision Inc",
    "SENS": "Senseonics Holdings",
    "TLRY": "Tilray Brands",
    "AMC":  "AMC Networks",
    "DWAC": "Digital World Acquisition Corp",
    "PHUN": "Phunware Inc",
    "SDC":  "SmileDirectClub",
    "RIDE":  "Lordstown Motors",
    "NKLA": "Nikola Corporation",
    "HYLN": "Hyliion Holdings",
    "GOEV": "Canoo Inc",
    "PSTH": "Pershing Square Tontine",
    "CCIV": "Churchill Capital Corp IV",
    "SOS":  "SOS Limited",
    "ATOS": "Atoss Software",
    "OCGN": "Ocugen Inc",
    "AGRX": "Agile Therapeutics",
    "ZNGA": "Zynga Inc",
    "DOGE": "Dogecoin",  # sometimes mentioned as ticker
    "TSLA": "Tesla Inc",
    "NVDA": "NVIDIA Corporation",
    "AMD":  "Advanced Micro Devices",
    "AAPL": "Apple Inc",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com Inc",
    "GOOG": "Alphabet Inc Class C",
    "GOOGL": "Alphabet Inc Class A",
    "META": "Meta Platforms",
    "NFLX": "Netflix Inc",
    "BABA": "Alibaba Group",
    "NIO":  "NIO Inc",
    "XPEV": "XPeng Inc",
    "LI":   "Li Auto Inc",
    "COIN": "Coinbase Global",
    "MSTR": "MicroStrategy Inc",
    "MARA": "Marathon Digital Holdings",
    "RIOT": "Riot Platforms",
    "HUT":  "Hut 8 Mining",
    "BTBT": "Bit Digital",
    "CAN":  "Canaan Inc",
    "EBON": "Ebang International",
    "ARBKF": "Argo Blockchain",
    "HIVE": "HIVE Blockchain Technologies",
    "CLSK": "CleanSpark Inc",
}


# ── Ticker universe ───────────────────────────────────────────────────────────

BROAD_TOKEN_RE = re.compile(r"\b([A-Z]{2,5})\b")


def build_data_driven_universe(
    con: duckdb.DuckDBPyConnection,
    min_mentions: int = 30,
    batch_size: int = 5000,
) -> set[str]:
    """
    Scan all comment bodies, count every uppercase 2-5 letter token,
    and return the set of tokens that appear at least min_mentions times.
    This gives us a data-driven ticker universe with no external dependency.
    """
    from collections import Counter
    counts: Counter[str] = Counter()

    total = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]  # type: ignore[index]
    offset = 0
    while True:
        rows = con.execute(
            "SELECT body FROM comments ORDER BY comment_id LIMIT ? OFFSET ?",
            [batch_size, offset],
        ).fetchall()
        if not rows:
            break
        for (body,) in rows:
            if not body:
                continue
            for m in BROAD_TOKEN_RE.finditer(body):
                tok = m.group(1)
                if tok not in STOPLIST:
                    counts[tok] += 1
        offset += batch_size
        if offset % 50000 == 0:
            log.info("universe.scan_progress", scanned=offset, total=total)

    universe = {tok for tok, cnt in counts.items() if cnt >= min_mentions}
    log.info("universe.built", total_tokens=len(counts), universe_size=len(universe), min_mentions=min_mentions)
    return universe


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_cashtags(body: str, universe: set[str]) -> list[str]:
    """Return deduplicated list of tickers found via $TICKER cashtag syntax."""
    seen: set[str] = set()
    found: list[str] = []
    for m in CASHTAG_RE.finditer(body):
        ticker = m.group(1)
        if ticker in universe and ticker not in STOPLIST and ticker not in seen:
            found.append(ticker)
            seen.add(ticker)
    return found


def _build_bare_ticker_pattern(universe: set[str]) -> re.Pattern[str]:
    """Build a single alternation regex matching any ticker as a bare word."""
    tickers = sorted(universe - STOPLIST, key=len, reverse=True)
    alternation = "|".join(re.escape(t) for t in tickers)
    return re.compile(r"\b(" + alternation + r")\b")


def _extract_bare_tickers(body: str, pattern: re.Pattern[str]) -> list[str]:
    """Return deduplicated list of tickers found as bare uppercase words."""
    seen: set[str] = set()
    found: list[str] = []
    for m in pattern.finditer(body):
        ticker = m.group(1)
        if ticker not in seen:
            found.append(ticker)
            seen.add(ticker)
    return found


def _build_name_patterns(universe: dict[str, str]) -> list[tuple[str, re.Pattern[str]]]:
    """Pre-compile one regex per company name for fast matching."""
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for ticker, name in universe.items():
        if ticker in STOPLIST:
            continue
        if not name or len(name) < 3:
            continue
        escaped = re.escape(name)
        try:
            pat = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
            patterns.append((ticker, pat))
        except re.error:
            pass
    return patterns


def _extract_names(
    body: str,
    name_patterns: list[tuple[str, re.Pattern[str]]],
) -> list[str]:
    """Return list of tickers found via company name matching."""
    found: list[str] = []
    for ticker, pat in name_patterns:
        if pat.search(body):
            found.append(ticker)
    return found


def _deduplicate(
    cashtags: list[str],
    bare_tickers: list[str],
    name_matches: list[str],
) -> list[tuple[str, str, float]]:
    """Merge results, priority: cashtag > bare_ticker > name_match."""
    rows: list[tuple[str, str, float]] = []
    seen: set[str] = set()

    for t in cashtags:
        rows.append((t, "cashtag", 1.0))
        seen.add(t)

    for t in bare_tickers:
        if t not in seen:
            rows.append((t, "bare_ticker", 0.90))
            seen.add(t)

    for t in name_matches:
        if t not in seen:
            rows.append((t, "name_match", 0.85))
            seen.add(t)

    return rows


# ── Storage ───────────────────────────────────────────────────────────────────

def _upsert_mentions(
    con: duckdb.DuckDBPyConnection,
    rows: list[tuple[str, str, str, float]],
) -> int:
    """Insert (comment_id, ticker, method, confidence) rows; ignore duplicates."""
    if not rows:
        return 0

    con.execute("CREATE TEMP TABLE IF NOT EXISTS _tm_staging (comment_id VARCHAR, ticker VARCHAR, match_method VARCHAR, match_confidence DOUBLE)")
    con.execute("DELETE FROM _tm_staging")
    con.executemany("INSERT INTO _tm_staging VALUES (?, ?, ?, ?)", rows)
    result = con.execute("""
        INSERT INTO ticker_mentions
        SELECT s.comment_id, s.ticker, s.match_method, s.match_confidence
        FROM _tm_staging s
        WHERE NOT EXISTS (
            SELECT 1 FROM ticker_mentions t
            WHERE t.comment_id = s.comment_id AND t.ticker = s.ticker
        )
    """)
    return result.rowcount if result and result.rowcount >= 0 else 0


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_extract() -> dict[str, Any]:
    """Full extraction pipeline.

    Reads config, loads the ticker universe, processes comments in batches,
    and upserts matches into `ticker_mentions`.

    Returns a dict with keys: comments_processed, mentions_inserted.
    """
    # ── Load config ───────────────────────────────────────────────────────────
    config_path = Path(__file__).parent.parent / "config.yaml"
    with config_path.open() as fh:
        cfg = yaml.safe_load(fh)

    db_path: str = cfg["storage"]["duckdb_path"]
    min_cashtag_conf: float = cfg["tickers"].get("min_cashtag_confidence", 0.95)
    min_name_conf: float    = cfg["tickers"].get("min_name_confidence", 0.80)
    batch_size: int         = 1000

    log.info(
        "extract.start",
        db_path=db_path,
        min_cashtag_conf=min_cashtag_conf,
        min_name_conf=min_name_conf,
    )

    # ── Open DB connection ────────────────────────────────────────────────────
    project_root = Path(__file__).parent.parent
    resolved_db  = str(project_root / db_path)
    con = duckdb.connect(resolved_db)

    # ── Ensure schema is current ──────────────────────────────────────────────
    schema_ddl = (project_root / "db" / "schema.sql").read_text()
    con.execute(schema_ddl)

    # ── Build data-driven universe (first pass over all comment bodies) ───────
    min_mentions: int = cfg["tickers"].get("min_mentions", 30)
    universe_set = build_data_driven_universe(con, min_mentions=min_mentions)

    # Pre-compile bare ticker pattern from data-driven universe.
    bare_ticker_pattern = _build_bare_ticker_pattern(universe_set)
    log.info("extract.patterns_compiled", universe_size=len(universe_set))

    # ── Count total comments for progress reporting ───────────────────────────
    total_comments: int = con.execute("SELECT COUNT(*) FROM comments").fetchone()[0]  # type: ignore[index]
    log.info("extract.total_comments", count=total_comments)

    comments_processed = 0
    mentions_inserted  = 0
    offset = 0

    while True:
        batch = con.execute(
            """
            SELECT comment_id, body
            FROM comments
            ORDER BY comment_id
            LIMIT ? OFFSET ?
            """,
            [batch_size, offset],
        ).fetchall()

        if not batch:
            break

        batch_mentions: list[tuple[str, str, str, float]] = []

        for comment_id, body in batch:
            if not body:
                continue

            cashtags     = _extract_cashtags(body, universe_set)
            bare_hits    = _extract_bare_tickers(body, bare_ticker_pattern)
            merged       = _deduplicate(cashtags, bare_hits, [])

            # Apply confidence thresholds from config.
            for ticker, method, confidence in merged:
                if method == "cashtag"     and confidence < min_cashtag_conf:
                    continue
                if method == "bare_ticker" and confidence < 0.85:
                    continue
                if method == "name_match"  and confidence < min_name_conf:
                    continue
                batch_mentions.append((comment_id, ticker, method, confidence))

        # Deduplicate within batch: higher-priority method wins
        METHOD_PRIORITY = {"cashtag": 0, "bare_ticker": 1, "name_match": 2}
        deduped: dict[tuple[str, str], tuple[str, str, str, float]] = {}
        for row in batch_mentions:
            key = (row[0], row[1])
            if key not in deduped or METHOD_PRIORITY[row[2]] < METHOD_PRIORITY[deduped[key][2]]:
                deduped[key] = row
        inserted = _upsert_mentions(con, list(deduped.values()))
        mentions_inserted  += inserted
        comments_processed += len(batch)
        offset             += batch_size

        log.info(
            "extract.batch",
            offset=offset,
            batch_size=len(batch),
            batch_mentions=len(batch_mentions),
            batch_inserted=inserted,
            total_processed=comments_processed,
        )

    con.close()

    totals: dict[str, Any] = {
        "comments_processed": comments_processed,
        "mentions_inserted":  mentions_inserted,
    }
    log.info("extract.done", **totals)
    return totals

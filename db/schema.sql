-- Single source of truth for all DDL.
-- Applied by scripts/init_db.py — idempotent (CREATE TABLE IF NOT EXISTS).

-- ── Run audit ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS runs (
    run_id          VARCHAR PRIMARY KEY,
    started_at      TIMESTAMP NOT NULL,
    completed_at    TIMESTAMP,
    stage           VARCHAR NOT NULL,   -- 'ingest'|'extract'|'score'|'signal'|'backtest'
    config_hash     VARCHAR NOT NULL,   -- sha256 of config.yaml at run time
    notes           TEXT
);

-- ── Raw comment snapshot ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS comments (
    comment_id      VARCHAR PRIMARY KEY,
    post_id         VARCHAR NOT NULL,
    subreddit       VARCHAR NOT NULL,
    author          VARCHAR NOT NULL,
    body            TEXT NOT NULL,
    created_utc     TIMESTAMP NOT NULL,
    score           INTEGER NOT NULL,
    parent_id       VARCHAR,
    ingested_at     TIMESTAMP NOT NULL,
    run_id          VARCHAR NOT NULL,
    raw_json        JSON NOT NULL
);

-- ── One row per (comment, ticker) pair ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS ticker_mentions (
    comment_id          VARCHAR NOT NULL,
    ticker              VARCHAR NOT NULL,
    match_method        VARCHAR NOT NULL,   -- 'cashtag'|'ner'|'company_name'
    match_confidence    DOUBLE NOT NULL,
    PRIMARY KEY (comment_id, ticker)
);

-- ── LLM scoring output ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS comment_scores (
    comment_id          VARCHAR NOT NULL,
    ticker              VARCHAR NOT NULL,
    stance              VARCHAR NOT NULL,   -- 'bullish'|'bearish'|'neutral'|'unclear'
    reasoning_tier      INTEGER NOT NULL,   -- 0..3
    q_specificity       DOUBLE NOT NULL,
    q_catalyst          DOUBLE NOT NULL,
    q_evidence          DOUBLE NOT NULL,
    q_falsifiability    DOUBLE NOT NULL,
    q_steelman          DOUBLE NOT NULL,
    q_composite         DOUBLE NOT NULL,
    is_naive            BOOLEAN NOT NULL,
    horizon_days        INTEGER,
    catalyst_date       DATE,
    rationale           TEXT,
    scorer_version      VARCHAR NOT NULL,
    scored_at           TIMESTAMP NOT NULL,
    parse_error         BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (comment_id, ticker, scorer_version)
);

-- ── Author credibility (point-in-time) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS author_credibility (
    author          VARCHAR NOT NULL,
    as_of_date      DATE NOT NULL,
    resolved_calls  INTEGER NOT NULL DEFAULT 0,
    brier_score     DOUBLE,
    w_credibility   DOUBLE NOT NULL DEFAULT 1.0,   -- 0.5..1.5
    PRIMARY KEY (author, as_of_date)
);

-- ── Final daily signal ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS daily_signal (
    ticker              VARCHAR NOT NULL,
    signal_date         DATE NOT NULL,
    reasoned_bull_mass  DOUBLE NOT NULL,
    reasoned_bear_mass  DOUBLE NOT NULL,
    net_reasoning       DOUBLE NOT NULL,
    conviction          DOUBLE NOT NULL,    -- bounded -1..1
    attention_z         DOUBLE,
    signal              DOUBLE,             -- conviction * attention_z; NULL if attention undefined
    n_comments_total    INTEGER NOT NULL,
    n_comments_naive    INTEGER NOT NULL,
    n_comments_scored   INTEGER NOT NULL,
    PRIMARY KEY (ticker, signal_date)
);

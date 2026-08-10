# Reddit Momentum Signal

Research pipeline that turns Reddit stock discussion into a daily numerical signal per ticker, then backtests whether that signal predicts forward returns.

**Core thesis:** Volume of opinion is noise. Volume of *well-reasoned* opinion may be signal.

## Setup

### 1. Register a Reddit app
1. Go to https://www.reddit.com/prefs/apps
2. Click **"create another app"** at the bottom
3. Choose **script** type
4. Name it anything (e.g. `momentum-signal`)
5. Set redirect URI to `http://localhost:8080`
6. Copy the `client_id` (under the app name) and `client_secret`

### 2. Configure credentials
```bash
cp .env.example .env
# fill in REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
```

### 3. Install dependencies
```bash
make install
```

### 4. Initialize the database
```bash
make init
```

### 5. Run the ingest pipeline
```bash
make ingest
```

## Architecture

```
ingest/       → pull Reddit posts + comments, snapshot immutably
extract/      → resolve ticker mentions, filter false positives
score/        → LLM-based stance + reasoning-quality classification
credibility/  → per-author track record, updated as calls resolve
signal/       → aggregate scored comments into daily per-ticker signal
backtest/     → align to market data, compute IC, compare vs baselines
```

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Ingest + storage | ✅ Done |
| 2 | Ticker extraction | ⬜ |
| 3 | Scorer + symmetry tests | ⬜ |
| 4 | Lead-lag test | ⬜ |
| 5 | Signal construction | ⬜ |
| 6 | Backtest + baselines | ⬜ |
| 7 | Credibility layer | ⬜ |
| 8 | Writeup | ⬜ |

## Results

*To be filled in after backtesting. Null results will be reported honestly.*

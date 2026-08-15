"""Flask leaderboard server — serves Reddit analyst rankings from DuckDB."""
from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import numpy as np
import pandas as pd
import yaml
from flask import Flask, jsonify, render_template

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_PATH    = PROJECT_ROOT / "data" / "leaderboard_cache.json"
PROFILES_PATH = PROJECT_ROOT / "data" / "analyst_profiles.json"
MIN_CALLS = 1


def _load_from_db() -> dict | None:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    db_path = str(PROJECT_ROOT / cfg["storage"]["duckdb_path"])
    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception:
        return None

    try:
        df = con.execute("""
            WITH author_comment_stats AS (
                SELECT
                    c.author,
                    COUNT(*)                        AS total_comments,
                    AVG(c.score)                    AS avg_upvotes,
                    COUNT(DISTINCT c.subreddit)     AS subreddits_active,
                    MIN(c.created_utc)              AS first_seen,
                    MAX(c.created_utc)              AS last_seen,
                    SUM(CASE WHEN EXISTS (
                        SELECT 1 FROM comment_scores cs
                        WHERE cs.comment_id = c.comment_id
                          AND cs.parse_error = FALSE
                          AND cs.is_naive = FALSE
                    ) THEN 1 ELSE 0 END)            AS scored_comments
                FROM comments c GROUP BY c.author
            ),
            subreddit_list AS (
                SELECT c.author,
                       STRING_AGG(DISTINCT c.subreddit, ',' ORDER BY c.subreddit) AS subs
                FROM comments c GROUP BY c.author
            )
            SELECT
                ac.author,
                ac.resolved_calls,
                ac.w_credibility,
                ac.brier_score,
                ac.acc_1d, ac.acc_5d, ac.acc_10d, ac.acc_20d,
                ac.acc_60d, ac.acc_90d, ac.acc_180d, ac.acc_365d,
                ROUND(acs.avg_upvotes, 1)   AS avg_upvotes,
                acs.scored_comments,
                sl.subs
            FROM author_credibility ac
            JOIN author_comment_stats acs ON acs.author = ac.author
            JOIN subreddit_list sl ON sl.author = ac.author
            WHERE ac.as_of_date = (SELECT MAX(as_of_date) FROM author_credibility)
              AND ac.resolved_calls >= ?
            ORDER BY ac.resolved_calls DESC, ac.w_credibility DESC
        """, [MIN_CALLS]).df()
    except Exception:
        con.close()
        return None
    con.close()

    if df.empty:
        return None

    df["adj_accuracy"] = df["w_credibility"] - 0.5
    df["volume_score"] = np.log1p(df["resolved_calls"])
    df["obscurity_bonus"] = 1.0 / np.log1p(df["avg_upvotes"] + 1)
    df["overall_score"] = df["adj_accuracy"] * df["volume_score"]
    df["gem_score"] = df["adj_accuracy"] * df["volume_score"] * df["obscurity_bonus"]

    ranked = df.sort_values("overall_score", ascending=False).head(100).reset_index(drop=True)

    analysts = []
    for i, row in ranked.iterrows():
        subs = [s.strip() for s in str(row["subs"]).split(",") if s.strip()][:3]
        acc = float(row["adj_accuracy"])   # adj_return (e.g. 0.123 = +12.3%)
        def _h(col):
            v = row.get(col)
            return round(float(v) * 100, 1) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None

        def _safe_float(v, decimals=3):
            try:
                f = float(v)
                return None if math.isnan(f) or math.isinf(f) else round(f, decimals)
            except (TypeError, ValueError):
                return None

        pct = acc * 100
        avg_return_pct = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"

        # brier_score stores adj_accuracy (% correct at 60d)
        raw_acc = _safe_float(row.get("brier_score"), 3)
        dir_accuracy = round(float(raw_acc) * 100, 1) if raw_acc is not None else None

        analysts.append({
            "rank": i + 1,
            "author": str(row["author"]),
            "accuracy_pct": avg_return_pct,        # avg return — primary display
            "accuracy_raw": round(pct, 1),          # signed %, for bar scaling
            "dir_accuracy": dir_accuracy,           # % calls in right direction
            "resolved_calls": int(row["resolved_calls"]),
            "avg_upvotes": _safe_float(row["avg_upvotes"], 1) or 0.0,
            "scored_comments": int(row["scored_comments"]),
            "gem_score": _safe_float(row["gem_score"]) or 0.0,
            "subreddits": subs,
            "overall_score": _safe_float(row["overall_score"]) or 0.0,
            "acc_1d":   _h("acc_1d"),
            "acc_5d":   _h("acc_5d"),
            "acc_10d":  _h("acc_10d"),
            "acc_20d":  _h("acc_20d"),
            "acc_60d":  _h("acc_60d"),
            "acc_90d":  _h("acc_90d"),
            "acc_180d": _h("acc_180d"),
            "acc_365d": _h("acc_365d"),
        })

    data = {
        "total_analysts": len(df),
        "avg_accuracy": f"+{df['adj_accuracy'].mean() * 100:.1f}%",
        "total_calls": int(df["resolved_calls"].sum()),
        "updated_at": datetime.now(timezone.utc).strftime("%H:%M UTC"),
        "analysts": analysts,
        "locked": False,
    }

    CACHE_PATH.write_text(json.dumps(data, indent=2))
    return data


def get_data() -> dict | None:
    data = _load_from_db()
    if data:
        return data
    if CACHE_PATH.exists():
        cached = json.loads(CACHE_PATH.read_text())
        cached["locked"] = True
        return cached
    return None


@app.route("/")
def index():
    return render_template("index.html")


DEMO_ANALYSTS = [
    ("ValueHunter99",      78, 41, 6.2,  ["investing","SecurityAnalysis","stocks"]),
    ("QuantEdge",          74, 38, 3.1,  ["options","thetagang","stocks"]),
    ("AlphaSeeker42",      71, 29, 8.7,  ["ValueInvesting","stocks"]),
    ("ThetaGangster",      69, 55, 12.4, ["thetagang","options"]),
    ("MomentumMike",       68, 22, 4.5,  ["stocks","StockMarket"]),
    ("DividendDave",       67, 18, 2.8,  ["dividends","Bogleheads"]),
    ("OptionsOracle",      66, 33, 9.1,  ["options","wallstreetbetsELITE"]),
    ("FundamentalsFred",   65, 47, 5.6,  ["SecurityAnalysis","ValueInvesting"]),
    ("RiskArbRaj",         64, 26, 3.3,  ["investing","stocks"]),
    ("MacroMaven",         63, 31, 7.2,  ["investing","ETFs"]),
    ("CatalystCarlos",     62, 19, 2.1,  ["stocks","StockMarket"]),
    ("DeepValueDan",       61, 44, 4.9,  ["ValueInvesting","SecurityAnalysis"]),
    ("VitardVince",        60, 28, 11.3, ["Vitards","stocks"]),
    ("BogleBot",           59, 36, 1.8,  ["Bogleheads","ETFs","dividends"]),
    ("SmallCapSam",        58, 23, 3.6,  ["stocks","investing"]),
    ("SectorRotator",      57, 41, 6.8,  ["ETFs","stocks","StockMarket"]),
    ("EarningsEdge",       56, 17, 2.4,  ["stocks","options"]),
    ("PatientPaula",       56, 52, 5.1,  ["investing","ValueInvesting"]),
    ("TechnicalTom",       55, 29, 8.9,  ["stocks","StockMarket"]),
    ("YieldYoda",          55, 38, 4.2,  ["dividends","investing"]),
    ("GrowthGuru",         54, 33, 7.6,  ["stocks","investing"]),
    ("CoveredCallCathy",   54, 27, 3.8,  ["thetagang","options"]),
    ("ARKWatcher",         53, 21, 9.4,  ["stocks","ETFs"]),
    ("SPACSpecialist",     53, 15, 2.7,  ["stocks","investing"]),
    ("MidCapMike",         52, 44, 5.3,  ["stocks","StockMarket"]),
    ("BalanceSheetBob",    52, 31, 3.1,  ["SecurityAnalysis","ValueInvesting"]),
    ("RealRateRuby",       51, 19, 6.7,  ["investing","Bogleheads"]),
    ("FlowTrader",         51, 26, 4.4,  ["options","stocks"]),
    ("HedgeFundHorace",    50, 38, 8.2,  ["investing","SecurityAnalysis"]),
    ("PatiencePays",       50, 47, 2.9,  ["ValueInvesting","dividends"]),
    ("IronCondorIvan",     49, 33, 5.5,  ["thetagang","options"]),
    ("NakedPutNina",       49, 21, 3.7,  ["thetagang","options"]),
    ("EarningsWhisperer",  48, 44, 7.1,  ["stocks","investing"]),
    ("TickerTerri",        48, 18, 2.2,  ["stocks","StockMarket"]),
    ("ShortSqueezeSteve",  47, 29, 14.6, ["stocks","wallstreetbetsELITE"]),
    ("QuietContrarianQ",   47, 36, 1.5,  ["investing","SecurityAnalysis"]),
    ("LeveragedLarry",     46, 24, 9.8,  ["options","stocks"]),
    ("MeanReversionMeg",   46, 31, 4.0,  ["stocks","investing"]),
    ("FCFFrank",           45, 42, 3.3,  ["SecurityAnalysis","ValueInvesting"]),
    ("BreakevenBrendan",   45, 19, 5.6,  ["options","thetagang"]),
    ("10KReader",          44, 51, 2.6,  ["SecurityAnalysis","investing"]),
    ("SpreadTraderSusan",  44, 27, 4.9,  ["options","thetagang"]),
    ("RegionalBankRon",    43, 22, 3.1,  ["stocks","investing"]),
    ("GoldBugGreta",       43, 17, 6.3,  ["investing","ETFs"]),
    ("InflationHawk",      42, 38, 5.0,  ["investing","Bogleheads"]),
    ("CallBuyerCalvin",    42, 23, 8.4,  ["options","stocks"]),
    ("FairValueFiona",     41, 44, 3.8,  ["ValueInvesting","SecurityAnalysis"]),
    ("NasdaqNick",         41, 32, 7.2,  ["stocks","StockMarket"]),
    ("ETFEnthusiast",      40, 55, 2.1,  ["ETFs","Bogleheads"]),
    ("InsiderFilingsIan",  40, 19, 3.5,  ["stocks","SecurityAnalysis"]),
    ("CyclicalCarla",      39, 28, 4.7,  ["stocks","investing"]),
    ("PEMasterPete",       39, 35, 5.9,  ["ValueInvesting","SecurityAnalysis"]),
    ("TailRiskTanya",      38, 22, 6.1,  ["options","investing"]),
    ("MoatBuilderMax",     38, 41, 3.2,  ["ValueInvesting","stocks"]),
    ("FreeCashFlowFelipe", 37, 33, 4.4,  ["SecurityAnalysis","investing"]),
    ("BullPutBrianna",     37, 18, 7.8,  ["thetagang","options"]),
    ("ConcentratedCarson", 36, 15, 2.9,  ["ValueInvesting","SecurityAnalysis"]),
    ("WatchlistWendy",     36, 29, 5.3,  ["stocks","StockMarket"]),
    ("MomentumMarcus",     35, 44, 8.1,  ["stocks","StockMarket"]),
    ("DCFDwight",          35, 26, 3.6,  ["SecurityAnalysis","ValueInvesting"]),
    ("BondProxyBeth",      34, 32, 2.4,  ["dividends","Bogleheads"]),
    ("SemiconductorSid",   34, 21, 6.9,  ["stocks","StockMarket"]),
    ("TurnaroundTracy",    33, 17, 4.1,  ["stocks","investing"]),
    ("NetNetNorman",       33, 38, 1.9,  ["ValueInvesting","SecurityAnalysis"]),
    ("VolatilityVince",    32, 25, 9.2,  ["options","thetagang"]),
    ("EnterpriseElla",     32, 31, 3.8,  ["SecurityAnalysis","stocks"]),
    ("REITRachel",         31, 42, 5.4,  ["dividends","investing","ETFs"]),
    ("GreenEnergyGary",    31, 19, 7.6,  ["stocks","ETFs"]),
    ("TurnaroundTed",      30, 28, 4.3,  ["stocks","investing"]),
    ("FactorInvestorFay",  30, 36, 2.7,  ["ETFs","Bogleheads","investing"]),
    ("ReversalRodrigo",    29, 23, 5.8,  ["stocks","StockMarket"]),
    ("WagTheDogWalt",      29, 17, 3.1,  ["investing","stocks"]),
    ("IntrinsicIrene",     28, 44, 2.3,  ["ValueInvesting","SecurityAnalysis"]),
    ("LongConvexLeo",      28, 21, 6.5,  ["options","investing"]),
    ("BioTechBrenda",      27, 15, 8.9,  ["stocks","StockMarket"]),
    ("MagazineCoverMatt",  27, 33, 4.6,  ["investing","stocks"]),
    ("LiquidityLynda",     26, 26, 3.3,  ["stocks","ETFs"]),
    ("CapStructCaprice",   26, 18, 5.1,  ["SecurityAnalysis","investing"]),
    ("QuantQuentin",       25, 38, 2.8,  ["stocks","options","investing"]),
    ("MeanMeanMeanMaria",  25, 22, 4.0,  ["stocks","StockMarket"]),
    ("MarginMaurice",      24, 29, 6.2,  ["options","stocks"]),
    ("CashPileCelia",      24, 35, 1.7,  ["ValueInvesting","SecurityAnalysis"]),
    ("PatientCapitalPat",  23, 47, 3.5,  ["ValueInvesting","investing"]),
    ("TurnaroundTina",     23, 19, 5.7,  ["stocks","investing"]),
    ("InsiderOwnershipOz", 22, 26, 2.9,  ["SecurityAnalysis","stocks"]),
    ("DeltaNeutralDan",    22, 21, 7.4,  ["options","thetagang"]),
    ("CyclePeakCyprian",   21, 33, 4.8,  ["stocks","StockMarket","investing"]),
    ("GapFillGloria",      21, 17, 6.1,  ["stocks","StockMarket"]),
    ("MoatMinerMiranda",   20, 41, 3.2,  ["ValueInvesting","SecurityAnalysis"]),
    ("ScuttlebuttSylvia",  20, 28, 5.5,  ["stocks","investing"]),
    ("ReinvestRex",        19, 36, 2.1,  ["dividends","Bogleheads","ETFs"]),
    ("PositionSizerPriya", 19, 22, 4.6,  ["options","investing"]),
    ("FloatRotterFred",    18, 18, 8.3,  ["stocks","StockMarket"]),
    ("SteadyEddieStu",     18, 44, 1.9,  ["Bogleheads","dividends"]),
    ("LowVolumeLeila",     17, 25, 2.4,  ["stocks","investing"]),
    ("ArbitrageAlex",      17, 19, 3.8,  ["investing","SecurityAnalysis"]),
    ("TaxLossHarvey",      16, 31, 2.6,  ["ETFs","Bogleheads"]),
    ("UnderCoverageUma",   16, 15, 3.1,  ["stocks","SecurityAnalysis"]),
    ("PatiencePatrick",    15, 52, 2.2,  ["ValueInvesting","investing","dividends"]),
    ("QuietMoneyQuinn",    15, 38, 1.6,  ["Bogleheads","ETFs","dividends"]),
]


def _demo_data():
    analysts = []
    for i, (author, acc, calls, upv, subs) in enumerate(DEMO_ANALYSTS):
        analysts.append({
            "rank": i + 1,
            "author": author,
            "accuracy_pct": f"{acc}%",
            "accuracy_raw": float(acc),
            "resolved_calls": calls,
            "avg_upvotes": upv,
            "scored_comments": calls + 5,
            "gem_score": round((acc / 100 - 0.5) * math.log1p(calls) / math.log1p(upv + 1), 3),
            "subreddits": subs,
            "overall_score": round((acc / 100 - 0.5) * math.log1p(calls), 3),
        })
    return {
        "total_analysts": 1247,
        "avg_accuracy": "54.3%",
        "total_calls": 8943,
        "updated_at": "DEMO MODE",
        "analysts": analysts,
        "locked": False,
        "demo": True,
    }


@app.route("/api/leaderboard")
def api_leaderboard():
    data = get_data()
    if data:
        return jsonify(data), 200
    return jsonify(_demo_data()), 200


@app.route("/api/analyst/<username>")
def analyst_profile(username):
    # Fast path: serve from pre-exported JSON (used in production)
    if PROFILES_PATH.exists():
        profiles = json.loads(PROFILES_PATH.read_text())
        if username in profiles:
            return jsonify(profiles[username])
        return jsonify({"error": "analyst not found"}), 404

    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    db_path = str(PROJECT_ROOT / cfg["storage"]["duckdb_path"])
    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception:
        return jsonify({"error": "db unavailable"}), 503

    def _sf(v, decimals=3):
        try:
            f = float(v)
            return None if math.isnan(f) or math.isinf(f) else round(f, decimals)
        except (TypeError, ValueError):
            return None

    def _si(v):
        try:
            f = float(v)
            return None if math.isnan(f) else int(f)
        except (TypeError, ValueError):
            return None

    try:
        cred = con.execute("""
            SELECT resolved_calls, w_credibility,
                   acc_1d, acc_5d, acc_10d, acc_20d, acc_60d, acc_90d, acc_180d, acc_365d,
                   calls_1d, calls_5d, calls_10d, calls_20d, calls_60d, calls_90d, calls_180d, calls_365d
            FROM author_credibility
            WHERE author = ?
              AND as_of_date = (SELECT MAX(as_of_date) FROM author_credibility)
        """, [username]).df()

        if cred.empty:
            con.close()
            return jsonify({"error": "analyst not found"}), 404

        preds = con.execute("""
            SELECT DISTINCT ON (c.comment_id, cs.ticker)
                cs.ticker,
                cs.stance,
                cs.reasoning_tier,
                cs.q_composite,
                cs.rationale,
                cs.horizon_days,
                LEFT(c.body, 600)   AS body_excerpt,
                c.created_utc,
                c.score             AS upvotes,
                c.subreddit
            FROM comment_scores cs
            JOIN comments c ON c.comment_id = cs.comment_id
            WHERE c.author = ?
              AND cs.parse_error = FALSE
              AND cs.is_naive = FALSE
              AND cs.stance IN ('bullish', 'bearish')
            ORDER BY c.comment_id, cs.ticker, cs.scored_at DESC
        """, [username]).df()

        comment_count = con.execute(
            "SELECT COUNT(*) FROM comments WHERE author = ?", [username]
        ).fetchone()[0]

        avg_upvotes = con.execute(
            "SELECT AVG(score) FROM comments WHERE author = ?", [username]
        ).fetchone()[0]

        subs = con.execute(
            "SELECT STRING_AGG(DISTINCT subreddit, ',') FROM comments WHERE author = ?", [username]
        ).fetchone()[0] or ""

    except Exception as exc:
        con.close()
        return jsonify({"error": str(exc)}), 500

    con.close()

    row = cred.iloc[0]
    w = _sf(row["w_credibility"]) or 1.0
    adj_acc = w - 0.5

    horizons = {}
    for h in [1, 5, 10, 20, 60, 90, 180, 365]:
        acc_raw = _sf(row.get(f"acc_{h}d"))
        horizons[str(h)] = {
            "acc": round(float(acc_raw) * 100, 1) if acc_raw is not None else None,
            "calls": _si(row.get(f"calls_{h}d")),
        }

    predictions = []
    preds_sorted = preds.sort_values("created_utc", ascending=False)
    for _, pr in preds_sorted.iterrows():
        body = str(pr["body_excerpt"]) if pr["body_excerpt"] else ""
        rationale = str(pr["rationale"]) if pr["rationale"] else None
        predictions.append({
            "ticker":         str(pr["ticker"]),
            "stance":         str(pr["stance"]),
            "reasoning_tier": _si(pr["reasoning_tier"]) or 0,
            "q_composite":    _sf(pr["q_composite"], 2),
            "rationale":      rationale,
            "horizon_days":   _si(pr["horizon_days"]),
            "body_excerpt":   body[:500] if body else None,
            "date":           str(pr["created_utc"])[:10],
            "upvotes":        _si(pr["upvotes"]) or 0,
            "subreddit":      str(pr["subreddit"]),
        })

    subreddit_list = [s.strip() for s in subs.split(",") if s.strip()]

    dir_acc = _sf(row.get("brier_score"), 3)

    return jsonify({
        "author":          username,
        "resolved_calls":  _si(row["resolved_calls"]) or 0,
        "avg_return_pct":  round(adj_acc * 100, 1),
        "dir_accuracy":    round(float(dir_acc) * 100, 1) if dir_acc is not None else None,
        "avg_upvotes":     _sf(avg_upvotes, 1) or 0.0,
        "total_comments":  int(comment_count),
        "subreddits":      subreddit_list,
        "horizons":        horizons,
        "predictions":     predictions,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5050)

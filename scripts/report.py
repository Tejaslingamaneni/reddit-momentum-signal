"""Generate summary report with charts for the Reddit Momentum Signal project."""

from __future__ import annotations
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import scipy.stats as stats
import duckdb
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.engine import _download_prices, _next_trading_day, _forward_return

warnings.filterwarnings("ignore")

OUT_DIR = Path(__file__).parent.parent / "reports"
OUT_DIR.mkdir(exist_ok=True)

DB = str(Path(__file__).parent.parent / "data" / "research.duckdb")

# ── palette ──────────────────────────────────────────────────────────────────
BULL   = "#2ecc71"
BEAR   = "#e74c3c"
NEUT   = "#95a5a6"
BLUE   = "#3498db"
DARK   = "#2c3e50"
GOLD   = "#f39c12"
PURPLE = "#9b59b6"

plt.rcParams.update({
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor":   "#ffffff",
    "axes.edgecolor":   "#dee2e6",
    "axes.grid":        True,
    "grid.color":       "#e9ecef",
    "grid.linestyle":   "-",
    "grid.linewidth":   0.5,
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "axes.labelsize":   9,
})


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_scoring():
    con = duckdb.connect(DB)
    tiers = con.execute("""
        SELECT reasoning_tier, stance, COUNT(*) n
        FROM comment_scores WHERE parse_error=FALSE
        GROUP BY reasoning_tier, stance
    """).df()
    totals = con.execute("""
        SELECT reasoning_tier, COUNT(*) n, ROUND(AVG(q_composite),3) avg_q
        FROM comment_scores WHERE parse_error=FALSE
        GROUP BY reasoning_tier ORDER BY reasoning_tier
    """).df()
    con.close()
    return tiers, totals


def load_signals():
    con = duckdb.connect(DB)
    df = con.execute("""
        SELECT ds.ticker, ds.signal_date, ds.conviction, ds.attention_z,
               ds.signal, ds.n_comments_scored,
               (SELECT (SUM(CASE WHEN cs.stance='bullish' THEN 1.0 ELSE 0.0 END)
                        - SUM(CASE WHEN cs.stance='bearish' THEN 1.0 ELSE 0.0 END))
                        / NULLIF(COUNT(*),0)
                FROM comment_scores cs
                JOIN comments c ON c.comment_id = cs.comment_id
                WHERE cs.ticker = ds.ticker
                  AND CAST(DATE_TRUNC('day', c.created_utc) AS DATE) = ds.signal_date
                  AND cs.parse_error=FALSE) naive_net
        FROM daily_signal ds WHERE ds.conviction IS NOT NULL
        ORDER BY ds.signal_date, ds.ticker
    """).df()
    con.close()
    return df


def build_return_panels(sdf, horizons):
    tickers = sdf["ticker"].unique().tolist()
    from datetime import timedelta
    start = (sdf["signal_date"].min() - timedelta(days=5)).strftime("%Y-%m-%d")
    end   = (sdf["signal_date"].max() + timedelta(days=35)).strftime("%Y-%m-%d")
    prices = _download_prices(tickers, start, end)
    if prices.empty:
        return {}
    trading_days = pd.DatetimeIndex(sorted(prices.index.unique()))
    valid = [t for t in tickers if t in prices.columns and not prices[t].dropna().empty]

    panels = {}
    for h in horizons:
        rows = []
        for _, row in sdf.iterrows():
            if row["ticker"] not in valid:
                continue
            fret = _forward_return(row["ticker"], pd.Timestamp(row["signal_date"]),
                                   h, prices, trading_days)
            rows.append({**row, "fwd_ret": fret})
        panels[h] = pd.DataFrame(rows).dropna(subset=["fwd_ret"])
    return panels


def compute_ic(s, r):
    mask = s.notna() & r.notna()
    if mask.sum() < 5:
        return np.nan, np.nan, mask.sum()
    ic, pv = stats.spearmanr(s[mask], r[mask])
    return float(ic), float(pv), int(mask.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────

def chart_tier_breakdown(ax, tiers, totals):
    tier_labels = {0: "Tier 0\nNaive/Spam", 1: "Tier 1\nGeneric",
                   2: "Tier 2\nSpecific", 3: "Tier 3\nFalsifiable"}
    colors = {"bullish": BULL, "bearish": BEAR, "neutral": NEUT, "unclear": "#bdc3c7"}

    stances = ["bullish", "bearish", "neutral", "unclear"]
    bottom  = np.zeros(4)
    for stance in stances:
        vals = []
        for t in range(4):
            sub = tiers[(tiers.reasoning_tier == t) & (tiers.stance == stance)]
            vals.append(int(sub["n"].sum()) if not sub.empty else 0)
        ax.bar([tier_labels[t] for t in range(4)], vals, bottom=bottom,
               color=colors[stance], label=stance.capitalize(), width=0.55, alpha=0.9)
        bottom += np.array(vals)

    ax.set_title("Comment Scores by Reasoning Tier & Stance")
    ax.set_ylabel("# Comments")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    total = int(tiers["n"].sum())
    ax.text(0.98, 0.97, f"N = {total:,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=DARK)


def chart_q_composite(ax, totals):
    colors = [NEUT, "#85c1e9", BLUE, PURPLE]
    bars = ax.bar(
        [f"Tier {r}" for r in totals["reasoning_tier"]],
        totals["avg_q"], color=colors, width=0.55, alpha=0.9
    )
    for bar, val in zip(bars, totals["avg_q"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_title("Average Quality Score (q_composite) by Tier")
    ax.set_ylabel("q_composite (0–1)")
    ax.set_ylim(0, 0.8)


def chart_ic_heatmap(ax, panels):
    horizons = sorted(panels.keys())
    signals  = ["full_signal", "conviction", "attention_z", "naive_net"]
    labels   = ["Full Signal\n(conv×attn)", "Conviction\n(quality-wtd)", "Attention\nz-score", "Naive Net\n(bull−bear)"]

    ic_mat = np.full((len(signals), len(horizons)), np.nan)
    pv_mat = np.full((len(signals), len(horizons)), np.nan)
    n_mat  = np.zeros((len(signals), len(horizons)), dtype=int)

    col_map = {"full_signal": "signal", "conviction": "conviction",
               "attention_z": "attention_z", "naive_net": "naive_net"}

    for j, h in enumerate(horizons):
        panel = panels[h]
        for i, sig in enumerate(signals):
            col = panel[col_map[sig]]
            ic, pv, n = compute_ic(col, panel["fwd_ret"])
            ic_mat[i, j] = ic
            pv_mat[i, j] = pv
            n_mat[i, j]  = n

    vmax = 0.4
    im = ax.imshow(ic_mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman IC", fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_yticks(range(len(signals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Spearman IC by Signal & Horizon")
    ax.set_xlabel("Forward Return Horizon")

    for i in range(len(signals)):
        for j in range(len(horizons)):
            ic = ic_mat[i, j]
            pv = pv_mat[i, j]
            n  = n_mat[i, j]
            if np.isnan(ic) or n < 5:
                txt = "n/a"
            else:
                star = "★" if pv < 0.05 else ("~" if pv < 0.15 else "")
                txt = f"{ic:+.2f}{star}\nn={n}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if abs(ic) > 0.25 else DARK)


def chart_attn_scatter(ax, panels):
    panel = panels.get(5, pd.DataFrame())
    if panel.empty or panel["attention_z"].isna().all():
        ax.set_title("attention_z vs 5-day Return (no data)")
        return
    p = panel.dropna(subset=["attention_z", "fwd_ret"])
    ic, pv, n = compute_ic(p["attention_z"], p["fwd_ret"])
    ax.scatter(p["attention_z"], p["fwd_ret"] * 100, alpha=0.5, s=25, color=BLUE, linewidths=0)
    m, b = np.polyfit(p["attention_z"], p["fwd_ret"] * 100, 1)
    xr = np.linspace(p["attention_z"].min(), p["attention_z"].max(), 100)
    ax.plot(xr, m * xr + b, color=DARK, lw=1.5, linestyle="--")
    ax.axhline(0, color=NEUT, lw=0.8)
    ax.axvline(0, color=NEUT, lw=0.8)
    ax.set_title(f"Attention Spike → 5-Day Return  (IC={ic:+.3f}, p={pv:.3f})")
    ax.set_xlabel("Attention z-score")
    ax.set_ylabel("5-Day Forward Return (%)")
    star = " ★ SIGNIFICANT" if pv < 0.05 else ""
    ax.text(0.97, 0.05, f"n={n}{star}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=DARK,
            fontweight="bold" if pv < 0.05 else "normal")


def chart_naive_scatter(ax, panels):
    panel = panels.get(1, pd.DataFrame())
    if panel.empty or panel["naive_net"].isna().all():
        ax.set_title("naive_net vs 1-day Return (no data)")
        return
    p = panel.dropna(subset=["naive_net", "fwd_ret"])
    ic, pv, n = compute_ic(p["naive_net"], p["fwd_ret"])
    colors = p["naive_net"].apply(lambda v: BULL if v > 0.1 else (BEAR if v < -0.1 else NEUT))
    ax.scatter(p["naive_net"], p["fwd_ret"] * 100, alpha=0.5, s=25, color=colors, linewidths=0)
    m, b = np.polyfit(p["naive_net"], p["fwd_ret"] * 100, 1)
    xr = np.linspace(p["naive_net"].min(), p["naive_net"].max(), 100)
    ax.plot(xr, m * xr + b, color=DARK, lw=1.5, linestyle="--")
    ax.axhline(0, color=NEUT, lw=0.8)
    ax.axvline(0, color=NEUT, lw=0.8)
    ax.set_title(f"Net Reddit Stance → 1-Day Return  (IC={ic:+.3f}, p={pv:.3f})")
    ax.set_xlabel("Naive Net Bullishness  (+1=all bull, −1=all bear)")
    ax.set_ylabel("1-Day Forward Return (%)")
    star = " ★ SIGNIFICANT" if pv < 0.05 else ""
    ax.text(0.97, 0.97, f"n={n}{star}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=DARK,
            fontweight="bold" if pv < 0.05 else "normal")
    legend = [Patch(facecolor=BULL, label="Net Bullish"), Patch(facecolor=BEAR, label="Net Bearish"),
              Patch(facecolor=NEUT, label="Neutral")]
    ax.legend(handles=legend, fontsize=7, loc="lower right")


def chart_signal_timeline(ax, sdf):
    # Pick the 4 tickers with the most signal days
    top = (sdf.dropna(subset=["signal"])
           .groupby("ticker").size().nlargest(5)
           .index.tolist())
    # Remove ETFs/non-stocks for cleaner view
    top = [t for t in top if t not in ("VOO", "SPY", "QQQ", "VT")][:4]

    colors = [BLUE, BULL, BEAR, GOLD]
    for ticker, color in zip(top, colors):
        sub = sdf[sdf["ticker"] == ticker].dropna(subset=["signal"]).sort_values("signal_date")
        if sub.empty:
            continue
        ax.plot(pd.to_datetime(sub["signal_date"]), sub["signal"],
                marker="o", ms=3, lw=1.5, label=ticker, color=color)

    ax.axhline(0, color=NEUT, lw=1, linestyle="--")
    ax.set_title("Signal Timeline — Top Individual Stock Tickers")
    ax.set_ylabel("Signal (conviction × attention_z)")
    ax.set_xlabel("")
    ax.legend(fontsize=8, loc="lower left")
    ax.tick_params(axis="x", rotation=30)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    tiers, totals = load_scoring()
    sdf = load_signals()

    print("Downloading prices & building return panels...")
    panels = build_return_panels(sdf, [1, 3, 5, 10])

    print("Rendering charts...")
    fig = plt.figure(figsize=(18, 13), facecolor="#f8f9fa")
    fig.suptitle(
        "Reddit Momentum Signal — Full Pipeline Summary",
        fontsize=16, fontweight="bold", color=DARK, y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.06, right=0.97, top=0.94, bottom=0.06)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, :2])   # IC heatmap — wide
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, 0])
    ax7 = fig.add_subplot(gs[2, 1])
    ax8 = fig.add_subplot(gs[2, 2])

    chart_tier_breakdown(ax1, tiers, totals)
    chart_q_composite(ax2, totals)

    # Stance pie
    stance_totals = tiers.groupby("stance")["n"].sum()
    colors_pie = [BULL, BEAR, NEUT, "#bdc3c7"]
    stances_order = ["bullish", "bearish", "neutral", "unclear"]
    vals = [int(stance_totals.get(s, 0)) for s in stances_order]
    wedges, texts, autotexts = ax3.pie(
        vals, labels=[s.capitalize() for s in stances_order],
        colors=colors_pie, autopct="%1.0f%%", startangle=140,
        textprops={"fontsize": 8}, pctdistance=0.75,
    )
    ax3.set_title("Stance Distribution")

    chart_ic_heatmap(ax4, panels)
    chart_signal_timeline(ax5, sdf)
    chart_attn_scatter(ax6, panels)
    chart_naive_scatter(ax7, panels)

    # Conviction histogram
    cv = sdf["conviction"].dropna()
    ax8.hist(cv, bins=30, color=PURPLE, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax8.axvline(0, color=DARK, lw=1.2, linestyle="--")
    ax8.set_title("Conviction Distribution")
    ax8.set_xlabel("Conviction (−1=pure bear, +1=pure bull)")
    ax8.set_ylabel("# Ticker-Days")
    ax8.text(0.97, 0.97, f"μ={cv.mean():+.2f}\nσ={cv.std():.2f}",
             transform=ax8.transAxes, ha="right", va="top", fontsize=8)

    out_path = OUT_DIR / "reddit_signal_report.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nReport saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Entry point for author credibility ranking — magnitude-based scoring."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from credibility.resolver import run_credibility

if __name__ == "__main__":
    result = run_credibility()
    if result is None or (isinstance(result, tuple) and len(result[0]) == 0):
        print("No results.")
        sys.exit(1)

    leaderboard, outcomes = result

    print("\n" + "=" * 70)
    print("REDDIT AUTHOR CREDIBILITY LEADERBOARD")
    print("Metric: avg raw return in predicted direction (magnitude-based)")
    print("=" * 70)

    print(f"\nTotal authors ranked: {len(leaderboard)}")
    print(f"Total resolved calls: {len(outcomes)}")
    print(f"Avg signed return:    {outcomes['signed_return'].mean():.2%}")
    print()

    cols = ["author", "resolved_calls", "adj_return", "credibility_weight"]
    available = [c for c in cols if c in leaderboard.columns]
    top = leaderboard.head(25)[available].copy()
    if "adj_return" in top.columns:
        top["adj_return"] = top["adj_return"].map("{:+.1%}".format)
    if "credibility_weight" in top.columns:
        top["credibility_weight"] = top["credibility_weight"].map("{:.3f}".format)
    top.index.name = "rank"

    print("Top 25 analysts by avg return per call:")
    print(top.to_string())

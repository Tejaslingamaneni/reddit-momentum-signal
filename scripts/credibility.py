#!/usr/bin/env python
"""Entry point for Milestone 7 – author credibility ranking."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from credibility.resolver import run_credibility
import pandas as pd

if __name__ == "__main__":
    result = run_credibility()
    if result is None or (isinstance(result, tuple) and len(result[0]) == 0):
        print("No results.")
        sys.exit(1)

    leaderboard, outcomes = result

    print("\n" + "=" * 75)
    print("REDDIT AUTHOR CREDIBILITY LEADERBOARD")
    print("Based on: direction accuracy of non-naive bullish/bearish predictions")
    print("=" * 75)

    print(f"\nTotal authors ranked: {len(leaderboard)}")
    print(f"Total resolved calls: {outcomes['correct'].count()}")
    print(f"Overall accuracy:     {outcomes['correct'].mean():.1%}")
    print()

    # Top 25
    top = leaderboard.head(25)[
        ["author", "resolved_calls", "correct_calls", "raw_accuracy",
         "credibility_weight", "avg_tier", "tickers_called"]
    ].copy()
    top["raw_accuracy"] = top["raw_accuracy"].map("{:.0%}".format)
    top["credibility_weight"] = top["credibility_weight"].map("{:.2f}x".format)
    top["avg_tier"] = top["avg_tier"].map("{:.1f}".format)
    top.index.name = "rank"

    print("Top 25 most active accurate authors:")
    print(top.to_string())

    print("\n--- Distribution of credibility weights ---")
    bins = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    labels = ["0.50-0.70 (contrarian)", "0.70-0.90 (below avg)",
              "0.90-1.10 (average)", "1.10-1.30 (above avg)", "1.30-1.50 (top)"]
    leaderboard["bucket"] = pd.cut(leaderboard["credibility_weight"], bins=bins, labels=labels)
    print(leaderboard["bucket"].value_counts().sort_index().to_string())

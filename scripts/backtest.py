#!/usr/bin/env python
"""Entry point for Milestone 6 – backtest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import run_backtest
import pandas as pd

if __name__ == "__main__":
    df = run_backtest()
    if df.empty:
        print("No results.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS — Spearman IC vs Forward Returns")
    print("=" * 70)

    for horizon in sorted(df["horizon_days"].unique()):
        sub = df[df["horizon_days"] == horizon].set_index("signal")
        print(f"\n  Horizon: {horizon} trading days")
        print(f"  {'Signal':<15} {'IC':>8} {'t-stat':>8} {'p-value':>8} {'n':>6}")
        print(f"  {'-'*47}")
        for sig in ["full_signal", "conviction", "attention_z", "naive_net"]:
            if sig in sub.index:
                r = sub.loc[sig]
                star = " *" if r["p_value"] < 0.05 else ("  ~" if r["p_value"] < 0.15 else "")
                print(f"  {sig:<15} {r['ic']:>8.4f} {r['t_stat']:>8.3f} {r['p_value']:>8.4f} {int(r['n']):>6}{star}")

    print()

#!/usr/bin/env python3
"""Entry point for `make score` (Milestone 3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import structlog

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(10))

from score.scorer import run_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--limit",
    type=int,
    default=100,
    help="Max comments to score (default 100)",
)
args = parser.parse_args()

totals = run_score(limit=args.limit)
print("\n── Score complete ──")
for k, v in totals.items():
    print(f"  {k}: {v:,}")

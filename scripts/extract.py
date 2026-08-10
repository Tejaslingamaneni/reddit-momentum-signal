#!/usr/bin/env python3
"""Entry point for `make extract` (Milestone 2 – ticker extraction)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv

load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(10))

from extract.tickers import run_extract

totals = run_extract()
print("\n── Extract complete ──")
for k, v in totals.items():
    print(f"  {k}: {v:,}")

#!/usr/bin/env python3
"""Entry point for `make ingest`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog
from dotenv import load_dotenv

load_dotenv()
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(10))

from ingest.fetch import run_ingest

totals = run_ingest()
print("\n── Ingest complete ──")
for sub, n in totals.items():
    print(f"  r/{sub}: {n:,} new comments")
print(f"  Total: {sum(totals.values()):,}")

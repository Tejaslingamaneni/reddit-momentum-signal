#!/usr/bin/env python
"""Entry point for Milestone 5 – signal construction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.builder import run_signal

if __name__ == "__main__":
    result = run_signal()
    print(result)

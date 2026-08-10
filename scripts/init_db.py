#!/usr/bin/env python3
"""Initialize the DuckDB database — safe to run multiple times."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from ingest.storage import init_db

cfg = yaml.safe_load(open("config.yaml"))
db_path = cfg["storage"]["duckdb_path"]
Path(db_path).parent.mkdir(parents=True, exist_ok=True)
init_db(db_path)
print(f"Database ready: {db_path}")

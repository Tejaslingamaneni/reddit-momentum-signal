.PHONY: install init ingest extract score signal backtest test lint typecheck

install:
	pip install -e ".[dev]"

init:
	python scripts/init_db.py

ingest:
	python scripts/ingest.py

extract:
	python scripts/extract.py

score:
	python scripts/score.py

credibility:
	python scripts/credibility.py

signal:
	python scripts/signals.py

backtest:
	python scripts/backtest.py

test:
	pytest tests/ -v

lint:
	ruff check .

typecheck:
	mypy . --ignore-missing-imports

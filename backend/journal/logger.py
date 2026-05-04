from __future__ import annotations

"""JSONL persistence for simulated and real-life decision records.

PRAIDS stores generated outputs in `artifacts/` using newline-delimited JSON so
records are append/read friendly and can be flattened into CSV by the frontend.
There are three logs:

- decision history: simulated recommendations across the backtest period
- portfolio ledger: simulated portfolio value, returns, drawdowns, allocations
- live journal: user-entered real-life decisions made after seeing PRAIDS output
"""

import json
from pathlib import Path


JOURNAL_PATH = Path("artifacts/decision_journal.jsonl")
LEDGER_PATH = Path("artifacts/portfolio_ledger.jsonl")
LIVE_JOURNAL_PATH = Path("artifacts/live_journal.jsonl")


def load_journal(path: str | Path = JOURNAL_PATH) -> list[dict]:
    """Load newline-delimited JSON records, returning an empty list if missing."""
    journal_path = Path(path)
    if not journal_path.exists():
        return []
    with journal_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def save_records(records: list[dict], path: str | Path) -> None:
    """Overwrite a JSONL file with the provided records."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")


def save_decision_history(records: list[dict], path: str | Path = JOURNAL_PATH) -> None:
    """Persist the full simulated decision history produced by a pipeline run."""
    save_records(records, path)


def save_portfolio_ledger(records: list[dict], path: str | Path = LEDGER_PATH) -> None:
    """Persist the full simulated portfolio ledger produced by a pipeline run."""
    save_records(records, path)


def load_portfolio_ledger(path: str | Path = LEDGER_PATH) -> list[dict]:
    """Load saved simulated portfolio ledger records."""
    return load_journal(path)


def append_live_journal_entry(entry: dict, path: str | Path = LIVE_JOURNAL_PATH) -> dict:
    """Append one user-entered real-life decision journal record."""
    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def load_live_journal(path: str | Path = LIVE_JOURNAL_PATH) -> list[dict]:
    """Load all real-life user journal entries."""
    return load_journal(path)

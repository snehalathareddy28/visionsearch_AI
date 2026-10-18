"""Persist and retrieve search history."""

import json
from datetime import datetime, timezone
from pathlib import Path

HISTORY_PATH = Path("data/search_history.json")


def save_search(query: str, mode: str, results=None):
    history = load_history()
    history.append({"query": query, "mode": mode, "result_count": len(results or []), "timestamp": datetime.now(timezone.utc).isoformat()})
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history[-100:], indent=2), encoding="utf-8")


def load_history():
    if not HISTORY_PATH.exists():
        return []
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))

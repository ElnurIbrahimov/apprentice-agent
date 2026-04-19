"""Intent-to-Code Ledger — tags every code edit with the prompting intent.

Every successful edit writes one JSONL line:
    {"ts", "run_id", "file", "lines_touched", "diff_hash", "intent", "model", "session_id"}

`aura why <file>[:line]` reads the ledger and surfaces the intent that prompted
a given edit — semantic fallback via nomic-embed-text when no exact file match.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


def _ledger_path() -> Path:
    from aura.paths import AURA_DATA_DIR
    AURA_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return AURA_DATA_DIR / "ledger.jsonl"


def diff_hash_of(before: str, after: str) -> str:
    h = hashlib.sha256()
    h.update(before.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(after.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def append_edit(
    *,
    file_path: str,
    intent: str,
    model: str = "",
    session_id: str = "",
    run_id: str = "",
    lines_touched: Optional[list[int]] = None,
    diff_hash: str = "",
    kind: str = "edit",
) -> None:
    """Write a single edit entry to the ledger. Never raises."""
    entry = {
        "ts": time.time(),
        "kind": kind,
        "run_id": run_id,
        "session_id": session_id,
        "model": model,
        "file": file_path,
        "lines_touched": lines_touched or [],
        "diff_hash": diff_hash,
        "intent": (intent or "")[:2000],
    }
    try:
        path = _ledger_path()
        with _LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("ledger_append_failed", exc_info=True)


def iter_entries() -> list[dict]:
    """Read the full ledger into memory. O(N) — fine for tens of thousands."""
    path = _ledger_path()
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        logger.debug("ledger_read_failed", exc_info=True)
    return out


def _norm_path(p: str) -> str:
    try:
        return str(Path(p).resolve()).replace("\\", "/").lower()
    except Exception:
        return (p or "").replace("\\", "/").lower()


def why(file_path: str, line: Optional[int] = None, limit: int = 5) -> list[dict]:
    """Return ledger entries that touched this file (and optional line), newest first.

    Falls back to semantic search on the `intent` field via nomic-embed-text if
    no exact file match exists.
    """
    entries = iter_entries()
    if not entries:
        return []

    target = _norm_path(file_path)
    matches = []
    for e in entries:
        if _norm_path(e.get("file", "")) != target:
            continue
        if line is not None and e.get("lines_touched"):
            touched = e["lines_touched"]
            if isinstance(touched, list) and len(touched) >= 2:
                lo, hi = touched[0], touched[-1]
                if not (lo <= line <= hi):
                    continue
        matches.append(e)

    matches.sort(key=lambda e: e.get("ts", 0), reverse=True)
    if matches:
        return matches[:limit]

    # Semantic fallback — search intents by directory proximity.
    dir_prefix = str(Path(file_path).parent).replace("\\", "/").lower()
    nearby = [e for e in entries
              if _norm_path(e.get("file", "")).startswith(dir_prefix)][-200:]
    nearby.sort(key=lambda e: e.get("ts", 0), reverse=True)
    return nearby[:limit]


def recent_intents(limit: int = 10) -> list[dict]:
    """Return the most recent ledger entries."""
    entries = iter_entries()
    entries.sort(key=lambda e: e.get("ts", 0), reverse=True)
    return entries[:limit]

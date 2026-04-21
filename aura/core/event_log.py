"""Append-only JSONL event logs for the future `/why` and learned-routing features.

Three files under AURA_DATA_DIR/events/:
  - edits.jsonl             — one record per successful mutating tool call
  - verifications.jsonl     — one per VerificationStage outcome
  - model_overrides.jsonl   — one per `/model` command

Writes are non-blocking (bg_pool) and lock-guarded per-file so concurrent
Aura processes (CLI + daemon + Telegram) can write safely. These logs are
not queried today — they accumulate so that `/why` and a learned router
can be built on top later without a cold-start data problem.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from aura.pools import bg_pool

logger = logging.getLogger(__name__)

_DATA_DIR_ENV = "AURA_DATA_DIR"
_DEFAULT_DATA_DIR = "data"
_EVENTS_SUBDIR = "events"

# Per-file mutex for lock-guarded appends. Keyed by absolute path.
_file_locks: dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


def _data_dir() -> Path:
    """Resolve AURA_DATA_DIR, defaulting to "data/" relative to cwd."""
    return Path(os.environ.get(_DATA_DIR_ENV, _DEFAULT_DATA_DIR))


def _events_path(name: str) -> Path:
    return _data_dir() / _EVENTS_SUBDIR / name


def _file_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _file_locks_guard:
        lock = _file_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _file_locks[key] = lock
        return lock


def _append_jsonl(path: Path, record: dict) -> None:
    """Synchronous append. Caller decides whether to dispatch to bg_pool."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("event_log mkdir failed", exc_info=True)
        return

    lock = _file_lock(path)
    with lock:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            logger.debug("event_log append failed for %s", path, exc_info=True)


def _async_append(path: Path, record: dict) -> None:
    """Fire-and-forget append via bg_pool. Never blocks the caller."""
    try:
        bg_pool().submit(_append_jsonl, path, record)
    except Exception:
        # bg_pool unavailable → fall back to synchronous append so we don't
        # silently drop the record. The caller's hot path won't be this fast
        # path anyway (IO-bound).
        _append_jsonl(path, record)


# ── Public API ───────────────────────────────────────────────────────────

def log_edit(
    session_id: str,
    tool: str,
    path: str,
    prompt: str = "",
    model: str = "",
    iteration: int = 0,
    extra: Optional[dict] = None,
) -> None:
    """Record a successful file-mutating tool call."""
    rec: dict[str, Any] = {
        "ts": time.time(),
        "session_id": session_id,
        "tool": tool,
        "path": path,
        "prompt": (prompt or "")[:500],
        "model": model,
        "iteration": iteration,
    }
    if extra:
        rec.update({k: v for k, v in extra.items() if k not in rec})
    _async_append(_events_path("edits.jsonl"), rec)


def log_verification(
    session_id: str,
    stage: str,
    runner: str,
    status: str,
    duration_s: float,
    failures: Optional[list] = None,
) -> None:
    """Record a VerificationStage outcome per sub-stage (typecheck or tests)."""
    rec: dict[str, Any] = {
        "ts": time.time(),
        "session_id": session_id,
        "stage": stage,
        "runner": runner,
        "status": status,
        "duration_s": round(float(duration_s or 0.0), 3),
        "failure_count": len(failures) if failures else 0,
        # Keep failure detail truncated so logs stay grep-able.
        "failures": (failures or [])[:20],
    }
    _async_append(_events_path("verifications.jsonl"), rec)


def log_model_override(
    session_id: str,
    from_model: str,
    to_model: str,
    prompt_context: str = "",
) -> None:
    """Record a /model override or apply_model_override call.

    Used later for learned-routing (log now, train when ~100+ samples exist).
    prompt_embedding is left None here; a separate bg pass can backfill it
    using the RAG server's nomic-embed-text model when desired.
    """
    rec: dict[str, Any] = {
        "ts": time.time(),
        "session_id": session_id,
        "from_model": from_model or "",
        "to_model": to_model or "",
        "prompt_context": (prompt_context or "")[:500],
        "prompt_embedding": None,
    }
    _async_append(_events_path("model_overrides.jsonl"), rec)


__all__ = ["log_edit", "log_verification", "log_model_override"]

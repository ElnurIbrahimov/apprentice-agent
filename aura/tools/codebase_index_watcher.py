"""Background watcher that keeps `CodebaseIndex` fresh as files change.

On any FS mutation under the project root, debounce 500ms and call `index()`.
The index internally uses mtime + SHA-256 skip tiers, so a full re-scan after
a single-file change is cheap — the scan walks the tree and skips almost
everything at tier-1.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", ".cache", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", "target", ".turbo", "aura_data", "data", "logs",
    ".aura-worktrees", ".aura_checkpoints", "models",
}
_IGNORED_EXT = {".log", ".lock", ".tmp", ".swp", ".swo", ".pyc", ".pyo"}
_DEBOUNCE_SEC = 0.5
_COOLDOWN_SEC = 2.0


def _should_ignore(path: str) -> bool:
    p = Path(path)
    parts = set(p.parts)
    if parts & _IGNORED_DIRS:
        return True
    if p.suffix.lower() in _IGNORED_EXT:
        return True
    if p.name.startswith("."):
        return True
    return False


class _DebouncedReindexer:
    def __init__(self, index_obj) -> None:
        self._index = index_obj
        self._lock = threading.Lock()
        self._pending = False
        self._last_run = 0.0
        self._timer: Optional[threading.Timer] = None

    def trigger(self) -> None:
        with self._lock:
            if self._timer is not None:
                try:
                    self._timer.cancel()
                except Exception:
                    pass
            self._pending = True
            self._timer = threading.Timer(_DEBOUNCE_SEC, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self) -> None:
        with self._lock:
            self._pending = False
            now = time.monotonic()
            if now - self._last_run < _COOLDOWN_SEC:
                # Too soon — reschedule
                self._timer = threading.Timer(_COOLDOWN_SEC, self._run)
                self._timer.daemon = True
                self._timer.start()
                return
            self._last_run = now

        try:
            from aura.pools import bg_submit
            bg_submit(self._index.index)
        except Exception:
            logger.debug("codebase_watcher_reindex_failed", exc_info=True)


_observer = None
_observer_lock = threading.Lock()


def start_watcher(index_obj, project_path: str) -> bool:
    """Start watching `project_path`. Safe to call multiple times — idempotent."""
    global _observer
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        logger.debug("codebase_watcher: watchdog not installed")
        return False

    with _observer_lock:
        if _observer is not None:
            return True

        reindexer = _DebouncedReindexer(index_obj)

        class _Handler(FileSystemEventHandler):
            def on_any_event(self, event):  # type: ignore[override]
                if event.is_directory:
                    return
                src = getattr(event, "src_path", "") or ""
                if _should_ignore(src):
                    return
                reindexer.trigger()

        obs = Observer()
        try:
            obs.schedule(_Handler(), project_path, recursive=True)
            obs.daemon = True
            obs.start()
            _observer = obs
            logger.info("[CodebaseIndex] Watcher started on %s", project_path)
            return True
        except Exception as e:
            logger.debug("codebase_watcher_start_failed", exc_info=True)
            return False


def stop_watcher() -> None:
    global _observer
    with _observer_lock:
        if _observer is not None:
            try:
                _observer.stop()
                _observer.join(timeout=2)
            except Exception:
                pass
            _observer = None

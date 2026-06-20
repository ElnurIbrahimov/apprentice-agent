"""Cached file index for fast @File completions.

Indexed on first use per cwd, cached for 30s with a simple staleness check
(fs mtime of the root directory). On large repos this drops the @File
latency from hundreds of ms to single-digit ms after the first indexing pass.
"""
from __future__ import annotations

import os
import time as _t
from typing import Optional

# Same rules as AtFileCompleter in input.py — keep in sync
default_ignore_dirs = frozenset({
    'node_modules', '.git', '__pycache__', '.venv', 'venv',
    '.tox', '.mypy_cache', '.pytest_cache', 'dist', 'build',
    '.next', '.nuxt', 'coverage', '.eggs',
})
default_code_exts = frozenset({
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java',
    '.json', '.yaml', '.yml', '.md', '.toml', '.html', '.css',
    '.sh', '.sql', '.c', '.cpp', '.h', '.cfg', '.ini', '.env',
})


class FileIndex:
    """Lazy, TTL-cached file tree index for code files.

    Builds on first use, reuses for up to *ttl_seconds* or until the
    root directory mtime changes (simple "something may have changed" signal).
    Thread-safe for single-producer (build) / concurrent-consumer (search)
    usage because the index is rebuilt atomically and searches read the
    current reference without locking.
    """

    def __init__(
        self,
        *,
        code_exts: Optional[set[str]] = None,
        ignore_dirs: Optional[set[str]] = None,
        max_depth: int = 4,
        ttl_seconds: float = 30.0,
        max_files: int = 5000,
    ) -> None:
        self._code_exts = code_exts or default_code_exts
        self._ignore_dirs = ignore_dirs or default_ignore_dirs
        self._max_depth = max_depth
        self._ttl_seconds = ttl_seconds
        self._max_files = max_files
        self._entries: list[tuple[str, str, str]] = []  # (rel_path, basename, ext)
        self._last_index_ts: float = 0.0
        self._last_index_cwd: str = ""
        self._last_root_mtime: float = 0.0

    # ─── Build / stale check ────────────────────────────────────────

    def _is_stale(self, cwd: str) -> bool:
        if not self._entries:
            return True
        if cwd != self._last_index_cwd:
            return True
        if _t.monotonic() - self._last_index_ts > self._ttl_seconds:
            return True
        # Quick staleness probe: root dir mtime changed → re-index
        try:
            curr_mtime = os.stat(cwd).st_mtime
            if curr_mtime != self._last_root_mtime:
                return True
        except OSError:
            pass
        return False

    def build(self, cwd: str) -> None:
        """Walk *cwd* and populate the index. May raise OSError on bad paths."""
        entries: list[tuple[str, str, str]] = []
        sep = os.sep
        code_exts = self._code_exts
        ignore_dirs = self._ignore_dirs
        max_depth = self._max_depth
        max_files = self._max_files

        for root, dirs, files in os.walk(cwd):
            # Prune
            dirs[:] = [
                d for d in dirs
                if d not in ignore_dirs and not d.endswith('.egg-info')
            ]
            rel_root = os.path.relpath(root, cwd)
            depth = rel_root.count(sep) + (0 if rel_root == '.' else 1)
            if depth > max_depth:
                dirs.clear()
                continue

            for fname in files:
                _, ext = os.path.splitext(fname)
                if ext.lower() not in code_exts:
                    continue
                rel_path = os.path.join(rel_root, fname).replace(sep, '/')
                if rel_path.startswith('./'):
                    rel_path = rel_path[2:]
                entries.append((rel_path, fname, ext))
                if len(entries) >= max_files:
                    break
            if len(entries) >= max_files:
                break

        # Atomic swap
        self._entries = entries
        self._last_index_cwd = cwd
        self._last_index_ts = _t.monotonic()
        try:
            self._last_root_mtime = os.stat(cwd).st_mtime
        except OSError:
            self._last_root_mtime = 0.0

    # ─── Search ───────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 20) -> list[tuple[str, str]]:
        """Return up to *max_results* (rel_path, ext) matching *query*.

        Sorting: starts-with > contains > shorter paths first.
        """
        ql = query.lower()
        matches: list[tuple[int, int, str, str]] = []  # (rank, len, rel_path, ext)
        for rel_path, _fname, ext in self._entries:
            pl = rel_path.lower()
            if ql in pl:
                rank = 0 if pl.startswith(ql) else 1
                matches.append((rank, len(pl), rel_path, ext))
        matches.sort(key=lambda t: (t[0], t[1]))
        return [(rel_path, ext) for _, _, rel_path, ext in matches[:max_results]]

    # ─── Entry point used by completer ────────────────────────────────

    def get(self, cwd: str, query: str, max_results: int = 20) -> list[tuple[str, str]]:
        """Build if stale, then search."""
        if self._is_stale(cwd):
            try:
                self.build(cwd)
            except OSError:
                return []
        return self.search(query, max_results)


# Module-level singleton so *all* completer instances share one index.
_shared_index: Optional[FileIndex] = None


def get_file_index(
    *,
    code_exts: Optional[set[str]] = None,
    ignore_dirs: Optional[set[str]] = None,
) -> FileIndex:
    """Return the shared FileIndex singleton, creating it on first call."""
    global _shared_index
    if _shared_index is None:
        _shared_index = FileIndex(
            code_exts=code_exts,
            ignore_dirs=ignore_dirs,
        )
    return _shared_index

# aura/cli/watch_mode.py
"""Watch mode — monitor files for AI comments and auto-respond."""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

# Patterns that trigger watch mode
WATCH_PATTERNS = [
    re.compile(r'#\s*AURA:\s*(.+)$', re.MULTILINE),       # # AURA: fix this
    re.compile(r'//\s*AURA:\s*(.+)$', re.MULTILINE),       # // AURA: fix this
    re.compile(r'#\s*AI:\s*(.+)$', re.MULTILINE),          # # AI: fix this
    re.compile(r'//\s*AI:\s*(.+)$', re.MULTILINE),         # // AI: fix this
    re.compile(r'/\*\s*AURA:\s*(.+?)\s*\*/', re.DOTALL),   # /* AURA: fix this */
]

# File extensions to watch
WATCH_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".cs", ".swift", ".kt",
    ".sh", ".yaml", ".yml", ".json", ".md", ".html", ".css",
    ".sql", ".r", ".scala", ".lua",
}

# Directories to skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".aura_checkpoints", "data",
}


@dataclass
class WatchHit:
    """A detected AI comment in a file."""
    file_path: str
    line_number: int
    instruction: str
    pattern_type: str  # "AURA" or "AI"
    resolved: bool = False
    response: str = ""


class FileWatcher:
    """Watches project files for AI comment markers."""

    def __init__(self, root: Optional[str] = None, poll_interval: float = 2.0):
        self._root = Path(root or os.getcwd())
        self._poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._hits: List[WatchHit] = []
        self._seen: Set[str] = set()  # file:line:instruction fingerprints
        self._on_hit: Optional[Callable[[WatchHit], None]] = None
        self._lock = threading.Lock()
        self._file_mtimes: Dict[str, float] = {}

    def set_callback(self, callback: Callable[[WatchHit], None]) -> None:
        """Set callback fired when a new AI comment is detected."""
        self._on_hit = callback

    def start(self) -> None:
        """Start watching in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="file-watcher")
        self._thread.start()

    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    def get_hits(self) -> List[WatchHit]:
        """Get all detected hits."""
        with self._lock:
            return list(self._hits)

    def get_unresolved(self) -> List[WatchHit]:
        """Get unresolved hits."""
        with self._lock:
            return [h for h in self._hits if not h.resolved]

    def mark_resolved(self, hit: WatchHit, response: str = "") -> None:
        """Mark a hit as resolved."""
        hit.resolved = True
        hit.response = response

    def scan_file(self, file_path: Path) -> List[WatchHit]:
        """Scan a single file for AI comments."""
        hits = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            return []

        for pattern in WATCH_PATTERNS:
            for match in pattern.finditer(content):
                instruction = match.group(1).strip()
                # Find line number
                line_num = content[:match.start()].count("\n") + 1
                fingerprint = f"{file_path}:{line_num}:{instruction}"

                with self._lock:
                    if fingerprint not in self._seen:
                        hit = WatchHit(
                            file_path=str(file_path),
                            line_number=line_num,
                            instruction=instruction,
                            pattern_type="AURA" if "AURA" in pattern.pattern else "AI",
                        )
                        hits.append(hit)
                        self._seen.add(fingerprint)
        return hits

    def scan_all(self) -> List[WatchHit]:
        """Scan all project files."""
        all_hits = []
        for root, dirs, files in os.walk(self._root):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in WATCH_EXTENSIONS:
                    continue

                fpath = Path(root) / fname
                hits = self.scan_file(fpath)
                if hits:
                    with self._lock:
                        self._hits.extend(hits)
                    all_hits.extend(hits)
        return all_hits

    def _poll_loop(self) -> None:
        """Background polling loop — checks for changed files."""
        while self._running:
            try:
                self._check_changed_files()
            except Exception:
                pass
            time.sleep(self._poll_interval)

    def _check_changed_files(self) -> None:
        """Check for files modified since last scan."""
        for root, dirs, files in os.walk(self._root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext not in WATCH_EXTENSIONS:
                    continue

                fpath = Path(root) / fname
                try:
                    mtime = fpath.stat().st_mtime
                except OSError:
                    continue

                fpath_str = str(fpath)
                with self._lock:
                    if fpath_str in self._file_mtimes and mtime <= self._file_mtimes[fpath_str]:
                        skip = True
                    else:
                        self._file_mtimes[fpath_str] = mtime
                        skip = False
                if skip:
                    continue
                hits = self.scan_file(fpath)
                if hits:
                    with self._lock:
                        self._hits.extend(hits)
                    for hit in hits:
                        if self._on_hit:
                            try:
                                self._on_hit(hit)
                            except Exception:
                                pass

    def clear(self) -> None:
        """Clear all hits and reset state."""
        with self._lock:
            self._hits.clear()
            self._seen.clear()
            self._file_mtimes.clear()


def remove_ai_comment(file_path: str, line_number: int) -> bool:
    """Remove an AI comment from a file after resolving it."""
    try:
        path = Path(file_path)
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        if 0 < line_number <= len(lines):
            line = lines[line_number - 1]
            # Remove the AI comment but keep the rest of the line
            # Skip DOTALL block-comment patterns — only apply line-level patterns
            for pattern in WATCH_PATTERNS:
                if pattern.flags & re.DOTALL:
                    continue
                cleaned = pattern.sub("", line).rstrip()
                if cleaned != line.rstrip():
                    if cleaned.strip():
                        lines[line_number - 1] = cleaned + "\n"
                    else:
                        lines[line_number - 1] = ""  # Remove entirely if line was just the comment
                    path.write_text("".join(lines))
                    return True
        return False
    except (OSError, PermissionError):
        return False


def create_watch_indicator(watcher: FileWatcher) -> str:
    """Status bar indicator for watch mode."""
    if not watcher.is_running:
        return ""
    unresolved = len(watcher.get_unresolved())
    if unresolved > 0:
        return f"[magenta]👁 {unresolved} comments[/magenta]"
    return "[dim]👁 watching[/dim]"

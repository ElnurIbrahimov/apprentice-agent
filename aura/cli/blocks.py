"""Block-based output manager for navigable conversation history.

Every agent response, tool call, tool result, error, info message, and
diff is registered as a discrete numbered block. The block registry enables
navigation (/blocks), per-block copy (/copy N), and future export features.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class OutputBlock:
    """A single output block in the conversation."""

    id: int
    block_type: str  # "response", "tool_call", "tool_result", "error", "diff", "info"
    title: str  # one-line summary shown in /blocks list
    content: str  # full content (markdown or plain text)
    expanded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BlockManager:
    """Tracks output blocks for navigation, copy, and export.

    Thread-safe. Capped at _MAX_BLOCKS to prevent unbounded memory growth
    in long-running sessions. When the cap is exceeded, the oldest block
    is dropped.
    """

    _MAX_BLOCKS = 200
    _MAX_CONTENT_BYTES = 8 * 1024  # per-block content cap

    def __init__(self) -> None:
        self._blocks: List[OutputBlock] = []
        self._counter: int = 0
        self._lock = threading.Lock()
        self._on_block_added: Optional[Callable[[OutputBlock], None]] = None

    # -- public API ---------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of blocks currently tracked."""
        with self._lock:
            return len(self._blocks)

    def add(
        self,
        block_type: str,
        title: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        expanded: bool = False,
    ) -> int:
        """Add a block and return its numeric ID."""
        # Truncate content and title to prevent unbounded memory growth
        safe_content = content
        if len(content) > self._MAX_CONTENT_BYTES:
            safe_content = content[:self._MAX_CONTENT_BYTES] + "\n\n\u2026 truncated"
        safe_title = title[:200] if len(title) > 200 else title
        with self._lock:
            self._counter += 1
            block = OutputBlock(
                id=self._counter,
                block_type=block_type,
                title=safe_title,
                content=safe_content,
                expanded=expanded,
                metadata=metadata or {},
            )
            self._blocks.append(block)
            # Prune oldest blocks when cap is exceeded
            while len(self._blocks) > self._MAX_BLOCKS:
                self._blocks.pop(0)

        # Fire callback outside the lock so callers can render without
        # risking deadlock if they call back into BlockManager.
        cb = self._on_block_added
        if cb is not None:
            try:
                cb(block)
            except Exception:
                pass

        return block.id

    def get(self, block_id: int) -> Optional[OutputBlock]:
        """Return the block with *block_id*, or None."""
        with self._lock:
            return next((b for b in self._blocks if b.id == block_id), None)

    def get_recent(self, n: int = 10) -> List[OutputBlock]:
        """Return the *n* most recent blocks."""
        if n <= 0:
            return []
        with self._lock:
            return list(self._blocks[-n:])

    def get_last(self) -> Optional[OutputBlock]:
        """Return the most recently added block, or None."""
        with self._lock:
            return self._blocks[-1] if self._blocks else None

    def set_on_block_added(self, callback: Callable[[OutputBlock], None]) -> None:
        """Register a callback fired every time a block is added."""
        self._on_block_added = callback

    def clear(self) -> None:
        """Remove all blocks (e.g., on /clear or new session)."""
        with self._lock:
            self._blocks.clear()
            self._counter = 0


# ---------------------------------------------------------------------------
# BlockManager is created per-session by ChatSession._init_ui_and_state().
# No process-wide singleton — each session gets its own block registry.
# ---------------------------------------------------------------------------

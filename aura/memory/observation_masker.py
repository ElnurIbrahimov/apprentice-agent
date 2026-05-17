"""Observation masking for verbose tool outputs.

The problem this solves:
    Long agent sessions die to context bloat. Every `read_file`,
    `grep`, `ls`, `code_search`, or `web_search` dumps hundreds or
    thousands of tokens into the conversation. Most of it is never
    read again, but it still consumes the LLM's context budget on
    every subsequent turn.

The fix (observation masking, per Cursor / Devin 2025):
    Intercept tool outputs BEFORE they enter the LLM loop. If an
    output is large, stash the full content under a content-hashed
    ID, and hand the LLM a compact placeholder:

        ⟦OBS:a3f9c2:180 lines · 6421 chars · tool=read_file⟧
        head> first N lines...
        tail> last M lines...

    Later, if the LLM actually needs the full content (e.g. to
    reason about line 120), it calls `expand_observation("a3f9c2")`
    and the masker returns the original text for one-shot injection
    into the next turn.

    Net effect: large outputs cost ~150 tokens in context instead
    of thousands, and nothing is lost — the store holds the full
    version for on-demand retrieval.

Design notes:
    - Content-addressed (SHA-256 of raw text) → identical outputs
      dedupe automatically.
    - LRU eviction with a token-weighted cap so the store stays
      bounded on long sessions.
    - Thread-safe for use from Aura's pool executors.
    - Zero external dependencies.

Usage:
    from aura.memory.observation_masker import ObservationMasker

    masker = ObservationMasker()

    # In your tool runner:
    raw = some_tool(...)
    payload = masker.mask(raw, tool_name="read_file", origin=path)
    # payload.display -> compact version for the LLM
    # payload.obs_id  -> short handle, None if not masked

    # Later, when the LLM wants the full content:
    full = masker.expand(obs_id)

    # Periodic housekeeping from idle loop:
    masker.evict_stale()
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Rough token estimator — 4 chars/token is the usual heuristic for English/code.
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class MaskedOutput:
    """Return value of `ObservationMasker.mask()`.

    `display` is what goes into the LLM context. If the content was
    masked, this is the compact placeholder + head/tail preview. If
    the content was short enough, this is simply the raw text.

    `obs_id` is set only when masking actually happened, so callers
    can distinguish the two cases without re-computing the hash.
    """
    display: str
    obs_id: Optional[str]
    was_masked: bool
    original_tokens: int
    masked_tokens: int

    @property
    def savings(self) -> int:
        return max(0, self.original_tokens - self.masked_tokens)


@dataclass
class _Entry:
    obs_id: str
    raw_text: str
    tool_name: str
    origin: str
    tokens: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


class ObservationMasker:
    """Content-addressable store with on-write masking.

    Thresholds are in characters, not tokens, so the masker is cheap
    (no tokenizer call on the hot path). `mask()` uses the estimator
    only for reporting, not for the decision.
    """

    def __init__(
        self,
        mask_threshold_chars: int = 2_400,   # ~600 tokens
        head_lines: int = 8,
        tail_lines: int = 3,
        max_store_tokens: int = 80_000,       # bounded so long sessions don't leak
        max_entries: int = 512,
    ) -> None:
        self.mask_threshold_chars = mask_threshold_chars
        self.head_lines = head_lines
        self.tail_lines = tail_lines
        self.max_store_tokens = max_store_tokens
        self.max_entries = max_entries

        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._total_tokens = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def mask(
        self,
        raw_text: str,
        tool_name: str = "",
        origin: str = "",
    ) -> MaskedOutput:
        """Wrap a tool output. Short outputs pass through unchanged."""
        if raw_text is None:
            raw_text = ""

        text = str(raw_text)
        original_tokens = _estimate_tokens(text)

        if len(text) <= self.mask_threshold_chars:
            return MaskedOutput(
                display=text,
                obs_id=None,
                was_masked=False,
                original_tokens=original_tokens,
                masked_tokens=original_tokens,
            )

        obs_id = self._store(text, tool_name=tool_name, origin=origin)
        display = self._build_placeholder(text, obs_id, tool_name, origin)
        return MaskedOutput(
            display=display,
            obs_id=obs_id,
            was_masked=True,
            original_tokens=original_tokens,
            masked_tokens=_estimate_tokens(display),
        )

    def expand(self, obs_id: str) -> Optional[str]:
        """Return the full original text for a previously-masked output."""
        with self._lock:
            entry = self._entries.get(obs_id)
            if entry is None:
                return None
            entry.last_accessed = time.time()
            entry.access_count += 1
            # LRU bump: move to the end
            self._entries.move_to_end(obs_id)
            return entry.raw_text

    def peek(self, obs_id: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a stored observation without touching LRU."""
        with self._lock:
            entry = self._entries.get(obs_id)
            if entry is None:
                return None
            return {
                "obs_id": entry.obs_id,
                "tool": entry.tool_name,
                "origin": entry.origin,
                "tokens": entry.tokens,
                "created_at": entry.created_at,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
                "chars": len(entry.raw_text),
            }

    def list_ids(self) -> List[str]:
        with self._lock:
            return list(self._entries.keys())

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total_tokens = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "total_tokens": self._total_tokens,
                "max_store_tokens": self.max_store_tokens,
                "max_entries": self.max_entries,
            }

    def evict_stale(self, max_age_seconds: float = 3_600.0) -> int:
        """Drop entries that haven't been accessed in a while. Returns count removed."""
        now = time.time()
        removed = 0
        with self._lock:
            stale_ids = [
                oid for oid, entry in self._entries.items()
                if (now - entry.last_accessed) > max_age_seconds
            ]
            for oid in stale_ids:
                self._drop(oid)
                removed += 1
        if removed:
            logger.debug("[ObservationMasker] evicted %d stale entries", removed)
        return removed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _store(self, text: str, tool_name: str, origin: str) -> str:
        obs_id = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:10]
        tokens = _estimate_tokens(text)

        with self._lock:
            existing = self._entries.get(obs_id)
            if existing is not None:
                existing.last_accessed = time.time()
                self._entries.move_to_end(obs_id)
                return obs_id

            entry = _Entry(
                obs_id=obs_id,
                raw_text=text,
                tool_name=tool_name,
                origin=origin,
                tokens=tokens,
            )
            self._entries[obs_id] = entry
            self._total_tokens += tokens
            self._enforce_caps_locked()
        return obs_id

    def _enforce_caps_locked(self) -> None:
        while (
            len(self._entries) > self.max_entries
            or self._total_tokens > self.max_store_tokens
        ) and self._entries:
            oldest_id, _ = next(iter(self._entries.items()))
            self._drop(oldest_id)

    def _drop(self, obs_id: str) -> None:
        entry = self._entries.pop(obs_id, None)
        if entry is not None:
            self._total_tokens = max(0, self._total_tokens - entry.tokens)

    def _build_placeholder(
        self,
        text: str,
        obs_id: str,
        tool_name: str,
        origin: str,
    ) -> str:
        lines = text.splitlines()
        line_count = len(lines)
        char_count = len(text)

        head = lines[: self.head_lines]
        tail = lines[-self.tail_lines:] if line_count > (self.head_lines + self.tail_lines) else []

        parts = [f"⟦OBS:{obs_id} · {line_count} lines · {char_count} chars"]
        if tool_name:
            parts.append(f"tool={tool_name}")
        if origin:
            parts.append(f"origin={origin}")
        header = " · ".join(parts) + "⟧"

        preview_lines = [header]
        if head:
            preview_lines.append("head>")
            preview_lines.extend(head)
        if tail:
            preview_lines.append(f"… ({line_count - self.head_lines - self.tail_lines} lines elided) …")
            preview_lines.append("tail>")
            preview_lines.extend(tail)
        preview_lines.append(
            f"⟦/OBS⟧ — call `expand_observation(\"{obs_id}\")` to inline the full text."
        )
        return "\n".join(preview_lines)


# ---------------------------------------------------------------------------
# Process-global default masker
# ---------------------------------------------------------------------------

_default_masker: Optional[ObservationMasker] = None
_default_lock = threading.Lock()


def get_default_masker() -> ObservationMasker:
    """Lazy singleton so most callers don't have to thread one through."""
    global _default_masker
    if _default_masker is None:
        with _default_lock:
            if _default_masker is None:
                _default_masker = ObservationMasker()
    return _default_masker


def mask_tool_output(raw_text: str, tool_name: str = "", origin: str = "") -> MaskedOutput:
    """Convenience wrapper around `get_default_masker().mask(...)`."""
    return get_default_masker().mask(raw_text, tool_name=tool_name, origin=origin)


def expand_observation(obs_id: str) -> Optional[str]:
    """Retrieve a previously-masked observation by its ID."""
    return get_default_masker().expand(obs_id)


__all__ = [
    "MaskedOutput",
    "ObservationMasker",
    "expand_observation",
    "get_default_masker",
    "mask_tool_output",
]

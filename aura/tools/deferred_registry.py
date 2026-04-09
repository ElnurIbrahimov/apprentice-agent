"""Deferred Tool Registry — lazy-load heavy tools on demand.

Stores name + description + a loader callable for tools that shouldn't be
instantiated at startup.  The LLM sees only name/description; full tool
instances are resolved on first use via resolve().

Singleton via module-level ``deferred_registry`` instance.

Author: Aura deferred-tools feature (2026-03)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class _DeferredEntry:
    """Internal bookkeeping for one deferred tool."""
    __slots__ = ("description", "instance", "loader", "name")

    def __init__(self, name: str, description: str, loader: Callable):
        self.name = name
        self.description = description
        self.loader = loader
        self.instance: Any = None  # populated on first resolve()


class DeferredToolRegistry:
    """Registry of tools that are deferred (not loaded at startup).

    Three search modes via ``search(query)``:
      - ``"select:ToolA,ToolB"`` — exact name match (comma-separated)
      - ``"+keyword term"`` — name must contain *keyword*, ranked by other terms
      - ``"keyword query"`` — regex match against name + description
    """

    def __init__(self) -> None:
        self._entries: Dict[str, _DeferredEntry] = {}

    # ------------------------------------------------------------------
    #  Registration
    # ------------------------------------------------------------------

    def register(self, name: str, description: str, loader: Callable) -> None:
        """Register a deferred tool.

        Args:
            name: Tool name (must be unique).
            description: Short human-readable description.
            loader: Zero-arg callable that returns the tool instance.
        """
        self._entries[name] = _DeferredEntry(name, description, loader)
        logger.debug("[DeferredRegistry] registered: %s", name)

    # ------------------------------------------------------------------
    #  Search
    # ------------------------------------------------------------------

    def search(self, query: str) -> List[Dict[str, str]]:
        """Search deferred tools.

        Modes:
          ``"select:name1,name2"`` — exact name lookup
          ``"+keyword other terms"`` — name must contain keyword, rank by rest
          ``"free text"`` — regex match on name + description
        """
        query = query.strip()
        if not query:
            return self.list_all()

        # --- Mode 1: select:name1,name2 ---
        if query.startswith("select:"):
            names = [n.strip() for n in query[7:].split(",") if n.strip()]
            results = []
            for n in names:
                entry = self._entries.get(n)
                if entry:
                    results.append({"name": entry.name, "description": entry.description})
            return results

        # --- Mode 2: +keyword other terms ---
        if query.startswith("+"):
            parts = query[1:].split(None, 1)
            required = parts[0].lower()
            extra = parts[1].lower() if len(parts) > 1 else ""

            candidates = []
            for entry in self._entries.values():
                if required in entry.name.lower():
                    score = 0
                    if extra:
                        haystack = f"{entry.name} {entry.description}".lower()
                        for term in extra.split():
                            if term in haystack:
                                score += 1
                    candidates.append((score, entry))

            candidates.sort(key=lambda x: -x[0])
            return [{"name": e.name, "description": e.description} for _, e in candidates]

        # --- Mode 3: regex/keyword search ---
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            # Fall back to literal substring match
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        results = []
        for entry in self._entries.values():
            haystack = f"{entry.name} {entry.description}"
            if pattern.search(haystack):
                results.append({"name": entry.name, "description": entry.description})
        return results

    # ------------------------------------------------------------------
    #  Resolution
    # ------------------------------------------------------------------

    def resolve(self, name: str) -> Any:
        """Instantiate and return a deferred tool by name.

        Returns the cached instance on subsequent calls.
        Returns None if the tool is not registered or loading fails.
        """
        entry = self._entries.get(name)
        if entry is None:
            return None

        if entry.instance is not None:
            return entry.instance

        try:
            inst = entry.loader()
            entry.instance = inst
            logger.info("[DeferredRegistry] resolved: %s", name)
            return inst
        except Exception as e:
            logger.warning("[DeferredRegistry] failed to resolve %s: %s", name, e)
            return None

    # ------------------------------------------------------------------
    #  Listing
    # ------------------------------------------------------------------

    def list_all(self) -> List[Dict[str, str]]:
        """Return ``[{name, description}]`` for every deferred tool."""
        return [
            {"name": e.name, "description": e.description}
            for e in self._entries.values()
        ]

    def has(self, name: str) -> bool:
        """Check if a tool is registered (deferred)."""
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


# ---------------------------------------------------------------------------
#  Module-level singleton
# ---------------------------------------------------------------------------

deferred_registry = DeferredToolRegistry()

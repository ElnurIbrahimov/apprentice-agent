"""Fuzzy-searchable command palette for AURA CLI.

Ctrl+K opens a palette showing slash commands, recent files, and sessions.
Uses prompt_toolkit Application (same pattern as model_picker / session_picker).
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PaletteItem:
    label: str
    action: str = ""  # returned on selection (defaults to label)
    category: str = "command"  # command, file, session, model
    description: str = ""
    def __post_init__(self) -> None:
        if not self.action:
            self.action = self.label

# ── Frecency tracking ───────────────────────────────────────────────────

import json
from pathlib import Path

_FRECENCY_PATH = Path.home() / ".aura" / "frecency.json"

_usage_data: Dict[str, tuple[int, float]] = {}  # action -> (count, last_used_time)
_usage_counts: Dict[str, tuple[int, float]] = _usage_data  # alias for tests


def _load_frecency() -> None:
    """Restore frecency from disk so it survives sessions."""
    global _usage_data
    if not _FRECENCY_PATH.exists():
        return
    try:
        with open(_FRECENCY_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            for action, entry in raw.items():
                if isinstance(entry, list) and len(entry) == 2:
                    _usage_data[action] = (int(entry[0]), float(entry[1]))
    except (OSError, ValueError, json.JSONDecodeError):
        pass


def _save_frecency() -> None:
    """Persist frecency to disk atomically."""
    try:
        _FRECENCY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _tmp = _FRECENCY_PATH.with_suffix(".tmp")
        serializable = {action: [count, ts] for action, (count, ts) in _usage_data.items()}
        with open(_tmp, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        _tmp.replace(_FRECENCY_PATH)
    except OSError:
        pass


def record_usage(action: str) -> None:
    count, _ = _usage_data.get(action, (0, 0.0))
    _usage_data[action] = (count + 1, _time.time())
    _save_frecency()

# Restore persisted frecency on first import so rankings survive sessions.
_load_frecency()

def _frecency_score(action: str) -> float:
    count, last_used = _usage_data.get(action, (0, 0.0))
    if count == 0:
        return 0.0
    recency = max(0.0, 1.0 - (_time.time() - last_used) / 86400)  # decay over 24h
    return count * 0.6 + recency * 40

# ── Fuzzy scoring ────────────────────────────────────────────────────────

def _fuzzy_score(query: str, item: PaletteItem) -> int:
    """Score an item against a query. 0 = no match. Higher = better."""
    q = query.lower()
    label = item.label.lower()
    desc = item.description.lower()

    # Exact match on label
    if q == label:
        return 100
    # Prefix match on label
    if label.startswith(q):
        return 50
    # Substring in label
    if q in label:
        return 25
    # Substring in description
    if q in desc:
        return 15
    # Subsequence match (all query chars in order in label)
    qi = 0
    for ch in label:
        if qi < len(q) and ch == q[qi]:
            qi += 1
    if qi == len(q):
        return 10
    # Same for description
    qi = 0
    for ch in desc:
        if qi < len(q) and ch == q[qi]:
            qi += 1
    if qi == len(q):
        return 5
    return 0

def _fuzzy_match(query: str, item: PaletteItem) -> bool:
    """Return True if query matches item (substring in label or description)."""
    if not query:
        return True
    return _fuzzy_score(query, item) > 0

def _sort_items(items: List[PaletteItem], query: str = "") -> List[PaletteItem]:
    if query:
        scored = [(it, _fuzzy_score(query, it)) for it in items]
        scored = [(it, s) for it, s in scored if s > 0]
        scored.sort(key=lambda x: (-x[1], -_frecency_score(x[0].action)))
        return [it for it, _ in scored]
    return sorted(items, key=lambda it: (-_frecency_score(it.action), it.category, it.label.lower()))

def build_items_from_commands(slash_commands: list[tuple[str, str]]) -> List[PaletteItem]:
    # Tag experimental commands so users can see they're not fully supported.
    try:
        from aura.cli.commands import EXPERIMENTAL_COMMANDS
    except ImportError:
        EXPERIMENTAL_COMMANDS = set()
    items: List[PaletteItem] = []
    for c, d in slash_commands:
        desc = f"[exp] {d}" if c in EXPERIMENTAL_COMMANDS else d
        items.append(PaletteItem(label=c, description=desc, category="command"))
    return items

def build_palette(slash_commands: list[tuple[str, str]], recent_files: Optional[List[str]] = None,
                  sessions: Optional[List[Dict[str, Any]]] = None) -> List[PaletteItem]:
    items = build_items_from_commands(slash_commands)
    for path in (recent_files or []):
        items.append(PaletteItem(label=path, description="Recent file", category="file"))
    for s in (sessions or []):
        title = s.get("title") or s.get("id", "Untitled")
        sid = s.get("id", title)
        items.append(PaletteItem(label=title, description="Session", category="session",
                                 action=f"/sessions switch {sid}"))
    return items

def open_palette(items: List[PaletteItem], console: Any = None) -> Optional[str]:
    sorted_items = _sort_items(items)
    try:
        result = _palette_interactive(sorted_items)
    except Exception:
        result = _palette_fallback(sorted_items, console)
    if result is not None:
        record_usage(result)
    return result

# ── Interactive (prompt_toolkit) ─────────────────────────────────────────

def _palette_interactive(items: List[PaletteItem]) -> Optional[str]:
    from aura.cli.picker import PickerItem, run_picker

    ICON = {"command": "/", "file": "\u25a1", "session": "\u25cb", "model": "\u25c7"}

    picker_items = [
        PickerItem(
            id=it.action,
            label=it.label,
            description=it.description,
            category=it.category,
            icon=ICON.get(it.category, " "),
        )
        for it in items
    ]

    return run_picker(
        picker_items,
        title="Command Palette",
        max_visible=16,
        style_overrides={"selected": "reverse bold"},
    )

# ── Fallback (numbered list) ────────────────────────────────────────────

def _palette_fallback(items: List[PaletteItem], console: Any = None) -> Optional[str]:
    p = console.print if console else print
    p("\n  Command Palette:")
    for i, it in enumerate(items[:20]):
        icon = {"command": "/", "file": "F", "session": "S", "model": "M"}.get(it.category, " ")
        p(f"  {i+1:>2}. [{icon}] {it.label:<30s}  {it.description}")
    try:
        choice = input("  Pick # (q to cancel) > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not choice or choice.lower() in ("q", "esc"):
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < min(20, len(items)):
            return items[idx].action
    except ValueError:
        pass
    return None

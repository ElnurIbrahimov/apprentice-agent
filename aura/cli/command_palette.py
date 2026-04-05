"""Fuzzy-searchable command palette for AURA CLI.

Ctrl+K opens a palette showing slash commands, recent files, and sessions.
Uses prompt_toolkit Application (same pattern as model_picker / session_picker).
"""
from __future__ import annotations
import time as _time
from dataclasses import dataclass
from typing import Any, Optional, List, Dict

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

_usage_data: Dict[str, tuple[int, float]] = {}  # action -> (count, last_used_time)

def record_usage(action: str) -> None:
    count, _ = _usage_data.get(action, (0, 0.0))
    _usage_data[action] = (count + 1, _time.time())

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

def _sort_items(items: List[PaletteItem], query: str = "") -> List[PaletteItem]:
    if query:
        scored = [(it, _fuzzy_score(query, it)) for it in items]
        scored = [(it, s) for it, s in scored if s > 0]
        scored.sort(key=lambda x: (-x[1], -_frecency_score(x[0].action)))
        return [it for it, _ in scored]
    return sorted(items, key=lambda it: (-_frecency_score(it.action), it.category, it.label.lower()))

def build_items_from_commands(slash_commands: list[tuple[str, str]]) -> List[PaletteItem]:
    return [PaletteItem(label=c, description=d, category="command") for c, d in slash_commands]

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

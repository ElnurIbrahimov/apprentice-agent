"""Fuzzy-searchable command palette for AURA CLI.

Ctrl+K opens a palette showing slash commands, recent files, and sessions.
Uses prompt_toolkit Application (same pattern as model_picker / session_picker).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, List, Dict

@dataclass
class PaletteItem:
    label: str
    description: str = ""
    category: str = "command"  # command, file, session, model
    action: str = ""  # returned on selection (defaults to label)
    def __post_init__(self) -> None:
        if not self.action:
            self.action = self.label

_usage_counts: Dict[str, int] = {}

def record_usage(action: str) -> None:
    _usage_counts[action] = _usage_counts.get(action, 0) + 1

def _fuzzy_match(query: str, item: PaletteItem) -> bool:
    q = query.lower()
    return q in item.label.lower() or q in item.description.lower()

def _sort_items(items: List[PaletteItem]) -> List[PaletteItem]:
    return sorted(items, key=lambda it: (-_usage_counts.get(it.action, 0), it.category, it.label.lower()))

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

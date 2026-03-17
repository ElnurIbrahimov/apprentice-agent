"""Fuzzy-searchable command palette for AURA CLI.

Ctrl+K opens a palette showing slash commands, recent files, and sessions.
Uses prompt_toolkit Application (same pattern as model_picker / session_picker).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class PaletteItem:
    label: str
    description: str = ""
    category: str = "command"  # command, file, session, model
    action: str = ""  # returned on selection (defaults to label)
    def __post_init__(self):
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

def build_items_from_commands(slash_commands: list) -> List[PaletteItem]:
    return [PaletteItem(label=c, description=d, category="command") for c, d in slash_commands]

def build_palette(slash_commands: list, recent_files: Optional[List[str]] = None,
                  sessions: Optional[List[Dict]] = None) -> List[PaletteItem]:
    items = build_items_from_commands(slash_commands)
    for path in (recent_files or []):
        items.append(PaletteItem(label=path, description="Recent file", category="file"))
    for s in (sessions or []):
        title = s.get("title") or s.get("id", "Untitled")
        sid = s.get("id", title)
        items.append(PaletteItem(label=title, description="Session", category="session",
                                 action=f"/sessions switch {sid}"))
    return items

def open_palette(items: List[PaletteItem], console=None) -> Optional[str]:
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
    from prompt_toolkit import Application
    from prompt_toolkit.layout import Layout, HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    ftxt, result, st = [""], [None], {"idx": 0, "scroll": 0}
    ICON = {"command": "/", "file": "\u25a1", "session": "\u25cb", "model": "\u25c7"}

    def _filtered():
        if not ftxt[0]:
            return list(enumerate(items))
        return [(i, it) for i, it in enumerate(items) if _fuzzy_match(ftxt[0], it)]

    def _find_pos(filtered):
        return next((j for j, (oi, _) in enumerate(filtered) if oi == st["idx"]), 0)

    def _reset_to_first():
        f = _filtered()
        if f:
            st["idx"], st["scroll"] = f[0][0], 0

    def _display():
        filtered = _filtered()
        fr = [("class:title", "  Command Palette\n")]
        if ftxt[0]:
            fr += [("class:dim", "  > "), ("class:filter", ftxt[0]), ("", "\n")]
        fr.append(("class:dim", "  " + "\u2500" * 55 + "\n"))
        if not filtered:
            fr.append(("class:dim", "  No matches.\n"))
            return fr
        mv = 16
        sp = _find_pos(filtered)
        if sp < st["scroll"]:
            st["scroll"] = sp
        elif sp >= st["scroll"] + mv:
            st["scroll"] = sp - mv + 1
        for _j, (oi, it) in enumerate(filtered[st["scroll"]:st["scroll"] + mv]):
            sel = oi == st["idx"]
            icon = ICON.get(it.category, " ")
            lbl = it.label if len(it.label) <= 30 else it.label[:27] + "..."
            fr.append(("class:selected" if sel else "", f"  {'▸' if sel else ' '} {icon} {lbl:<32s}"))
            fr += [("class:dim", f" {it.description}"), ("", "\n")]
        total = len(filtered)
        if total > mv:
            if st["scroll"] > 0:
                fr.append(("class:dim", "    ... more above\n"))
            rem = total - st["scroll"] - mv
            if rem > 0:
                fr.append(("class:dim", f"    ... {rem} more below\n"))
        fr += [("class:dim", "  " + "\u2500" * 55 + "\n"),
               ("class:hint", "  \u2191\u2193"), ("class:dim", " navigate  "),
               ("class:hint", "Enter"), ("class:dim", " select  "),
               ("class:hint", "Esc"), ("class:dim", " cancel  "),
               ("class:hint", "Type"), ("class:dim", " to filter")]
        return fr

    kb = KeyBindings()

    @kb.add("up")
    def _up(e):
        f = _filtered()
        if f:
            p = _find_pos(f)
            if p > 0:
                st["idx"] = f[p - 1][0]

    @kb.add("down")
    def _down(e):
        f = _filtered()
        if f:
            p = _find_pos(f)
            if p < len(f) - 1:
                st["idx"] = f[p + 1][0]

    @kb.add("enter")
    def _select(e):
        for oi, it in _filtered():
            if oi == st["idx"]:
                result[0] = it.action
                break
        e.app.exit()

    @kb.add("escape")
    def _cancel(e): e.app.exit()

    @kb.add("c-c")
    def _cc(e): e.app.exit()

    @kb.add("backspace")
    def _bs(e):
        if ftxt[0]:
            ftxt[0] = ftxt[0][:-1]
            _reset_to_first()

    @kb.add("<any>")
    def _type(e):
        ch = e.data
        if ch and len(ch) == 1 and ch.isprintable():
            ftxt[0] += ch
            _reset_to_first()

    app = Application(
        layout=Layout(HSplit([Window(content=FormattedTextControl(_display), wrap_lines=False)])),
        key_bindings=kb,
        style=Style.from_dict({"title": "bold cyan", "dim": "#666666", "filter": "bold yellow",
                                "selected": "reverse bold", "hint": "bold cyan"}),
        full_screen=False, mouse_support=False)
    app.run()
    return result[0]

# ── Fallback (numbered list) ────────────────────────────────────────────

def _palette_fallback(items: List[PaletteItem], console=None) -> Optional[str]:
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

"""Reusable interactive picker for AURA CLI.

Provides a prompt_toolkit-based fuzzy-search picker used by
model_picker, command_palette, and any future list-selection UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class PickerItem:
    id: str
    label: str
    description: str = ""
    category: str = ""
    icon: str = ""


def run_picker(
    items: List[PickerItem],
    placeholder: str = "Search...",
    title: str = "Picker",
    max_visible: int = 18,
    style_overrides: Optional[dict] = None,
) -> Optional[str]:
    """Show an interactive picker and return the selected item's id, or None.

    Args:
        items: List of PickerItem to display.
        placeholder: Shown when filter is empty (currently unused in UI but reserved).
        title: Header text.
        max_visible: Max rows shown before scrolling.
        style_overrides: Extra prompt_toolkit style dict entries merged on top of defaults.

    Returns:
        The ``id`` of the selected PickerItem, or None if cancelled.
    """
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    if not items:
        return None

    # ── mutable closure state ────────────────────────────────────────────
    filter_text = [""]
    result: list[Optional[str]] = [None]
    state = {"idx": 0, "scroll": 0}

    # ── filtering with fuzzy scoring ────────────────────────────────────
    def _score_item(q: str, it: PickerItem) -> int:
        """Score picker item against query. 0 = no match. Higher = better."""
        label = it.label.lower()
        desc = it.description.lower()
        if q == label:
            return 100
        if label.startswith(q):
            return 50
        if q in label:
            return 25
        if q in desc:
            return 15
        qi = 0
        for ch in label:
            if qi < len(q) and ch == q[qi]:
                qi += 1
        if qi == len(q):
            return 10
        qi = 0
        for ch in desc:
            if qi < len(q) and ch == q[qi]:
                qi += 1
        if qi == len(q):
            return 5
        return 0

    def _filtered() -> list[tuple[int, PickerItem]]:
        q = filter_text[0].lower()
        if not q:
            return list(enumerate(items))
        scored = [(i, it, _score_item(q, it)) for i, it in enumerate(items)]
        scored = [(i, it, s) for i, it, s in scored if s > 0]
        scored.sort(key=lambda x: -x[2])
        return [(i, it) for i, it, _ in scored]

    def _find_pos(filtered: list[tuple[int, PickerItem]]) -> int:
        return next((j for j, (oi, _) in enumerate(filtered) if oi == state["idx"]), 0)

    def _reset_to_first() -> None:
        f = _filtered()
        if f:
            state["idx"], state["scroll"] = f[0][0], 0

    # ── display ──────────────────────────────────────────────────────────
    def _display() -> list[tuple[str, str]]:
        filtered = _filtered()
        fr: list = [("class:title", f"  {title}\n")]

        if filter_text[0]:
            fr += [("class:dim", "  > "), ("class:filter", filter_text[0]), ("", "\n")]

        fr.append(("class:dim", "  " + "\u2500" * 55 + "\n"))

        if not filtered:
            fr.append(("class:dim", "  No matches.\n"))
            return fr

        sp = _find_pos(filtered)
        if sp < state["scroll"]:
            state["scroll"] = sp
        elif sp >= state["scroll"] + max_visible:
            state["scroll"] = sp - max_visible + 1

        for _j, (oi, it) in enumerate(filtered[state["scroll"]:state["scroll"] + max_visible]):
            sel = oi == state["idx"]
            icon_str = f"{it.icon} " if it.icon else ""
            lbl = it.label if len(it.label) <= 30 else it.label[:27] + "..."
            cursor = "\u25b8" if sel else " "
            cls = "class:selected" if sel else ""
            fr.append((cls, f"  {cursor} {icon_str}{lbl:<32s}"))
            if it.description:
                fr.append(("class:dim", f" {it.description}"))
            fr.append(("", "\n"))

        total = len(filtered)
        if total > max_visible:
            if state["scroll"] > 0:
                fr.append(("class:dim", "    ... more above\n"))
            rem = total - state["scroll"] - max_visible
            if rem > 0:
                fr.append(("class:dim", f"    ... {rem} more below\n"))

        fr += [
            ("class:dim", "  " + "\u2500" * 55 + "\n"),
            ("class:hint", "  \u2191\u2193"), ("class:dim", " navigate  "),
            ("class:hint", "Enter"), ("class:dim", " select  "),
            ("class:hint", "Esc"), ("class:dim", " cancel  "),
            ("class:hint", "Type"), ("class:dim", " to filter"),
        ]
        return fr

    # ── keybindings ──────────────────────────────────────────────────────
    kb = KeyBindings()

    @kb.add("up")
    def _up(e: Any) -> None:
        f = _filtered()
        if f:
            p = _find_pos(f)
            if p > 0:
                state["idx"] = f[p - 1][0]

    @kb.add("down")
    def _down(e: Any) -> None:
        f = _filtered()
        if f:
            p = _find_pos(f)
            if p < len(f) - 1:
                state["idx"] = f[p + 1][0]

    @kb.add("enter")
    def _select(e: Any) -> None:
        for oi, it in _filtered():
            if oi == state["idx"]:
                result[0] = it.id
                break
        e.app.exit()

    @kb.add("escape")
    def _cancel(e: Any) -> None:
        e.app.exit()

    @kb.add("c-c")
    def _cc(e: Any) -> None:
        e.app.exit()

    @kb.add("backspace")
    def _bs(e: Any) -> None:
        if filter_text[0]:
            filter_text[0] = filter_text[0][:-1]
            _reset_to_first()

    @kb.add("<any>")
    def _type(e: Any) -> None:
        ch = e.data
        if ch and len(ch) == 1 and ch.isprintable():
            filter_text[0] += ch
            _reset_to_first()

    # ── style ────────────────────────────────────────────────────────────
    base_style = {
        "title": "bold cyan",
        "dim": "#666666",
        "filter": "bold yellow",
        "selected": "reverse bold",
        "hint": "bold cyan",
    }
    if style_overrides:
        base_style.update(style_overrides)

    app = Application(
        layout=Layout(HSplit([
            Window(content=FormattedTextControl(_display), wrap_lines=False),
        ])),
        key_bindings=kb,
        style=Style.from_dict(base_style),
        full_screen=False,
        mouse_support=False,
    )
    app.run()
    return result[0]

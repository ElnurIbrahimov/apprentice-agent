# aura/cli/session_picker.py
"""Interactive session browser with arrow keys, preview, and metadata."""
from __future__ import annotations
import time
from typing import Optional, List, Dict
from rich.console import Console


def pick_session(console: Console, sessions: List[Dict], current_session_id: str = "") -> Optional[Dict]:
    """Interactive session picker. Returns selected session dict or None if cancelled.

    Falls back to simple numbered list if prompt_toolkit is unavailable.
    """
    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return None

    try:
        return _pick_session_interactive(sessions, current_session_id)
    except Exception:
        return _pick_session_fallback(console, sessions, current_session_id)


def _format_session_line(session: Dict, width: int = 60, is_current: bool = False) -> str:
    """Format a session for display in the picker."""
    title = (session.get("title") or "Untitled")[:35]

    # Time formatting
    updated = session.get("updated_at", session.get("created_at", 0))
    if isinstance(updated, (int, float)) and updated > 0:
        elapsed = time.time() - updated
        if elapsed < 3600:
            time_str = f"{int(elapsed / 60)}m ago"
        elif elapsed < 86400:
            time_str = f"{int(elapsed / 3600)}h ago"
        elif elapsed < 604800:
            time_str = f"{int(elapsed / 86400)}d ago"
        else:
            time_str = time.strftime("%b %d", time.localtime(updated))
    else:
        time_str = "unknown"

    msg_count = session.get("stats", {}).get("message_count", session.get("message_count", 0))
    model = (session.get("model") or "auto")[:15]

    marker = " \u2190" if is_current else ""
    return f"{title:<35} {msg_count:>3} msgs  {model:<15} {time_str:>8}{marker}"


def _pick_session_interactive(sessions: List[Dict], current_session_id: str) -> Optional[Dict]:
    """Full interactive picker using prompt_toolkit."""
    from prompt_toolkit import Application
    from prompt_toolkit.layout import Layout, HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style

    filter_text = [""]  # mutable container for closure
    result = [None]  # mutable container for closure
    state = {"idx": 0, "scroll_offset": 0}

    # Find current session index
    for i, s in enumerate(sessions):
        if s.get("id") == current_session_id:
            state["idx"] = i
            break

    def get_filtered():
        ft = filter_text[0].lower()
        if not ft:
            return list(enumerate(sessions))
        return [(i, s) for i, s in enumerate(sessions)
                if ft in (s.get("title") or "").lower()
                or ft in (s.get("model") or "").lower()
                or ft in (s.get("project") or "").lower()]

    def get_display():
        filtered = get_filtered()
        fragments = []

        # Header
        header = f" {'Session':<35} {'Msgs':>5}  {'Model':<15} {'Last Active':>8}"
        fragments.append(("class:header", header + "\n"))
        fragments.append(("class:separator", " " + "\u2500" * 75 + "\n"))

        if not filtered:
            fragments.append(("class:dim", " (no matching sessions)\n"))
            return fragments

        # Scrolling
        max_visible = 20
        sel_pos = 0
        for j, (orig_i, _) in enumerate(filtered):
            if orig_i == state["idx"]:
                sel_pos = j
                break

        if sel_pos < state["scroll_offset"]:
            state["scroll_offset"] = sel_pos
        elif sel_pos >= state["scroll_offset"] + max_visible:
            state["scroll_offset"] = sel_pos - max_visible + 1

        offset = state["scroll_offset"]
        visible = filtered[offset:offset + max_visible]

        for j, (orig_i, session) in enumerate(visible):
            is_selected = (orig_i == state["idx"])
            is_current = session.get("id", "") == current_session_id
            text = _format_session_line(session, is_current=is_current)

            if is_selected:
                style = "class:selected"
                prefix = "\u25b8 "
            else:
                style = "class:current" if is_current else ""
                prefix = "  "
            fragments.append((style, prefix + text + "\n"))

        # Scroll indicators
        total = len(filtered)
        if total > max_visible:
            if offset > 0:
                fragments.append(("class:dim", "  ... more above\n"))
            if offset + max_visible < total:
                fragments.append(("class:dim", f"  ... {total - offset - max_visible} more below\n"))

        # Footer
        fragments.append(("class:separator", " " + "\u2500" * 75 + "\n"))
        if filter_text[0]:
            fragments.append(("class:filter", f" Filter: {filter_text[0]}"))
            fragments.append(("class:dim", "  |  "))
        fragments.append(("class:hint", " \u2191\u2193"))
        fragments.append(("class:dim", " navigate  "))
        fragments.append(("class:hint", "Enter"))
        fragments.append(("class:dim", " select  "))
        fragments.append(("class:hint", "Del"))
        fragments.append(("class:dim", " delete  "))
        fragments.append(("class:hint", "Esc"))
        fragments.append(("class:dim", " cancel  "))
        fragments.append(("class:hint", "Type"))
        fragments.append(("class:dim", " to filter"))

        return fragments

    control = FormattedTextControl(get_display)
    window = Window(content=control, always_hide_cursor=True, wrap_lines=False)

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        filtered = get_filtered()
        if not filtered:
            return
        pos = 0
        for j, (orig_i, _) in enumerate(filtered):
            if orig_i == state["idx"]:
                pos = j
                break
        if pos > 0:
            state["idx"] = filtered[pos - 1][0]

    @kb.add("down")
    def _down(event):
        filtered = get_filtered()
        if not filtered:
            return
        pos = 0
        for j, (orig_i, _) in enumerate(filtered):
            if orig_i == state["idx"]:
                pos = j
                break
        if pos < len(filtered) - 1:
            state["idx"] = filtered[pos + 1][0]

    @kb.add("enter")
    def _select(event):
        filtered = get_filtered()
        for orig_i, session in filtered:
            if orig_i == state["idx"]:
                result[0] = session
                event.app.exit()
                return
        event.app.exit()

    @kb.add("escape")
    def _cancel(event):
        event.app.exit()

    @kb.add("c-c")
    def _ctrl_c(event):
        event.app.exit()

    @kb.add("delete")
    def _delete(event):
        filtered = get_filtered()
        for orig_i, session in filtered:
            if orig_i == state["idx"]:
                result[0] = {"__action__": "delete", "session": session}
                event.app.exit()
                return
        event.app.exit()

    # Type to filter
    @kb.add("<any>")
    def _type(event):
        char = event.data
        if char and len(char) == 1 and char.isprintable():
            filter_text[0] += char
            filtered = get_filtered()
            if filtered:
                state["idx"] = filtered[0][0]
                state["scroll_offset"] = 0

    @kb.add("backspace")
    def _backspace(event):
        if filter_text[0]:
            filter_text[0] = filter_text[0][:-1]
            filtered = get_filtered()
            if filtered:
                state["idx"] = filtered[0][0]
                state["scroll_offset"] = 0

    style = Style.from_dict({
        "header": "bold cyan",
        "separator": "#666666",
        "selected": "reverse bold",
        "current": "green",
        "dim": "#666666",
        "filter": "bold yellow",
        "hint": "bold cyan",
    })

    app = Application(
        layout=Layout(window),
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
    )
    app.run()
    return result[0]


def _pick_session_fallback(console: Console, sessions: List[Dict], current_session_id: str) -> Optional[Dict]:
    """Simple numbered list fallback."""
    console.print("\n[bold cyan]Sessions:[/bold cyan]")
    for i, session in enumerate(sessions[:20]):
        is_current = session.get("id", "") == current_session_id
        line = _format_session_line(session, is_current=is_current)
        marker = "[green]\u2192[/green]" if is_current else " "
        console.print(f"  {marker} {i + 1:>2}. {line}")

    console.print(f"\n  0. Cancel")
    try:
        choice = input("Select session: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if choice.isdigit() and 0 < int(choice) <= min(20, len(sessions)):
        return sessions[int(choice) - 1]
    return None

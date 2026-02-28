# D:/Aura/aura/cli/input.py
"""prompt_toolkit-based input for AURA CLI — history, autocomplete.

Falls back to plain input() when prompt_toolkit can't attach to the console
(piped stdin, MSYS2 without winpty, etc.).
"""

from pathlib import Path

HISTORY_FILE = Path.home() / ".aura_history"

_session_ok = True  # whether prompt_toolkit session is usable


def create_session():
    """Create a prompt_toolkit session with persistent history.

    Returns a PromptSession if prompt_toolkit works on this terminal,
    otherwise returns None (signals get_input to use plain input()).
    """
    global _session_ok
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.styles import Style

        _style = Style.from_dict({"prompt": "bold cyan"})
        session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            style=_style,
        )
        _session_ok = True
        return session
    except Exception:
        _session_ok = False
        return None


def get_input(session) -> "str | None":
    """
    Get one line of user input.
    Returns None on Ctrl+D / Ctrl+C (signal to exit).
    Uses prompt_toolkit if available, plain input() otherwise.
    """
    try:
        if session is not None and _session_ok:
            return session.prompt("\nYou: ").strip()
        else:
            return input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

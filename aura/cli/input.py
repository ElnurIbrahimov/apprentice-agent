# D:/Aura/aura/cli/input.py
"""prompt_toolkit-based input for AURA CLI — history, autocomplete."""

from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style

HISTORY_FILE = Path.home() / ".aura_history"

_STYLE = Style.from_dict({
    "prompt": "bold cyan",
})


def create_session() -> PromptSession:
    """Create a prompt_toolkit session with persistent history."""
    return PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        style=_STYLE,
    )


def get_input(session: PromptSession) -> "str | None":
    """
    Get one line of user input.
    Returns None on Ctrl+D / Ctrl+C (signal to exit).
    """
    try:
        return session.prompt("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

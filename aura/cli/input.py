"""prompt_toolkit-based input for AURA CLI — styled prompt, keybindings, model picker.

Falls back to plain input() when prompt_toolkit can't attach to the console.
"""

from pathlib import Path

HISTORY_FILE = Path.home() / ".aura_history"

_session_ok = True
_model_pick_requested = False  # flag for Ctrl+M


def create_session():
    """Create a prompt_toolkit session with styled prompt, history, and keybindings."""
    global _session_ok
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.styles import Style
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.key_binding import KeyBindings

        _style = Style.from_dict({
            "prompt": "bold cyan",
            "placeholder": "#666666 italic",
        })

        kb = KeyBindings()

        @kb.add("c-m")  # Ctrl+M
        def _ctrl_m(event):
            global _model_pick_requested
            _model_pick_requested = True
            event.app.exit(result="__MODEL_PICK__")

        session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            style=_style,
            key_bindings=kb,
            placeholder=HTML('<style fg="#666666"><i>Type a message, / for commands...</i></style>'),
        )
        _session_ok = True
        return session
    except Exception:
        _session_ok = False
        return None


def get_input(session) -> "str | None":
    """Get user input. Returns None on exit, '__MODEL_PICK__' for Ctrl+M."""
    try:
        if session is not None and _session_ok:
            result = session.prompt([("class:prompt", "\n  ❯ ")]).strip()
            return result
        else:
            return input("\n  ❯ ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

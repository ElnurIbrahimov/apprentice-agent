"""prompt_toolkit-based input for AURA CLI — styled prompt, keybindings, completions.

Falls back to plain input() when prompt_toolkit can't attach to the console.
"""

from pathlib import Path

HISTORY_FILE = Path.home() / ".aura_history"

_session_ok = True

# All slash commands with descriptions (for autocomplete)
SLASH_COMMANDS = [
    ("/quit", "Exit AURA"),
    ("/exit", "Exit AURA"),
    ("/clear", "Clear conversation history"),
    ("/model", "View/set model (auto, <name>)"),
    ("/compact", "Compact conversation history"),
    ("/plan", "Create and execute a plan"),
    ("/shell", "Execute shell command"),
    ("/bash", "Execute shell command"),
    ("/run", "Execute shell command"),
    ("/grep", "Search code content"),
    ("/search", "Search files by pattern"),
    ("/find", "Find definitions/references"),
    ("/edit", "Read file with line numbers"),
    ("/project", "Project info/context/index"),
    ("/agent", "Run specialist agent"),
    ("/sessions", "Manage sessions"),
    ("/browse", "Browse web pages"),
    ("/hook", "Manage hooks"),
    ("/speak", "Text-to-speech"),
    ("/recall", "Search memories"),
    ("/goal", "Run a goal"),
]


def create_session():
    """Create a prompt_toolkit session with styled prompt, history, completions, and keybindings."""
    global _session_ok
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.styles import Style
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.completion import Completer, Completion

        class SlashCompleter(Completer):
            """Show slash command completions when typing /."""
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor.lstrip()
                if text.startswith("/"):
                    prefix = text.lower()
                    for cmd, desc in SLASH_COMMANDS:
                        if cmd.startswith(prefix):
                            yield Completion(
                                cmd,
                                start_position=-len(text),
                                display_meta=desc,
                            )

        _style = Style.from_dict({
            "prompt": "bold cyan",
            "placeholder": "#666666 italic",
            "completion-menu.completion": "bg:#1a1a2e #e0e0e0",
            "completion-menu.completion.current": "bg:#16213e #00d2ff bold",
            "completion-menu.meta.completion": "bg:#1a1a2e #888888",
            "completion-menu.meta.completion.current": "bg:#16213e #aaaaaa",
        })

        kb = KeyBindings()

        @kb.add("escape", "m")  # Alt+M or Esc then M
        def _model_pick(event):
            event.app.exit(result="__MODEL_PICK__")

        session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=SlashCompleter(),
            complete_while_typing=True,
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
    """Get user input. Returns None on exit, '__MODEL_PICK__' for Alt+M."""
    try:
        if session is not None and _session_ok:
            result = session.prompt([("class:prompt", "\n  > ")]).strip()
            return result
        else:
            return input("\n  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

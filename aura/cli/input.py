"""prompt_toolkit-based input for AURA CLI — styled prompt, keybindings, completions.

Falls back to plain input() when prompt_toolkit can't attach to the console.
"""

from pathlib import Path

HISTORY_FILE = Path.home() / ".aura_history"

_session_ok = True

# Signal constants for keybindings — returned as pseudo-input from prompt_toolkit
SIGNAL_MODEL_PICK = "__MODEL_PICK__"
SIGNAL_CLEAR_SCREEN = "__CLEAR_SCREEN__"
SIGNAL_NEW_SESSION = "__NEW_SESSION__"
SIGNAL_COMMAND_PALETTE = "__CMD_PALETTE__"
SIGNAL_OPEN_EDITOR = "__OPEN_EDITOR__"
SIGNAL_REWIND = "__REWIND__"
SIGNAL_CYCLE_PERMS = "__CYCLE_PERMS__"

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
    ("/say", "Text-to-speech (alias)"),
    ("/recall", "Search memories"),
    ("/goal", "Run a goal"),
    ("/trust", "Enable trust mode (auto-approve all tools)"),
    ("/cost", "Show session cost breakdown"),
    ("/context", "Show context window usage"),
    ("/rewind", "Rewind file changes to a checkpoint"),
    ("/theme", "Switch color theme"),
]

# Subcommand completions for commands that accept them
SUBCOMMANDS: dict[str, list[tuple[str, str]]] = {
    "/model": [
        ("auto", "Auto-select best model"),
        ("deepseek-r1:8b", "DeepSeek R1 8B (local)"),
        ("qwen3:8b", "Qwen 3 8B (local)"),
        ("qwen2.5-coder:7b", "Qwen 2.5 Coder 7B (local)"),
        ("devstral-2:123b", "Devstral 2 123B (cloud)"),
        ("cogito-2.1:671b", "Cogito 2.1 671B (cloud)"),
        ("qwen3-coder:480b", "Qwen 3 Coder 480B (cloud)"),
    ],
    "/project": [
        ("info", "Show project summary"),
        ("init", "Initialize project config"),
        ("context", "Show/set project context"),
        ("index", "Index project files"),
        ("search", "Search project index"),
    ],
    "/sessions": [
        ("list", "List saved sessions"),
        ("switch", "Switch to a session"),
        ("new", "Start a new session"),
    ],
    "/agent": [
        ("research", "Research specialist"),
        ("coder", "Coding specialist"),
        ("analyst", "Analysis specialist"),
        ("creative", "Creative specialist"),
        ("parallel", "Run parallel agents"),
    ],
    "/browse": [
        ("search", "Web search query"),
        ("text", "Extract page text"),
        ("screenshot", "Take page screenshot"),
        ("click", "Click an element"),
        ("links", "List page links"),
    ],
    "/theme": [
        ("dark", "Default dark theme"),
        ("light", "Light background theme"),
        ("monokai", "Monokai color scheme"),
        ("dracula", "Dracula color scheme"),
        ("solarized", "Solarized dark theme"),
        ("nord", "Nord color scheme"),
    ],
}


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
            """Show slash command and subcommand completions when typing /."""

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor.lstrip()
                if not text.startswith("/"):
                    return

                parts = text.split(None, 1)  # split on first whitespace
                base_cmd = parts[0].lower()

                if len(parts) == 1 and " " not in text:
                    # Still typing the command itself — match against SLASH_COMMANDS
                    prefix = text.lower()
                    for cmd, desc in SLASH_COMMANDS:
                        if cmd.startswith(prefix):
                            yield Completion(
                                cmd,
                                start_position=-len(text),
                                display=cmd,
                                display_meta=desc,
                            )
                else:
                    # Command is complete, offer subcommands
                    sub_prefix = parts[1].lower() if len(parts) > 1 else ""
                    subs = SUBCOMMANDS.get(base_cmd)
                    if subs:
                        for sub, desc in subs:
                            if sub.startswith(sub_prefix):
                                yield Completion(
                                    sub,
                                    start_position=-len(sub_prefix),
                                    display=sub,
                                    display_meta=desc,
                                )

        _style = Style.from_dict({
            "prompt": "bold cyan",
            "placeholder": "#666666 italic",
            # Completion dropdown — dark background, light text
            "completion-menu": "bg:#1a1a2e #e0e0e0",
            "completion-menu.completion": "bg:#1a1a2e #e0e0e0",
            "completion-menu.completion.current": "bg:#0f3460 #00d2ff bold",
            "completion-menu.meta.completion": "bg:#1a1a2e #777777",
            "completion-menu.meta.completion.current": "bg:#0f3460 #bbbbbb",
            # Scrollbar styling
            "scrollbar.background": "bg:#1a1a2e",
            "scrollbar.button": "bg:#333355",
            "scrollbar.arrow": "bg:#333355 #aaaaaa",
        })

        kb = KeyBindings()

        @kb.add("escape", "m")  # Alt+M or Esc then M
        def _model_pick(event):
            event.app.exit(result=SIGNAL_MODEL_PICK)

        @kb.add('c-l')
        def _clear(event):
            event.app.current_buffer.text = SIGNAL_CLEAR_SCREEN
            event.app.current_buffer.validate_and_handle()

        @kb.add('c-n')
        def _new_session(event):
            event.app.current_buffer.text = SIGNAL_NEW_SESSION
            event.app.current_buffer.validate_and_handle()

        @kb.add('c-k')
        def _palette(event):
            event.app.current_buffer.text = SIGNAL_COMMAND_PALETTE
            event.app.current_buffer.validate_and_handle()

        @kb.add('c-g')
        def _editor(event):
            event.app.current_buffer.text = SIGNAL_OPEN_EDITOR
            event.app.current_buffer.validate_and_handle()

        @kb.add('s-tab')
        def _cycle_perms(event):
            event.app.current_buffer.text = SIGNAL_CYCLE_PERMS
            event.app.current_buffer.validate_and_handle()

        @kb.add('escape', 'escape')
        def _rewind(event):
            event.app.current_buffer.text = SIGNAL_REWIND
            event.app.current_buffer.validate_and_handle()

        session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=SlashCompleter(),
            complete_while_typing=True,
            complete_in_thread=True,
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
    """Get user input. Returns None on exit, or a SIGNAL_* constant for keybindings."""
    try:
        if session is not None and _session_ok:
            result = session.prompt([("class:prompt", "\n  > ")]).strip()
            return result
        else:
            return input("\n  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

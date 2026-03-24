"""prompt_toolkit-based input for AURA CLI — styled prompt, keybindings, completions.

Falls back to plain input() when prompt_toolkit can't attach to the console.

NOTE: Keybindings below are hardcoded. See keybindings.py for a
KeybindingsRegistry that will eventually replace these with
user-customizable shortcuts loaded from ~/.aura/keybindings.json.
"""

from pathlib import Path

HISTORY_FILE = Path.home() / ".aura_history"

_session_ok = True

# ---------------------------------------------------------------------------
# Persistent bottom toolbar (updated by display.show_status_bar)
# ---------------------------------------------------------------------------
_bottom_toolbar_content = None


def set_bottom_toolbar(content):
    """Update the persistent status bar content."""
    global _bottom_toolbar_content
    _bottom_toolbar_content = content


def _get_bottom_toolbar():
    """Called by prompt_toolkit to render the bottom toolbar."""
    return _bottom_toolbar_content


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
    ("/edit", "View file contents with line numbers"),
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
    ("/fleet", "Run parallel sub-agents"),
    ("/tasks", "Show background tasks"),
    ("/research", "Start research mode"),
    ("/sources", "Show research sources"),
    ("/export", "Export research to Markdown"),
    ("/mood", "Show emotional state"),
    ("/pr", "Create pull request"),
    ("/branch", "Create git branch"),
    ("/stash", "Smart git stash"),
    ("/blame", "Git blame with context"),
    ("/test", "Run tests"),
    ("/watch", "Watch files for AI comments"),
    ("/evolve", "Evolve skills with GEPA"),
    ("/diff", "Show git diff with syntax highlighting"),
    ("/git", "Run read-only git commands"),
    ("/mcp", "Manage MCP server connections"),
    ("/audit", "Inspect Merkle audit chain"),
    ("/hand", "Manage autonomous Hands"),
    ("/retry", "Re-run the last prompt"),
    ("/undo", "Undo last file edit"),
    ("/debate", "Multi-model debate on a question"),
    ("/fork", "Fork conversation into a new branch"),
    ("/branches", "List conversation branches"),
    ("/checkout", "Switch to a conversation branch"),
    ("/merge", "Merge branch back to parent"),
    ("/chain", "Run prompt pipelines (step1 -> step2 -> ...)"),
    ("/channels", "Show active channel bridges and status"),
]

# Subcommand completions for commands that accept them
SUBCOMMANDS: dict[str, list[tuple[str, str]]] = {
    "/model": [
        ("auto", "Auto-select best model"),
        ("minimax-m2.7:cloud", "MiniMax M2.7 (cloud, 1M ctx)"),
        ("minimax-m2.5:cloud", "MiniMax M2.5 (cloud, SWE 80.2%)"),
        ("kimi-k2.5:cloud", "Kimi K2.5 (cloud, agentic)"),
        ("qwen3.5:397b-cloud", "Qwen 3.5 397B (cloud, reasoning)"),
        ("deepseek-v3.2:cloud", "DeepSeek V3.2 (cloud, all-rounder)"),
        ("qwen3-coder:480b-cloud", "Qwen 3 Coder 480B (cloud, code)"),
        ("nemotron-3-super:cloud", "Nemotron 3 Super (cloud, fast)"),
        ("glm-5:cloud", "GLM-5 (cloud, general)"),
        ("gpt-oss:120b-cloud", "GPT-OSS 120B (cloud)"),
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
        ("new", "Start a new session"),
        ("delete", "Delete a session"),
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
    "/hook": [
        ("list", "Show registered hooks"),
        ("add", "Add a new hook (event command)"),
        ("remove", "Remove a hook by name"),
    ],
    "/export": [
        ("research", "Export research session to Markdown"),
    ],
    "/git": [
        ("status", "Show working tree status"),
        ("log", "Show commit log"),
        ("diff", "Show changes"),
        ("branch", "List branches"),
        ("blame", "Show file blame"),
        ("stash", "Show stash list"),
        ("show", "Show commit details"),
    ],
    "/mcp": [],
    "/audit": [
        ("verify", "Verify chain integrity"),
        ("tail", "Show last N entries"),
        ("count", "Show total entry count"),
    ],
    "/hand": [
        ("list", "List active Hands"),
        ("activate", "Activate a hand for scheduling"),
        ("deactivate", "Deactivate a hand"),
        ("run", "Run a hand immediately"),
        ("status", "Show detailed hand status"),
    ],
    "/evolve": [
        ("--skill-ids", "Comma-separated skill IDs to evolve"),
        ("--dry-run", "Preview without running"),
        ("--max-iterations", "Max evolution iterations (default 5)"),
    ],
    "/checkout": [
        ("main", "Switch to main branch"),
        ("1", "Switch to fork-1"),
        ("2", "Switch to fork-2"),
        ("3", "Switch to fork-3"),
    ],
    "/chain": [
        ("list", "List saved chains"),
        ("save", "Save a named chain"),
        ("run", "Run a saved chain"),
        ("delete", "Delete a saved chain"),
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
            "project": "#ffffff bold",
            "branch": "#888888",
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
            # Persistent bottom toolbar
            "bottom-toolbar": "bg:#1a1a1a #cccccc",
        })

        kb = KeyBindings()

        @kb.add("escape", "m")  # Alt+M or Esc then M
        def _model_pick(event):
            event.app.exit(result=SIGNAL_MODEL_PICK)

        @kb.add('c-l')
        def _clear(event):
            event.app.exit(result=SIGNAL_CLEAR_SCREEN)

        @kb.add('c-n')
        def _new_session(event):
            event.app.exit(result=SIGNAL_NEW_SESSION)

        @kb.add('c-k')
        def _palette(event):
            event.app.exit(result=SIGNAL_COMMAND_PALETTE)

        @kb.add('c-g')
        def _editor(event):
            event.app.exit(result=SIGNAL_OPEN_EDITOR)

        @kb.add('s-tab')
        def _cycle_perms(event):
            event.app.exit(result=SIGNAL_CYCLE_PERMS)

        @kb.add('c-z')
        def _rewind(event):
            event.app.exit(result=SIGNAL_REWIND)

        @kb.add('escape', 'enter')  # Alt+Enter inserts a newline
        def _newline(event):
            event.current_buffer.insert_text('\n')

        @kb.add('enter')  # Enter always submits (even in multiline mode)
        def _submit(event):
            event.current_buffer.validate_and_handle()

        session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=SlashCompleter(),
            complete_while_typing=True,
            complete_in_thread=True,
            multiline=True,
            style=_style,
            key_bindings=kb,
            placeholder=HTML('<style fg="#555555"><i>  Type a message, / for commands, ? for help</i></style>'),
            bottom_toolbar=_get_bottom_toolbar,
        )
        _session_ok = True
        return session
    except Exception:
        _session_ok = False
        return None


def _get_prompt_prefix() -> list:
    """Build prompt prefix with project name and git branch."""
    try:
        import subprocess
        import os
        # Get git branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=2, cwd="."
        )
        branch = result.stdout.strip() if result.returncode == 0 else ""

        # Get project name (directory name)
        project = os.path.basename(os.getcwd())

        if branch:
            return [
                ("class:project", f"\n  {project}"),
                ("class:branch", f" ({branch})"),
                ("class:prompt", " > "),
            ]
    except Exception:
        pass
    return [("class:prompt", "\n  > ")]


def get_input(session) -> "str | None":
    """Get user input. Returns None on exit, or a SIGNAL_* constant for keybindings."""
    try:
        if session is not None and _session_ok:
            result = session.prompt(_get_prompt_prefix()).strip()
            return result
        else:
            return input("\n  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

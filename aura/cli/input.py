"""prompt_toolkit-based input for AURA CLI — styled prompt, keybindings, completions.

Falls back to plain input() when prompt_toolkit can't attach to the console.
Keybindings are loaded from KeybindingsRegistry, which supports user
customization via ~/.aura/keybindings.json.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Slash commands for autocomplete — single source of truth lives in
# aura.cli.commands. Re-exported here so prompt_toolkit completers (and
# the command palette in chat_session_signals.py) import from the CLI
# entry point without a circular module layout.
from aura.cli.commands import SLASH_COMMANDS

HISTORY_FILE = Path.home() / ".aura_history"


@dataclass
class _InputState:
    """Module-level input state — mutable globals consolidated for testability.

    The CLI runs as a single instance per process so module-level state is
    safe for production, but tests that import this module can call
    ``_input_state.reset()`` to get a clean slate.
    """
    session_ok: bool = True
    bottom_toolbar_content: str = ""
    git_branch_cache: str = ""
    git_branch_ts: float = 0.0
    model_cache: list = field(default_factory=list)
    model_cache_ts: float = 0.0

    def reset(self) -> None:
        """Reset all mutable state to defaults (useful for test isolation)."""
        self.session_ok = True
        self.bottom_toolbar_content = ""
        self.git_branch_cache = ""
        self.git_branch_ts = 0.0
        self.model_cache = []
        self.model_cache_ts = 0.0


_input_state = _InputState()

_GIT_BRANCH_TTL: float = 10.0
_MODEL_CACHE_TTL: float = 30.0


def set_bottom_toolbar(content):
    """Update the persistent status bar content."""
    _input_state.bottom_toolbar_content = content


def _get_bottom_toolbar():
    """Called by prompt_toolkit to render the bottom toolbar."""
    return _input_state.bottom_toolbar_content


# Signal constants for keybindings — returned as pseudo-input from prompt_toolkit
SIGNAL_MODEL_PICK = "__MODEL_PICK__"
SIGNAL_CLEAR_SCREEN = "__CLEAR_SCREEN__"
SIGNAL_NEW_SESSION = "__NEW_SESSION__"
SIGNAL_COMMAND_PALETTE = "__CMD_PALETTE__"
SIGNAL_OPEN_EDITOR = "__OPEN_EDITOR__"
SIGNAL_REWIND = "__REWIND__"
SIGNAL_CYCLE_PERMS = "__CYCLE_PERMS__"

# ---------------------------------------------------------------------------
# Dynamic /model completer — reads the live provider registry so the list
# doesn't drift when models are added or removed. Cached to avoid rebuilding
# on every keystroke.


def _dynamic_models() -> list[tuple[str, str]]:
    """Return ('model_id', 'display') tuples from configured providers.

    Falls back to a single 'auto' entry on import failure so the completer
    never crashes the prompt.
    """
    import time as _t
    now = _t.time()
    if _input_state.model_cache and (now - _input_state.model_cache_ts) < _MODEL_CACHE_TTL:
        return _input_state.model_cache
    try:
        from aura.providers import list_all_provider_models
        items: list[tuple[str, str]] = [("auto", "Auto-select best model")]
        for model, display in list_all_provider_models():
            items.append((model, display))
        _input_state.model_cache = items
        _input_state.model_cache_ts = now
        return items
    except Exception:
        return [("auto", "Auto-select best model")]


# Subcommand completions for commands that accept them.
# Values are either a static list OR a zero-arg callable returning the list.
SubcommandList = list[tuple[str, str]]
SUBCOMMANDS: dict[str, "SubcommandList | Callable[[], SubcommandList]"] = {
    "/model": _dynamic_models,
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
        ("export", "Export session to markdown"),
    ],
    "/trace": [
        ("last", "Show the last run's events"),
        ("runs", "Show recent run summaries"),
        ("failures", "Show recent failed runs"),
        ("10", "Show the last 10 events"),
        ("25", "Show the last 25 events"),
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
    # /evolve accepts flags (--skill-ids, --dry-run, --max-iterations), not
    # subcommands. Listing them here would pollute the subcommand completer
    # with leading `--` tokens. Users can `/evolve --help` for the flag list.
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
    "/snippet": [
        ("save", "Save a new snippet"),
        ("list", "List all snippets"),
        ("delete", "Delete a snippet"),
    ],
}


def create_session():
    """Create a prompt_toolkit session with styled prompt, history, completions, and keybindings."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import Completer, Completion, merge_completers
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style

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
                    if callable(subs):
                        try:
                            subs = subs()
                        except Exception:
                            subs = []
                    if subs:
                        for sub, desc in subs:
                            if sub.startswith(sub_prefix):
                                yield Completion(
                                    sub,
                                    start_position=-len(sub_prefix),
                                    display=sub,
                                    display_meta=desc,
                                )

        class FilePathCompleter(Completer):
            """Complete file paths when not typing a slash command."""
            _CODE_EXTS = frozenset({'.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java',
                          '.json', '.yaml', '.yml', '.md', '.toml', '.html', '.css',
                          '.sh', '.sql', '.c', '.cpp', '.h', '.cfg', '.ini', '.env'})
            _MAX = 15

            def get_completions(self, document, complete_event):
                import os
                text = document.text_before_cursor
                if text.lstrip().startswith("/"):
                    return

                # Find the word being typed
                word = document.get_word_before_cursor(WORD=True)
                if not word or len(word) < 2:
                    return

                # Try to interpret as a path
                try:
                    if os.sep in word or "/" in word:
                        parent = os.path.dirname(word)
                        prefix = os.path.basename(word).lower()
                    else:
                        parent = "."
                        prefix = word.lower()

                    if not os.path.isdir(parent):
                        return

                    count = 0
                    for entry in sorted(os.listdir(parent)):
                        if count >= self._MAX:
                            break
                        if entry.startswith("."):
                            continue  # Skip hidden files

                        full = os.path.join(parent, entry)
                        entry_lower = entry.lower()

                        if not entry_lower.startswith(prefix):
                            continue

                        is_dir = os.path.isdir(full)
                        _, ext = os.path.splitext(entry)

                        if not is_dir and ext.lower() not in self._CODE_EXTS:
                            continue

                        display = entry + "/" if is_dir else entry
                        completion_text = os.path.join(parent, entry) if parent != "." else entry
                        if is_dir:
                            completion_text += "/"

                        yield Completion(
                            completion_text,
                            start_position=-len(word),
                            display=display,
                            display_meta="dir" if is_dir else ext,
                        )
                        count += 1
                except (OSError, PermissionError):
                    return

        class AtFileCompleter(Completer):
            """Complete file paths when user types @ anywhere in the input.

            Triggered by @ prefix on the current word. Fuzzy-matches file
            paths, respects .gitignore-like patterns, and limits to 20 results.
            """
            _CODE_EXTS = frozenset({
                '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java',
                '.json', '.yaml', '.yml', '.md', '.toml', '.html', '.css',
                '.sh', '.sql', '.c', '.cpp', '.h', '.cfg', '.ini', '.env',
            })
            _IGNORE_DIRS = frozenset({
                'node_modules', '.git', '__pycache__', '.venv', 'venv',
                '.tox', '.mypy_cache', '.pytest_cache', 'dist', 'build',
                '.next', '.nuxt', 'coverage', '.eggs', '*.egg-info',
            })
            _MAX = 20

            def get_completions(self, document, complete_event):
                word = document.get_word_before_cursor(WORD=True)
                if not word or not word.startswith('@'):
                    before = document.text[:document.cursor_position]
                    at_idx = before.rfind('@')
                    if at_idx < 0:
                        return
                    word = before[at_idx:]
                    if ' ' in word or '\n' in word:
                        return

                query = word[1:]  # strip the @
                if len(query) < 1:
                    return

                import os
                cwd = os.getcwd()
                try:
                    from ..file_index import get_file_index
                    idx = get_file_index(
                        code_exts=self._CODE_EXTS,
                        ignore_dirs=self._IGNORE_DIRS,
                    )
                    results = idx.get(cwd, query, max_results=self._MAX)
                except Exception:
                    # Fallback to the old synchronous walk on any indexing failure
                    try:
                        results = self._fallback_walk(cwd, query)
                    except Exception:
                        return

                for rel_path, ext in results:
                    fname = os.path.basename(rel_path)
                    yield Completion(
                        rel_path,
                        start_position=-len(word),
                        display=fname,
                        display_meta=ext.lstrip('.'),
                    )

            def _fallback_walk(self, cwd: str, query: str) -> list[tuple[str, str]]:
                """Old synchronous walk for graceful degradation."""
                import os
                query_lower = query.lower()
                results = []
                for root, dirs, files in os.walk(cwd):
                    dirs[:] = [
                        d for d in dirs
                        if d not in self._IGNORE_DIRS and not d.endswith('.egg-info')
                    ]
                    rel_root = os.path.relpath(root, cwd)
                    depth = rel_root.count(os.sep) + (0 if rel_root == '.' else 1)
                    if depth > 4:
                        dirs.clear()
                        continue
                    for fname in files:
                        _, ext = os.path.splitext(fname)
                        if ext.lower() not in self._CODE_EXTS:
                            continue
                        rel_path = os.path.relpath(os.path.join(root, fname), cwd)
                        rel_path = rel_path.replace(os.sep, '/')
                        if query_lower in rel_path.lower():
                            results.append((rel_path, ext))
                    if len(results) >= self._MAX * 3:
                        break
                def _sort_key(item):
                    p, _ = item
                    pl = p.lower()
                    return (0 if pl.startswith(query_lower) else 1, len(pl), pl)
                results.sort(key=_sort_key)
                return results[:self._MAX]

        # Get theme accent for prompt styling
        try:
            from aura.cli.themes import get_theme
            _theme = get_theme()
            _accent = _theme.accent
            _accent_dim = _theme.accent_dim
        except (ImportError, AttributeError):
            from aura.cli.themes import AuraTheme
            _fb = AuraTheme(name="fallback")
            _accent = _fb.accent
            _accent_dim = _fb.accent_dim

        _style = Style.from_dict({
            "prompt": f"bold {_accent}",
            "prompt.sep": "#444444",
            "project": "#ffffff bold",
            "branch": "#888888",
            "mode": f"bold {_accent}",
            "placeholder": "#555555 italic",
            # Completion dropdown
            "completion-menu": "bg:#1a1a2e #e0e0e0",
            "completion-menu.completion": "bg:#1a1a2e #e0e0e0",
            "completion-menu.completion.current": f"bg:#0f3460 {_accent} bold",
            "completion-menu.meta.completion": "bg:#1a1a2e #777777",
            "completion-menu.meta.completion.current": "bg:#0f3460 #bbbbbb",
            # Scrollbar styling
            "scrollbar.background": "bg:#1a1a2e",
            "scrollbar.button": "bg:#333355",
            "scrollbar.arrow": "bg:#333355 #aaaaaa",
            # Bottom toolbar — darker background, clear text
            "bottom-toolbar": "bg:#111115 #aaaaaa",
            "bottom-toolbar.text": "#cccccc",
        })

        kb = KeyBindings()

        # Load keybindings from registry (supports user overrides via ~/.aura/keybindings.json)
        from aura.cli.keybindings import (
            ACTION_CLEAR_SCREEN,
            ACTION_COMMAND_PALETTE,
            ACTION_CYCLE_PERMISSIONS,
            ACTION_MODEL_PICKER,
            ACTION_NEW_SESSION,
            ACTION_OPEN_EDITOR,
            ACTION_REWIND,
            KeybindingsRegistry,
            parse_key_to_pt,
        )
        _registry = KeybindingsRegistry()

        # Map actions to signal constants
        _ACTION_TO_SIGNAL = {
            ACTION_MODEL_PICKER: SIGNAL_MODEL_PICK,
            ACTION_CLEAR_SCREEN: SIGNAL_CLEAR_SCREEN,
            ACTION_NEW_SESSION: SIGNAL_NEW_SESSION,
            ACTION_COMMAND_PALETTE: SIGNAL_COMMAND_PALETTE,
            ACTION_OPEN_EDITOR: SIGNAL_OPEN_EDITOR,
            ACTION_CYCLE_PERMISSIONS: SIGNAL_CYCLE_PERMS,
            ACTION_REWIND: SIGNAL_REWIND,
        }

        for action, signal in _ACTION_TO_SIGNAL.items():
            key_str = _registry.get_key_for_action(action)
            if key_str:
                pt_keys = parse_key_to_pt(key_str)

                def _make_handler(sig):
                    def _handler(event):
                        event.app.exit(result=sig)
                    return _handler

                kb.add(*pt_keys)(_make_handler(signal))

        @kb.add('escape', 'enter')  # Alt+Enter inserts a newline
        def _newline(event):
            event.current_buffer.insert_text('\n')

        @kb.add('enter')  # Enter always submits (even in multiline mode)
        def _submit(event):
            event.current_buffer.validate_and_handle()

        # Ctrl+V — paste image or text from clipboard. Only register if the
        # user hasn't bound ctrl+v to a different action in ~/.aura/keybindings.json.
        # `get_action("ctrl+v")` returns the custom action if the user mapped
        # ctrl+v themselves, None otherwise — in which case our clipboard
        # paste fallback takes over.
        if _registry.get_action("ctrl+v") is None:
            @kb.add('c-v')
            def _paste(event):
                try:
                    from .clipboard import read_clipboard_image, read_clipboard_text
                except Exception:
                    return
                img_path = read_clipboard_image()
                if img_path:
                    event.current_buffer.insert_text(f"[image: {img_path}]")
                    return
                text = read_clipboard_text()
                if text:
                    event.current_buffer.insert_text(text)

        session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=merge_completers([SlashCompleter(), AtFileCompleter(), FilePathCompleter()]),
            complete_while_typing=True,
            complete_in_thread=True,
            multiline=True,
            style=_style,
            key_bindings=kb,
            placeholder=HTML('<style fg="#555555"><i>Message, / commands, ? help, Alt+M model, Alt+Enter newline</i></style>'),
            bottom_toolbar=_get_bottom_toolbar,
        )
        _input_state.session_ok = True
        return session
    except Exception:
        _input_state.session_ok = False
        return None


def _get_git_branch() -> str:
    """Get current git branch with TTL cache to avoid subprocess per prompt."""
    import time
    now = time.monotonic()
    if now - _input_state.git_branch_ts < _GIT_BRANCH_TTL:
        return _input_state.git_branch_cache
    _input_state.git_branch_ts = now
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=2, cwd=".",
        )
        _input_state.git_branch_cache = result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        _input_state.git_branch_cache = ""
    return _input_state.git_branch_cache


def _get_prompt_prefix() -> list:
    """Build a visually distinct prompt with project context and styled caret.

    Renders as:
      ╭─
      │ project (branch) ❯
    """  # noqa: RUF002 — docstring shows actual prompt visual
    import os

    # Show meaningful directory name — collapse home dir.
    # Use normcase+normpath so Windows case-insensitive FS compares work
    # (C:\Users\Asus vs c:\users\asus) and `/` vs `\` differences collapse.
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    def _norm(p):
        return os.path.normcase(os.path.normpath(p))
    if _norm(cwd) == _norm(home):
        project = "~"
    else:
        project = os.path.basename(cwd)

    branch = _get_git_branch()

    # Prompt with left border for visual structure
    parts = []
    parts.append(("class:prompt.sep", "\n  \u256d\u2500\n"))
    parts.append(("class:prompt.sep", "  \u2502 "))
    parts.append(("class:project", project))
    if branch:
        parts.append(("class:branch", f" ({branch})"))
    parts.append(("class:prompt", " \u276f "))
    return parts


def get_input(session) -> "str | None":
    """Get user input. Returns None on exit, or a SIGNAL_* constant for keybindings."""
    try:
        if session is not None and _input_state.session_ok:
            result = session.prompt(_get_prompt_prefix()).strip()
            return result
        else:
            return input("\n  > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

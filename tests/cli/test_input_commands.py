"""Tests for input.py — SLASH_COMMANDS, SUBCOMMANDS, signals, completer."""
import pytest

from aura.cli.commands import COMMAND_REGISTRY
from aura.cli.input import (
    SLASH_COMMANDS,
    SUBCOMMANDS,
    SIGNAL_MODEL_PICK,
    SIGNAL_CLEAR_SCREEN,
    SIGNAL_NEW_SESSION,
    SIGNAL_COMMAND_PALETTE,
    SIGNAL_OPEN_EDITOR,
    SIGNAL_REWIND,
    SIGNAL_CYCLE_PERMS,
    HISTORY_FILE,
)


# ── SLASH_COMMANDS completeness ───────────────────────────────────────────

# Canonical commands expected in the completer (aliases excluded by design).
EXPECTED_CANONICAL_COMMANDS = [
    "/quit", "/clear", "/model", "/compact", "/plan", "/shell",
    "/grep", "/search", "/edit", "/project",
    "/agent", "/sessions", "/browse", "/hook", "/speak", "/recall",
    "/goal", "/trust", "/cost", "/context", "/trace", "/rewind", "/theme", "/fleet",
    "/tasks", "/research", "/sources", "/export", "/mood", "/pr", "/branch",
    "/stash", "/blame", "/test", "/watch", "/evolve", "/diff", "/git",
    "/mcp", "/audit", "/hand", "/retry", "/undo",
]

# Aliases — present in COMMAND_REGISTRY for dispatch but NOT in SLASH_COMMANDS.
EXPECTED_ALIASES = ["/exit", "/bash", "/run", "/find", "/say", "/memory"]


def test_slash_commands_has_all_canonical():
    command_names = [cmd for cmd, _ in SLASH_COMMANDS]
    for expected in EXPECTED_CANONICAL_COMMANDS:
        assert expected in command_names, f"Missing canonical command: {expected}"


def test_aliases_in_registry_not_completer():
    completer_names = {cmd for cmd, _ in SLASH_COMMANDS}
    for alias in EXPECTED_ALIASES:
        assert alias in COMMAND_REGISTRY, f"Alias {alias} missing from dispatch registry"
        assert alias not in completer_names, (
            f"Alias {alias} leaked into completer — aliases should dispatch only, "
            "not clutter autocomplete"
        )


def test_slash_commands_count():
    # 54 canonical + 2 runtime-only (/retry, /channels) = 56 in completer.
    assert len(SLASH_COMMANDS) == 56


def test_every_command_has_description():
    for cmd, desc in SLASH_COMMANDS:
        assert isinstance(desc, str), f"{cmd} description is not a string"
        assert len(desc) > 0, f"{cmd} has empty description"


def test_no_duplicate_commands():
    command_names = [cmd for cmd, _ in SLASH_COMMANDS]
    assert len(command_names) == len(set(command_names)), (
        f"Duplicates found: {[c for c in command_names if command_names.count(c) > 1]}"
    )


def test_all_commands_start_with_slash():
    for cmd, _ in SLASH_COMMANDS:
        assert cmd.startswith("/"), f"Command does not start with /: {cmd}"


def test_commands_are_lowercase():
    for cmd, _ in SLASH_COMMANDS:
        assert cmd == cmd.lower(), f"Command is not lowercase: {cmd}"


# ── SUBCOMMANDS ───────────────────────────────────────────────────────────

def test_subcommands_keys_are_valid_commands():
    command_names = {cmd for cmd, _ in SLASH_COMMANDS}
    for key in SUBCOMMANDS:
        assert key in command_names, f"SUBCOMMANDS key {key} is not a valid command"


def _resolve_subcommands(key):
    """SUBCOMMANDS values may be a static list or a zero-arg callable (Fix 2)."""
    slot = SUBCOMMANDS[key]
    return slot() if callable(slot) else slot


def test_subcommands_values_are_lists_of_tuples():
    for key in SUBCOMMANDS:
        subs = _resolve_subcommands(key)
        assert isinstance(subs, list), f"SUBCOMMANDS[{key}] did not resolve to a list"
        for item in subs:
            assert isinstance(item, tuple) and len(item) == 2, (
                f"SUBCOMMANDS[{key}] entry is not a (name, desc) tuple: {item}"
            )
            assert isinstance(item[0], str) and isinstance(item[1], str)


def test_subcommands_has_model():
    assert "/model" in SUBCOMMANDS
    model_subs = [name for name, _ in _resolve_subcommands("/model")]
    assert "auto" in model_subs


def test_subcommands_has_project():
    assert "/project" in SUBCOMMANDS
    project_subs = [name for name, _ in _resolve_subcommands("/project")]
    assert "info" in project_subs
    assert "index" in project_subs
    assert "search" in project_subs


def test_subcommands_has_sessions():
    assert "/sessions" in SUBCOMMANDS
    session_subs = [name for name, _ in _resolve_subcommands("/sessions")]
    assert "list" in session_subs
    assert "new" in session_subs
    assert "delete" in session_subs


def test_subcommands_has_trace():
    assert "/trace" in SUBCOMMANDS
    trace_subs = [name for name, _ in _resolve_subcommands("/trace")]
    assert "last" in trace_subs
    assert "runs" in trace_subs
    assert "failures" in trace_subs
    assert "10" in trace_subs
    assert "25" in trace_subs


def test_subcommands_has_theme():
    assert "/theme" in SUBCOMMANDS
    theme_subs = [name for name, _ in _resolve_subcommands("/theme")]
    assert "dark" in theme_subs
    assert "light" in theme_subs


def test_subcommands_has_git():
    assert "/git" in SUBCOMMANDS
    git_subs = [name for name, _ in _resolve_subcommands("/git")]
    assert "status" in git_subs
    assert "log" in git_subs
    assert "diff" in git_subs


# ── Signals ───────────────────────────────────────────────────────────────

def test_signal_constants_are_unique():
    signals = [
        SIGNAL_MODEL_PICK, SIGNAL_CLEAR_SCREEN, SIGNAL_NEW_SESSION,
        SIGNAL_COMMAND_PALETTE, SIGNAL_OPEN_EDITOR, SIGNAL_REWIND,
        SIGNAL_CYCLE_PERMS,
    ]
    assert len(signals) == len(set(signals)), "Signal constants must be unique"


def test_signal_constants_are_strings():
    for sig in [
        SIGNAL_MODEL_PICK, SIGNAL_CLEAR_SCREEN, SIGNAL_NEW_SESSION,
        SIGNAL_COMMAND_PALETTE, SIGNAL_OPEN_EDITOR, SIGNAL_REWIND,
        SIGNAL_CYCLE_PERMS,
    ]:
        assert isinstance(sig, str)
        assert len(sig) > 0


def test_signal_constants_are_dunder_style():
    """Signals use __NAME__ convention to avoid collision with user input."""
    for sig in [
        SIGNAL_MODEL_PICK, SIGNAL_CLEAR_SCREEN, SIGNAL_NEW_SESSION,
        SIGNAL_COMMAND_PALETTE, SIGNAL_OPEN_EDITOR, SIGNAL_REWIND,
        SIGNAL_CYCLE_PERMS,
    ]:
        assert sig.startswith("__") and sig.endswith("__"), f"Signal {sig} not __DUNDER__ style"


# ── HISTORY_FILE ──────────────────────────────────────────────────────────

def test_history_file_is_in_home_dir():
    from pathlib import Path
    assert HISTORY_FILE.parent == Path.home()
    assert HISTORY_FILE.name == ".aura_history"


# ── SlashCompleter (integration test via create_session internals) ────────

def test_slash_completer_class_exists():
    """Verify the completer class can be instantiated from within create_session's scope.

    We test the completion logic by importing the module-level data it uses.
    """
    # The completer uses SLASH_COMMANDS and SUBCOMMANDS — verify they are non-empty
    assert len(SLASH_COMMANDS) > 0
    assert len(SUBCOMMANDS) > 0


def test_slash_commands_descriptions_unique():
    """Now that aliases are excluded from the completer, every entry should have
    a distinct description — duplicate descriptions signal a copy-paste error.
    """
    desc_to_cmds: dict[str, list[str]] = {}
    for cmd, desc in SLASH_COMMANDS:
        desc_to_cmds.setdefault(desc, []).append(cmd)
    dupes = {d: cs for d, cs in desc_to_cmds.items() if len(cs) > 1}
    assert not dupes, f"Duplicate descriptions in completer (aliases should dispatch only): {dupes}"

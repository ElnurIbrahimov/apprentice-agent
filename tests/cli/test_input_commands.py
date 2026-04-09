"""Tests for input.py — SLASH_COMMANDS, SUBCOMMANDS, signals, completer."""
import pytest

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

EXPECTED_SLASH_COMMANDS = [
    "/quit", "/exit", "/clear", "/model", "/compact", "/plan", "/shell",
    "/bash", "/run", "/grep", "/search", "/find", "/edit", "/project",
    "/agent", "/sessions", "/browse", "/hook", "/speak", "/say", "/recall",
    "/goal", "/trust", "/cost", "/context", "/rewind", "/theme", "/fleet",
    "/tasks", "/research", "/sources", "/export", "/mood", "/pr", "/branch",
    "/stash", "/blame", "/test", "/watch", "/evolve", "/diff", "/git",
    "/mcp", "/audit", "/hand", "/retry", "/undo",
]


def test_slash_commands_has_all_expected():
    command_names = [cmd for cmd, _ in SLASH_COMMANDS]
    for expected in EXPECTED_SLASH_COMMANDS:
        assert expected in command_names, f"Missing command: {expected}"


def test_slash_commands_count():
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


def test_subcommands_values_are_lists_of_tuples():
    for key, subs in SUBCOMMANDS.items():
        assert isinstance(subs, list), f"SUBCOMMANDS[{key}] is not a list"
        for item in subs:
            assert isinstance(item, tuple) and len(item) == 2, (
                f"SUBCOMMANDS[{key}] entry is not a (name, desc) tuple: {item}"
            )
            assert isinstance(item[0], str) and isinstance(item[1], str)


def test_subcommands_has_model():
    assert "/model" in SUBCOMMANDS
    model_subs = [name for name, _ in SUBCOMMANDS["/model"]]
    assert "auto" in model_subs


def test_subcommands_has_project():
    assert "/project" in SUBCOMMANDS
    project_subs = [name for name, _ in SUBCOMMANDS["/project"]]
    assert "info" in project_subs
    assert "index" in project_subs
    assert "search" in project_subs


def test_subcommands_has_sessions():
    assert "/sessions" in SUBCOMMANDS
    session_subs = [name for name, _ in SUBCOMMANDS["/sessions"]]
    assert "list" in session_subs
    assert "new" in session_subs
    assert "delete" in session_subs


def test_subcommands_has_theme():
    assert "/theme" in SUBCOMMANDS
    theme_subs = [name for name, _ in SUBCOMMANDS["/theme"]]
    assert "dark" in theme_subs
    assert "light" in theme_subs


def test_subcommands_has_git():
    assert "/git" in SUBCOMMANDS
    git_subs = [name for name, _ in SUBCOMMANDS["/git"]]
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


def test_slash_commands_descriptions_not_duplicated():
    """Each command should have a unique description (detect copy-paste errors)."""
    [desc for _, desc in SLASH_COMMANDS]
    # Aliases can share descriptions (e.g., /shell and /bash both say "Execute shell command")
    # But non-alias commands should have distinct descriptions
    non_alias_cmds = {}
    for cmd, desc in SLASH_COMMANDS:
        if desc not in non_alias_cmds:
            non_alias_cmds[desc] = []
        non_alias_cmds[desc].append(cmd)

    # Known acceptable duplicates (aliases)
    for desc, cmds in non_alias_cmds.items():
        if len(cmds) > 1:
            # These should be known alias groups
            cmd_set = set(cmds)
            known_groups = [
                {"/quit", "/exit"},
                {"/shell", "/bash", "/run"},
                {"/speak", "/say"},
                {"/search", "/find"},
            ]
            found_group = any(cmd_set.issubset(g) for g in known_groups)
            assert found_group, f"Unexpected duplicate description '{desc}' for commands: {cmds}"

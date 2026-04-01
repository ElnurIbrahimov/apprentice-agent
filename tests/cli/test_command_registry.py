"""Tests for command registry — registration, dispatch, aliases."""
import pytest
from unittest.mock import MagicMock, patch

from aura.cli.commands import COMMAND_REGISTRY, handle_command
from aura.cli.commands.handlers import (
    handle_quit, handle_help, handle_goal, handle_recall, handle_clear,
    handle_speak, handle_model, handle_compact, handle_plan, handle_hand,
    handle_audit, handle_browse, handle_grep, handle_search, handle_edit,
    handle_project, handle_shell, handle_agent, handle_evolve, handle_fleet,
    handle_tasks, handle_research, handle_sources, handle_export, handle_mood,
    handle_hook, handle_sessions, handle_theme, handle_trust, handle_context,
    handle_rewind, handle_cost, handle_undo, handle_diff,
    handle_git, handle_pr, handle_branch, handle_stash, handle_blame,
    handle_test, handle_watch, handle_mcp,
)


# ── Registry completeness ─────────────────────────────────────────────────

EXPECTED_COMMANDS = [
    "/quit", "/exit", "/help", "/goal", "/recall", "/clear", "/speak", "/say",
    "/model", "/compact", "/plan", "/hand", "/audit", "/browse", "/grep",
    "/search", "/find", "/edit", "/project", "/shell", "/bash", "/run",
    "/agent", "/evolve", "/fleet", "/tasks", "/research", "/sources",
    "/export", "/mood", "/hook", "/sessions", "/theme", "/trust", "/context",
    "/rewind", "/cost", "/undo", "/diff", "/git", "/pr", "/branch",
    "/stash", "/blame", "/test", "/watch", "/mcp",
]


def test_registry_has_all_expected_commands():
    for cmd in EXPECTED_COMMANDS:
        assert cmd in COMMAND_REGISTRY, f"Missing command: {cmd}"


def test_registry_count():
    assert len(COMMAND_REGISTRY) == 55


def test_every_registry_value_is_callable():
    for cmd, handler in COMMAND_REGISTRY.items():
        assert callable(handler), f"Handler for {cmd} is not callable"


# ── Aliases ───────────────────────────────────────────────────────────────

def test_exit_aliases_to_quit():
    assert COMMAND_REGISTRY["/exit"] is COMMAND_REGISTRY["/quit"]
    assert COMMAND_REGISTRY["/exit"] is handle_quit


def test_bash_aliases_to_shell():
    assert COMMAND_REGISTRY["/bash"] is COMMAND_REGISTRY["/shell"]
    assert COMMAND_REGISTRY["/bash"] is handle_shell


def test_run_aliases_to_shell():
    assert COMMAND_REGISTRY["/run"] is COMMAND_REGISTRY["/shell"]
    assert COMMAND_REGISTRY["/run"] is handle_shell


def test_say_aliases_to_speak():
    assert COMMAND_REGISTRY["/say"] is COMMAND_REGISTRY["/speak"]
    assert COMMAND_REGISTRY["/say"] is handle_speak


def test_find_aliases_to_search():
    assert COMMAND_REGISTRY["/find"] is COMMAND_REGISTRY["/search"]
    assert COMMAND_REGISTRY["/find"] is handle_search


# ── Dispatch via handle_command ───────────────────────────────────────────

def test_handle_command_dispatches_to_correct_handler():
    """handle_command should call the right handler with (agent, arg, context)."""
    agent = MagicMock()
    with patch("aura.cli.commands.handlers.handle_recall") as mock_recall:
        # We need to patch it in the registry too
        original = COMMAND_REGISTRY["/recall"]
        COMMAND_REGISTRY["/recall"] = mock_recall
        try:
            handle_command(agent, "/recall some query")
            mock_recall.assert_called_once_with(agent, "some query", {"speak": False})
        finally:
            COMMAND_REGISTRY["/recall"] = original


def test_handle_command_with_speak_flag():
    agent = MagicMock()
    mock_handler = MagicMock()
    COMMAND_REGISTRY["/test_fake"] = mock_handler
    try:
        handle_command(agent, "/test_fake arg1", speak=True)
        mock_handler.assert_called_once_with(agent, "arg1", {"speak": True})
    finally:
        del COMMAND_REGISTRY["/test_fake"]


def test_handle_command_unknown_command(capsys):
    agent = MagicMock()
    handle_command(agent, "/nonexistent_cmd")
    captured = capsys.readouterr()
    assert "Unknown command" in captured.out
    assert "/nonexistent_cmd" in captured.out


def test_handle_command_case_insensitive():
    """Commands should be lowercased before lookup."""
    agent = MagicMock()
    mock_handler = MagicMock()
    original = COMMAND_REGISTRY["/help"]
    COMMAND_REGISTRY["/help"] = mock_handler
    try:
        handle_command(agent, "/HELP")
        mock_handler.assert_called_once()
    finally:
        COMMAND_REGISTRY["/help"] = original


def test_handle_command_no_arg():
    """Command with no argument should pass empty string as arg."""
    agent = MagicMock()
    mock_handler = MagicMock()
    original = COMMAND_REGISTRY["/help"]
    COMMAND_REGISTRY["/help"] = mock_handler
    try:
        handle_command(agent, "/help")
        mock_handler.assert_called_once_with(agent, "", {"speak": False})
    finally:
        COMMAND_REGISTRY["/help"] = original


def test_handle_command_export_research_routes_to_export():
    """'/export research ...' should route to handle_export."""
    agent = MagicMock()
    mock_export = MagicMock()
    original = COMMAND_REGISTRY["/export"]
    COMMAND_REGISTRY["/export"] = mock_export
    try:
        handle_command(agent, "/export research session1")
        mock_export.assert_called_once_with(agent, "research session1", {"speak": False})
    finally:
        COMMAND_REGISTRY["/export"] = original


# ── Handler-to-command mapping ────────────────────────────────────────────

HANDLER_MAP = {
    "/quit": handle_quit,
    "/help": handle_help,
    "/goal": handle_goal,
    "/recall": handle_recall,
    "/clear": handle_clear,
    "/speak": handle_speak,
    "/model": handle_model,
    "/compact": handle_compact,
    "/plan": handle_plan,
    "/hand": handle_hand,
    "/audit": handle_audit,
    "/browse": handle_browse,
    "/grep": handle_grep,
    "/search": handle_search,
    "/edit": handle_edit,
    "/project": handle_project,
    "/shell": handle_shell,
    "/agent": handle_agent,
    "/evolve": handle_evolve,
    "/fleet": handle_fleet,
    "/tasks": handle_tasks,
    "/research": handle_research,
    "/sources": handle_sources,
    "/export": handle_export,
    "/mood": handle_mood,
    "/hook": handle_hook,
    "/sessions": handle_sessions,
    "/theme": handle_theme,
    "/trust": handle_trust,
    "/context": handle_context,
    "/rewind": handle_rewind,
    "/cost": handle_cost,
    "/undo": handle_undo,
    "/diff": handle_diff,
    "/git": handle_git,
    "/pr": handle_pr,
    "/branch": handle_branch,
    "/stash": handle_stash,
    "/blame": handle_blame,
    "/test": handle_test,
    "/watch": handle_watch,
    "/mcp": handle_mcp,
}


def test_each_command_maps_to_correct_handler():
    for cmd, expected_handler in HANDLER_MAP.items():
        assert COMMAND_REGISTRY[cmd] is expected_handler, (
            f"{cmd} maps to {COMMAND_REGISTRY[cmd].__name__}, expected {expected_handler.__name__}"
        )

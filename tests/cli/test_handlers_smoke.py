"""Smoke tests for command handlers — verify they don't crash with mock agent/context."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Optional

from aura.cli.context import CLIContext


def _make_mock_agent():
    """Create a mock agent with common attributes handlers expect."""
    agent = MagicMock()
    agent.brain = MagicMock()
    agent.brain._model_override = "test-model"
    agent.brain.get_session_stats.return_value = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "cost_usd": 0.001,
        "queries": 5,
    }
    agent.brain.clear_history.return_value = None
    agent.brain.compact_history.return_value = "Summary"
    agent.brain.think.return_value = {"response": "test response"}
    agent.tools = {}
    agent.recall_memories.return_value = []
    agent._speak = MagicMock()
    return agent


def _make_mock_cli_ctx(agent=None, *, with_loop=False, with_permissions=False,
                       with_session=False, with_bg=False, with_research=False,
                       with_hooks=False):
    """Create a mock CLIContext with optional subsystems."""
    if agent is None:
        agent = _make_mock_agent()
    ctx = CLIContext(agent=agent)
    if with_loop:
        ctx.agentic_loop = MagicMock()
    if with_permissions:
        ctx.permissions = MagicMock()
    if with_session:
        ctx.session = MagicMock()
    if with_bg:
        ctx.bg_manager = MagicMock()
    if with_research:
        ctx.research_ctx = MagicMock()
    if with_hooks:
        ctx.hook_manager = MagicMock()
    return ctx


def _patch_ctx(cli_ctx):
    """Patch the module-level _current in context.py."""
    return patch("aura.cli.context._current", cli_ctx)


def _ctx(speak=False):
    return {"speak": speak}


# ── handle_help ───────────────────────────────────────────────────────────

def test_handle_help_smoke():
    from aura.cli.commands.handlers import handle_help
    with patch("aura.cli.display.show_help"):
        handle_help(_make_mock_agent(), "", _ctx())


# ── handle_recall ─────────────────────────────────────────────────────────

def test_handle_recall_with_query(capsys):
    from aura.cli.commands.handlers import handle_recall
    agent = _make_mock_agent()
    agent.recall_memories.return_value = [{"content": "memory 1"}]
    handle_recall(agent, "test query", _ctx())
    out = capsys.readouterr().out
    assert "Recalled" in out


def test_handle_recall_no_query(capsys):
    from aura.cli.commands.handlers import handle_recall
    handle_recall(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_clear ──────────────────────────────────────────────────────────

def test_handle_clear_smoke(capsys):
    from aura.cli.commands.handlers import handle_clear
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no agentic_loop
    with _patch_ctx(cli_ctx):
        handle_clear(agent, "--force", _ctx())
    out = capsys.readouterr().out
    assert "cleared" in out.lower()


def test_handle_clear_with_agentic_loop(capsys):
    from aura.cli.commands.handlers import handle_clear
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_loop=True)
    with _patch_ctx(cli_ctx):
        handle_clear(agent, "--force", _ctx())
    cli_ctx.agentic_loop.clear_history.assert_called_once()
    out = capsys.readouterr().out
    assert "cleared" in out.lower()


# ── handle_speak ──────────────────────────────────────────────────────────

def test_handle_speak_with_text(capsys):
    from aura.cli.commands.handlers import handle_speak
    agent = _make_mock_agent()
    handle_speak(agent, "hello world", _ctx())
    agent._speak.assert_called_once_with("hello world")
    out = capsys.readouterr().out
    assert "Spoke" in out


def test_handle_speak_no_text(capsys):
    from aura.cli.commands.handlers import handle_speak
    handle_speak(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_model ──────────────────────────────────────────────────────────

def test_handle_model_set_by_name(capsys):
    from aura.cli.commands.handlers import handle_model
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no loop
    with _patch_ctx(cli_ctx):
        handle_model(agent, "devstral-2:cloud", _ctx())
    agent.brain.set_model_override.assert_called_once_with("devstral-2:cloud")
    out = capsys.readouterr().out
    assert "devstral-2:cloud" in out


def test_handle_model_set_auto(capsys):
    from aura.cli.commands.handlers import handle_model
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)
    with _patch_ctx(cli_ctx):
        handle_model(agent, "auto", _ctx())
    agent.brain.set_model_override.assert_called_once_with(None)
    out = capsys.readouterr().out
    assert "auto" in out.lower()


def test_handle_model_set_with_agentic_loop(capsys):
    from aura.cli.commands.handlers import handle_model
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_loop=True)
    with _patch_ctx(cli_ctx):
        handle_model(agent, "test-model", _ctx())
    agent.brain.set_model_override.assert_called_once_with("test-model")
    assert cli_ctx.agentic_loop.model_override == "test-model"


# ── handle_compact ────────────────────────────────────────────────────────

def test_handle_compact_smoke(capsys):
    from aura.cli.commands.handlers import handle_compact
    agent = _make_mock_agent()
    handle_compact(agent, "", _ctx())
    agent.brain.compact_history.assert_called_once()
    out = capsys.readouterr().out
    assert "Compact" in out


def test_handle_compact_with_focus(capsys):
    from aura.cli.commands.handlers import handle_compact
    agent = _make_mock_agent()
    handle_compact(agent, "python code", _ctx())
    agent.brain.compact_history.assert_called_once_with(focus="python code")


# ── handle_plan ───────────────────────────────────────────────────────────

def test_handle_plan_no_arg(capsys):
    from aura.cli.commands.handlers import handle_plan
    handle_plan(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_browse ─────────────────────────────────────────────────────────

def test_handle_browse_no_arg(capsys):
    from aura.cli.commands.handlers import handle_browse
    handle_browse(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_grep ───────────────────────────────────────────────────────────

def test_handle_grep_no_arg(capsys):
    from aura.cli.commands.handlers import handle_grep
    handle_grep(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_search ─────────────────────────────────────────────────────────

def test_handle_search_no_arg(capsys):
    from aura.cli.commands.handlers import handle_search
    handle_search(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_edit ───────────────────────────────────────────────────────────

def test_handle_edit_no_arg(capsys):
    from aura.cli.commands.handlers import handle_edit
    handle_edit(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_shell ──────────────────────────────────────────────────────────

def test_handle_shell_no_arg(capsys):
    from aura.cli.commands.handlers import handle_shell
    handle_shell(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out or "shell" in out.lower()


# ── handle_fleet ──────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_fleet_no_arg(mock_console):
    from aura.cli.commands.handlers import handle_fleet
    handle_fleet(_make_mock_agent(), "", _ctx())
    mock_console.print.assert_called()


# ── handle_tasks ──────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_tasks_no_bg_manager(mock_console):
    from aura.cli.commands.handlers import handle_tasks
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no bg_manager
    with _patch_ctx(cli_ctx):
        handle_tasks(agent, "", _ctx())
    mock_console.print.assert_called()


# ── handle_research ───────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_research_no_arg(mock_console):
    from aura.cli.commands.handlers import handle_research
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no research_ctx
    with _patch_ctx(cli_ctx):
        handle_research(agent, "", _ctx())
    mock_console.print.assert_called()


@patch("aura.cli.display.console")
def test_handle_research_with_topic(mock_console):
    from aura.cli.commands.handlers import handle_research
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no research_ctx initially
    with _patch_ctx(cli_ctx):
        handle_research(agent, "quantum computing", _ctx())
    mock_console.print.assert_called()
    # Should have set research_ctx on the CLIContext
    assert cli_ctx.research_ctx is not None


# ── handle_sources ────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_sources_no_research(mock_console):
    from aura.cli.commands.handlers import handle_sources
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)
    with _patch_ctx(cli_ctx):
        handle_sources(agent, "", _ctx())
    mock_console.print.assert_called()


# ── handle_export ─────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_export_no_research(mock_console):
    from aura.cli.commands.handlers import handle_export
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)
    with _patch_ctx(cli_ctx):
        handle_export(agent, "", _ctx())
    mock_console.print.assert_called()


# ── handle_mood ───────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_mood_no_engine(mock_console):
    from aura.cli.commands.handlers import handle_mood
    with patch("aura.emotion.alma_engine.get_alma_engine", side_effect=Exception("no engine")):
        handle_mood(_make_mock_agent(), "", _ctx())
    # Should print "not available" message without crashing
    mock_console.print.assert_called()


# ── handle_trust ──────────────────────────────────────────────────────────

def test_handle_trust_smoke(capsys):
    from aura.cli.commands.handlers import handle_trust
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_permissions=True)
    with _patch_ctx(cli_ctx):
        handle_trust(agent, "", _ctx())
    cli_ctx.permissions.set_trust_mode.assert_called_once_with(True)
    out = capsys.readouterr().out
    assert "Trust mode" in out


def test_handle_trust_off(capsys):
    from aura.cli.commands.handlers import handle_trust
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_permissions=True)
    with _patch_ctx(cli_ctx):
        handle_trust(agent, "off", _ctx())
    cli_ctx.permissions.set_trust_mode.assert_called_once_with(False)


# ── handle_context ────────────────────────────────────────────────────────

def test_handle_context_no_agentic_loop(capsys):
    from aura.cli.commands.handlers import handle_context
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no loop
    with _patch_ctx(cli_ctx):
        handle_context(agent, "", _ctx())
    out = capsys.readouterr().out
    assert "not available" in out.lower()


# ── handle_rewind ─────────────────────────────────────────────────────────

def test_handle_rewind_no_agentic_loop(capsys):
    from aura.cli.commands.handlers import handle_rewind
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no loop
    with _patch_ctx(cli_ctx):
        handle_rewind(agent, "", _ctx())
    out = capsys.readouterr().out
    assert "No checkpoint" in out


# ── handle_cost ───────────────────────────────────────────────────────────

def test_handle_cost_smoke(capsys):
    from aura.cli.commands.handlers import handle_cost
    agent = _make_mock_agent()
    handle_cost(agent, "", _ctx())
    out = capsys.readouterr().out
    assert "Session Cost" in out
    assert "100" in out  # input tokens
    assert "50" in out   # output tokens


# ── handle_undo ───────────────────────────────────────────────────────────

def test_handle_undo_no_agentic_loop(capsys):
    from aura.cli.commands.handlers import handle_undo
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no loop
    with _patch_ctx(cli_ctx):
        handle_undo(agent, "", _ctx())
    out = capsys.readouterr().out
    assert "No active" in out


# ── handle_diff ───────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_diff_no_changes(mock_console):
    from aura.cli.commands.handlers import handle_diff
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", stderr="")
        handle_diff(_make_mock_agent(), "", _ctx())
    mock_console.print.assert_called()


@patch("aura.cli.display.console")
def test_handle_diff_blocks_unsafe_flag(mock_console):
    from aura.cli.commands.handlers import handle_diff
    handle_diff(_make_mock_agent(), "-c evil", _ctx())
    # Should have called show_error via console.print
    assert mock_console.print.called


# ── handle_git ────────────────────────────────────────────────────────────

def test_handle_git_no_arg(capsys):
    from aura.cli.commands.handlers import handle_git
    handle_git(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


def test_handle_git_blocks_push(capsys):
    from aura.cli.commands.handlers import handle_git
    handle_git(_make_mock_agent(), "push", _ctx())
    out = capsys.readouterr().out
    assert "Blocked" in out


def test_handle_git_blocks_dangerous_flag(capsys):
    from aura.cli.commands.handlers import handle_git
    handle_git(_make_mock_agent(), "log -c evil", _ctx())
    out = capsys.readouterr().out
    assert "Blocked" in out or "dangerous" in out.lower()


@patch("aura.cli.display.console")
def test_handle_git_status(mock_console):
    from aura.cli.commands.handlers import handle_git
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="On branch main", stderr="")
        handle_git(_make_mock_agent(), "status", _ctx())
    mock_console.print.assert_called()


# ── handle_branch ─────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_branch_no_arg(mock_console):
    from aura.cli.commands.handlers import handle_branch
    handle_branch(_make_mock_agent(), "", _ctx())
    mock_console.print.assert_called()


# ── handle_blame ──────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_blame_bad_format(mock_console):
    from aura.cli.commands.handlers import handle_blame
    handle_blame(_make_mock_agent(), "nolinenum", _ctx())
    mock_console.print.assert_called()


# ── handle_mcp ────────────────────────────────────────────────────────────

def test_handle_mcp_no_loop(capsys):
    from aura.cli.commands.handlers import handle_mcp
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no loop
    with _patch_ctx(cli_ctx):
        handle_mcp(agent, "", _ctx())
    out = capsys.readouterr().out
    assert "No MCP" in out or "Configure" in out


# ── handle_sessions ───────────────────────────────────────────────────────

def test_handle_sessions_delete_not_found(capsys):
    from aura.cli.commands.handlers import handle_sessions
    agent = _make_mock_agent()
    with patch("aura.core.session.AgenticSession") as MockSes:
        mock_ses = MagicMock()
        MockSes.return_value = mock_ses
        mock_ses.list_sessions.return_value = []
        mock_ses.delete.return_value = False
        agent.brain.list_conversations.return_value = []
        handle_sessions(agent, "delete nonexistent", _ctx())
    out = capsys.readouterr().out
    assert "not found" in out.lower()


# ── handle_theme ──────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_theme_show_current(mock_console):
    from aura.cli.commands.handlers import handle_theme
    handle_theme(_make_mock_agent(), "", _ctx())
    mock_console.print.assert_called()


@patch("aura.cli.display.console")
def test_handle_theme_set_unknown(mock_console):
    from aura.cli.commands.handlers import handle_theme
    handle_theme(_make_mock_agent(), "nonexistent_theme", _ctx())
    # Should print error about unknown theme
    mock_console.print.assert_called()


# ── handle_hook ───────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_hook_list(mock_console):
    from aura.cli.commands.handlers import handle_hook
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no hook_manager initially
    with _patch_ctx(cli_ctx):
        handle_hook(agent, "", _ctx())
    # Should create a HookManager and call render_hooks_table


# ── handle_watch ──────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_watch_stop(mock_console):
    from aura.cli.commands.handlers import handle_watch
    agent = _make_mock_agent()
    mock_watcher = MagicMock()
    cli_ctx = _make_mock_cli_ctx(agent)
    cli_ctx.file_watcher = mock_watcher
    with _patch_ctx(cli_ctx):
        handle_watch(agent, "stop", _ctx())
    mock_watcher.stop.assert_called_once()

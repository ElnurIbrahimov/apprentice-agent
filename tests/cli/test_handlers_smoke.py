"""Smoke tests for command handlers — verify they don't crash with mock agent/context."""
import os
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
    from aura.cli.commands import handle_help
    with patch("aura.cli.display.show_help"):
        handle_help(_make_mock_agent(), "", _ctx())


# ── handle_recall ─────────────────────────────────────────────────────────

def test_handle_recall_with_query(capsys):
    from aura.cli.commands import handle_recall
    agent = _make_mock_agent()
    agent.recall_memories.return_value = [{"content": "memory 1"}]
    handle_recall(agent, "test query", _ctx())
    out = capsys.readouterr().out
    assert "Recalled" in out


def test_handle_recall_no_query(capsys):
    from aura.cli.commands import handle_recall
    handle_recall(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_clear ──────────────────────────────────────────────────────────

def test_handle_clear_smoke(capsys):
    from aura.cli.commands import handle_clear
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no agentic_loop
    with _patch_ctx(cli_ctx):
        handle_clear(agent, "--force", _ctx())
    out = capsys.readouterr().out
    assert "cleared" in out.lower()


def test_handle_clear_with_agentic_loop(capsys):
    from aura.cli.commands import handle_clear
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_loop=True)
    with _patch_ctx(cli_ctx):
        handle_clear(agent, "--force", _ctx())
    cli_ctx.agentic_loop.clear_history.assert_called_once()
    out = capsys.readouterr().out
    assert "cleared" in out.lower()


def test_handle_clear_uses_permissions_manager_for_confirmation(capsys):
    from aura.cli.commands import handle_clear

    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_permissions=True)
    cli_ctx.permissions.check.return_value = False

    with _patch_ctx(cli_ctx), patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
        handle_clear(agent, "", _ctx())

    cli_ctx.permissions.check.assert_called_once_with(
        "clear_history",
        {"scope": "conversation_history"},
    )
    out = capsys.readouterr().out
    assert "cancelled" in out.lower()


# ── handle_speak ──────────────────────────────────────────────────────────

def test_handle_speak_with_text(capsys):
    from aura.cli.commands import handle_speak
    agent = _make_mock_agent()
    handle_speak(agent, "hello world", _ctx())
    agent._speak.assert_called_once_with("hello world")
    out = capsys.readouterr().out
    assert "Spoke" in out


def test_handle_speak_no_text(capsys):
    from aura.cli.commands import handle_speak
    handle_speak(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_model ──────────────────────────────────────────────────────────

def test_handle_model_set_by_name(capsys):
    from aura.cli.commands import handle_model
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no loop
    with _patch_ctx(cli_ctx):
        handle_model(agent, "devstral-2:cloud", _ctx())
    agent.brain.set_model_override.assert_called_once_with("devstral-2:cloud")
    out = capsys.readouterr().out
    assert "devstral-2:cloud" in out


def test_handle_model_set_auto(capsys):
    from aura.cli.commands import handle_model
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)
    with _patch_ctx(cli_ctx):
        handle_model(agent, "auto", _ctx())
    agent.brain.set_model_override.assert_called_once_with(None)
    out = capsys.readouterr().out
    assert "auto" in out.lower()


def test_handle_model_set_with_agentic_loop(capsys):
    from aura.cli.commands import handle_model
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_loop=True)
    with _patch_ctx(cli_ctx):
        handle_model(agent, "test-model", _ctx())
    agent.brain.set_model_override.assert_called_once_with("test-model")
    assert cli_ctx.agentic_loop.model_override == "test-model"


# ── handle_compact ────────────────────────────────────────────────────────

def test_handle_compact_smoke(capsys):
    from aura.cli.commands import handle_compact
    agent = _make_mock_agent()
    handle_compact(agent, "", _ctx())
    agent.brain.compact_history.assert_called_once()
    out = capsys.readouterr().out
    assert "Compact" in out


def test_handle_compact_with_focus(capsys):
    from aura.cli.commands import handle_compact
    agent = _make_mock_agent()
    handle_compact(agent, "python code", _ctx())
    agent.brain.compact_history.assert_called_once_with(focus="python code")


# ── handle_plan ───────────────────────────────────────────────────────────

def test_handle_plan_no_arg(capsys):
    from aura.cli.commands import handle_plan
    handle_plan(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_browse ─────────────────────────────────────────────────────────

def test_handle_browse_no_arg(capsys):
    from aura.cli.commands import handle_browse
    handle_browse(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_grep ───────────────────────────────────────────────────────────

def test_handle_grep_no_arg(capsys):
    from aura.cli.commands import handle_grep
    handle_grep(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_search ─────────────────────────────────────────────────────────

def test_handle_search_no_arg(capsys):
    from aura.cli.commands import handle_search
    handle_search(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


# ── handle_edit ───────────────────────────────────────────────────────────

def test_handle_edit_no_arg(capsys):
    from aura.cli.commands import handle_edit
    handle_edit(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


def test_handle_test_autofix_uses_permissions_manager_to_deny():
    from aura.cli.commands import handle_test

    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_loop=True, with_permissions=True)
    cli_ctx.permissions.check.return_value = False

    fake_result = MagicMock()
    fake_result.success = False
    fake_result.failures = ["tests/test_app.py::test_fail"]
    fake_result.output = "failure output"

    with (
        _patch_ctx(cli_ctx),
        patch("aura.cli.test_runner.run_tests", return_value=fake_result),
        patch("aura.cli.test_runner.render_test_results"),
        patch("builtins.input", side_effect=AssertionError("unexpected prompt")),
    ):
        handle_test(agent, "pytest -q", _ctx())

    cli_ctx.permissions.check.assert_called_once_with(
        "auto_fix_tests",
        {"command": "pytest -q", "failure_count": 1},
    )
    cli_ctx.agentic_loop.run.assert_not_called()


def test_handle_test_autofix_uses_permissions_manager_to_allow():
    from aura.cli.commands import handle_test

    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_loop=True, with_permissions=True)
    cli_ctx.permissions.check.return_value = True
    cli_ctx.agentic_loop.run.return_value = {"response": "fixed"}

    fake_result = MagicMock()
    fake_result.success = False
    fake_result.failures = ["tests/test_app.py::test_fail"]
    fake_result.output = "failure output"

    with (
        _patch_ctx(cli_ctx),
        patch("aura.cli.test_runner.run_tests", return_value=fake_result),
        patch("aura.cli.test_runner.render_test_results"),
        patch("builtins.input", side_effect=AssertionError("unexpected prompt")),
    ):
        handle_test(agent, "pytest -q", _ctx())

    cli_ctx.permissions.check.assert_called_once_with(
        "auto_fix_tests",
        {"command": "pytest -q", "failure_count": 1},
    )
    cli_ctx.agentic_loop.run.assert_called_once()


# ── handle_shell ──────────────────────────────────────────────────────────

def test_handle_shell_no_arg(capsys):
    from aura.cli.commands import handle_shell
    handle_shell(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out or "shell" in out.lower()


def test_handle_shell_prompts_when_not_full_auto():
    from aura.cli.commands import handle_shell

    agent = _make_mock_agent()
    shell_tool = MagicMock()
    agent.tools["shell_executor"] = shell_tool
    cli_ctx = _make_mock_cli_ctx(agent, with_permissions=True)
    cli_ctx.permissions.check.return_value = False

    with _patch_ctx(cli_ctx), patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
        handle_shell(agent, "echo hi", _ctx())

    cli_ctx.permissions.check.assert_called_once_with("shell", {"command": "echo hi", "cwd": os.getcwd()})
    shell_tool.run_streaming.assert_not_called()


def test_handle_shell_skips_prompt_in_full_auto():
    from aura.cli.commands import handle_shell

    agent = _make_mock_agent()
    shell_tool = MagicMock()
    shell_tool.run_streaming.return_value = {"success": True, "exit_code": 0, "elapsed": 0.1}
    agent.tools["shell_executor"] = shell_tool
    cli_ctx = _make_mock_cli_ctx(agent, with_permissions=True)
    cli_ctx.permissions.check.return_value = True

    with _patch_ctx(cli_ctx), patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
        handle_shell(agent, "echo hi", _ctx())

    cli_ctx.permissions.check.assert_called_once_with("shell", {"command": "echo hi", "cwd": os.getcwd()})
    shell_tool.run_streaming.assert_called_once()


# ── handle_fleet ──────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_fleet_no_arg(mock_console):
    from aura.cli.commands import handle_fleet
    handle_fleet(_make_mock_agent(), "", _ctx())
    mock_console.print.assert_called()


def test_handle_agent_prompts_when_not_full_auto():
    from aura.cli.commands import handle_agent

    agent = _make_mock_agent()
    agent.permissions = MagicMock()
    agent.permissions.check.return_value = False
    agent.orchestrator = MagicMock()
    agent.orchestrator.specialists = {"coder": MagicMock(description="Writes code")}

    with patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
        handle_agent(agent, "coder investigate", _ctx())

    agent.permissions.check.assert_called_once_with(
        "spawn_agent",
        {"task": "investigate", "specialist": "coder"},
    )
    agent.orchestrator._execute_single.assert_not_called()


def test_handle_agent_skips_prompt_in_full_auto():
    from aura.cli.commands import handle_agent

    agent = _make_mock_agent()
    agent.permissions = MagicMock()
    agent.permissions.check.return_value = True
    agent.orchestrator = MagicMock()
    agent.orchestrator.specialists = {"coder": MagicMock(description="Writes code")}
    agent.orchestrator._execute_single.return_value = MagicMock(success=True, response="done")

    with patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
        handle_agent(agent, "coder investigate", _ctx())

    agent.permissions.check.assert_called_once_with(
        "spawn_agent",
        {"task": "investigate", "specialist": "coder"},
    )
    agent.orchestrator._execute_single.assert_called_once()


# ── handle_tasks ──────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_tasks_no_bg_manager(mock_console):
    from aura.cli.commands import handle_tasks
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no bg_manager
    with _patch_ctx(cli_ctx):
        handle_tasks(agent, "", _ctx())
    mock_console.print.assert_called()


# ── handle_research ───────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_research_no_arg(mock_console):
    from aura.cli.commands import handle_research
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no research_ctx
    with _patch_ctx(cli_ctx):
        handle_research(agent, "", _ctx())
    mock_console.print.assert_called()


@patch("aura.cli.display.console")
def test_handle_research_with_topic(mock_console):
    from aura.cli.commands import handle_research
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
    from aura.cli.commands import handle_sources
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)
    with _patch_ctx(cli_ctx):
        handle_sources(agent, "", _ctx())
    mock_console.print.assert_called()


# ── handle_export ─────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_export_no_research(mock_console):
    from aura.cli.commands import handle_export
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)
    with _patch_ctx(cli_ctx):
        handle_export(agent, "", _ctx())
    mock_console.print.assert_called()


# ── handle_mood ───────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_mood_no_engine(mock_console):
    from aura.cli.commands import handle_mood
    with patch("aura.emotion.alma_engine.get_alma_engine", side_effect=Exception("no engine")):
        handle_mood(_make_mock_agent(), "", _ctx())
    # Should print "not available" message without crashing
    mock_console.print.assert_called()


# ── handle_trust ──────────────────────────────────────────────────────────

def test_handle_trust_smoke(capsys):
    from aura.cli.commands import handle_trust
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_permissions=True)
    with _patch_ctx(cli_ctx):
        handle_trust(agent, "", _ctx())
    cli_ctx.permissions.set_trust_mode.assert_called_once_with(True)
    out = capsys.readouterr().out
    assert "Trust mode" in out


def test_handle_trust_off(capsys):
    from aura.cli.commands import handle_trust
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_permissions=True)
    with _patch_ctx(cli_ctx):
        handle_trust(agent, "off", _ctx())
    cli_ctx.permissions.set_trust_mode.assert_called_once_with(False)


# ── handle_context ────────────────────────────────────────────────────────

def test_handle_context_no_agentic_loop(capsys):
    from aura.cli.commands import handle_context
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no loop
    with _patch_ctx(cli_ctx):
        handle_context(agent, "", _ctx())
    out = capsys.readouterr().out
    assert "not available" in out.lower()


# ── handle_rewind ─────────────────────────────────────────────────────────

def test_handle_rewind_no_agentic_loop(capsys):
    from aura.cli.commands import handle_rewind
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no loop
    with _patch_ctx(cli_ctx):
        handle_rewind(agent, "", _ctx())
    out = capsys.readouterr().out
    assert "No checkpoint" in out


# ── handle_cost ───────────────────────────────────────────────────────────

def test_handle_cost_smoke(capsys):
    from aura.cli.commands import handle_cost
    agent = _make_mock_agent()
    handle_cost(agent, "", _ctx())
    out = capsys.readouterr().out
    assert "Session Cost" in out
    assert "100" in out  # input tokens
    assert "50" in out   # output tokens


# ── handle_undo ───────────────────────────────────────────────────────────

def test_handle_undo_no_agentic_loop(capsys):
    from aura.cli.commands import handle_undo
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no loop
    with _patch_ctx(cli_ctx):
        handle_undo(agent, "", _ctx())
    out = capsys.readouterr().out
    assert "No active" in out


# ── handle_diff ───────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_diff_no_changes(mock_console):
    from aura.cli.commands import handle_diff
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", stderr="")
        handle_diff(_make_mock_agent(), "", _ctx())
    mock_console.print.assert_called()


@patch("aura.cli.display.console")
def test_handle_diff_blocks_unsafe_flag(mock_console):
    from aura.cli.commands import handle_diff
    handle_diff(_make_mock_agent(), "-c evil", _ctx())
    # Should have called show_error via console.print
    assert mock_console.print.called


# ── handle_git ────────────────────────────────────────────────────────────

def test_handle_git_no_arg(capsys):
    from aura.cli.commands import handle_git
    handle_git(_make_mock_agent(), "", _ctx())
    out = capsys.readouterr().out
    assert "Usage" in out


def test_handle_git_blocks_push(capsys):
    from aura.cli.commands import handle_git
    handle_git(_make_mock_agent(), "push", _ctx())
    out = capsys.readouterr().out
    assert "Blocked" in out


def test_handle_git_blocks_dangerous_flag(capsys):
    from aura.cli.commands import handle_git
    handle_git(_make_mock_agent(), "log -c evil", _ctx())
    out = capsys.readouterr().out
    assert "Blocked" in out or "dangerous" in out.lower()


@patch("aura.cli.display.console")
def test_handle_git_status(mock_console):
    from aura.cli.commands import handle_git
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="On branch main", stderr="")
        handle_git(_make_mock_agent(), "status", _ctx())
    mock_console.print.assert_called()


# ── handle_branch ─────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_branch_no_arg(mock_console):
    from aura.cli.commands import handle_branch
    handle_branch(_make_mock_agent(), "", _ctx())
    mock_console.print.assert_called()


# ── handle_blame ──────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_blame_bad_format(mock_console):
    from aura.cli.commands import handle_blame
    handle_blame(_make_mock_agent(), "nolinenum", _ctx())
    mock_console.print.assert_called()


# ── handle_mcp ────────────────────────────────────────────────────────────

def test_handle_mcp_no_loop(capsys):
    from aura.cli.commands import handle_mcp
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no loop
    with _patch_ctx(cli_ctx):
        handle_mcp(agent, "", _ctx())
    out = capsys.readouterr().out
    assert "No MCP" in out or "Configure" in out


# ── handle_sessions ───────────────────────────────────────────────────────

def test_handle_sessions_delete_not_found(capsys):
    from aura.cli.commands import handle_sessions
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


def test_handle_trace_no_active_session(capsys):
    from aura.cli.commands import handle_trace

    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)
    with _patch_ctx(cli_ctx):
        handle_trace(agent, "", _ctx())

    out = capsys.readouterr().out
    assert "No active session trace" in out


@patch("aura.cli.display.console")
def test_handle_trace_renders_recent_events(mock_console):
    from aura.cli.commands import handle_trace

    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_session=True)
    cli_ctx.session.session_id = "ses_demo"
    cli_ctx.session.events = [
        {"type": "response", "run_id": "run_a", "iteration": 1, "payload": {"text": "Started work"}},
        {"type": "tool_start", "run_id": "run_a", "iteration": 2, "payload": {"tool_name": "shell", "tool_args": {"command": "pytest -q"}}},
        {"type": "tool_result", "run_id": "run_a", "iteration": 2, "payload": {"tool_name": "shell", "tool_result": "{\"ok\": true}"}},
        {"type": "run_finished", "run_id": "run_a", "iteration": 2, "payload": {"status": "completed", "model": "qwen"}},
    ]

    with _patch_ctx(cli_ctx):
        handle_trace(agent, "3", _ctx())

    printed = [call.args[0] for call in mock_console.print.call_args_list]
    assert "Trace for ses_demo recent (3/4 events)" in printed[0]
    assert "[2] tool start: shell pytest -q" in printed[1]
    assert "[2] tool result: shell ok" in printed[2]
    assert "[2] run finished: completed (qwen)" in printed[3]


@patch("aura.cli.display.console")
def test_handle_trace_last_groups_latest_run(mock_console):
    from aura.cli.commands import handle_trace

    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_session=True)
    cli_ctx.session.session_id = "ses_demo"
    cli_ctx.session.events = [
        {"type": "response", "run_id": "run_old", "iteration": 1, "payload": {"text": "old run"}},
        {"type": "run_finished", "run_id": "run_old", "iteration": 1, "payload": {"status": "completed", "model": "qwen"}},
        {"type": "tool_start", "run_id": "run_new", "iteration": 1, "payload": {"tool_name": "grep", "tool_args": {"pattern": "todo"}}},
        {"type": "tool_result", "run_id": "run_new", "iteration": 1, "payload": {"tool_name": "grep", "tool_result": "{\"ok\": true}"}},
        {"type": "run_finished", "run_id": "run_new", "iteration": 1, "payload": {"status": "guard_tripped", "model": "qwen"}},
    ]

    with _patch_ctx(cli_ctx):
        handle_trace(agent, "last", _ctx())

    printed = [call.args[0] for call in mock_console.print.call_args_list]
    assert "Trace for ses_demo last run (3/5 events)" in printed[0]
    assert "[1] tool start: grep todo" in printed[1]
    assert "[1] tool result: grep ok" in printed[2]
    assert "[1] run finished: guard_tripped (qwen)" in printed[3]


@patch("aura.cli.display.console")
def test_handle_trace_failures_renders_failed_run_summaries(mock_console):
    from aura.cli.commands import handle_trace

    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_session=True)
    cli_ctx.session.session_id = "ses_demo"
    cli_ctx.session.events = [
        {"type": "run_finished", "run_id": "run_ok", "iteration": 1, "payload": {"status": "completed", "model": "qwen", "tool_calls": 1, "response": "ok"}},
        {"type": "response", "run_id": "run_timeout", "iteration": 1, "payload": {"text": "trying again"}},
        {"type": "run_finished", "run_id": "run_timeout", "iteration": 1, "payload": {"status": "model_timeout", "model": "qwen", "tool_calls": 0, "response": "timed out"}},
        {"type": "tool_start", "run_id": "run_guard", "iteration": 2, "payload": {"tool_name": "shell", "tool_args": {"command": "pytest"}}},
        {"type": "run_finished", "run_id": "run_guard", "iteration": 2, "payload": {"status": "guard_tripped", "model": "qwen", "tool_calls": 1, "response": "blocked"}},
    ]

    with _patch_ctx(cli_ctx):
        handle_trace(agent, "failures", _ctx())

    printed = [call.args[0] for call in mock_console.print.call_args_list]
    assert "Trace for ses_demo failed runs (2 runs)" in printed[0]
    assert "run 1 [run_timeout]: model_timeout (qwen), 0 tool calls -> timed out" in printed[1]
    assert "run 2 [run_guard]: guard_tripped (qwen), 1 tool calls -> blocked" in printed[2]


@patch("aura.cli.display.console")
def test_handle_trace_runs_renders_recent_run_summaries_with_run_ids(mock_console):
    from aura.cli.commands import handle_trace

    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent, with_session=True)
    cli_ctx.session.session_id = "ses_demo"
    cli_ctx.session.events = [
        {"type": "response", "run_id": "run_a", "iteration": 1, "payload": {"text": "first"}},
        {"type": "run_finished", "run_id": "run_a", "iteration": 1, "payload": {"status": "completed", "model": "qwen", "tool_calls": 1, "response": "done one"}},
        {"type": "tool_start", "run_id": "run_b", "iteration": 1, "payload": {"tool_name": "shell", "tool_args": {"command": "pytest"}}},
        {"type": "run_finished", "run_id": "run_b", "iteration": 1, "payload": {"status": "cancelled", "model": "qwen", "tool_calls": 1, "response": "stopped"}},
    ]

    with _patch_ctx(cli_ctx):
        handle_trace(agent, "runs", _ctx())

    printed = [call.args[0] for call in mock_console.print.call_args_list]
    assert "Trace for ses_demo recent runs (2 runs)" in printed[0]
    assert "run 1 [run_a]: completed (qwen), 1 tool calls -> done one" in printed[1]
    assert "run 2 [run_b]: cancelled (qwen), 1 tool calls -> stopped" in printed[2]


# ── handle_theme ──────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_theme_show_current(mock_console):
    from aura.cli.commands import handle_theme
    handle_theme(_make_mock_agent(), "", _ctx())
    mock_console.print.assert_called()


@patch("aura.cli.display.console")
def test_handle_theme_set_unknown(mock_console):
    from aura.cli.commands import handle_theme
    handle_theme(_make_mock_agent(), "nonexistent_theme", _ctx())
    # Should print error about unknown theme
    mock_console.print.assert_called()


# ── handle_hook ───────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_hook_list(mock_console):
    from aura.cli.commands import handle_hook
    agent = _make_mock_agent()
    cli_ctx = _make_mock_cli_ctx(agent)  # no hook_manager initially
    with _patch_ctx(cli_ctx):
        handle_hook(agent, "", _ctx())
    # Should create a HookManager and call render_hooks_table


# ── handle_watch ──────────────────────────────────────────────────────────

@patch("aura.cli.display.console")
def test_handle_watch_stop(mock_console):
    from aura.cli.commands import handle_watch
    agent = _make_mock_agent()
    mock_watcher = MagicMock()
    cli_ctx = _make_mock_cli_ctx(agent)
    cli_ctx.file_watcher = mock_watcher
    with _patch_ctx(cli_ctx):
        handle_watch(agent, "stop", _ctx())
    mock_watcher.stop.assert_called_once()

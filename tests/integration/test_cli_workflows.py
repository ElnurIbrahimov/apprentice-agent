"""Transcript-style CLI workflow tests.

These tests exercise realistic multi-step operator flows across command
handlers and top-level subcommands so permission consistency is validated
at the workflow level, not just per handler.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from aura.cli.chat_loop import _rewind_picker
from aura.cli.checkpoint import CheckpointManager
from aura.cli.commands.session_commands import handle_sessions
from aura.cli.commands.session_commands import handle_retry
from aura.cli.commands.tool_commands import handle_test
from aura.cli.context import CLIContext
from aura.core.commands import cmd_commit
from aura.core.session import AgenticSession


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.brain = MagicMock()
    agent.brain._model_override = "test-model:latest"
    agent.tools = {}
    return agent


def _make_cli_ctx(agent: MagicMock) -> CLIContext:
    ctx = CLIContext(agent=agent)
    ctx.permissions = MagicMock()
    ctx.agentic_loop = MagicMock()
    return ctx


def test_failure_recovery_workflow_transcript_routes_permissions_consistently():
    agent = _make_agent()
    ctx = _make_cli_ctx(agent)
    loop = ctx.agentic_loop
    loop._conversation_history = [{"role": "user", "content": "fix auth bug"}]
    loop._loop_error = True
    loop._current_tier = "fast"
    loop.router = SimpleNamespace(tier="fast")
    loop.run.side_effect = [
        {"response": "applied test fixes", "model": "test-model:latest"},
        {"response": "retry succeeded", "model": "test-model:latest"},
    ]
    ctx.permissions.check.return_value = True

    failing_tests = SimpleNamespace(
        success=False,
        failures=["tests/test_auth.py::test_refresh"],
        output="AssertionError: refresh token mismatch",
    )

    with (
        patch("aura.cli.context._current", ctx),
        patch("aura.cli.test_runner.run_tests", return_value=failing_tests),
        patch("aura.cli.test_runner.render_test_results"),
    ):
        handle_test(agent, "pytest -q", {"speak": False})
        handle_retry(agent, "", {"speak": False})

    assert ctx.permissions.check.call_args_list == [
        call("auto_fix_tests", {"command": "pytest -q", "failure_count": 1}),
        call(
            "retry_tier_escalation",
            {
                "from_tier": "fast",
                "to_tier": "balanced",
                "prompt": "fix auth bug",
            },
        ),
    ]
    assert loop.router.tier == "balanced"
    assert loop.run.call_count == 2
    assert loop.run.call_args_list[0].args[0].startswith("These tests failed:")
    assert loop.run.call_args_list[1].args[0] == "fix auth bug"


@patch("aura.ApprenticeAgent")
@patch("aura.tools.git_tool.GitTool")
@patch("aura.core.commands._create_subcommand_permission_manager")
@patch("aura.core.commands.subprocess.run")
def test_recovery_then_commit_workflow_transcript_keeps_mutation_gates_centralized(
    mock_run,
    mock_permissions_factory,
    mock_git_cls,
    mock_agent_cls,
):
    agent = _make_agent()
    ctx = _make_cli_ctx(agent)
    loop = ctx.agentic_loop
    loop._conversation_history = [{"role": "user", "content": "stabilize auth flow"}]
    loop._loop_error = True
    loop._current_tier = "fast"
    loop.router = SimpleNamespace(tier="fast")
    loop.run.side_effect = [
        {"response": "auto-fixed tests", "model": "test-model:latest"},
        {"response": "retry completed", "model": "test-model:latest"},
    ]
    ctx.permissions.check.return_value = True

    failing_tests = SimpleNamespace(
        success=False,
        failures=["tests/test_auth.py::test_refresh"],
        output="AssertionError: refresh token mismatch",
    )

    subcommand_permissions = MagicMock()
    subcommand_permissions.check.side_effect = [True, True]
    mock_permissions_factory.return_value = subcommand_permissions

    git = MagicMock()
    git.status.return_value = {"success": True, "dirty_count": 1}
    git.diff.return_value = {"diff": "diff --git a/auth.py b/auth.py\n+fix"}
    git.add.return_value = {"success": True}
    git.commit.return_value = {"success": True}
    mock_git_cls.return_value = git

    commit_agent = MagicMock()
    commit_agent.brain.think.return_value = "fix: stabilize auth flow"
    mock_agent_cls.return_value = commit_agent

    mock_run.side_effect = [
        MagicMock(stdout="", returncode=0),
        MagicMock(stdout="diff --git a/auth.py b/auth.py\n+fix", returncode=0),
    ]

    with (
        patch("aura.cli.context._current", ctx),
        patch("aura.cli.test_runner.run_tests", return_value=failing_tests),
        patch("aura.cli.test_runner.render_test_results"),
        patch("builtins.input", return_value=""),
    ):
        handle_test(agent, "pytest -q", {"speak": False})
        handle_retry(agent, "", {"speak": False})
        result = cmd_commit(SimpleNamespace(all=True))

    assert result == 0
    assert ctx.permissions.check.call_args_list == [
        call("auto_fix_tests", {"command": "pytest -q", "failure_count": 1}),
        call(
            "retry_tier_escalation",
            {
                "from_tier": "fast",
                "to_tier": "balanced",
                "prompt": "stabilize auth flow",
            },
        ),
    ]
    assert subcommand_permissions.check.call_args_list == [
        call("git", {"action": "add", "files": "."}),
        call("git", {"action": "commit", "message": "fix: stabilize auth flow"}),
    ]
    git.commit.assert_called_once_with(os.getcwd(), message="fix: stabilize auth flow")


def test_new_session_rewind_resume_workflow_restores_file_and_history(mock_brain, tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.chdir(project_root)

    target_file = project_root / "app.py"
    target_file.write_text("print('original')\n", encoding="utf-8")

    sessions_dir = tmp_path / "sessions"
    checkpoints_dir = project_root / ".aura_checkpoints"

    original_session = AgenticSession(sessions_dir=str(sessions_dir))
    original_session.new(project_root=str(project_root), model="test-model:latest")
    original_session.append({"role": "user", "content": "edit app"})
    original_session.append({"role": "assistant", "content": "done"})
    original_session.save()
    original_session_id = original_session.session_id

    from aura.core.agentic_loop import AgenticLoop

    loop = AgenticLoop(
        brain=mock_brain,
        project_root=str(project_root),
        max_iterations=1,
        session=original_session,
    )
    loop._conversation_history = list(original_session.messages)

    checkpoint_mgr = CheckpointManager(checkpoint_dir=checkpoints_dir)
    checkpoint_mgr.snapshot(str(target_file), label="before edit")
    target_file.write_text("print('edited')\n", encoding="utf-8")

    agent = _make_agent()
    agent.brain.list_conversations.return_value = []
    ctx = CLIContext(agent=agent, agentic_loop=loop, session=original_session)

    with patch("aura.cli.context._current", ctx):
        handle_sessions(agent, "new", {"speak": False})

    assert ctx.session is not original_session
    assert ctx.session.session_id != original_session_id
    assert loop.session is ctx.session
    assert loop._conversation_history == []

    console = MagicMock()
    with patch("builtins.input", return_value="1"):
        assert _rewind_picker(checkpoint_mgr, console) is True

    assert target_file.read_text(encoding="utf-8") == "print('original')\n"

    resumed_session = AgenticSession(sessions_dir=str(sessions_dir))
    resumed_loop = AgenticLoop(
        brain=mock_brain,
        project_root=str(project_root),
        max_iterations=1,
        session=resumed_session,
    )

    assert resumed_loop.load_session(original_session_id) is True
    assert [m["content"] for m in resumed_loop._conversation_history] == ["edit app", "done"]


def test_sessions_command_can_switch_live_loop_to_saved_session(mock_brain, tmp_path):
    sessions_dir = tmp_path / "sessions"

    saved_session = AgenticSession(sessions_dir=str(sessions_dir))
    saved_session.new(project_root=str(tmp_path), model="test-model:latest")
    saved_session.append({"role": "user", "content": "saved prompt"})
    saved_session.append({"role": "assistant", "content": "saved response"})
    saved_session.save()
    saved_summary = saved_session.list_sessions(limit=1)[0]

    current_session = AgenticSession(sessions_dir=str(sessions_dir))
    current_session.new(project_root=str(tmp_path), model="test-model:latest")
    current_session.append({"role": "user", "content": "current prompt"})
    current_session.save()

    from aura.core.agentic_loop import AgenticLoop

    loop = AgenticLoop(
        brain=mock_brain,
        project_root=str(tmp_path),
        max_iterations=1,
        session=current_session,
    )
    loop._conversation_history = list(current_session.messages)

    agent = _make_agent()
    agent.brain.list_conversations.return_value = []
    ctx = CLIContext(agent=agent, agentic_loop=loop, session=current_session)

    session_mgr = AgenticSession(sessions_dir=str(sessions_dir))

    with (
        patch("aura.cli.context._current", ctx),
        patch("aura.core.session.AgenticSession", return_value=session_mgr),
        patch("aura.cli.session_picker.pick_session", return_value=saved_summary),
    ):
        handle_sessions(agent, "", {"speak": False})

    assert [m["content"] for m in loop._conversation_history] == ["saved prompt", "saved response"]

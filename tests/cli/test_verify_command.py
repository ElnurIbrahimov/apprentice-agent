"""Tests for the /verify slash command."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aura.cli.commands.verify_commands import handle_verify


def _fake_ctx(hot_files=None, agentic_loop=True, aura_config=None):
    agentic = MagicMock() if agentic_loop else None
    if agentic:
        agentic._hot_files = hot_files or []
        agentic.project_root = "."
        agentic.aura_config = aura_config or {}
        agentic._verification_stage = None  # force fresh build
    ctx = SimpleNamespace(
        agentic_loop=agentic,
        session=SimpleNamespace(session_id="s1"),
    )
    return ctx


def test_no_active_session_prints_warning(capsys):
    with patch("aura.cli.commands.verify_commands.get_ctx", return_value=None):
        handle_verify(MagicMock(), "", {})
    out = capsys.readouterr().out
    assert "No active session" in out


def test_no_hot_files_prints_dim_notice(capsys):
    ctx = _fake_ctx(hot_files=[])
    with patch("aura.cli.commands.verify_commands.get_ctx", return_value=ctx):
        handle_verify(MagicMock(), "", {})
    out = capsys.readouterr().out
    assert "No files edited" in out


def test_verify_runs_stage_with_hot_files():
    ctx = _fake_ctx(hot_files=["foo.py"])
    fake_stage = MagicMock()
    fake_stage.mode = "typecheck"
    fake_stage.run.return_value = SimpleNamespace(
        mode="typecheck", success=True, duration_s=0.1,
        stages=[{"name": "typecheck", "runner": "mypy", "success": True,
                 "duration_s": 0.1, "failures": []}],
        changed_files=["foo.py"], skipped_reason="",
    )
    with patch("aura.cli.commands.verify_commands.get_ctx", return_value=ctx), \
         patch("aura.core.verification_stage.VerificationStage", return_value=fake_stage):
        handle_verify(MagicMock(), "", {})

    fake_stage.run.assert_called_once()
    args, kwargs = fake_stage.run.call_args
    assert "foo.py" in args[0]


def test_verify_override_mode_is_scoped():
    """/verify tests should override stage.mode temporarily but restore it."""
    ctx = _fake_ctx(hot_files=["foo.py"])
    fake_stage = MagicMock()
    fake_stage.mode = "typecheck"  # starting mode
    fake_stage.run.return_value = SimpleNamespace(
        mode="tests", success=True, duration_s=0.2,
        stages=[{"name": "tests", "runner": "pytest", "success": True,
                 "duration_s": 0.2, "failures": []}],
        changed_files=["foo.py"], skipped_reason="",
    )

    with patch("aura.cli.commands.verify_commands.get_ctx", return_value=ctx), \
         patch("aura.core.verification_stage.VerificationStage", return_value=fake_stage):
        handle_verify(MagicMock(), "tests", {})

    # After the call, original mode should be restored.
    assert fake_stage.mode == "typecheck"

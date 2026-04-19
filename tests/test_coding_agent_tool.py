"""Tests for aura.tools.coding_agent.CodingAgentTool."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from aura.core.permissions import SandboxTier, get_sandbox_tier, set_sandbox_tier
from aura.tools.coding_agent import AGENT_NAMES, CodingAgentTool


@pytest.fixture(autouse=True)
def _restore_sandbox():
    original = get_sandbox_tier()
    yield
    set_sandbox_tier(original)


def test_available_agents_reports_all(monkeypatch):
    """available_agents returns a dict with every agent name as a key."""
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/bin/{name}")
    avail = CodingAgentTool.available_agents()
    assert set(avail.keys()) == set(AGENT_NAMES)
    assert all(avail.values())


def test_build_command_claude():
    cmd = CodingAgentTool._build_command("claude", "fix the bug")
    assert cmd[0] == "claude"
    assert "--print" in cmd
    assert "--permission-mode" in cmd
    assert "bypassPermissions" in cmd
    assert cmd[-1] == "fix the bug"


def test_build_command_codex():
    cmd = CodingAgentTool._build_command("codex", "refactor X")
    assert cmd[:2] == ["codex", "exec"]
    assert cmd[-1] == "refactor X"


def test_build_command_aider():
    cmd = CodingAgentTool._build_command("aider", "add tests")
    assert cmd[0] == "aider"
    assert "--message" in cmd
    assert "add tests" in cmd
    assert "--yes-always" in cmd


def test_build_command_opencode():
    cmd = CodingAgentTool._build_command("opencode", "do X")
    assert cmd[:2] == ["opencode", "run"]


def test_build_command_goose():
    cmd = CodingAgentTool._build_command("goose", "do Y")
    assert cmd[:3] == ["goose", "run", "--text"]
    assert "do Y" in cmd


def test_build_command_unknown_agent_raises():
    with pytest.raises(ValueError, match="Unknown agent"):
        CodingAgentTool._build_command("cursor", "whatever")


def test_delegate_unknown_agent_raises():
    with pytest.raises(ValueError):
        CodingAgentTool.delegate("cursor", "whatever")


def test_delegate_agent_not_on_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = CodingAgentTool.delegate("claude", "test prompt")
    assert result.success is False
    assert result.exit_code == 127
    assert "not found" in result.stderr.lower()


def test_delegate_read_only_sandbox_blocks(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    set_sandbox_tier(SandboxTier.READ_ONLY)
    result = CodingAgentTool.delegate("claude", "test")
    assert result.success is False
    assert "BLOCKED" in result.stderr
    assert "READ_ONLY" in result.stderr


def test_delegate_successful_subprocess(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = "did the thing"
    fake_proc.stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_proc)

    result = CodingAgentTool.delegate("claude", "fix X")
    assert result.success is True
    assert result.exit_code == 0
    assert "did the thing" in result.stdout
    assert result.agent == "claude"


def test_delegate_nonzero_exit(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    fake_proc = MagicMock()
    fake_proc.returncode = 2
    fake_proc.stdout = ""
    fake_proc.stderr = "syntax error"
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_proc)

    result = CodingAgentTool.delegate("codex", "fix X")
    assert result.success is False
    assert result.exit_code == 2
    assert "syntax error" in result.stderr


def test_delegate_timeout(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")

    def _raise_timeout(*_, **__):
        raise subprocess.TimeoutExpired(cmd=["claude", "test"], timeout=10)

    monkeypatch.setattr("subprocess.run", _raise_timeout)
    result = CodingAgentTool.delegate("claude", "test", timeout=10)
    assert result.success is False
    assert result.exit_code == -9
    assert "TIMEOUT" in result.stderr


def test_tool_schema_is_valid():
    schema = CodingAgentTool.tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "coding_agent"
    props = schema["function"]["parameters"]["properties"]
    assert "agent" in props
    assert "prompt" in props
    required = schema["function"]["parameters"]["required"]
    assert "agent" in required and "prompt" in required
    # agent enum lists all supported agents
    assert set(props["agent"]["enum"]) == set(AGENT_NAMES)


def test_result_summary_truncates_long_stdout(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    long_output = "x" * 5000
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = long_output
    fake_proc.stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_proc)
    result = CodingAgentTool.delegate("claude", "test")
    summary = result.summary(max_chars=1000)
    assert "truncated" in summary
    assert len(summary) < len(long_output) + 500


def test_run_returns_summary_string(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/fake/{name}")
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = "hello from agent"
    fake_proc.stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_proc)
    out = CodingAgentTool.run("claude", "say hi")
    assert isinstance(out, str)
    assert "hello from agent" in out
    assert "exit=0" in out

"""Tests for aura.core.permissions.SandboxTier clamp behavior."""
from __future__ import annotations

import pytest

from aura.core.permissions import (
    AUTO,
    BLOCKED,
    PROMPT,
    PermissionManager,
    SandboxTier,
    _clamp_tier_for_sandbox,
    get_sandbox_tier,
    set_sandbox_tier,
)


@pytest.fixture(autouse=True)
def _reset_sandbox_tier():
    """Ensure each test starts with UNRESTRICTED and restores after."""
    original = get_sandbox_tier()
    set_sandbox_tier(SandboxTier.UNRESTRICTED)
    yield
    set_sandbox_tier(original)


def test_default_is_unrestricted():
    assert get_sandbox_tier() == SandboxTier.UNRESTRICTED


def test_read_only_blocks_write_file():
    set_sandbox_tier(SandboxTier.READ_ONLY)
    assert _clamp_tier_for_sandbox("write_file", AUTO) == BLOCKED


def test_read_only_blocks_shell():
    set_sandbox_tier(SandboxTier.READ_ONLY)
    assert _clamp_tier_for_sandbox("shell", PROMPT) == BLOCKED


def test_read_only_allows_read_file():
    set_sandbox_tier(SandboxTier.READ_ONLY)
    assert _clamp_tier_for_sandbox("read_file", AUTO) == AUTO


def test_read_only_allows_git_status():
    set_sandbox_tier(SandboxTier.READ_ONLY)
    assert _clamp_tier_for_sandbox("git.status", AUTO) == AUTO


def test_read_only_blocks_git_commit():
    set_sandbox_tier(SandboxTier.READ_ONLY)
    assert _clamp_tier_for_sandbox("git.commit", PROMPT) == BLOCKED


def test_workspace_write_escalates_shell_from_auto_to_prompt():
    set_sandbox_tier(SandboxTier.WORKSPACE_WRITE)
    # Even if shell is AUTO, workspace-write forces prompt
    assert _clamp_tier_for_sandbox("shell", AUTO) == PROMPT


def test_workspace_write_leaves_write_file_unclamped():
    set_sandbox_tier(SandboxTier.WORKSPACE_WRITE)
    # write_file defaults to AUTO; workspace-write doesn't touch it
    assert _clamp_tier_for_sandbox("write_file", AUTO) == AUTO


def test_workspace_write_escalates_spawn_agent():
    set_sandbox_tier(SandboxTier.WORKSPACE_WRITE)
    assert _clamp_tier_for_sandbox("spawn_agent", AUTO) == PROMPT


def test_unrestricted_is_noop():
    set_sandbox_tier(SandboxTier.UNRESTRICTED)
    assert _clamp_tier_for_sandbox("write_file", AUTO) == AUTO
    assert _clamp_tier_for_sandbox("shell", PROMPT) == PROMPT


def test_permission_manager_check_respects_read_only():
    """High-level check() must enforce READ_ONLY sandbox."""
    pm = PermissionManager()
    set_sandbox_tier(SandboxTier.READ_ONLY)
    # write_file would normally be AUTO → now BLOCKED
    assert pm.check("write_file", {"path": "foo.txt", "content": "x"}) is False
    # read_file still allowed
    assert pm.check("read_file", {"path": "foo.txt"}) is True


def test_permission_manager_read_only_beats_trust_mode():
    """READ_ONLY sandbox must be stronger than trust mode (safety first)."""
    pm = PermissionManager()
    pm.set_trust_mode(True)
    set_sandbox_tier(SandboxTier.READ_ONLY)
    # Trust mode would normally auto-approve, but READ_ONLY blocks
    assert pm.check("write_file", {"path": "foo.txt", "content": "x"}) is False


def test_permission_manager_trust_mode_works_when_unrestricted():
    pm = PermissionManager()
    pm.set_trust_mode(True)
    set_sandbox_tier(SandboxTier.UNRESTRICTED)
    assert pm.check("write_file", {"path": "foo.txt", "content": "x"}) is True
    assert pm.check("shell", {"command": "ls"}) is True

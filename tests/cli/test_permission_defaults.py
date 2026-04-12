from aura.core.permissions import PermissionManager


def test_read_only_git_actions_remain_auto_approved():
    permissions = PermissionManager()

    assert permissions.check("git", {"action": "status"})
    assert permissions.check("git", {"action": "diff"})


def test_git_mutations_require_explicit_approval_by_default():
    permissions = PermissionManager()

    assert not permissions.check("git", {"action": "push"})
    assert not permissions.check("git", {"action": "pull"})
    assert not permissions.check("git", {"action": "commit", "message": "save work"})


def test_shell_auto_approval_is_limited_to_read_only_commands():
    permissions = PermissionManager()

    assert permissions.check("shell", {"command": "pwd"})
    assert permissions.check("shell", {"command": "Get-ChildItem"})
    assert not permissions.check("shell", {"command": "git status"})
    assert not permissions.check("shell", {"command": "npm install"})


def test_spawn_agent_requires_explicit_approval_by_default():
    permissions = PermissionManager()

    assert not permissions.check("spawn_agent", {"task": "inspect repo", "specialist": "coder"})


def test_clear_history_requires_explicit_approval_by_default():
    permissions = PermissionManager()

    assert not permissions.check("clear_history", {"scope": "conversation_history"})


def test_auto_fix_tests_requires_explicit_approval_by_default():
    permissions = PermissionManager()

    assert not permissions.check("auto_fix_tests", {"command": "pytest", "failure_count": 2})


def test_retry_tier_escalation_requires_explicit_approval_by_default():
    permissions = PermissionManager()

    assert not permissions.check(
        "retry_tier_escalation",
        {"from_tier": "fast", "to_tier": "balanced", "prompt": "fix auth bug"},
    )


def test_mode_property_tracks_base_mode_and_trust_override():
    permissions = PermissionManager()

    permissions.set_mode("auto_edit")
    assert permissions.mode == "auto_edit"
    assert permissions.current_mode == "auto_edit"

    permissions.set_mode("full_auto")
    assert permissions.mode == "full_auto"
    assert permissions.current_mode == "full_auto"

    permissions.set_trust_mode(False)
    assert permissions.mode == "auto_edit"
    assert permissions.current_mode == "auto_edit"

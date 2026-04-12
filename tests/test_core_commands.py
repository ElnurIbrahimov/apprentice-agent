import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aura.core.commands import cmd_commit


def _commit_args(*, all_changes: bool = False):
    return SimpleNamespace(all=all_changes)


def _mock_subprocess_run():
    return [
        MagicMock(stdout="", returncode=0),
        MagicMock(stdout="diff --git a/app.py b/app.py\n+change", returncode=0),
    ]


@patch("aura.ApprenticeAgent")
@patch("aura.tools.git_tool.GitTool")
@patch("aura.core.commands._create_subcommand_permission_manager")
@patch("aura.core.commands.subprocess.run")
def test_cmd_commit_uses_permission_manager_for_stage_and_commit(
    mock_run,
    mock_permissions_factory,
    mock_git_cls,
    mock_agent_cls,
):
    permissions = MagicMock()
    permissions.check.side_effect = [True, True]
    mock_permissions_factory.return_value = permissions

    git = MagicMock()
    git.status.return_value = {"success": True, "dirty_count": 1}
    git.diff.return_value = {"diff": "diff --git a/app.py b/app.py\n+change"}
    git.add.return_value = {"success": True}
    git.commit.return_value = {"success": True}
    mock_git_cls.return_value = git

    agent = MagicMock()
    agent.brain.think.return_value = "feat: improve commit flow"
    mock_agent_cls.return_value = agent

    mock_run.side_effect = _mock_subprocess_run()

    with patch("builtins.input", return_value=""):
        result = cmd_commit(_commit_args(all_changes=True))

    assert result == 0
    permissions.check.assert_any_call("git", {"action": "add", "files": "."})
    permissions.check.assert_any_call("git", {"action": "commit", "message": "feat: improve commit flow"})
    git.add.assert_called_once()
    git.commit.assert_called_once_with(os.getcwd(), message="feat: improve commit flow")


@patch("aura.ApprenticeAgent")
@patch("aura.tools.git_tool.GitTool")
@patch("aura.core.commands._create_subcommand_permission_manager")
def test_cmd_commit_cancels_when_stage_denied(
    mock_permissions_factory,
    mock_git_cls,
    mock_agent_cls,
):
    permissions = MagicMock()
    permissions.check.return_value = False
    mock_permissions_factory.return_value = permissions

    git = MagicMock()
    git.status.return_value = {"success": True, "dirty_count": 1}
    git.diff.return_value = {"diff": "diff --git a/app.py b/app.py\n+change"}
    mock_git_cls.return_value = git

    mock_agent_cls.return_value = MagicMock()

    result = cmd_commit(_commit_args(all_changes=True))

    assert result == 0
    permissions.check.assert_called_once_with("git", {"action": "add", "files": "."})
    git.add.assert_not_called()
    git.commit.assert_not_called()


@patch("aura.ApprenticeAgent")
@patch("aura.tools.git_tool.GitTool")
@patch("aura.core.commands._create_subcommand_permission_manager")
@patch("aura.core.commands.subprocess.run")
def test_cmd_commit_cancels_when_commit_denied(
    mock_run,
    mock_permissions_factory,
    mock_git_cls,
    mock_agent_cls,
):
    permissions = MagicMock()
    permissions.check.side_effect = [False]
    mock_permissions_factory.return_value = permissions

    git = MagicMock()
    git.status.return_value = {"success": True, "dirty_count": 1}
    git.diff.return_value = {"diff": "diff --git a/app.py b/app.py\n+change"}
    mock_git_cls.return_value = git

    agent = MagicMock()
    agent.brain.think.return_value = "feat: improve commit flow"
    mock_agent_cls.return_value = agent

    mock_run.side_effect = _mock_subprocess_run()

    with patch("builtins.input", return_value=""):
        result = cmd_commit(_commit_args(all_changes=False))

    assert result == 0
    permissions.check.assert_called_once_with("git", {"action": "commit", "message": "feat: improve commit flow"})
    git.commit.assert_not_called()


@patch("aura.ApprenticeAgent")
@patch("aura.tools.git_tool.GitTool")
@patch("aura.core.commands._create_subcommand_permission_manager")
@patch("aura.core.commands.subprocess.run")
def test_cmd_commit_allows_message_edit_before_approval(
    mock_run,
    mock_permissions_factory,
    mock_git_cls,
    mock_agent_cls,
):
    permissions = MagicMock()
    permissions.check.return_value = True
    mock_permissions_factory.return_value = permissions

    git = MagicMock()
    git.status.return_value = {"success": True, "dirty_count": 1}
    git.diff.return_value = {"diff": "diff --git a/app.py b/app.py\n+change"}
    git.commit.return_value = {"success": True}
    mock_git_cls.return_value = git

    agent = MagicMock()
    agent.brain.think.return_value = "feat: improve commit flow"
    mock_agent_cls.return_value = agent

    mock_run.side_effect = _mock_subprocess_run()

    with patch("builtins.input", side_effect=["y", "feat: custom message"]):
        result = cmd_commit(_commit_args(all_changes=False))

    assert result == 0
    permissions.check.assert_called_once_with("git", {"action": "commit", "message": "feat: custom message"})
    git.commit.assert_called_once_with(os.getcwd(), message="feat: custom message")

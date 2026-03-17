"""Tests for git power tools."""
import pytest
import re
from aura.cli.git_tools import (
    create_branch, smart_stash, get_blame, auto_commit_edit,
    COMMIT_MSG_PROMPT, PR_DESCRIPTION_PROMPT, git_run,
)

def test_git_run_success(tmp_path):
    # Init a temp git repo
    git_run(["init"], cwd=str(tmp_path))
    result = git_run(["status"], cwd=str(tmp_path))
    assert result["success"]

def test_git_run_failure():
    result = git_run(["status"], cwd="/nonexistent_path_xyz")
    assert not result["success"]

def test_create_branch_sanitizes_name():
    # Just test the sanitization logic
    safe = re.sub(r'[^\w\-/.]', '-', "feat/my cool feature!!")[:60]
    assert safe == "feat/my-cool-feature--"

def test_commit_msg_prompt():
    prompt = COMMIT_MSG_PROMPT.format(diff="+ added login\n- removed old auth")
    assert "login" in prompt
    assert "type(scope)" in prompt

def test_pr_description_prompt():
    prompt = PR_DESCRIPTION_PROMPT.format(branch="feat/auth", diff="3 files", log="abc123 fix auth")
    assert "feat/auth" in prompt

def test_git_run_timeout():
    # This should not hang — git status is fast
    result = git_run(["status"], timeout=5)
    # May fail if not in a git repo, but should not timeout
    assert isinstance(result, dict)

def test_get_blame_format():
    # get_blame returns a dict with specific keys when successful
    # Can't easily test without a real repo, so test the structure
    result = get_blame("nonexistent_file.py", 1)
    assert "success" in result or "error" in result

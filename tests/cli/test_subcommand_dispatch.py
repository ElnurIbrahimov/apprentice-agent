"""Regression tests for subcommand parsing with positional arguments.

Historical bug: the top-level `goal` positional was registered as `nargs="*"`
on the same parser as the subparsers. argparse greedily swallowed the
subcommand token into `goal`, then tried to match the trailing positional
as a subparser choice and failed with 'invalid choice: <value>'.

These tests lock in the contract that every subcommand that accepts its
own positional arguments (`recall`, `why`, `worktree`, `log`, `ide`, `exec`)
can be invoked without colliding with the top-level `goal` positional.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main as aura_main  # noqa: E402


def _parse(argv):
    """Simulate main's two-step parse: build parser for these argv, then parse.

    Uses monkey-patched sys.argv so the pre-scan sees the right tokens.
    Returns the argparse.Namespace.
    """
    saved = sys.argv
    sys.argv = ["aura"] + list(argv)
    try:
        parser, use_subparsers = aura_main._build_argument_parser()
        args = parser.parse_args(argv)
        if not use_subparsers:
            # mirror main()'s post-hoc normalization
            args.command = None
        return args
    finally:
        sys.argv = saved


class TestSubcommandsWithPositionals:
    """Previously all these raised SystemExit(2) via argparse 'invalid choice'."""

    def test_recall_accepts_positional(self):
        args = _parse(["recall", "telegram"])
        assert args.command == "recall"
        assert args.recall_query == ["telegram"]

    def test_recall_accepts_multiple_words(self):
        args = _parse(["recall", "broadmind", "training", "results"])
        assert args.command == "recall"
        assert args.recall_query == ["broadmind", "training", "results"]

    def test_recall_with_limit_flag(self):
        args = _parse(["recall", "telegram", "--limit", "3"])
        assert args.command == "recall"
        assert args.recall_query == ["telegram"]
        assert args.recall_limit == 3

    def test_why_accepts_path(self):
        args = _parse(["why", "main.py"])
        assert args.command == "why"
        assert args.why_target == "main.py"

    def test_why_accepts_path_with_line(self):
        args = _parse(["why", "aura/brain.py:150"])
        assert args.command == "why"
        assert args.why_target == "aura/brain.py:150"

    def test_worktree_accepts_name(self):
        args = _parse(["worktree", "myfeature"])
        assert args.command == "worktree"
        assert args.worktree_name == "myfeature"

    def test_worktree_list_flag(self):
        args = _parse(["worktree", "--list"])
        assert args.command == "worktree"
        assert args.worktree_list is True

    def test_log_recent_action(self):
        args = _parse(["log", "recent"])
        assert args.command == "log"
        assert args.action == "recent"

    def test_log_search_with_query(self):
        args = _parse(["log", "search", "telegram", "bot"])
        assert args.command == "log"
        assert args.action == "search"
        assert args.query == ["telegram", "bot"]

    def test_ide_validate_action(self):
        args = _parse(["ide", "validate"])
        assert args.command == "ide"
        assert args.action == "validate"

    def test_ide_reset_action(self):
        args = _parse(["ide", "reset"])
        assert args.command == "ide"
        assert args.action == "reset"

    def test_exec_accepts_prompt(self):
        args = _parse(["exec", "say hello"])
        assert args.command == "exec"
        assert args.exec_prompt == "say hello"

    def test_exec_with_timeout(self):
        args = _parse(["exec", "say hi", "--timeout", "5"])
        assert args.command == "exec"
        assert args.exec_prompt == "say hi"
        assert args.exec_timeout == 5


class TestDirectGoalPath:
    """When no subcommand is given, first positional becomes the goal prompt."""

    def test_single_word_goal(self):
        args = _parse(["hello"])
        assert args.command is None
        assert args.goal == ["hello"]

    def test_multi_token_goal(self):
        args = _parse(["fix", "the", "login", "bug"])
        assert args.command is None
        assert args.goal == ["fix", "the", "login", "bug"]

    def test_quoted_goal_single_token(self):
        args = _parse(["fix the login bug"])
        assert args.command is None
        assert args.goal == ["fix the login bug"]

    def test_goal_with_flags(self):
        args = _parse(["--fast", "--model", "kimi-k2.6:cloud", "say ACK"])
        assert args.command is None
        assert args.goal == ["say ACK"]
        assert args.fast is True
        assert args.model == "kimi-k2.6:cloud"

    def test_flag_only_no_goal(self):
        args = _parse(["--fast"])
        assert args.command is None
        assert args.goal == []


class TestFlagOnlySubcommands:
    """Subcommands that take no positionals still dispatch."""

    def test_doctor(self):
        args = _parse(["doctor"])
        assert args.command == "doctor"

    def test_status(self):
        args = _parse(["status"])
        assert args.command == "status"

    def test_commit_with_all_flag(self):
        args = _parse(["commit", "--all"])
        assert args.command == "commit"
        assert args.all is True

    def test_cost_with_breakdown(self):
        args = _parse(["cost", "--by-model"])
        assert args.command == "cost"
        assert args.by_model is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

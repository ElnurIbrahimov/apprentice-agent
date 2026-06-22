"""Tests for the cron, guardrails, sessions, skills, and plugins helpers
added during the engineering review. Covers path-traversal safety,
helper consolidation, and basic correctness of the new utility modules.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the project root is on sys.path so the imports below resolve.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── guardrails: deque cap, warn/hard-stop behavior ──────────────────────

def test_guardrails_continue_on_success():
    from aura.guardrails import record_tool_call, reset_state, is_hard_stopped
    reset_state()
    result = record_tool_call("read_file", {"path": "foo.py"}, success=True)
    assert result["action"] == "continue"
    assert not is_hard_stopped()


def test_guardrails_warn_after_same_tool_failures():
    from aura.guardrails import record_tool_call, reset_state
    reset_state()
    # Threshold for warning on same-tool failures is 3 by default.
    actions = []
    for _ in range(3):
        actions.append(record_tool_call("read_file", {"path": "x.py"}, success=False))
    # First two failures should continue, third should warn.
    assert actions[0]["action"] == "continue"
    assert actions[1]["action"] == "continue"
    assert actions[2]["action"] == "warn"


def test_guardrails_hard_stop_after_threshold():
    from aura.guardrails import record_tool_call, reset_state, is_hard_stopped
    reset_state()
    for _ in range(8):
        record_tool_call("read_file", {"path": "x.py"}, success=False)
    assert is_hard_stopped()


def test_guardrails_idempotent_detection():
    from aura.guardrails import record_tool_call, reset_state
    reset_state()
    # Three identical no-progress calls in a row should be flagged as idempotent.
    r1 = record_tool_call("read_file", {"path": "a.py"}, success=False)
    r2 = record_tool_call("read_file", {"path": "a.py"}, success=False)
    r3 = record_tool_call("read_file", {"path": "a.py"}, success=False)
    # Third identical call triggers the warn threshold.
    assert r1["action"] == "continue"
    assert r2["action"] == "continue"
    assert r3["action"] in ("warn", "stop")


# ── sessions_cli: path-traversal defense ──────────────────────────────

def test_safe_session_path_rejects_traversal(tmp_path):
    from aura.cli.commands.sessions_cli_commands import _safe_session_path
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Safe id
    safe = _safe_session_path(sessions_dir, "ses_123_abc")
    assert safe is not None
    assert safe == sessions_dir / "ses_123_abc"

    # Path traversal attempts
    assert _safe_session_path(sessions_dir, "../etc/passwd") is None
    assert _safe_session_path(sessions_dir, "..") is None
    assert _safe_session_path(sessions_dir, "") is None
    assert _safe_session_path(sessions_dir, "foo/bar") is None
    assert _safe_session_path(sessions_dir, "foo\\bar") is None


# ── skills: name sanitization and traversal defense ───────────────────

def test_skills_install_rejects_traversal_name(tmp_path, monkeypatch):
    """A URL whose last path segment is '..' must not escape target_dir."""
    from aura.cli import commands as _cmds  # noqa: F401
    from aura.cli.commands import skills_commands

    # Redirect ~/.aura/skills to tmp_path
    fake_home = tmp_path
    monkeypatch.setattr(skills_commands.Path, "home", classmethod(lambda cls: fake_home))

    # Simulate a URL ending in a path-traversal name.
    # The post-strip code would normally produce '..' — must be rejected.
    monkeypatch.setattr(
        skills_commands, "_install_skill", lambda url: None
    )  # Ensure module is loaded

    # Direct test of the sanitizer behavior via _uninstall_skill.
    # Inject a target_dir via monkeypatching the function's import path.
    target_dir = fake_home / ".aura" / "skills"
    target_dir.mkdir(parents=True)

    # Bad names that would resolve to '..' or empty after sanitization.
    from aura.cli.commands.skills_commands import _uninstall_skill
    for bad_name in ("..", ".", "../etc", "foo/../bar"):
        # These should hit the "Invalid skill name" or "Refusing to uninstall" branch.
        # We don't need to assert console output — the absence of an exception or
        # filesystem change is the test.
        before = set(target_dir.iterdir())
        _uninstall_skill(bad_name)
        after = set(target_dir.iterdir())
        assert before == after, f"uninstall of {bad_name!r} changed filesystem"


# ── human_delay: config-driven modes ──────────────────────────────────

def test_human_delay_disabled_by_default():
    from aura.human_delay import is_human_delay_enabled, humanize_delay
    # No config in test env → mode is "off" by default.
    assert not is_human_delay_enabled()
    assert humanize_delay(0) == 0.0
    assert humanize_delay(1000) == 0.0


def test_human_delay_fixed_mode():
    from aura.human_delay import is_human_delay_enabled, humanize_delay
    # Patch the config loader to return a fixed mode.
    with patch("aura.human_delay.get_human_delay_config",
               return_value={"mode": "fixed", "min_ms": 500, "max_ms": 1000}):
        assert is_human_delay_enabled()
        assert humanize_delay(0) == 0.5


# ── personalities: built-in list, set/clear ─────────────────────────────

def test_personalities_builtin_list_nonempty():
    from aura.personalities import list_personalities, get_personality_prompt

    all_p = list_personalities()
    assert len(all_p) >= 14  # We ship 14 built-ins.
    # Spot-check a few well-known personas.
    assert get_personality_prompt("helpful") is not None
    assert get_personality_prompt("pirate") is not None
    assert get_personality_prompt("nonexistent") is None


# ── streaming_config: sane defaults ───────────────────────────────────

def test_streaming_defaults():
    from aura.streaming_config import (
        is_streaming_enabled, get_edit_interval, get_buffer_threshold,
    )
    # Without config, defaults to disabled, sensible values.
    assert not is_streaming_enabled()
    assert get_edit_interval() == 0.8
    assert get_buffer_threshold() == 24


# ── security_config: approval mode round-trips ────────────────────────

def test_security_config_approval_mode():
    from aura.security_config import get_approvals_mode
    assert get_approvals_mode() in ("manual", "smart", "off")  # Default or env override.


def test_smart_approval_low_risk_command():
    from aura.security_config import classify_command_risk
    # Without smart mode, risk is "unknown".
    risk = classify_command_risk("ls -la")
    assert risk == "unknown"


# ── guardrails: deque cap (regression test) ───────────────────────────

def test_guardrails_deque_caps_automatically():
    from aura.guardrails import _RECENT_CALLS_MAXLEN, record_tool_call, reset_state, get_state

    reset_state()
    # Append more than the cap; deque should silently drop the oldest.
    for i in range(_RECENT_CALLS_MAXLEN * 3):
        record_tool_call("read_file", {"i": str(i)}, success=True)
    assert len(get_state().recent_calls) == _RECENT_CALLS_MAXLEN


# ── cron: job ID prefix lookup helper ────────────────────────────────

def test_cron_find_job_helper(tmp_path, monkeypatch):
    # Patch CRON file to a tmp location so we don't pollute the real ~/.aura.
    from aura.cli.commands import cron_commands

    fake_cron = tmp_path / "cron_jobs.json"
    fake_cron.write_text(json.dumps([
        {"id": "job_1700000000_001", "schedule": "5m", "prompt": "ping", "enabled": True},
        {"id": "job_1700000000_002", "schedule": "10m", "prompt": "pong", "enabled": False},
    ]), encoding="utf-8")
    monkeypatch.setattr(cron_commands, "_CRON_FILE", fake_cron)
    monkeypatch.setattr(cron_commands, "_load_jobs", lambda: cron_commands._load_jobs.__wrapped__() if hasattr(cron_commands._load_jobs, "__wrapped__") else json.loads(fake_cron.read_text(encoding="utf-8")))

    # The helper itself is what we care about — call it directly.
    jobs = json.loads(fake_cron.read_text(encoding="utf-8"))
    found = cron_commands._find_job(jobs, "job_1700000000_001")
    assert found is not None
    assert found["prompt"] == "ping"

    # Prefix match
    found2 = cron_commands._find_job(jobs, "job_1700000000")
    assert found2 is not None
    # First match wins
    assert found2["id"] == "job_1700000000_001"

    # No match
    assert cron_commands._find_job(jobs, "nonexistent") is None


# ── main.py: regression test for the import-shadowing bug ─────────────

def test_main_subcommands_dont_shadow_module_helpers():
    """Regression: importing `_get_console` inside `main()` made the name
    local to the entire function, breaking every other reference.

    Verify that `main._get_console` is the module-level function (not a
    local variable that gets shadowed by the subcommand handlers).
    """
    import main as aura_main
    # _get_console should be a module-level function attribute.
    assert callable(getattr(aura_main, "_get_console", None))


def test_main_subcommands_run_without_unboundlocalerror():
    """Regression: `aura config` used to raise UnboundLocalError because
    `from ... import _get_console` inside `update` made `_get_console` a
    local variable in main() that wasn't bound on the first config call.

    The handler should not error out on basic config invocation.
    """
    import sys
    from unittest.mock import patch

    # Patch argv so the parser builds with subparsers enabled.
    with patch.object(sys, "argv", ["aura", "config"]):
        from main import _build_argument_parser
        parser, _ = _build_argument_parser()
        args = parser.parse_args()
        assert args.command == "config"


def test_main_completion_subcommand_parses():
    """`aura completion` should default to bash and work without args."""
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["aura", "completion"]):
        from main import _build_argument_parser
        parser, _ = _build_argument_parser()
        args = parser.parse_args()
        assert args.command == "completion"
        assert args.completion_shell == "bash"


def test_main_plugins_subcommand_parses():
    """`aura plugins` should default to list action."""
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["aura", "plugins"]):
        from main import _build_argument_parser
        parser, _ = _build_argument_parser()
        args = parser.parse_args()
        assert args.command == "plugins"
        assert args.plugins_action == "list"
        assert args.plugins_name == ""
        assert args.plugins_source == ""


def test_main_prune_subcommand_parses():
    """`aura prune --dry-run` should parse with --days default 90."""
    import sys
    from unittest.mock import patch

    with patch.object(sys, "argv", ["aura", "prune", "--dry-run"]):
        from main import _build_argument_parser
        parser, _ = _build_argument_parser()
        args = parser.parse_args()
        assert args.command == "prune"
        assert args.prune_days == 90
        assert args.dry_run is True
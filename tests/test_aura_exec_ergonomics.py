"""Tests for `aura exec` new flags."""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aura.cli.oneshot import _install_wallclock_timeout


def test_install_wallclock_timeout_short_noop(monkeypatch):
    """Short timeout should still install cleanly (no exception)."""
    monkeypatch.setattr("os._exit", lambda code: None)
    _install_wallclock_timeout(0)  # 0 = no-op
    # No raise = success


def test_install_wallclock_timeout_fires(monkeypatch):
    """Timer-based timeout calls os._exit after the delay."""
    called = {}

    def _fake_exit(code):
        called["code"] = code

    monkeypatch.setattr("os._exit", _fake_exit)
    _install_wallclock_timeout(1)  # 1 second
    time.sleep(1.3)
    assert called.get("code") == 124


def test_main_argparse_registers_exec_flags():
    """--timeout, --quiet, --output-failures are registered on the exec subparser."""
    import sys
    from contextlib import redirect_stdout
    import io
    with patch.object(sys, "argv", ["aura", "exec"]):
        from main import _build_argument_parser
        parser, use_subparsers = _build_argument_parser()
        assert use_subparsers
        # Walk subparsers to find the exec subparser and inspect ITS help
        subparsers_action = None
        for action in parser._actions:
            if getattr(action, "choices", None) and "exec" in action.choices:
                subparsers_action = action
                break
        assert subparsers_action is not None
        exec_parser = subparsers_action.choices["exec"]
        help_text = exec_parser.format_help()
        assert "--timeout" in help_text
        assert "--quiet" in help_text
        assert "--output-failures" in help_text

def test_main_exec_default_timeout_is_zero():
    """Default --timeout is 0 (no timeout)."""
    import sys
    with patch.object(sys, "argv", ["aura", "exec", "do something"]):
        from main import _build_argument_parser
        parser, _ = _build_argument_parser()
        args = parser.parse_args()
        assert getattr(args, "exec_timeout", 0) == 0
        assert not getattr(args, "quiet", False)
        assert not getattr(args, "output_failures", False)


def test_main_exec_quiet_flag():
    import sys
    with patch.object(sys, "argv", ["aura", "exec", "do something", "--quiet"]):
        from main import _build_argument_parser
        parser, _ = _build_argument_parser()
        args = parser.parse_args()
        assert args.quiet is True


def test_main_exec_timeout_flag():
    import sys
    with patch.object(sys, "argv", ["aura", "exec", "do X", "--timeout", "30"]):
        from main import _build_argument_parser
        parser, _ = _build_argument_parser()
        args = parser.parse_args()
        assert args.exec_timeout == 30


def test_main_exec_output_failures_flag():
    import sys
    with patch.object(sys, "argv", ["aura", "exec", "do X", "--output-failures"]):
        from main import _build_argument_parser
        parser, _ = _build_argument_parser()
        args = parser.parse_args()
        assert args.output_failures is True

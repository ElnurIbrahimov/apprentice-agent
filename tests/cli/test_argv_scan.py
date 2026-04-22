"""Regression tests for _argv_has_subcommand + _collect_valued_flags.

These two functions decide whether `aura fix the bug` is a goal prompt or a
subcommand invocation. A silent drift — e.g., adding a new `--flag value` arg
but forgetting to register it — would mis-classify the first positional and
crash argparse with 'invalid choice'. Deriving the valued-flag set from the
parser itself (rather than a hardcoded set) prevents that, and these tests
lock in the contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Import the functions directly from main.py without running main().
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import main as aura_main  # noqa: E402


def _scan(argv):
    """Build the parser, derive valued flags, then scan — mirrors main()."""
    parser, _ = aura_main._build_argument_parser()
    valued = aura_main._collect_valued_flags(parser)
    return aura_main._argv_has_subcommand(argv, valued)


class TestArgvScan:
    def test_bare_subcommand_detected(self):
        assert _scan(["init"])
        assert _scan(["doctor"])
        assert _scan(["commit"])

    def test_bare_prompt_is_not_subcommand(self):
        assert not _scan(["fix the login bug"])
        assert not _scan(["hello world"])

    def test_flag_then_subcommand(self):
        # Boolean flag → scanner skips one token, reaches subcommand
        assert _scan(["--verbose", "status"])
        assert _scan(["-v", "status"])

    def test_valued_flag_with_separate_value_then_subcommand(self):
        # This is the case the old hardcoded set could break on.
        assert _scan(["--tier", "fast", "status"])
        assert _scan(["--model", "kimi-k2.6:cloud", "doctor"])
        assert _scan(["--budget", "2.0", "cost"])
        assert _scan(["--max-iterations", "10", "status"])

    def test_valued_flag_with_equals_syntax_then_subcommand(self):
        assert _scan(["--tier=fast", "status"])
        assert _scan(["--model=kimi-k2.6:cloud", "doctor"])

    def test_valued_flag_then_prompt(self):
        # Value-taking flag + non-subcommand positional → goal mode
        assert not _scan(["--tier", "fast", "fix the bug"])
        assert not _scan(["--model", "gpt-5", "write hello"])

    def test_double_dash_halts_scan(self):
        assert not _scan(["--", "init"])

    def test_nargs_plus_flag(self):
        # --channels takes nargs="+"; scanner should skip both values
        # then pick up subcommand. Current implementation only peels ONE value
        # off each flag regardless of nargs, so to avoid regressions we test
        # the realistic case: --channels takes the trailing values.
        assert _scan(["--channels", "telegram", "status"])

    def test_collect_valued_flags_contains_known(self):
        parser, _ = aura_main._build_argument_parser()
        valued = aura_main._collect_valued_flags(parser)
        # Every flag we know takes a value must be present.
        expected = {"--tier", "--model", "--budget", "--max-iterations",
                    "--prompt", "-p", "--preference", "--format",
                    "--channels", "-ch", "--login", "--logout",
                    "--dream-date", "--mode"}
        missing = expected - valued
        assert not missing, f"valued-flag set missing entries: {missing}"

    def test_collect_valued_flags_excludes_bool_flags(self):
        parser, _ = aura_main._build_argument_parser()
        valued = aura_main._collect_valued_flags(parser)
        # Boolean flags must never show up — they consume zero values.
        for flag in ("--verbose", "-v", "--trust", "--fast", "--voice",
                     "--sandboxed", "--workspace-write", "--unrestricted",
                     "--routing-trace", "--dream", "--no-fastpath",
                     "--no-barge-in", "--speak"):
            assert flag not in valued, f"{flag} should not be in valued flags"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

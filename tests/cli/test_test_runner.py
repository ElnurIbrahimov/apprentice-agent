"""Tests for test runner integration."""
import pytest
from aura.cli.test_runner import TestResult, run_tests, _parse_output, TestHistory, render_test_results
from rich.console import Console
from io import StringIO

def test_result_defaults():
    r = TestResult()
    assert r.total == 0
    assert r.success
    assert r.summary

def test_result_with_failures():
    r = TestResult(passed=5, failed=2, skipped=1)
    assert r.total == 8
    assert not r.success
    assert "2 failed" in r.summary
    assert "5 passed" in r.summary

def test_parse_pytest_output():
    r = TestResult()
    _parse_output(r, "===== 10 passed, 2 failed, 1 skipped in 3.2s =====")
    assert r.passed == 10
    assert r.failed == 2
    assert r.skipped == 1

def test_parse_jest_output():
    r = TestResult()
    _parse_output(r, "Tests:  3 failed, 12 passed, 15 total")
    assert r.failed == 3
    assert r.passed == 12

def test_run_tests_command_not_found():
    result = run_tests("nonexistent_test_command_xyz")
    assert result.errors > 0

def test_history():
    h = TestHistory()
    h.add(TestResult(passed=5))
    h.add(TestResult(passed=3, failed=1))
    assert h.total_runs == 2
    assert h.pass_rate == 0.5

def test_history_summary():
    h = TestHistory()
    assert "No test runs" in h.summary()
    h.add(TestResult(passed=1))
    assert "1 runs" in h.summary()

def test_render_results():
    console = Console(file=StringIO(), force_terminal=True, width=80)
    r = TestResult(passed=5, failed=1, duration=2.3, output="5 passed\n1 failed\n")
    render_test_results(console, r)
    output = console.file.getvalue()
    assert "passed" in output
    assert "failed" in output

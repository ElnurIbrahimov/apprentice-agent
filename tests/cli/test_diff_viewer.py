"""Tests for diff viewer."""
import pytest
from aura.cli.diff_viewer import render_diff, generate_diff, diff_summary, render_diff_compact

def test_generate_diff_basic():
    old = "line1\nline2\nline3\n"
    new = "line1\nmodified\nline3\n"
    diff = generate_diff(old, new, filename="test.py")
    assert "line2" in diff
    assert "modified" in diff

def test_generate_diff_empty_old():
    diff = generate_diff("", "new content\n", filename="new.py")
    assert "new content" in diff

def test_generate_diff_no_change():
    diff = generate_diff("same\n", "same\n", filename="test.py")
    assert diff == ""

def test_render_diff_returns_panel():
    old = "def hello():\n    pass\n"
    new = "def hello():\n    return 'world'\n"
    result = render_diff(old, new, filename="test.py")
    assert result is not None
    from rich.panel import Panel
    assert isinstance(result, Panel)

def test_render_diff_no_change_returns_none():
    result = render_diff("same\n", "same\n", filename="test.py")
    assert result is None

def test_diff_summary():
    old = "a\nb\nc\n"
    new = "a\nB\nc\nd\n"
    summary = diff_summary(old, new, filename="test.py")
    assert "test.py" in summary
    assert "+" in summary

def test_diff_summary_no_change():
    summary = diff_summary("same\n", "same\n", filename="test.py")
    assert "no changes" in summary

def test_render_diff_compact():
    old = "a\n"
    new = "b\n"
    compact = render_diff_compact(old, new, filename="test.py")
    assert "edit" in compact
    assert "test.py" in compact

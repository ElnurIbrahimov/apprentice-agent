"""Tests for smart session picker."""
import pytest
from aura.cli.session_picker import _format_session_line, _pick_session_fallback
from rich.console import Console
from io import StringIO
import time


def test_format_session_line_basic():
    session = {
        "title": "Fix login bug",
        "updated_at": time.time() - 300,  # 5 min ago
        "stats": {"message_count": 12},
        "model": "kimi-k2.6:cloud",
    }
    line = _format_session_line(session)
    assert "Fix login bug" in line
    assert "12" in line
    assert "kimi-k2.6:cloud" in line
    assert "m ago" in line


def test_format_session_line_current():
    session = {"title": "Active", "id": "ses_123"}
    line = _format_session_line(session, is_current=True)
    assert "\u2190" in line


def test_format_session_line_old():
    session = {"title": "Old", "updated_at": time.time() - 86400 * 3}
    line = _format_session_line(session)
    assert "d ago" in line


def test_format_session_line_no_time():
    session = {"title": "No Time"}
    line = _format_session_line(session)
    assert "unknown" in line


def test_pick_session_empty():
    from aura.cli.session_picker import pick_session
    console = Console(file=StringIO())
    result = pick_session(console, [], "")
    assert result is None


def test_fallback_cancel(monkeypatch):
    console = Console(file=StringIO())
    sessions = [{"title": "Test", "id": "s1", "updated_at": time.time()}]
    monkeypatch.setattr("builtins.input", lambda _: "0")
    result = _pick_session_fallback(console, sessions, "")
    assert result is None


def test_fallback_select(monkeypatch):
    console = Console(file=StringIO())
    sessions = [
        {"title": "First", "id": "s1", "updated_at": time.time()},
        {"title": "Second", "id": "s2", "updated_at": time.time()},
    ]
    monkeypatch.setattr("builtins.input", lambda _: "2")
    result = _pick_session_fallback(console, sessions, "")
    assert result["id"] == "s2"

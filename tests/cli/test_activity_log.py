"""Tests for activity log."""
import pytest
from aura.cli.activity_log import ActivityLog

@pytest.fixture
def log(tmp_path):
    return ActivityLog(db_path=tmp_path / "test_log.db")

def test_log_and_retrieve(log):
    row_id = log.log(prompt="hello", response="world", model="test")
    assert row_id > 0
    recent = log.get_recent(limit=1)
    assert len(recent) == 1
    assert recent[0]["prompt"] == "hello"

def test_search(log):
    log.log(prompt="fix authentication bug", response="I'll fix the auth")
    log.log(prompt="add new feature", response="Adding feature X")
    results = log.search("authentication")
    assert len(results) >= 1
    assert "authentication" in results[0]["prompt"]

def test_search_empty(log):
    results = log.search("")
    assert results == []

def test_stats(log):
    log.log(prompt="a", tokens_in=100, tokens_out=200, cost=0.01)
    log.log(prompt="b", tokens_in=150, tokens_out=300, cost=0.02)
    stats = log.get_stats()
    assert stats["total_interactions"] == 2
    assert stats["total_tokens_in"] == 250

def test_export_markdown(log):
    log.log(prompt="hello", response="hi", model="test", session_id="ses_123")
    md = log.export_session("ses_123", format="markdown")
    assert "hello" in md
    assert "hi" in md

def test_export_json(log):
    import json
    log.log(prompt="hello", response="hi", session_id="ses_123")
    data = json.loads(log.export_session("ses_123", format="json"))
    assert len(data) == 1

def test_multiple_sessions(log):
    log.log(prompt="a", session_id="s1")
    log.log(prompt="b", session_id="s2")
    md = log.export_session("s1")
    assert "a" in md
    assert "b" not in md

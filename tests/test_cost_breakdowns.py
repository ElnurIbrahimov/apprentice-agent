"""Tests for cost command breakdown queries."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aura.cli.activity_log import ActivityLog


@pytest.fixture
def tmp_log(tmp_path, monkeypatch) -> ActivityLog:
    """Create an ActivityLog pointing at a temp SQLite DB."""
    db_path = tmp_path / "activity.sqlite"
    monkeypatch.setenv("AURA_ACTIVITY_LOG_PATH", str(db_path))
    log = ActivityLog(db_path=db_path)
    return log


def _insert(log: ActivityLog, session: str, model: str, tokens_in: int, tokens_out: int, cost: float):
    conn = sqlite3.connect(str(log._db_path))
    try:
        conn.execute(
            "INSERT INTO interactions (session_id, timestamp, model, tokens_in, tokens_out, cost, tool_calls, prompt, response) "
            "VALUES (?, strftime('%s','now'), ?, ?, ?, ?, 0, '?', '?')",
            (session, model, tokens_in, tokens_out, cost),
        )
        conn.commit()
    finally:
        conn.close()


def test_get_stats_empty(tmp_log):
    stats = tmp_log.get_stats()
    assert stats["total_interactions"] == 0
    assert stats["total_cost"] == 0.0


def test_get_stats_session_filter(tmp_log):
    _insert(tmp_log, "sess_a", "kimi:cloud", 100, 50, 0.01)
    _insert(tmp_log, "sess_b", "qwen:cloud", 200, 100, 0.02)
    _insert(tmp_log, "sess_a", "kimi:cloud", 50, 25, 0.005)

    all_stats = tmp_log.get_stats()
    assert all_stats["total_interactions"] == 3
    assert abs(all_stats["total_cost"] - 0.035) < 1e-6

    a_stats = tmp_log.get_stats(session_id="sess_a")
    assert a_stats["total_interactions"] == 2
    assert abs(a_stats["total_cost"] - 0.015) < 1e-6


def test_get_stats_by_model(tmp_log):
    _insert(tmp_log, "s1", "kimi:cloud", 100, 50, 0.01)
    _insert(tmp_log, "s1", "kimi:cloud", 200, 100, 0.02)
    _insert(tmp_log, "s1", "qwen:cloud", 300, 150, 0.03)

    rows = tmp_log.get_stats_by_model()
    assert len(rows) == 2
    # Ordered by cost desc
    assert rows[0]["model"] == "qwen:cloud"
    assert rows[0]["interactions"] == 1
    assert rows[1]["model"] == "kimi:cloud"
    assert rows[1]["interactions"] == 2
    assert rows[1]["tokens_in"] == 300


def test_get_stats_by_provider_groups_by_prefix(tmp_log):
    _insert(tmp_log, "s1", "kimi-k2.5:cloud", 100, 50, 0.01)
    _insert(tmp_log, "s1", "kimi-k2.5:cloud", 200, 100, 0.02)
    _insert(tmp_log, "s1", "qwen3-coder:480b-cloud", 300, 150, 0.03)

    rows = tmp_log.get_stats_by_provider()
    providers = {r["provider"] for r in rows}
    assert "kimi-k2.5" in providers or "kimi" in providers
    # qwen3-coder should be its own provider bucket
    assert "qwen3-coder" in providers


def test_by_model_respects_session(tmp_log):
    _insert(tmp_log, "s1", "kimi:cloud", 100, 50, 0.01)
    _insert(tmp_log, "s2", "kimi:cloud", 999, 999, 0.99)
    rows = tmp_log.get_stats_by_model(session_id="s1")
    assert len(rows) == 1
    assert rows[0]["interactions"] == 1
    assert abs(rows[0]["cost"] - 0.01) < 1e-6

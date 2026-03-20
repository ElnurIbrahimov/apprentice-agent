"""Smoke tests for SOTA predictive_tasks upgrades."""
import pytest
import tempfile
from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch, tmp_path):
    """Point DB_PATH to a temp directory so tests never touch real data."""
    import aura.tools.predictive_tasks as mod
    monkeypatch.setattr(mod, "DB_PATH", tmp_path / "task_events.db")


def _tool():
    from aura.tools.predictive_tasks import PredictiveTaskTool
    return PredictiveTaskTool()


# 1. Instantiation
def test_instantiation():
    t = _tool()
    assert t.name == "predictive_tasks"
    assert t._engine is not None


# 2. execute("predict") returns dict with "success"
def test_predict():
    result = _tool().execute("predict")
    assert isinstance(result, dict)
    assert result["success"] is True


# 3. execute("log") inserts successfully
def test_log():
    result = _tool().execute("log", tool="test", action_name="run")
    assert result["success"] is True
    assert "logged" in result


# 4. execute("feedback") works
def test_feedback():
    result = _tool().execute(
        "feedback",
        prediction_id="test123",
        tool="test",
        action_name="run",
        accepted=True,
    )
    assert result["success"] is True
    assert result["accepted"] is True


# 5. execute("sequence") returns valid dict
def test_sequence():
    result = _tool().execute("sequence")
    assert isinstance(result, dict)
    assert result["success"] is True


# 6. execute("combined") returns valid dict
def test_combined():
    result = _tool().execute("combined")
    assert isinstance(result, dict)
    assert result["success"] is True
    assert "sources" in result


# 7. execute("context") returns valid dict
def test_context():
    result = _tool().execute("context", context_hint="test_project")
    assert isinstance(result, dict)
    assert result["success"] is True


# 8. DB migration is idempotent
def test_db_idempotent():
    from aura.tools.predictive_tasks import _get_db
    conn1 = _get_db()
    conn1.close()
    conn2 = _get_db()  # second call must not raise
    conn2.close()


# 9. execute("stats") and execute("heatmap") still work
def test_stats():
    result = _tool().execute("stats")
    assert result["success"] is True
    assert "total_events" in result


def test_heatmap():
    result = _tool().execute("heatmap")
    assert result["success"] is True
    assert "heatmap" in result


# 10. execute("patterns") still works
def test_patterns():
    result = _tool().execute("patterns")
    assert result["success"] is True
    assert "patterns" in result


# Bonus: round-trip log -> stats confirms data is stored
def test_log_then_stats():
    t = _tool()
    t.execute("log", tool="git", action_name="commit")
    t.execute("log", tool="git", action_name="commit")
    stats = t.execute("stats")
    assert stats["success"] is True
    assert stats["total_events"] >= 2

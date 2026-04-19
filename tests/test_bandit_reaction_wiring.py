"""Tests for the message_id ↔ request_id mapping and reaction→bandit wiring."""

import time

import pytest


@pytest.fixture
def bandit(tmp_path, monkeypatch):
    """Isolated StrategyBandit instance using a temp sqlite DB."""
    monkeypatch.setenv("AURA_DATA_DIR", str(tmp_path))
    from aura.consciousness.strategy_bandit import StrategyBandit
    return StrategyBandit(enabled=True, db_path=str(tmp_path / "aura_meta.db"))


def test_map_and_get_roundtrip(bandit):
    bandit.map_message_to_request(12345, "abc123def456")
    assert bandit.get_request_id_for_message(12345) == "abc123def456"


def test_get_returns_none_for_unknown_message(bandit):
    assert bandit.get_request_id_for_message(999999) is None


def test_map_handles_zero_and_empty_gracefully(bandit):
    bandit.map_message_to_request(0, "abc")
    bandit.map_message_to_request(123, "")
    # Neither should have been stored
    assert bandit.get_request_id_for_message(0) is None
    assert bandit.get_request_id_for_message(123) is None


def test_map_different_surfaces_separately(bandit):
    bandit.map_message_to_request(42, "telegram_rid", surface="telegram")
    bandit.map_message_to_request(42, "web_rid", surface="web")
    assert bandit.get_request_id_for_message(42, "telegram") == "telegram_rid"
    assert bandit.get_request_id_for_message(42, "web") == "web_rid"


def test_map_upsert_overwrites_old_mapping(bandit):
    bandit.map_message_to_request(1, "old_rid")
    bandit.map_message_to_request(1, "new_rid")
    assert bandit.get_request_id_for_message(1) == "new_rid"


def test_context_var_is_set_by_select_strategy(bandit):
    from aura.consciousness.strategy_bandit import get_last_bandit_request_id

    selection = bandit.select_strategy("calculate 2 + 2")
    assert get_last_bandit_request_id() == selection.request_id


def test_context_var_none_before_any_selection():
    """In a fresh context, get_last_bandit_request_id returns None."""
    import contextvars
    from aura.consciousness.strategy_bandit import get_last_bandit_request_id

    ctx = contextvars.Context()
    assert ctx.run(get_last_bandit_request_id) is None


def test_record_user_feedback_works_on_mapped_outcome(bandit):
    """End-to-end: select strategy, fake outcome, map message, record reaction feedback."""
    selection = bandit.select_strategy("calculate something")
    # Fake a minimal outcome so record_user_feedback has something to update
    bandit.record_outcome(
        request_id=selection.request_id,
        strategy=selection.strategy,
        category=selection.category,
        latency_ms=500.0,
        response_length=100,
        metrics={"judge_score": 0.5},
    )
    # Telegram sends a message with id=777 that carried this request
    bandit.map_message_to_request(777, selection.request_id)
    found = bandit.get_request_id_for_message(777)
    assert found == selection.request_id

    # A positive reaction arrives later — record_user_feedback must not raise
    bandit.record_user_feedback(found, 1.0)

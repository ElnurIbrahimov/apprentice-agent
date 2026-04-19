"""Tests for the dedicated world-model layer in SystemPromptBuilder."""

import time
from unittest.mock import patch


def _make_builder():
    from aura.prompt_builder import SystemPromptBuilder
    return SystemPromptBuilder()


def test_world_model_empty_context_is_pass_through():
    b = _make_builder()
    with patch("aura.consciousness.world_model.get_world_model") as mock_get:
        mock_get.return_value.get_context_summary.return_value = ""
        result = b._inject_world_model("IDENTITY")
    assert result == "IDENTITY"


def test_world_model_context_is_appended():
    b = _make_builder()
    with patch("aura.consciousness.world_model.get_world_model") as mock_get:
        mock_get.return_value.get_context_summary.return_value = "[World State]\nActive projects: Foo (2d stale)"
        result = b._inject_world_model("IDENTITY")
    assert "[World State]" in result
    assert "Active projects: Foo" in result
    assert result.startswith("IDENTITY")


def test_world_model_cap_applied_to_huge_output():
    b = _make_builder()
    huge = "X\n" * 5000  # 10_000 chars
    with patch("aura.consciousness.world_model.get_world_model") as mock_get:
        mock_get.return_value.get_context_summary.return_value = huge
        result = b._inject_world_model("IDENTITY")
    # Appended portion must respect the 2000-char cap
    appended = result[len("IDENTITY") + 2:]  # skip the "\n\n"
    assert len(appended) <= 2000


def test_world_model_cache_hit_does_not_recall_engine():
    b = _make_builder()
    with patch("aura.consciousness.world_model.get_world_model") as mock_get:
        mock_get.return_value.get_context_summary.return_value = "[World State]\nActive projects: Foo"
        b._inject_world_model("IDENTITY")  # miss, populates cache
        b._inject_world_model("IDENTITY")  # hit, should reuse cache
    # get_world_model() called exactly once — cache short-circuits the second call
    assert mock_get.call_count == 1


def test_world_model_cache_invalidated_after_ttl():
    b = _make_builder()
    with patch("aura.consciousness.world_model.get_world_model") as mock_get:
        mock_get.return_value.get_context_summary.return_value = "[World State]\ninitial"
        b._inject_world_model("IDENTITY")
        # Force cache expiry
        b._world_model_ts = time.time() - 999
        mock_get.return_value.get_context_summary.return_value = "[World State]\nrefreshed"
        result = b._inject_world_model("IDENTITY")
    assert "refreshed" in result
    assert mock_get.call_count == 2


def test_world_model_injection_survives_engine_exception():
    b = _make_builder()
    with patch("aura.consciousness.world_model.get_world_model") as mock_get:
        mock_get.side_effect = RuntimeError("engine down")
        result = b._inject_world_model("IDENTITY")
    # Injection must not raise; pass through unchanged.
    assert result == "IDENTITY"


def test_world_model_requested_with_800_tokens():
    b = _make_builder()
    with patch("aura.consciousness.world_model.get_world_model") as mock_get:
        mock_get.return_value.get_context_summary.return_value = "[World State]"
        b._inject_world_model("IDENTITY")
    mock_get.return_value.get_context_summary.assert_called_with(max_tokens=800)

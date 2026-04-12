from __future__ import annotations

from aura.core.agentic_loop_outcomes import LoopOutcome, ToolBatchResult


def test_loop_outcome_to_result_dict_exposes_status_and_success():
    outcome = LoopOutcome.model_timeout("qwen")

    result = outcome.to_result_dict(iterations=3, tool_calls=2, model="qwen")

    assert result == {
        "success": False,
        "status": "model_timeout",
        "response": (
            "Request timed out for qwen.\n"
            "  - The model may be overloaded or too large.\n"
            "  - Try a smaller model with: /model <name>"
        ),
        "iterations": 3,
        "tool_calls": 2,
        "model": "qwen",
    }


def test_completed_outcome_is_successful():
    outcome = LoopOutcome.completed("done")

    assert outcome.error is False
    assert outcome.status == "completed"
    assert outcome.response == "done"


def test_tool_batch_result_defaults_to_continue():
    result = ToolBatchResult()

    assert result.should_break is False
    assert result.outcome is None

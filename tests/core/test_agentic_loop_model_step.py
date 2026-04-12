from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from aura.core.agentic_loop_model_step import ModelStepController


def _make_loop() -> SimpleNamespace:
    brain = MagicMock()
    return SimpleNamespace(
        brain=brain,
        _loop_error=False,
        _empty_response_count=0,
        _thinking_nudge_count=0,
        _verify_completion=True,
        _verification_done=False,
        iteration=1,
        tool_calls_total=0,
        _verify_task_completion=MagicMock(return_value=None),
    )


def test_request_step_returns_tool_calls_ready_from_stream():
    loop = _make_loop()
    loop.brain.think_with_tools_stream.return_value = iter(
        [
            ("content", "Searching"),
            ("tool_calls", [{"function": {"name": "shell", "arguments": {}}}]),
            ("done", {"model": "test-model", "content": "Searching"}),
        ]
    )
    controller = ModelStepController(loop)

    result = controller.request_step(
        messages=[{"role": "user", "content": "hi"}],
        active_tools=[],
        step_model="test-model",
        on_chunk=lambda _: None,
    )

    assert result.status == "tool_calls_ready"
    assert result.tool_calls == [{"function": {"name": "shell", "arguments": {}}}]
    assert result.content == "Searching"
    assert result.model_used == "test-model"
    assert result.delivery == "stream"


def test_request_step_falls_back_to_blocking_call():
    loop = _make_loop()
    loop.brain.think_with_tools_stream.side_effect = RuntimeError("stream broke")
    loop.brain.think_with_tools.return_value = {
        "message": {"content": "Done", "tool_calls": None},
        "model": "fallback-model",
    }
    controller = ModelStepController(loop)

    result = controller.request_step(
        messages=[{"role": "user", "content": "hi"}],
        active_tools=[],
        step_model="test-model",
        on_chunk=lambda _: None,
    )

    assert result.status == "content_ready"
    assert result.content == "Done"
    assert result.model_used == "fallback-model"
    assert result.delivery == "blocking"


def test_resolve_content_only_returns_retry_for_empty_response():
    loop = _make_loop()
    controller = ModelStepController(loop)

    result = controller.resolve_content_only(prompt="hi", content="", delivery="stream")

    assert result.status == "retry_empty_response"
    assert result.extra_messages == [
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Continue. Execute the task using tools."},
    ]


def test_resolve_content_only_returns_retry_for_verification_failure():
    loop = _make_loop()
    loop.iteration = 2
    loop.tool_calls_total = 1
    loop._verify_task_completion.return_value = "tests still failing"
    controller = ModelStepController(loop)

    result = controller.resolve_content_only(
        prompt="fix it",
        content="All done.",
        delivery="stream",
    )

    assert result.status == "retry_verification"
    assert result.extra_messages is not None
    assert result.extra_messages[0] == {"role": "assistant", "content": "All done."}
    assert "tests still failing" in result.extra_messages[1]["content"]


def test_resolve_content_only_returns_completed_for_blocking_delivery():
    loop = _make_loop()
    controller = ModelStepController(loop)

    result = controller.resolve_content_only(
        prompt="fix it",
        content="Finished.",
        delivery="blocking",
    )

    assert result.status == "terminal"
    assert result.outcome is not None
    assert result.outcome.status == "completed"
    assert result.outcome.response == "Finished."

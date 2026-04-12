from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from aura.core.agentic_loop_events import LoopEventEmitter


def test_loop_event_emitter_dispatches_structured_and_legacy_callbacks():
    loop = SimpleNamespace(iteration=3, _current_run_id="run_demo")
    on_event = MagicMock()
    on_chunk = MagicMock()
    on_response = MagicMock()
    on_tool_start = MagicMock()
    on_tool_call = MagicMock()
    emitter = LoopEventEmitter(
        loop,
        on_event=on_event,
        on_chunk=on_chunk,
        on_response=on_response,
        on_tool_start=on_tool_start,
        on_tool_call=on_tool_call,
    )

    chunk_event = emitter.emit("chunk", text="he")
    response_event = emitter.emit("response", text="done", delivery="blocking")
    start_event = emitter.emit("tool_start", tool_name="shell", tool_args={"command": "pwd"})
    result_event = emitter.emit(
        "tool_result",
        tool_name="shell",
        tool_args={"command": "pwd"},
        tool_result={"ok": True},
    )

    assert chunk_event.type == "chunk"
    assert chunk_event.run_id == "run_demo"
    assert response_event.payload["delivery"] == "blocking"
    assert start_event.iteration == 3
    assert result_event.payload["tool_result"] == {"ok": True}
    assert on_event.call_count == 4
    on_chunk.assert_called_once_with("he")
    on_response.assert_called_once_with("done", 3)
    on_tool_start.assert_called_once_with("shell", {"command": "pwd"})
    on_tool_call.assert_called_once_with("shell", {"command": "pwd"}, {"ok": True})


def test_loop_event_emitter_reports_listener_availability():
    loop = SimpleNamespace(iteration=1, _current_run_id="run_demo")
    emitter = LoopEventEmitter(loop, on_tool_call=lambda *_: None)

    assert emitter.listens_for("tool_result") is True
    assert emitter.listens_for("tool_start") is False

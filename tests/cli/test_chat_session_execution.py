from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aura.cli.chat_session_execution import SessionExecutionController
from aura.core.agentic_loop_events import LoopEvent


def _make_session() -> SimpleNamespace:
    brain = MagicMock()
    brain._model_override = "qwen"
    brain.get_session_stats.return_value = {"cost_usd": 1.25}
    agent = SimpleNamespace(
        brain=brain,
        mood={"mood": "focused"},
        memory=SimpleNamespace(memories=["a", "b"]),
        _speak=MagicMock(),
    )
    agentic = SimpleNamespace(
        _conversation_history=[{"role": "user", "content": "hi"}],
        _hot_files=["src/app.py"],
        iteration=2,
        max_iterations=5,
        model_override="qwen",
        _run_auto_test=MagicMock(return_value="pytest failed"),
    )
    return SimpleNamespace(
        agent=agent,
        agentic=agentic,
        agentic_session=SimpleNamespace(session_id="sess-1"),
        console=MagicMock(),
        steering=MagicMock(),
        hook_mgr=MagicMock(),
        _HookEvent=SimpleNamespace(
            PRE_TOOL_CALL="pre_tool_call",
            POST_TOOL_CALL="post_tool_call",
            POST_EDIT="post_edit",
            POST_RESPONSE="post_response",
        ),
        current_model="qwen",
        _streamer_displayed=False,
        activity_log=MagicMock(),
        _cm_conv_id=None,
        _pending_follow_up=None,
        _follow_up_depth=0,
        _MAX_FOLLOW_UP_DEPTH=3,
        msg_count=0,
        session_title="",
        token_used=0,
        token_limit=0,
        perm_mode="careful",
        _project_type="python",
        _show_bar=MagicMock(),
        speak=True,
        _auto_test_enabled=True,
    )


def test_process_normal_result_updates_follow_up_and_post_response_state():
    session = _make_session()
    session.steering.pop_follow_up.return_value = "do the next step"
    controller = SessionExecutionController(session)

    with (
        patch("aura.cli.context_bar.estimate_messages_tokens", return_value=321),
        patch("aura.cli.context_bar.get_context_limit", return_value=123456),
        patch("aura.cli.display.show_context_summary") as show_context_summary,
        patch("aura.cli.display.show_response") as show_response,
    ):
        handled = controller.process_normal_result(
            "ship it",
            {"response": "done", "model": "qwen", "tool_calls": 2},
        )

    assert handled is True
    show_context_summary.assert_called_once_with(
        memory_count=2,
        mood="focused",
        model="qwen",
        tool_count=2,
    )
    show_response.assert_called_once_with("done", model="qwen", stream=False)
    session.activity_log.log.assert_called_once()
    assert session._pending_follow_up == "do the next step"
    assert session._follow_up_depth == 1
    assert session.msg_count == 1
    assert session.session_title == "ship it"
    assert session.token_used == 321
    assert session.token_limit == 123456
    session._show_bar.assert_called_once()
    session.hook_mgr.fire.assert_any_call(
        "post_response",
        {"response": "done", "model": "qwen"},
    )
    session.agent._speak.assert_called_once_with("done")


def test_process_normal_result_rejects_error_results():
    session = _make_session()
    controller = SessionExecutionController(session)

    with patch("aura.cli.display.show_error") as show_error:
        handled = controller.process_normal_result(
            "ship it",
            {"response": "[LLM Error] failed", "model": "qwen"},
        )

    assert handled is False
    show_error.assert_called_once_with("[LLM Error] failed")
    session._show_bar.assert_not_called()


def test_process_normal_result_drops_follow_up_after_depth_limit():
    session = _make_session()
    session._follow_up_depth = 3
    session.steering.pop_follow_up.return_value = "do the next step"
    controller = SessionExecutionController(session)

    with (
        patch("aura.cli.context_bar.estimate_messages_tokens", return_value=321),
        patch("aura.cli.context_bar.get_context_limit", return_value=123456),
        patch("aura.cli.display.show_context_summary"),
        patch("aura.cli.display.show_response"),
        patch("aura.cli.display.show_info") as show_info,
    ):
        handled = controller.process_normal_result(
            "ship it",
            {"response": "done", "model": "qwen", "tool_calls": 0},
        )

    assert handled is True
    assert session._pending_follow_up is None
    assert session._follow_up_depth == 3
    show_info.assert_called_once_with("Max auto-follow-up depth reached, dropping follow-up.")


def test_run_agent_handles_tool_callbacks_and_auto_test_feedback():
    session = _make_session()
    controller = SessionExecutionController(session)
    streamer = MagicMock()

    def _run(prompt, *, on_event, steering_queue):
        assert prompt == "fix auth"
        assert steering_queue is session.steering
        on_event(LoopEvent(type="chunk", run_id="run_demo", iteration=1, payload={"text": "hello"}))
        on_event(
            LoopEvent(
                type="tool_start",
                run_id="run_demo",
                iteration=1,
                payload={"tool_name": "edit_file", "tool_args": {"path": "src/app.py"}},
            )
        )
        on_event(
            LoopEvent(
                type="tool_result",
                run_id="run_demo",
                iteration=1,
                payload={
                    "tool_name": "edit_file",
                    "tool_args": {"path": "src/app.py"},
                    "tool_result": {"success": True},
                },
            )
        )
        return {"response": "done", "tokens": 42}

    session.agentic.run = _run

    with (
        patch("aura.cli.display.StreamingResponse", return_value=streamer),
        patch("aura.cli.display.show_tool_call") as show_tool_call,
        patch("aura.cli.display.show_tool_result_inline") as show_tool_result_inline,
        patch("aura.cli.display.show_response_attribution") as show_response_attribution,
    ):
        result = controller.run_agent("fix auth")

    assert result == {"response": "done", "tokens": 42}
    streamer.start.assert_called_once()
    streamer.chunk.assert_called_once_with("hello")
    streamer.pause.assert_called()
    streamer.resume.assert_called_once()
    streamer.finish.assert_called_once()
    show_tool_call.assert_called_once()
    show_tool_result_inline.assert_called_once_with("edit_file", {"success": True})
    show_response_attribution.assert_called_once()
    session.hook_mgr.fire.assert_any_call(
        "pre_tool_call",
        {"tool_name": "edit_file", "tool_args": "{'path': 'src/app.py'}"},
    )
    session.hook_mgr.fire.assert_any_call(
        "post_tool_call",
        {"tool_name": "edit_file", "tool_args": "{'path': 'src/app.py'}"},
    )
    session.hook_mgr.fire.assert_any_call(
        "post_edit",
        {"tool_name": "edit_file", "file_path": "src/app.py"},
    )
    assert session.agentic._conversation_history[-1]["content"] == (
        "[Auto-test failed after editing] pytest failed"
    )
    assert session._streamer_displayed is True

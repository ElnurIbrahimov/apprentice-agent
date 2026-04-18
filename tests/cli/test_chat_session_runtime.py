from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aura.cli.chat_session_runtime import SessionRuntimeController


def _make_session() -> SimpleNamespace:
    bridge = MagicMock()
    bridge.has_pending.return_value = False
    hook_mgr = MagicMock()
    return SimpleNamespace(
        bridge=bridge,
        _channel_lock=threading.Lock(),
        agentic=SimpleNamespace(run=MagicMock()),
        console=MagicMock(),
        agent=SimpleNamespace(brain=MagicMock()),
        bg_manager=MagicMock(),
        _session_initialized=False,
        agentic_session=MagicMock(),
        _pending_follow_up=None,
        _follow_up_depth=0,
        _pt_session=object(),
        _injected_input=None,
        _handle_signal=MagicMock(),
        _dispatch_command=MagicMock(),
        _run_plan_mode=MagicMock(),
        _run_agent=MagicMock(),
        _process_normal_result=MagicMock(return_value=True),
        _show_bar=MagicMock(),
        perm_mode="careful",
        current_model="qwen",
        _project_root="D:/Aura",
        _project_type="python",
        msg_count=0,
        session_title="",
        token_used=0,
        token_limit=0,
        _cm_conv_id=None,
        _last_ipc_heartbeat=0.0,
        last_user_input="",
        hook_mgr=hook_mgr,
        _HookEvent=SimpleNamespace(SESSION_END="session_end"),
    )


def test_submit_background_starts_task_and_reports_id():
    session = _make_session()
    session.bg_manager.submit.return_value = SimpleNamespace(id="bg-1")
    # Background tasks now go through AgenticLoop.clone_for_background() so
    # they pick up a restricted PermissionManager instead of calling
    # brain.think() directly. The stub loop accepts any permissions and
    # returns a canned result.
    expected = {"success": True, "response": "done", "iterations": 1, "tool_calls": 0}
    bg_loop = SimpleNamespace(run=MagicMock(return_value=expected))
    session.agentic.clone_for_background = MagicMock(return_value=bg_loop)
    controller = SessionRuntimeController(session)

    controller.submit_background("& fix it")

    session.bg_manager.submit.assert_called_once()
    submitted_prompt, submitted_fn = session.bg_manager.submit.call_args.args
    assert submitted_prompt == "fix it"
    assert submitted_fn("fix it") == expected
    session.agentic.clone_for_background.assert_called_once()
    (bg_perms,), _ = session.agentic.clone_for_background.call_args
    assert bg_perms.current_mode == "careful"
    bg_loop.run.assert_called_once_with("fix it")
    session.console.print.assert_called_with("[cyan]Background task started: bg-1[/cyan]")


def test_drain_channels_processes_one_message_and_sends_response():
    session = _make_session()
    session.bridge.has_pending.return_value = True
    ch_msg = SimpleNamespace(text="hello", channel="discord")
    session.bridge.get_pending_message.return_value = ch_msg
    session.agentic.run.return_value = {"response": "world"}
    controller = SessionRuntimeController(session)

    class _ImmediateThread:
        def __init__(self, target, daemon, name):
            self._target = target

        def start(self):
            self._target()

    with (
        patch("threading.Thread", _ImmediateThread),
        patch("aura.cli.chat_loop._display_channel_response") as display_response,
    ):
        controller.drain_channels()

    display_response.assert_called_once_with(session.console, ch_msg, "world")
    session.bridge.send_response.assert_called_once_with(ch_msg, "world")
    assert not session._channel_lock.locked()


def test_run_exits_cleanly_when_input_returns_none():
    session = _make_session()
    controller = SessionRuntimeController(session)

    with patch("aura.cli.input.get_input", return_value=None):
        controller.run()

    session.bridge.stop.assert_called_once_with()
    session.hook_mgr.fire.assert_called_once_with(
        "session_end",
        {"reason": "user_exit"},
    )
    session.console.print.assert_called_with("\n[dim]Goodbye.[/dim]\n")


def test_run_routes_background_prefix_to_submit_background():
    session = _make_session()
    controller = SessionRuntimeController(session)

    with (
        patch("aura.cli.input.get_input", side_effect=["& fix auth", None]),
        patch.object(controller, "submit_background") as submit_background,
    ):
        controller.run()

    submit_background.assert_called_once_with("& fix auth")

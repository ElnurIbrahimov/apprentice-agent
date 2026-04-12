from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aura.core.agentic_loop_outcomes import ToolBatchResult
from aura.core.agentic_loop_tool_calls import ToolCallCoordinator


def _make_loop() -> SimpleNamespace:
    executor = SimpleNamespace(
        _TOOL_ALIASES={"Glob": "glob"},
        execute=MagicMock(),
    )
    planner = SimpleNamespace(current_plan=True, advance_step=MagicMock())
    return SimpleNamespace(
        session=MagicMock(),
        permissions=MagicMock(),
        executor=executor,
        tool_calls_total=0,
        _show_tool_status=MagicMock(),
        _edits_this_turn=0,
        _has_edits=False,
        _last_tools_were_reads=True,
        _track_hot_file=MagicMock(),
        _planner=planner,
        _tool_result_has_error=lambda result: '"error"' in result,
        _cancel_event=SimpleNamespace(is_set=lambda: False),
        _loop_error=False,
        iteration=2,
    )


def test_parse_tool_calls_skips_malformed_arguments_and_records_tool_error():
    loop = _make_loop()
    coordinator = ToolCallCoordinator(loop)
    messages: list[dict] = []

    parsed = coordinator.parse_tool_calls(
        [
            {"function": {"name": "shell", "arguments": '{"command": "pwd"}'}},
            {"function": {"name": "shell", "arguments": "{bad json"}},
        ],
        messages,
    )

    assert parsed == [("shell", {"command": "pwd"})]
    assert len(messages) == 1
    assert "Malformed arguments for shell" in messages[0]["content"]
    loop.session.append.assert_called_once_with(messages[0])


def test_approve_and_execute_applies_permissions_and_keeps_order():
    loop = _make_loop()
    loop.permissions.check.side_effect = [False, True]
    loop.executor.execute.return_value = json.dumps({"ok": True})
    coordinator = ToolCallCoordinator(loop)
    on_tool_start = MagicMock()

    approved = coordinator.approve_and_execute(
        [("Glob", {"pattern": "*.py"}), ("shell", {"command": "pwd"})],
        on_tool_start=on_tool_start,
    )

    assert approved == [
        ("Glob", {"pattern": "*.py"}, json.dumps({"error": "Permission denied by user"})),
        ("shell", {"command": "pwd"}, json.dumps({"ok": True})),
    ]
    assert loop.tool_calls_total == 2
    loop.permissions.check.assert_any_call("glob", {"pattern": "*.py"})
    loop.permissions.check.assert_any_call("shell", {"command": "pwd"})
    loop._show_tool_status.assert_any_call("Glob", {"pattern": "*.py"}, denied=True)
    loop._show_tool_status.assert_any_call("shell", {"command": "pwd"})
    on_tool_start.assert_called_once_with("shell", {"command": "pwd"})


def test_collect_results_updates_loop_state_and_stops_on_guard():
    loop = _make_loop()
    coordinator = ToolCallCoordinator(loop)
    messages: list[dict] = []
    on_tool_call = MagicMock()
    guard = MagicMock()
    guard.record.side_effect = [
        None,
        SimpleNamespace(triggered=True, fallback_message="stopped by guard"),
    ]

    result = coordinator.collect_results(
        [
            ("edit_file", {"path": "a.py"}, json.dumps({"ok": True})),
            ("shell", {"command": "pytest"}, json.dumps({"ok": True})),
        ],
        messages,
        guard,
        on_tool_call=on_tool_call,
    )

    assert isinstance(result, ToolBatchResult)
    assert result.should_break is True
    assert result.outcome is not None
    assert result.outcome.status == "guard_tripped"
    assert result.outcome.response == "stopped by guard"
    assert loop._loop_error is True
    assert loop._edits_this_turn == 1
    assert loop._has_edits is True
    assert loop._last_tools_were_reads is False
    assert len(messages) == 2
    on_tool_call.assert_any_call("edit_file", {"path": "a.py"}, json.dumps({"ok": True}))
    on_tool_call.assert_any_call("shell", {"command": "pytest"}, json.dumps({"ok": True}))
    loop._track_hot_file.assert_any_call("edit_file", {"path": "a.py"}, json.dumps({"ok": True}))
    loop._track_hot_file.assert_any_call("shell", {"command": "pytest"}, json.dumps({"ok": True}))
    loop._planner.advance_step.assert_any_call(result=json.dumps({"ok": True})[:100])


def test_approve_and_execute_parallel_path_preserves_result_slots():
    loop = _make_loop()
    loop.permissions.check.return_value = True
    coordinator = ToolCallCoordinator(loop)

    class _DoneFuture:
        def __init__(self, value):
            self._value = value

        def result(self, timeout=None):
            return self._value

    class _Pool:
        def __init__(self):
            self.calls = []

        def submit(self, fn, tool_name, args):
            self.calls.append((tool_name, args))
            return _DoneFuture(json.dumps({"tool": tool_name}))

    pool = _Pool()

    with patch("aura.core.agentic_loop_tool_calls._get_tool_pool", return_value=pool):
        approved = coordinator.approve_and_execute(
            [("read_file", {"path": "a.py"}), ("shell", {"command": "pwd"})],
        )

    assert approved == [
        ("read_file", {"path": "a.py"}, json.dumps({"tool": "read_file"})),
        ("shell", {"command": "pwd"}, json.dumps({"tool": "shell"})),
    ]
    assert pool.calls == [
        ("read_file", {"path": "a.py"}),
        ("shell", {"command": "pwd"}),
    ]

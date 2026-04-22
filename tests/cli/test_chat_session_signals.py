from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aura.cli.chat_session_signals import SessionSignalController


def _make_session() -> SimpleNamespace:
    brain = MagicMock()
    brain._model_override = "qwen"
    agent = SimpleNamespace(brain=brain)
    agentic_session = MagicMock()
    agentic = SimpleNamespace(
        session=agentic_session,
        _conversation_history=[{"role": "user", "content": "hi"}],
        model_override="qwen",
    )
    session = SimpleNamespace(
        console=MagicMock(),
        agent=agent,
        agentic=agentic,
        checkpoint_mgr=MagicMock(),
        msg_count=3,
        token_used=456,
        token_limit=999,
        current_model="qwen",
        session_title="hello",
        perm_mode="careful",
        permissions=MagicMock(),
        _project_type="python",
        _show_bar=MagicMock(),
        _show_perm_banner=MagicMock(),
    )

    # Mirror the real ChatSession.apply_model_override helper so the test
    # exercises the single-source-of-truth path introduced in 3.6.
    def _apply(model):
        resolved = None if (model is None or model == "auto") else model
        brain.set_model_override(resolved)
        agentic.model_override = resolved
        session.current_model = resolved or "auto"

    session.apply_model_override = _apply
    return session


def test_signal_command_palette_injects_selected_command():
    session = _make_session()
    controller = SessionSignalController(session)

    with (
        patch("aura.cli.command_palette.build_palette", return_value=["/model auto"]),
        patch("aura.cli.command_palette.open_palette", return_value="/model auto"),
        patch("aura.cli.command_palette.record_usage") as record_usage,
    ):
        result = controller.handle("__CMD_PALETTE__")

    assert result is not None
    assert result.should_continue_loop is False
    assert result.injected_input == "/model auto"
    record_usage.assert_called_once_with("/model auto")


def test_signal_cycle_perms_updates_mode_and_refreshes_ui():
    session = _make_session()
    controller = SessionSignalController(session)

    with (
        patch("aura.cli.permissions_ui.cycle_permission_mode", return_value="auto_edit"),
        patch(
            "aura.cli.permissions_ui.get_mode_description",
            return_value="Auto-edit mode",
        ),
    ):
        result = controller.handle("__CYCLE_PERMS__")

    assert result is not None
    assert result.should_continue_loop is True
    assert session.perm_mode == "auto_edit"
    session.permissions.set_mode.assert_called_once_with("auto_edit")
    session._show_perm_banner.assert_called_once_with("auto_edit")
    session._show_bar.assert_called_once()


def test_signal_model_picker_updates_model_and_status_bar():
    session = _make_session()
    controller = SessionSignalController(session)

    with (
        patch("aura.cli.model_picker.pick_model", return_value="auto"),
        patch("aura.cli.context_bar.get_context_limit", return_value=123456),
        patch("aura.cli.display.show_info") as show_info,
    ):
        result = controller.handle("__MODEL_PICK__")

    assert result is not None
    assert result.should_continue_loop is True
    session.agent.brain.set_model_override.assert_called_once_with(None)
    assert session.current_model == "auto"
    assert session.agentic.model_override is None
    assert session.token_limit == 123456
    show_info.assert_called_once_with("Model set to auto-routing")
    session._show_bar.assert_called_once()


def test_signal_new_session_clears_runtime_state_and_checkpoints():
    session = _make_session()
    controller = SessionSignalController(session)

    result = controller.handle("__NEW_SESSION__")

    assert result is not None
    assert result.should_continue_loop is True
    session.agentic.session.save.assert_called_once_with()
    assert session.agentic._conversation_history == []
    session.checkpoint_mgr.clear.assert_called_once_with()
    assert session.msg_count == 0
    assert session.token_used == 0
    session._show_bar.assert_called_once()

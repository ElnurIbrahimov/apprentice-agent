"""Tests for ChatSession.apply_model_override and _dispatch_command.

These methods are part of the 727-line ChatSession god-class. The
class itself is hard to unit-test (the __init__ runs session bootstrap,
reads the filesystem, and wires 4 controllers), so we don't try.

Instead, we create a minimal stand-in object with the attributes
these methods touch, then bind the methods to it via __get__, and
verify the contract:

  apply_model_override(model) must update three mirrors atomically:
    - agent.brain.set_model_override (the routing decision)
    - agentic.model_override         (the agentic loop)
    - session.current_model          (the displayed value)

  _dispatch_command(user_input) must wrap handle_command in a
  try/except that converts any exception into show_error, while
  still running the post-dispatch status-bar sync.

If a future refactor breaks the 3-mirror contract or the try/except,
these tests fail immediately.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_session_for_apply_model():
    """Minimal stand-in: only the attributes apply_model_override touches."""
    session = SimpleNamespace()
    session.current_model = "auto"  # initial state

    # agent.brain.set_model_override
    brain = MagicMock()
    agent = MagicMock()
    agent.brain = brain

    # agentic.model_override
    agentic = SimpleNamespace(model_override=None)

    session.agent = agent
    session.agentic = agentic
    session.agentic_session = None  # event_log path will skip
    return session, brain, agentic


# ── apply_model_override ──────────────────────────────────────────────────


def test_apply_model_override_sets_all_three_mirrors():
    from aura.cli.chat_session import ChatSession
    session, brain, agentic = _make_session_for_apply_model()
    method = ChatSession.apply_model_override.__get__(session, type(session))
    method("qwen3:8b")
    brain.set_model_override.assert_called_once_with("qwen3:8b")
    assert agentic.model_override == "qwen3:8b"
    assert session.current_model == "qwen3:8b"


def test_apply_model_override_normalizes_auto_to_none():
    from aura.cli.chat_session import ChatSession
    session, brain, agentic = _make_session_for_apply_model()
    method = ChatSession.apply_model_override.__get__(session, type(session))
    method("auto")
    # set_model_override should be called with None (the canonical auto)
    brain.set_model_override.assert_called_once_with(None)
    assert agentic.model_override is None
    # current_model should display as "auto" (the user-facing form)
    assert session.current_model == "auto"


def test_apply_model_override_normalizes_none_to_auto():
    from aura.cli.chat_session import ChatSession
    session, brain, agentic = _make_session_for_apply_model()
    method = ChatSession.apply_model_override.__get__(session, type(session))
    method(None)
    brain.set_model_override.assert_called_once_with(None)
    assert agentic.model_override is None
    assert session.current_model == "auto"


def test_apply_model_override_tolerates_brain_failure():
    """If brain.set_model_override raises, the other mirrors must still
    update. The brain call is wrapped in try/except in the source —
    this locks that behavior in.

    The first call (here: a pre-existing override) raises; the second
    call (the one we make inside apply_model_override) succeeds.
    """
    from aura.cli.chat_session import ChatSession
    session, brain, agentic = _make_session_for_apply_model()
    brain.set_model_override.side_effect = RuntimeError("brain exploded")
    method = ChatSession.apply_model_override.__get__(session, type(session))
    # Should not raise.
    method("kimi-k2.6")
    # Other mirrors still got updated.
    assert agentic.model_override == "kimi-k2.6"
    assert session.current_model == "kimi-k2.6"


# ── _dispatch_command ───────────────────────────────────────────────────


def _make_session_for_dispatch(agent, brain_override=None):
    """Minimal stand-in for _dispatch_command's body.

    The method touches: self.agent, self.speak, self.agentic_session,
    self.agentic._conversation_history, self.current_model,
    self.token_used, self.token_limit, self._project_type,
    self.session_title, self.msg_count, self.perm_mode,
    plus calls show_error and handle_command. We mock all of them.
    """
    session = SimpleNamespace()
    session.agent = agent
    session.speak = False
    session.current_model = "auto"
    session.token_used = 0
    session.token_limit = 8192
    session.agentic_session = None
    session.agentic = SimpleNamespace(_conversation_history=[])
    session._project_type = ""
    session.session_title = ""
    session.msg_count = 0
    session.perm_mode = "auto"
    # Stub the status-bar sync helper
    session._show_bar = MagicMock()
    return session


def test_dispatch_command_runs_handler_and_syncs_status_bar():
    from aura.cli.chat_session import ChatSession
    agent = MagicMock()
    session = _make_session_for_dispatch(agent)
    method = ChatSession._dispatch_command.__get__(session, type(session))

    with patch("aura.cli.commands.handle_command") as mock_handle, \
         patch("aura.cli.display.show_error") as mock_err, \
         patch("aura.cli.context_bar.estimate_messages_tokens", return_value=42), \
         patch("aura.cli.context_bar.get_context_limit", return_value=4096):
        method("/help")

    mock_handle.assert_called_once_with(agent, "/help", speak=False)
    mock_err.assert_not_called()
    session._show_bar.assert_called_once()
    assert session.token_used == 42
    assert session.token_limit == 4096


def test_dispatch_command_converts_handler_exception_to_show_error():
    from aura.cli.chat_session import ChatSession
    agent = MagicMock()
    session = _make_session_for_dispatch(agent)
    method = ChatSession._dispatch_command.__get__(session, type(session))

    with patch("aura.cli.commands.handle_command", side_effect=RuntimeError("boom")), \
         patch("aura.cli.display.show_error") as mock_err, \
         patch("aura.cli.context_bar.estimate_messages_tokens", return_value=0), \
         patch("aura.cli.context_bar.get_context_limit", return_value=4096):
        method("/crash")

    mock_err.assert_called_once()
    err_msg = mock_err.call_args[0][0]
    assert "Command failed" in err_msg
    assert "boom" in err_msg
    # Status bar must still re-sync after the failure.
    session._show_bar.assert_called_once()


def test_dispatch_command_resyncs_model_from_brain_if_current_model_is_auto():
    """Legacy guard: if current_model is empty/auto and the brain has an
    override (set by something that bypassed apply_model_override),
    re-sync session.current_model from brain.get_model_override().
    """
    from aura.cli.chat_session import ChatSession
    agent = MagicMock()
    agent.brain.get_model_override.return_value = "gpt-4o"
    session = _make_session_for_dispatch(agent)
    session.current_model = "auto"  # empty/auto
    method = ChatSession._dispatch_command.__get__(session, type(session))

    with patch("aura.cli.commands.handle_command"), \
         patch("aura.cli.display.show_error"), \
         patch("aura.cli.context_bar.estimate_messages_tokens", return_value=0), \
         patch("aura.cli.context_bar.get_context_limit", return_value=4096):
        method("/whatever")

    agent.brain.get_model_override.assert_called()
    assert session.current_model == "gpt-4o"
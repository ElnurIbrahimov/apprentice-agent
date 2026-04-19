"""Tests for aura.channels.slack_channel.SlackChannel (mocked slack-sdk)."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from aura.channels.bridge import ChannelResponse, ChannelSource


def _install_fake_slack_sdk(monkeypatch):
    """Install a fake slack_sdk module so we can exercise SlackChannel without the real lib."""
    slack_sdk = types.ModuleType("slack_sdk")
    slack_sdk.WebClient = MagicMock(name="WebClient")  # type: ignore[attr-defined]

    socket_mode_mod = types.ModuleType("slack_sdk.socket_mode")
    socket_mode_mod.SocketModeClient = MagicMock(name="SocketModeClient")  # type: ignore[attr-defined]

    request_mod = types.ModuleType("slack_sdk.socket_mode.request")
    class _FakeRequest:
        def __init__(self, type: str = "", payload: dict | None = None, envelope_id: str = "eid"):
            self.type = type
            self.payload = payload or {}
            self.envelope_id = envelope_id
    request_mod.SocketModeRequest = _FakeRequest  # type: ignore[attr-defined]

    response_mod = types.ModuleType("slack_sdk.socket_mode.response")
    class _FakeResponse:
        def __init__(self, envelope_id: str = ""):
            self.envelope_id = envelope_id
    response_mod.SocketModeResponse = _FakeResponse  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "slack_sdk", slack_sdk)
    monkeypatch.setitem(sys.modules, "slack_sdk.socket_mode", socket_mode_mod)
    monkeypatch.setitem(sys.modules, "slack_sdk.socket_mode.request", request_mod)
    monkeypatch.setitem(sys.modules, "slack_sdk.socket_mode.response", response_mod)
    return slack_sdk, socket_mode_mod, request_mod, response_mod


def test_source_is_slack():
    from aura.channels.slack_channel import SlackChannel
    ch = SlackChannel(bot_token="xoxb-fake", app_token="xapp-fake")
    assert ch.source == ChannelSource.SLACK


def test_missing_tokens_does_not_start(monkeypatch, caplog):
    from aura.channels.slack_channel import SlackChannel
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
    ch = SlackChannel()
    ch.start(on_message=lambda msg: None)
    assert not ch.is_running


def test_allowed_channels_filter(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOWED_CHANNELS", "C1,C2,C3")
    from aura.channels.slack_channel import SlackChannel
    ch = SlackChannel(bot_token="xoxb", app_token="xapp")
    assert ch._allowed == {"C1", "C2", "C3"}


def test_dispatch_ignores_bot_messages(monkeypatch):
    from aura.channels.slack_channel import SlackChannel
    ch = SlackChannel(bot_token="xoxb", app_token="xapp")
    ch._bot_user_id = "U_BOT"
    calls = []
    ch._on_message = lambda msg: calls.append(msg)

    ch._dispatch_event({"type": "message", "bot_id": "BBOT", "text": "hi", "user": "U_x", "channel": "C1"})
    ch._dispatch_event({"type": "message", "user": "U_BOT", "text": "hi", "channel": "C1"})
    ch._dispatch_event({"type": "channel_joined", "user": "U_HUMAN", "channel": "C1"})

    assert calls == []


def test_dispatch_routes_user_message(monkeypatch):
    from aura.channels.slack_channel import SlackChannel
    ch = SlackChannel(bot_token="xoxb", app_token="xapp")
    ch._bot_user_id = "U_BOT"
    calls = []
    ch._on_message = lambda msg: calls.append(msg)

    ch._dispatch_event({
        "type": "message", "user": "U_HUMAN",
        "text": "hey bot", "channel": "C1", "ts": "123.456",
        "thread_ts": "100.001",
    })

    assert len(calls) == 1
    msg = calls[0]
    assert msg.source == ChannelSource.SLACK
    assert msg.text == "hey bot"
    assert msg.user_id == "U_HUMAN"
    assert msg.chat_id == "C1"
    assert msg.metadata["thread_ts"] == "100.001"
    assert msg.metadata["event_type"] == "message"


def test_dispatch_respects_allowlist(monkeypatch):
    from aura.channels.slack_channel import SlackChannel
    ch = SlackChannel(bot_token="xoxb", app_token="xapp", allowed_channels="C_ALLOW")
    calls = []
    ch._on_message = lambda msg: calls.append(msg)

    ch._dispatch_event({"type": "message", "user": "U", "text": "hi", "channel": "C_OTHER"})
    assert not calls

    ch._dispatch_event({"type": "message", "user": "U", "text": "hi", "channel": "C_ALLOW"})
    assert len(calls) == 1


def test_send_uses_web_client(monkeypatch):
    _install_fake_slack_sdk(monkeypatch)
    from aura.channels.slack_channel import SlackChannel
    ch = SlackChannel(bot_token="xoxb", app_token="xapp")
    ch._web_client = MagicMock()
    ch.send(ChannelResponse(text="hi", target_source=ChannelSource.SLACK, chat_id="C1"))
    ch._web_client.chat_postMessage.assert_called_once()


def test_send_without_client_is_safe(monkeypatch):
    from aura.channels.slack_channel import SlackChannel
    ch = SlackChannel(bot_token="xoxb", app_token="xapp")
    ch._web_client = None
    # Should not raise
    ch.send(ChannelResponse(text="x", target_source=ChannelSource.SLACK, chat_id="C1"))


def test_stop_cleans_up_when_never_started():
    from aura.channels.slack_channel import SlackChannel
    ch = SlackChannel(bot_token="xoxb", app_token="xapp")
    ch.stop()
    assert not ch.is_running

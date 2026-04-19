"""Slack channel adapter using slack-sdk Socket Mode.

Socket Mode removes the need to run a public webhook server — Slack pushes
events over a WebSocket from your workspace's app. Requires two tokens:

    SLACK_BOT_TOKEN   xoxb-… (bot user token)
    SLACK_APP_TOKEN   xapp-… (app-level token with connections:write scope)

Optional:
    SLACK_ALLOWED_CHANNELS   comma-separated channel IDs; empty = allow all

Creating the Slack app:
    1. https://api.slack.com/apps → Create New App (from scratch)
    2. Socket Mode: enable → generate app-level token (connections:write)
    3. OAuth & Permissions: bot scopes — chat:write, app_mentions:read,
       channels:history, groups:history, im:history
    4. Install to workspace → copy Bot User OAuth Token (xoxb-…)
    5. Event Subscriptions (Socket): app_mention, message.channels,
       message.im
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Optional

from .bridge import ChannelAdapter, ChannelMessage, ChannelResponse, ChannelSource

logger = logging.getLogger(__name__)


class SlackChannel(ChannelAdapter):
    """Slack Socket-Mode adapter."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        app_token: Optional[str] = None,
        allowed_channels: Optional[str] = None,
    ):
        self._bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN", "")
        self._app_token = app_token or os.getenv("SLACK_APP_TOKEN", "")
        raw_allow = allowed_channels if allowed_channels is not None else os.getenv("SLACK_ALLOWED_CHANNELS", "")
        self._allowed: set[str] = {c.strip() for c in raw_allow.split(",") if c.strip()}
        self._on_message: Optional[Callable[[ChannelMessage], None]] = None
        self._web_client: Any = None
        self._socket_client: Any = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._bot_user_id: str = ""

    @property
    def source(self) -> ChannelSource:
        return ChannelSource.SLACK

    @property
    def is_running(self) -> bool:
        return self._running

    def _check_configured(self) -> bool:
        if not self._bot_token or not self._app_token:
            logger.warning(
                "[Slack] Missing tokens — set SLACK_BOT_TOKEN and SLACK_APP_TOKEN. "
                "Channel will not start."
            )
            return False
        return True

    def start(self, on_message: Callable[[ChannelMessage], None]) -> None:
        if not self._check_configured():
            return
        self._on_message = on_message
        try:
            from slack_sdk import WebClient
            from slack_sdk.socket_mode import SocketModeClient
            from slack_sdk.socket_mode.request import SocketModeRequest
            from slack_sdk.socket_mode.response import SocketModeResponse
        except ImportError:
            logger.error(
                "[Slack] slack-sdk not installed. Run: pip install slack-sdk"
            )
            return

        self._web_client = WebClient(token=self._bot_token)
        try:
            auth = self._web_client.auth_test()
            self._bot_user_id = auth.get("user_id", "")
            logger.info(
                "[Slack] authenticated as %s (team: %s)",
                auth.get("user", "?"), auth.get("team", "?"),
            )
        except Exception as exc:
            logger.error("[Slack] auth_test failed: %s", exc)
            return

        self._socket_client = SocketModeClient(
            app_token=self._app_token,
            web_client=self._web_client,
        )

        def _handle(client, req: "SocketModeRequest") -> None:
            try:
                client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))
            except Exception as exc:
                logger.debug("[Slack] ack failed: %s", exc)

            if req.type != "events_api":
                return
            event = (req.payload or {}).get("event", {}) or {}
            self._dispatch_event(event)

        self._socket_client.socket_mode_request_listeners.append(_handle)

        def _run():
            try:
                self._socket_client.connect()
                self._running = True
                logger.info("[Slack] socket-mode connected")
                self._socket_client.wait_socket_mode_client_is_running()
            except Exception as exc:
                logger.error("[Slack] socket mode run failed: %s", exc)
            finally:
                self._running = False

        self._thread = threading.Thread(target=_run, daemon=True, name="slack-socket")
        self._thread.start()

    def _dispatch_event(self, event: dict) -> None:
        """Route incoming event to the on_message callback."""
        if not self._on_message:
            return
        # Skip bot's own messages + messages from other bots
        if event.get("bot_id") or event.get("user") == self._bot_user_id:
            return
        event_type = event.get("type", "")
        if event_type not in ("message", "app_mention"):
            return

        channel_id = event.get("channel", "")
        if self._allowed and channel_id not in self._allowed:
            return

        text = event.get("text", "") or ""
        if not text.strip():
            return

        msg = ChannelMessage(
            source=ChannelSource.SLACK,
            text=text,
            user_id=event.get("user", ""),
            user_name=event.get("user", ""),  # Slack IDs only; display name needs users.info
            chat_id=channel_id,
            message_id=event.get("ts", ""),
            metadata={
                "thread_ts": event.get("thread_ts"),
                "team": event.get("team"),
                "event_type": event_type,
            },
        )
        try:
            self._on_message(msg)
        except Exception as exc:
            logger.exception("[Slack] on_message callback failed: %s", exc)

    def send(self, response: ChannelResponse) -> None:
        if not self._web_client:
            logger.warning("[Slack] send called but client not initialized")
            return
        try:
            thread_ts = None
            if response.reply_to and response.reply_to.metadata:
                thread_ts = response.reply_to.metadata.get("thread_ts") or response.reply_to.message_id
            self._web_client.chat_postMessage(
                channel=response.chat_id,
                text=response.text,
                thread_ts=thread_ts,
            )
        except Exception as exc:
            logger.error("[Slack] send failed: %s", exc)

    def stop(self) -> None:
        self._running = False
        if self._socket_client:
            try:
                self._socket_client.disconnect()
            except Exception as exc:
                logger.debug("[Slack] disconnect failed: %s", exc)
        self._socket_client = None
        self._web_client = None

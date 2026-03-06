"""LINE channel adapter for AURA.

Uses the official LINE Messaging API Python SDK v3 (line-bot-sdk-python).
Receives messages via webhook (LINE pushes POST requests to your server).
Sends replies via the Push API (not Reply API, so no 30s token expiry).

Setup
-----
1. Create a LINE Messaging API channel at https://developers.line.biz/
2. Note your Channel Access Token and Channel Secret
3. Set your webhook URL to https://your-server/line/webhook
4. Set environment variables::

       LINE_CHANNEL_ACCESS_TOKEN=your_token_here
       LINE_CHANNEL_SECRET=your_secret_here
       LINE_WEBHOOK_HOST=0.0.0.0   # optional, default
       LINE_WEBHOOK_PORT=8443       # optional, default

Requires: pip install line-bot-sdk aiohttp

Architecture
------------
LINE is webhook-only — it calls your server when messages arrive.
This adapter starts an aiohttp HTTP server internally to receive those calls.
For HTTPS in production, put nginx or Cloudflare Tunnel in front.

Features
--------
- Text and sticker messages
- Image / file attachment metadata
- Group and 1:1 chat support
- Typing indicator via LoadingAnimation API (official, max 60s)
- Multi-user / multi-group support
- Signature verification on every webhook call

Usage
-----
    from aura.channels.line_adapter import LINEAdapter
    from aura.channels.channel_manager import get_channel_manager

    adapter = LINEAdapter(
        access_token="your_token",
        channel_secret="your_secret",
        webhook_port=8443,
    )
    get_channel_manager().register(adapter)
    await get_channel_manager().start_all()
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from base64 import b64decode, b64encode
from typing import Any, Dict, List, Optional

from .base import (
    ChannelAdapter,
    ChannelCapability,
    InboundMessage,
    OutboundMessage,
)

logger = logging.getLogger(__name__)

try:
    import aiohttp
    from aiohttp import web as aiohttp_web
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    aiohttp_web = None
    AIOHTTP_AVAILABLE = False

try:
    from linebot.v3 import WebhookParser
    from linebot.v3.messaging import (
        AsyncApiClient,
        AsyncMessagingApi,
        Configuration,
        PushMessageRequest,
        TextMessage,
        ImageMessage,
    )
    from linebot.v3.webhooks import (
        MessageEvent,
        TextMessageContent,
        ImageMessageContent,
        FileMessageContent,
        StickerMessageContent,
        FollowEvent,
        JoinEvent,
    )
    LINE_SDK_AVAILABLE = True
except ImportError:
    LINE_SDK_AVAILABLE = False
    WebhookParser = None


class LINEAdapter(ChannelAdapter):
    """AURA channel adapter for LINE Messaging API.

    Parameters
    ----------
    access_token:
        LINE Channel Access Token. Falls back to LINE_CHANNEL_ACCESS_TOKEN env var.
    channel_secret:
        LINE Channel Secret (for webhook signature verification).
        Falls back to LINE_CHANNEL_SECRET env var.
    webhook_host:
        Host to bind the webhook HTTP server. Default: 0.0.0.0
    webhook_port:
        Port to bind the webhook HTTP server. Default: 8443
    group_whitelist:
        If non-empty, only respond in these group/room IDs.
    """

    channel_id = "line"
    display_name = "LINE"
    capabilities = [
        ChannelCapability.TEXT,
        ChannelCapability.IMAGES,
        ChannelCapability.FILE_UPLOAD,
        ChannelCapability.BUTTONS,
    ]

    def __init__(
        self,
        access_token: Optional[str] = None,
        channel_secret: Optional[str] = None,
        webhook_host: str = "0.0.0.0",
        webhook_port: int = 8443,
        group_whitelist: Optional[List[str]] = None,
    ):
        super().__init__()
        self._token = access_token or os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
        self._secret = channel_secret or os.getenv("LINE_CHANNEL_SECRET", "")
        self._host = webhook_host
        self._port = int(os.getenv("LINE_WEBHOOK_PORT", str(webhook_port)))
        self._group_whitelist = set(group_whitelist or [])

        self._api_client: Optional[Any] = None      # AsyncApiClient
        self._messaging_api: Optional[Any] = None   # AsyncMessagingApi
        self._parser: Optional[Any] = None          # WebhookParser
        self._app_runner: Optional[Any] = None      # aiohttp AppRunner
        self._server_task: Optional[asyncio.Task] = None
        self._loading_anim_lock: Dict[str, float] = {}  # chat_id → last sent time

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp not installed. Run: pip install aiohttp")
        if not LINE_SDK_AVAILABLE:
            raise ImportError(
                "line-bot-sdk not installed. Run: pip install line-bot-sdk"
            )
        if not self._token:
            raise ValueError(
                "LINE_CHANNEL_ACCESS_TOKEN not set. "
                "Export it or pass access_token= to LINEAdapter()."
            )
        if not self._secret:
            raise ValueError(
                "LINE_CHANNEL_SECRET not set. "
                "Export it or pass channel_secret= to LINEAdapter()."
            )

        config = Configuration(access_token=self._token)
        self._api_client = AsyncApiClient(config)
        self._messaging_api = AsyncMessagingApi(self._api_client)
        self._parser = WebhookParser(self._secret)

        # Start aiohttp webhook server
        app = aiohttp_web.Application()
        app.router.add_post("/line/webhook", self._handle_webhook)
        app.router.add_get("/line/health", self._handle_health)

        self._app_runner = aiohttp_web.AppRunner(app)
        await self._app_runner.setup()
        site = aiohttp_web.TCPSite(self._app_runner, self._host, self._port)
        await site.start()

        self._running = True
        logger.info(f"[LINE] Webhook server listening on {self._host}:{self._port}/line/webhook")

    async def stop(self) -> None:
        self._running = False
        if self._app_runner:
            await self._app_runner.cleanup()
        if self._api_client:
            await self._api_client.close()
        logger.info("[LINE] Adapter stopped.")

    # ── Send ──────────────────────────────────────────────────────────────────

    async def send(self, message: OutboundMessage) -> bool:
        if not self._messaging_api:
            return False

        # Chunk to LINE's 5000-char limit per message
        text = message.text
        chunks = [text[i:i + 4999] for i in range(0, len(text), 4999)]
        line_messages = [TextMessage(type="text", text=chunk) for chunk in chunks[:5]]

        try:
            await self._messaging_api.push_message(
                PushMessageRequest(
                    to=message.chat_id,
                    messages=line_messages,
                )
            )
            return True
        except Exception as e:
            logger.error(f"[LINE] Push error: {e}")
            return False

    # ── Typing / loading animation ────────────────────────────────────────────

    async def ack(self, chat_id: str, message_id: str, emoji: str = "⏳") -> None:
        """Show LINE loading animation (official API, displays for up to 60s)."""
        if not self._messaging_api:
            return
        import time
        last = self._loading_anim_lock.get(chat_id, 0)
        if time.time() - last < 5:
            return  # Don't spam; LINE rate-limits this
        self._loading_anim_lock[chat_id] = time.time()
        try:
            await self._messaging_api.show_loading_animation_with_http_info(
                {"chatId": chat_id, "loadingSeconds": 20}
            )
        except Exception:
            pass

    # ── Health ────────────────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        return {
            "channel": self.channel_id,
            "running": self._running,
            "webhook_port": self._port,
            "sdk_available": LINE_SDK_AVAILABLE,
        }

    # ── Webhook handler ───────────────────────────────────────────────────────

    async def _handle_health(self, request: Any) -> Any:
        return aiohttp_web.Response(text="OK", status=200)

    async def _handle_webhook(self, request: Any) -> Any:
        """Receive and verify LINE webhook POST request."""
        body_bytes = await request.read()

        # Verify LINE signature
        signature = request.headers.get("X-Line-Signature", "")
        if not self._verify_signature(body_bytes, signature):
            logger.warning("[LINE] Invalid webhook signature — rejected.")
            return aiohttp_web.Response(status=400, text="Invalid signature")

        try:
            events = self._parser.parse(body_bytes.decode("utf-8"), signature)
        except Exception as e:
            logger.error(f"[LINE] Webhook parse error: {e}")
            return aiohttp_web.Response(status=400, text="Parse error")

        for event in events:
            asyncio.create_task(self._handle_event(event))

        return aiohttp_web.Response(status=200, text="OK")

    async def _handle_event(self, event: Any) -> None:
        """Dispatch a single LINE event."""
        # Only process MessageEvents with text content
        if not LINE_SDK_AVAILABLE:
            return
        if not isinstance(event, MessageEvent):
            return

        # Determine chat_id and user
        source = event.source
        source_type = getattr(source, "type", "user")

        if source_type == "group":
            chat_id = source.group_id
        elif source_type == "room":
            chat_id = source.room_id
        else:
            chat_id = source.user_id

        user_id = getattr(source, "user_id", chat_id)

        # Apply group whitelist
        if source_type in ("group", "room") and self._group_whitelist:
            if chat_id not in self._group_whitelist:
                return

        # Extract text content
        content = event.message
        if isinstance(content, TextMessageContent):
            text = content.text.strip()
        elif isinstance(content, StickerMessageContent):
            text = f"[Sticker: {content.package_id}/{content.sticker_id}]"
        elif isinstance(content, ImageMessageContent):
            text = "[Image received]"
        elif isinstance(content, FileMessageContent):
            text = f"[File: {content.file_name}]"
        else:
            return  # Unsupported content type

        # Resolve display name (best effort — requires profile API)
        user_name = user_id
        if self._messaging_api:
            try:
                profile = await self._messaging_api.get_profile(user_id)
                user_name = profile.display_name
            except Exception:
                pass

        inbound = InboundMessage(
            channel_id=self.channel_id,
            chat_id=chat_id,
            user_id=user_id,
            user_name=user_name,
            text=text,
            message_id=event.message.id if hasattr(event, "message") else None,
            metadata={
                "source_type": source_type,
                "reply_token": getattr(event, "reply_token", None),
                "timestamp": event.timestamp,
            },
        )

        await self._dispatch(inbound)

    # ── Signature verification ────────────────────────────────────────────────

    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify LINE webhook X-Line-Signature header."""
        if not self._secret:
            return True  # No secret configured — skip verification
        try:
            digest = hmac.new(
                self._secret.encode("utf-8"), body, hashlib.sha256
            ).digest()
            expected = b64encode(digest).decode("utf-8")
            return hmac.compare_digest(expected, signature)
        except Exception:
            return False

"""Signal channel adapter for AURA.

Uses the signal-cli REST API (https://github.com/bbernhard/signal-cli-rest-api)
which wraps signal-cli in a Docker container and exposes a simple HTTP/WebSocket API.

Setup
-----
1. Run the Docker container::

       docker run -d \
         -p 8080:8080 \
         -v /path/to/signal-cli-config:/home/.local/share/signal-cli \
         bbernhard/signal-cli-rest-api

2. Register or link your Signal number (one-time)::

       # Link to existing account
       curl -X GET 'http://localhost:8080/v1/qrcodelink?device_name=aura'
       # Scan the QR code in Signal > Linked Devices

3. Set environment variables::

       SIGNAL_API_URL=http://localhost:8080   # default
       SIGNAL_PHONE_NUMBER=+1234567890        # your registered number

Requires: pip install aiohttp

Features
--------
- WebSocket receive loop (no polling)
- Typing indicator while processing (send "typing" notification)
- Supports text, attachments, group messages
- Message deduplication via timestamp+sender key
- Auto-reconnect on WebSocket disconnect

Usage
-----
    from aura.channels.signal_adapter import SignalAdapter
    from aura.channels.channel_manager import get_channel_manager

    adapter = SignalAdapter(
        phone_number="+1234567890",
        api_url="http://localhost:8080",
    )
    get_channel_manager().register(adapter)
    await get_channel_manager().start_all()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

from .base import (
    ChannelAdapter,
    ChannelCapability,
    InboundMessage,
    OutboundMessage,
)

logger = logging.getLogger(__name__)

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False


class SignalAdapter(ChannelAdapter):
    """AURA channel adapter for Signal via signal-cli REST API.

    Parameters
    ----------
    phone_number:
        The Signal phone number registered with signal-cli (e.g. "+1234567890").
        Falls back to SIGNAL_PHONE_NUMBER env var.
    api_url:
        Base URL of the signal-cli REST API. Falls back to SIGNAL_API_URL env var.
        Default: http://localhost:8080
    group_whitelist:
        If non-empty, only respond in these group IDs (base64 group IDs).
    reconnect_delay:
        Seconds to wait before reconnecting after a WebSocket disconnect.
    """

    channel_id = "signal"
    display_name = "Signal"
    capabilities = [
        ChannelCapability.TEXT,
        ChannelCapability.IMAGES,
        ChannelCapability.VOICE,
        ChannelCapability.FILE_UPLOAD,
    ]

    def __init__(
        self,
        phone_number: Optional[str] = None,
        api_url: Optional[str] = None,
        group_whitelist: Optional[List[str]] = None,
        reconnect_delay: float = 5.0,
    ):
        super().__init__()
        self._phone = phone_number or os.getenv("SIGNAL_PHONE_NUMBER", "")
        self._api_url = (api_url or os.getenv("SIGNAL_API_URL", "http://localhost:8080")).rstrip("/")
        self._group_whitelist: Set[str] = set(group_whitelist or [])
        self._reconnect_delay = reconnect_delay
        self._session: Optional[Any] = None   # aiohttp.ClientSession
        self._ws_task: Optional[asyncio.Task] = None
        self._seen: Dict[str, float] = {}     # dedup: key → timestamp
        self._seen_ttl = 60.0                 # seconds to remember seen messages

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp not installed. Run: pip install aiohttp")
        if not self._phone:
            raise ValueError(
                "SIGNAL_PHONE_NUMBER not set. Export it or pass phone_number= to SignalAdapter()."
            )

        self._session = aiohttp.ClientSession()
        self._running = True
        logger.info(f"[Signal] Starting adapter for {self._phone} @ {self._api_url}")
        self._ws_task = asyncio.create_task(self._receive_loop(), name="signal-ws")

    async def stop(self) -> None:
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._session:
            await self._session.close()
        logger.info("[Signal] Adapter stopped.")

    # ── Send ──────────────────────────────────────────────────────────────────

    async def send(self, message: OutboundMessage) -> bool:
        if not self._session:
            return False

        # chat_id is either a phone number (1:1) or a base64 group ID
        is_group = not message.chat_id.startswith("+")
        endpoint = f"{self._api_url}/v2/send"

        payload: Dict[str, Any] = {
            "message": self._strip_markdown(message.text),
            "number": self._phone,
        }
        if is_group:
            payload["recipients"] = [{"group_id": message.chat_id}]
        else:
            payload["recipients"] = [{"number": message.chat_id}]

        # Attach files if present
        if message.attachments:
            payload["base64_attachments"] = message.attachments

        try:
            async with self._session.post(endpoint, json=payload) as resp:
                if resp.status in (200, 201):
                    return True
                body = await resp.text()
                logger.warning(f"[Signal] Send failed {resp.status}: {body[:200]}")
                return False
        except Exception as e:
            logger.error(f"[Signal] Send error: {e}")
            return False

    # ── Typing indicator (no formal reactions in Signal) ───────────────────────

    async def ack(self, chat_id: str, message_id: str, emoji: str = "⏳") -> None:
        """Send a typing indicator as acknowledgement."""
        await self._set_typing(chat_id, typing=True)

    async def remove_ack(self, chat_id: str, message_id: str, emoji: str = "⏳") -> None:
        """Stop typing indicator."""
        await self._set_typing(chat_id, typing=False)

    async def _set_typing(self, recipient: str, typing: bool) -> None:
        if not self._session:
            return
        is_group = not recipient.startswith("+")
        payload: Dict[str, Any] = {
            "number": self._phone,
            "typing": typing,
        }
        if is_group:
            payload["group_id"] = recipient
        else:
            payload["recipient"] = recipient
        try:
            await self._session.put(
                f"{self._api_url}/v1/typing-indicator/{self._phone}",
                json=payload,
            )
        except Exception:
            pass

    # ── Health ────────────────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        reachable = False
        if self._session:
            try:
                async with self._session.get(
                    f"{self._api_url}/v1/health", timeout=aiohttp.ClientTimeout(total=3)
                ) as resp:
                    reachable = resp.status == 200
            except Exception:
                pass
        return {
            "channel": self.channel_id,
            "running": self._running,
            "api_reachable": reachable,
            "phone": self._phone,
        }

    # ── WebSocket receive loop ────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """Persistent WebSocket loop with auto-reconnect."""
        ws_url = self._api_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/v1/receive/{self._phone}"

        while self._running:
            try:
                async with self._session.ws_connect(ws_url) as ws:
                    logger.info(f"[Signal] WebSocket connected: {ws_url}")
                    async for raw in ws:
                        if not self._running:
                            break
                        if raw.type == aiohttp.WSMsgType.TEXT:
                            await self._on_ws_message(raw.data)
                        elif raw.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                            logger.warning(f"[Signal] WebSocket closed: {raw.type}")
                            break
            except asyncio.CancelledError:
                return
            except Exception as e:
                if self._running:
                    logger.warning(f"[Signal] WebSocket error, reconnecting in {self._reconnect_delay}s: {e}")
                    await asyncio.sleep(self._reconnect_delay)

    async def _on_ws_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        envelope = data.get("envelope", data)
        if not envelope:
            return

        sender = envelope.get("source") or envelope.get("sourceNumber", "")
        sender_name = envelope.get("sourceName") or sender

        # Skip own messages
        if sender == self._phone:
            return

        # Find the actual message content
        data_msg = (
            envelope.get("dataMessage")
            or envelope.get("syncMessage", {}).get("sentMessage")
        )
        if not data_msg:
            return

        text = data_msg.get("message", "").strip()
        if not text:
            return

        timestamp = str(data_msg.get("timestamp", int(time.time() * 1000)))
        dedup_key = f"{sender}:{timestamp}"

        # Dedup: skip already-seen messages
        now = time.time()
        self._seen = {k: v for k, v in self._seen.items() if now - v < self._seen_ttl}
        if dedup_key in self._seen:
            return
        self._seen[dedup_key] = now

        # Determine chat_id: group or 1:1
        group_info = data_msg.get("groupInfo") or {}
        group_id = group_info.get("groupId", "")
        chat_id = group_id if group_id else sender

        # Apply group whitelist
        if group_id and self._group_whitelist and group_id not in self._group_whitelist:
            return

        attachments = [
            att.get("filename", att.get("id", ""))
            for att in data_msg.get("attachments", [])
            if att
        ]

        inbound = InboundMessage(
            channel_id=self.channel_id,
            chat_id=chat_id,
            user_id=sender,
            user_name=sender_name,
            text=text,
            message_id=timestamp,
            attachments=attachments,
            metadata={
                "group_id": group_id,
                "group_name": group_info.get("groupName", ""),
                "is_group": bool(group_id),
            },
        )
        await self._dispatch(inbound)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Signal doesn't render markdown — strip common formatting."""
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # bold
        text = re.sub(r"_(.*?)_", r"\1", text)           # italic
        text = re.sub(r"`(.*?)`", r"\1", text)           # code
        text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)  # headings
        return text.strip()

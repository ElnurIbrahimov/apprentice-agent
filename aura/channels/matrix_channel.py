"""
MatrixChannel — Lightweight Matrix adapter for CLI Bridge.

Runs matrix-nio in a background thread, forwards messages to the CLI agent
via ChannelBridge, and sends responses back to Matrix rooms.

Thin relay — the CLI's agent handles all processing.

Requires: pip install matrix-nio>=0.24.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Callable, List, Optional, Set

from .bridge import (
    ChannelAdapter,
    ChannelMessage,
    ChannelResponse,
    ChannelSource,
)

logger = logging.getLogger(__name__)

_MATRIX_MSG_LIMIT = 65536  # Matrix has generous limits


def _split_message(text: str, limit: int = _MATRIX_MSG_LIMIT) -> List[str]:
    """Split a long message for Matrix (rarely needed)."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind('\n', 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    return chunks


class MatrixChannel(ChannelAdapter):
    """Matrix adapter — relays messages between Matrix rooms and the CLI agent."""

    def __init__(
        self,
        homeserver: Optional[str] = None,
        user_id: Optional[str] = None,
        access_token: Optional[str] = None,
        allowed_rooms: Optional[Set[str]] = None,
    ):
        self._homeserver = homeserver or os.environ.get("MATRIX_HOMESERVER", "")
        self._user_id = user_id or os.environ.get("MATRIX_USER_ID", "")
        self._access_token = access_token or os.environ.get("MATRIX_ACCESS_TOKEN", "")
        self._allowed_rooms = allowed_rooms or set()
        env_rooms = os.environ.get("MATRIX_ALLOWED_ROOMS", "")
        if env_rooms and not self._allowed_rooms:
            self._allowed_rooms = {r.strip() for r in env_rooms.split(",") if r.strip()}

        self._running = False
        self._client = None
        self._on_message: Optional[Callable[[ChannelMessage], None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def source(self) -> ChannelSource:
        return ChannelSource.MATRIX

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, on_message: Callable[[ChannelMessage], None]) -> None:
        """Start the Matrix client in a background thread."""
        if not self._homeserver or not self._user_id or not self._access_token:
            logger.error("[Matrix] Missing MATRIX_HOMESERVER, MATRIX_USER_ID, or MATRIX_ACCESS_TOKEN")
            return
        if self._running:
            return

        self._on_message = on_message
        self._running = True

        self._thread = threading.Thread(
            target=self._run_client,
            daemon=True,
            name="matrix-channel",
        )
        self._thread.start()
        logger.info("[Matrix] Channel adapter starting...")

    def _run_client(self):
        """Run the matrix-nio client in its own event loop."""
        try:
            from nio import AsyncClient, LoginResponse, RoomMessageText

            async def _main():
                client = AsyncClient(self._homeserver, self._user_id)
                client.access_token = self._access_token
                client.user_id = self._user_id
                # Skip device/key setup for simple relay
                client.device_id = "AURA_BRIDGE"
                self._client = client
                self._loop = asyncio.get_running_loop()

                # Perform an initial sync to get the since token
                # (so we only process new messages, not history)
                resp = await client.sync(timeout=10000)
                if hasattr(resp, 'next_batch'):
                    logger.info(f"[Matrix] Connected as {self._user_id}, initial sync done")
                else:
                    logger.warning(f"[Matrix] Initial sync response: {resp}")

                # Register message callback
                async def _on_room_message(room, event):
                    # Ignore own messages
                    if event.sender == self._user_id:
                        return

                    # Filter by allowed rooms
                    if self._allowed_rooms and room.room_id not in self._allowed_rooms:
                        return

                    text = event.body if hasattr(event, 'body') else ""
                    if not text:
                        return

                    ch_msg = ChannelMessage(
                        source=ChannelSource.MATRIX,
                        text=text,
                        user_id=event.sender,
                        user_name=event.sender.split(":")[0].lstrip("@"),
                        chat_id=room.room_id,
                        message_id=event.event_id,
                        metadata={
                            "room_name": room.display_name if hasattr(room, 'display_name') else room.room_id,
                            "homeserver": self._homeserver,
                        },
                    )

                    if self._on_message:
                        self._on_message(ch_msg)

                client.add_event_callback(_on_room_message, RoomMessageText)

                # Sync loop
                while self._running:
                    try:
                        await client.sync(timeout=30000)
                    except Exception as e:
                        if self._running:
                            logger.warning(f"[Matrix] Sync error: {e}")
                            await asyncio.sleep(5)

                await client.close()

            asyncio.run(_main())

        except ImportError:
            logger.error("[Matrix] matrix-nio not installed. Run: pip install matrix-nio>=0.24.0")
            self._running = False
        except Exception as e:
            logger.error(f"[Matrix] Client crashed: {e}")
            self._running = False

    def send(self, response: ChannelResponse) -> None:
        """Send a response back through Matrix."""
        if not self._client or not self._loop or self._loop.is_closed():
            logger.warning("[Matrix] Cannot send — client not running")
            return

        room_id = response.chat_id
        text = response.text or ""

        async def _send():
            try:
                from nio import RoomSendResponse
                for chunk in _split_message(text):
                    content = {
                        "msgtype": "m.text",
                        "body": chunk,
                        # Send as markdown-formatted
                        "format": "org.matrix.custom.html",
                        "formatted_body": chunk.replace("\n", "<br>"),
                    }
                    resp = await self._client.room_send(
                        room_id=room_id,
                        message_type="m.room.message",
                        content=content,
                    )
                    if not isinstance(resp, RoomSendResponse):
                        logger.warning(f"[Matrix] Send to {room_id} returned: {resp}")
            except Exception as e:
                logger.error(f"[Matrix] Send failed to {room_id}: {e}")

        asyncio.run_coroutine_threadsafe(_send(), self._loop)

    def stop(self) -> None:
        """Stop the Matrix client."""
        self._running = False
        # The sync loop will break on next iteration
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("[Matrix] Channel adapter stopped")

"""AURA Channel Manager — routes messages between channels and the AURA agent.

Inspired by ClawdBot's multi-channel gateway layer.

Usage
-----
    import asyncio
    from aura.channels import get_channel_manager
    from aura.channels.discord_adapter import DiscordAdapter

    manager = get_channel_manager()
    manager.register(DiscordAdapter(token="..."))
    asyncio.run(manager.start_all())
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable, Dict, List, Optional

from .base import ChannelAdapter, InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)


class ChannelManager:
    """Central hub — registers adapters and wires them to the AURA agent."""

    def __init__(self):
        self._adapters: Dict[str, ChannelAdapter] = {}
        self._agent_handler: Optional[Callable[[InboundMessage], str]] = None
        self._lock = threading.Lock()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, adapter: ChannelAdapter) -> None:
        """Register a channel adapter. Call before start_all()."""
        with self._lock:
            self._adapters[adapter.channel_id] = adapter
        adapter.on_message(self._handle_inbound)
        logger.info(f"[ChannelManager] Registered: {adapter.display_name} ({adapter.channel_id})")

    def set_agent_handler(self, handler: Callable[[InboundMessage], str]) -> None:
        """Set the AURA agent callable that processes messages.

        The handler receives an InboundMessage and returns a response string.

        Example::
            def my_handler(msg: InboundMessage) -> str:
                return agent.chat(msg.text, chat_id=msg.chat_id)["response"]

            manager.set_agent_handler(my_handler)
        """
        self._agent_handler = handler

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start_all(self) -> None:
        """Start all registered adapters concurrently."""
        with self._lock:
            adapters = list(self._adapters.values())

        if not adapters:
            logger.warning("[ChannelManager] No adapters registered — nothing to start.")
            return

        logger.info(f"[ChannelManager] Starting {len(adapters)} channel(s): "
                    f"{[a.channel_id for a in adapters]}")
        await asyncio.gather(*(a.start() for a in adapters), return_exceptions=True)

    async def stop_all(self) -> None:
        """Gracefully stop all adapters."""
        with self._lock:
            adapters = list(self._adapters.values())
        await asyncio.gather(*(a.stop() for a in adapters), return_exceptions=True)
        logger.info("[ChannelManager] All channels stopped.")

    # ── Outbound ──────────────────────────────────────────────────────────────

    async def send(self, channel_id: str, message: OutboundMessage) -> bool:
        """Send a message through a specific channel."""
        adapter = self._adapters.get(channel_id)
        if not adapter:
            logger.warning(f"[ChannelManager] Channel not found: {channel_id}")
            return False
        return await adapter.send(message)

    async def broadcast(self, text: str, chat_ids: Dict[str, str]) -> None:
        """Send a message to multiple channels. chat_ids: {channel_id: chat_id}."""
        tasks = []
        for channel_id, chat_id in chat_ids.items():
            msg = OutboundMessage(text=text, chat_id=chat_id)
            tasks.append(self.send(channel_id, msg))
        await asyncio.gather(*tasks, return_exceptions=True)

    # ── Inbound pipeline ──────────────────────────────────────────────────────

    async def _handle_inbound(self, msg: InboundMessage) -> None:
        """Receive a normalised message, call agent, send response back."""
        if not self._agent_handler:
            logger.warning("[ChannelManager] No agent handler set — dropping message.")
            return

        logger.debug(
            f"[ChannelManager] [{msg.channel_id}] {msg.user_name}: {msg.text[:60]}"
        )

        try:
            # Run sync handler in thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(
                None, self._agent_handler, msg
            )
        except Exception as e:
            logger.error(f"[ChannelManager] Agent handler error: {e}")
            response_text = "I encountered an error processing your request."

        if not response_text:
            return

        out = OutboundMessage(
            text=response_text,
            chat_id=msg.chat_id,
            reply_to_id=msg.message_id,
        )
        adapter = self._adapters.get(msg.channel_id)
        if adapter:
            await adapter.send(out)
            # Remove ack reaction after reply (if supported)
            if msg.message_id:
                await adapter.remove_ack(msg.chat_id, msg.message_id)

    # ── Health ────────────────────────────────────────────────────────────────

    async def health_all(self) -> Dict[str, dict]:
        """Return health status for all channels."""
        results = {}
        for cid, adapter in self._adapters.items():
            try:
                results[cid] = await adapter.health_check()
            except Exception as e:
                results[cid] = {"channel": cid, "error": str(e)}
        return results

    def list_channels(self) -> List[str]:
        with self._lock:
            return list(self._adapters.keys())


# ── Singleton ─────────────────────────────────────────────────────────────────

_manager: Optional[ChannelManager] = None
_manager_lock = threading.Lock()


def get_channel_manager() -> ChannelManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ChannelManager()
    return _manager

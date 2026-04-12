"""Conversation event bridge for the chat WebSocket surface."""

from __future__ import annotations

import logging
import threading

from api.services.websocket_hub import websocket_hub

logger = logging.getLogger(__name__)

_conv_listener_registered = False
_conv_listener_lock = threading.Lock()


def ensure_conversation_listener() -> None:
    """Register the ConversationManager listener once per process."""

    global _conv_listener_registered
    if _conv_listener_registered:
        return

    with _conv_listener_lock:
        if _conv_listener_registered:
            return

        try:
            from aura.core.conversation_manager import get_conversation_manager

            manager = get_conversation_manager()

            async def _on_conv_event(event) -> None:
                payload = {
                    "type": "conv_sync",
                    "event": event.event_type,
                    "conversation_id": event.conversation_id,
                    "surface": event.surface,
                    "data": event.data,
                    "timestamp": event.timestamp,
                }
                await websocket_hub.broadcast_json(payload)

            manager.register_async_listener(_on_conv_event)
            _conv_listener_registered = True
            logger.info("[Chat] ConversationManager async listener registered")
        except Exception as exc:
            logger.debug("[Chat] ConvManager listener registration deferred: %s", exc)

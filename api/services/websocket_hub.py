"""Shared WebSocket event broadcasting for API routes and background services."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketHub:
    """Manage connected API WebSocket clients and fan out shared events."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = threading.Lock()

    def register(self, websocket: WebSocket) -> None:
        """Register a new connected client."""

        with self._lock:
            self._connections.append(websocket)

    def unregister(self, websocket: WebSocket) -> None:
        """Remove a disconnected or failed client."""

        with self._lock:
            try:
                self._connections.remove(websocket)
            except ValueError:
                pass

    @property
    def connection_count(self) -> int:
        """Return the current number of connected clients."""

        with self._lock:
            return len(self._connections)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        """Send a JSON payload to every connected client."""

        with self._lock:
            targets = list(self._connections)

        for websocket in targets:
            try:
                await websocket.send_json(payload)
            except Exception:
                logger.debug("broadcast_ws_send_failed", exc_info=True)
                self.unregister(websocket)

    async def broadcast_proactive_message(self, message: Any) -> None:
        """Push a proactive message event to all connected clients."""

        payload = {
            "type": "proactive",
            "content": message.content,
            "action": (
                message.action.value
                if hasattr(message.action, "value")
                else str(message.action)
            ),
            "priority": (
                message.priority.name
                if hasattr(message.priority, "name")
                else str(message.priority)
            ),
            "timestamp": (
                message.timestamp.isoformat()
                if hasattr(message.timestamp, "isoformat")
                else str(message.timestamp)
            ),
            "metadata": getattr(message, "metadata", {}),
        }
        await self.broadcast_json(payload)

    async def broadcast_hand_event(self, result_dict: dict[str, Any]) -> None:
        """Push a completed hand event."""

        await self.broadcast_json({"type": "hand_event", **result_dict})

    async def broadcast_action_trace(
        self,
        hand_name: str,
        step: int,
        description: str,
    ) -> None:
        """Push a live hand execution step."""

        payload = {
            "type": "action_trace",
            "hand": hand_name,
            "step": step,
            "description": description,
            "timestamp": time.time(),
        }
        await self.broadcast_json(payload)

    async def broadcast_hand_approval_request(
        self,
        request_dict: dict[str, Any],
    ) -> None:
        """Push a hand approval request."""

        await self.broadcast_json({"type": "hand_approval_request", **request_dict})


websocket_hub = WebSocketHub()

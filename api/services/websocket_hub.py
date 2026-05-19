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

    async def broadcast_agent_event(
        self,
        *,
        kind: str,
        run_id: str,
        iteration: int,
        payload: dict[str, Any],
        timestamp: float | None = None,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        """Push a typed agent event (tool_start, tool_result, chunk, response, ...)
        to all connected clients. Emitted by LoopEventEmitter via Telegram's
        agent_core mixin and consumed by both the Mini App and web SPA for live
        tool-progress rendering."""

        await self.broadcast_json({
            "type": "agent_event",
            "kind": kind,
            "run_id": run_id,
            "iteration": iteration,
            "payload": payload,
            "timestamp": timestamp if timestamp is not None else time.time(),
            "user_id": user_id,
            "conversation_id": conversation_id,
        })

    async def broadcast_proactive_card(self, card: dict[str, Any]) -> None:
        """Push a rich proactive card (from a Hand or webhook trigger)
        to all connected clients. Rendered by Mini App's ProactiveCard
        component with action buttons (ack/more/snooze)."""

        await self.broadcast_json({"type": "proactive_card", **card})

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

    async def broadcast_message_append(
        self,
        role: str,
        content: str,
        *,
        surface: str = "telegram",
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Push a chat message append so every connected surface mirrors it."""

        await self.broadcast_json({
            "type": "message_append",
            "role": role,
            "content": content,
            "surface": surface,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timestamp": time.time(),
        })

    async def broadcast_typing(
        self,
        *,
        active: bool,
        surface: str = "telegram",
        user_id: str | None = None,
    ) -> None:
        """Push a typing start/end signal."""

        await self.broadcast_json({
            "type": "typing",
            "active": bool(active),
            "surface": surface,
            "user_id": user_id,
            "timestamp": time.time(),
        })

    async def broadcast_alma_state(self, state: dict[str, Any]) -> None:
        """Push an ALMA mood/neuromod snapshot so the Mini App dashboard live-updates."""

        await self.broadcast_json({"type": "alma_state", "state": state, "timestamp": time.time()})

    async def broadcast_bandit_pull(
        self,
        *,
        arm: str,
        reward: float | None,
        totals: dict[str, Any] | None = None,
    ) -> None:
        """Push a Strategy Bandit arm-pull event so live charts update."""

        await self.broadcast_json({
            "type": "bandit_pull",
            "arm": arm,
            "reward": reward,
            "totals": totals or {},
            "timestamp": time.time(),
        })

    async def broadcast_hand_state_update(
        self,
        *,
        hand_name: str,
        status: str,
        last_decision: dict[str, Any] | None = None,
    ) -> None:
        """Push a Hand status snapshot — idle/running/completed + last decision."""

        await self.broadcast_json({
            "type": "hand_state",
            "hand": hand_name,
            "status": status,
            "last_decision": last_decision,
            "timestamp": time.time(),
        })

    async def broadcast_inner_thought(
        self,
        *,
        thought: str,
        level: str = "summary",
        source: str = "inner_thoughts",
    ) -> None:
        """Push an inner-monologue fragment (consciousness/intrinsic motivation output)."""

        await self.broadcast_json({
            "type": "inner_thought",
            "thought": thought,
            "level": level,
            "source": source,
            "timestamp": time.time(),
        })


websocket_hub = WebSocketHub()


# ---------------------------------------------------------------------------
# Thread-safe fire-and-forget helpers for sync code paths
# ---------------------------------------------------------------------------
def _schedule_broadcast(coro) -> None:
    """Schedule an async broadcast from sync code without needing the event loop.

    Safe to call from any thread. If no loop is running (startup, background
    workers, atexit), drops silently rather than crashing.
    """
    import asyncio as _asyncio
    try:
        loop = _asyncio.get_running_loop()
        if loop.is_running():
            _asyncio.run_coroutine_threadsafe(coro, loop)
            return
    except RuntimeError:
        pass
    # No running loop — try to run it synchronously as a one-shot
    try:
        _asyncio.run(coro)
    except Exception:
        logger.debug("[websocket_hub] fire-and-forget broadcast dropped", exc_info=True)


def push_message(role: str, content: str, *, surface: str = "telegram", user_id: str | None = None, conversation_id: str | None = None) -> None:
    """Fire-and-forget message broadcast from sync code (Telegram bot thread)."""
    _schedule_broadcast(
        websocket_hub.broadcast_message_append(
            role=role,
            content=content,
            surface=surface,
            conversation_id=conversation_id,
            user_id=user_id,
        )
    )


def push_typing(active: bool, *, surface: str = "telegram", user_id: str | None = None) -> None:
    """Fire-and-forget typing signal broadcast from sync code."""
    _schedule_broadcast(
        websocket_hub.broadcast_typing(active=active, surface=surface, user_id=user_id)
    )


def push_bandit_pull(arm: str, reward: float | None = None, totals: dict[str, Any] | None = None) -> None:
    """Fire-and-forget bandit-pull broadcast from sync code."""
    _schedule_broadcast(
        websocket_hub.broadcast_bandit_pull(arm=arm, reward=reward, totals=totals)
    )


def push_hand_state(hand_name: str, status: str, last_decision: dict[str, Any] | None = None) -> None:
    """Fire-and-forget hand-state broadcast from sync code."""
    _schedule_broadcast(
        websocket_hub.broadcast_hand_state_update(
            hand_name=hand_name, status=status, last_decision=last_decision,
        )
    )


def push_inner_thought(thought: str, level: str = "summary", source: str = "inner_thoughts") -> None:
    """Fire-and-forget inner-thought broadcast from sync code."""
    _schedule_broadcast(
        websocket_hub.broadcast_inner_thought(thought=thought, level=level, source=source)
    )


def push_agent_event(
    kind: str,
    run_id: str,
    iteration: int,
    payload: dict[str, Any],
    *,
    user_id: str | None = None,
    conversation_id: str | None = None,
) -> None:
    """Fire-and-forget typed agent event broadcast from sync code.

    Called from Telegram's agent thread on every LoopEvent so the Mini App
    and web SPA see live tool_start / tool_result / chunk / response events
    keyed by run_id."""
    _schedule_broadcast(
        websocket_hub.broadcast_agent_event(
            kind=kind,
            run_id=run_id,
            iteration=iteration,
            payload=payload,
            user_id=user_id,
            conversation_id=conversation_id,
        )
    )


def push_proactive_card(card: dict[str, Any]) -> None:
    """Fire-and-forget proactive-card broadcast from sync code."""
    _schedule_broadcast(websocket_hub.broadcast_proactive_card(card))


__all__ = [
    "WebSocketHub",
    "push_agent_event",
    "push_bandit_pull",
    "push_hand_state",
    "push_inner_thought",
    "push_message",
    "push_proactive_card",
    "push_typing",
    "websocket_hub",
]

"""Structured event emission for AgenticLoop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class LoopEvent:
    """Structured event emitted during an agentic run."""

    type: str
    run_id: str
    iteration: int
    payload: dict[str, Any]


class LoopEventEmitter:
    """Dispatch structured loop events and adapt them to legacy callbacks."""

    def __init__(
        self,
        loop: Any,
        *,
        on_emit: Callable[[LoopEvent], None] | None = None,
        on_event: Callable[[LoopEvent], None] | None = None,
        on_tool_call: Callable[[str, dict[str, Any], Any], None] | None = None,
        on_response: Callable[[str, int], None] | None = None,
        on_chunk: Callable[[str], None] | None = None,
        on_tool_start: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._loop = loop
        self._on_emit = on_emit
        self._on_event = on_event
        self._on_tool_call = on_tool_call
        self._on_response = on_response
        self._on_chunk = on_chunk
        self._on_tool_start = on_tool_start

    def listens_for(self, event_type: str) -> bool:
        if self._on_event is not None:
            return True
        if event_type == "tool_result":
            return self._on_tool_call is not None
        if event_type == "response":
            return self._on_response is not None
        if event_type == "chunk":
            return self._on_chunk is not None
        if event_type == "tool_start":
            return self._on_tool_start is not None
        return False

    def emit(self, event_type: str, **payload: Any) -> LoopEvent:
        event = LoopEvent(
            type=event_type,
            run_id=str(getattr(self._loop, "_current_run_id", "")),
            iteration=getattr(self._loop, "iteration", 0),
            payload=payload,
        )

        if self._on_emit is not None:
            self._on_emit(event)

        if self._on_event is not None:
            self._on_event(event)

        if event_type == "chunk" and self._on_chunk is not None:
            self._on_chunk(str(payload.get("text", "")))
        elif event_type == "response" and self._on_response is not None:
            self._on_response(str(payload.get("text", "")), event.iteration)
        elif event_type == "tool_start" and self._on_tool_start is not None:
            self._on_tool_start(
                str(payload.get("tool_name", "")),
                dict(payload.get("tool_args", {})),
            )
        elif event_type == "tool_result" and self._on_tool_call is not None:
            self._on_tool_call(
                str(payload.get("tool_name", "")),
                dict(payload.get("tool_args", {})),
                payload.get("tool_result"),
            )

        return event

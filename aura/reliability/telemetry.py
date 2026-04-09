"""
Structured Telemetry / Flight Recorder — Phase 1.

Records events for:
  - memory decisions (from write gate)
  - browser action executions
  - loop guard triggers
  - model routing choices
  - contradiction detections
  - dream consolidation

Storage: local JSONL file, ring-buffered in memory.
No external dependencies beyond stdlib.

Author: Aura reliability upgrade (2026-03)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from aura.jsonl_utils import rotate_jsonl_if_needed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------------

class TelemetryKind:
    MEMORY_DECISION    = "memory_decision"
    BROWSER_ACTION     = "browser_action"
    LOOP_GUARD         = "loop_guard"
    MODEL_ROUTING      = "model_routing"
    CONTRADICTION      = "contradiction_detected"
    DREAM_CYCLE        = "dream_cycle"
    TASK_RESULT        = "task_result"
    TOOL_CALL          = "tool_call"


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class TelemetryEvent:
    kind: str
    ts: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    session_id: str = ""
    user_id: str = "default_user"
    task_id: str = ""
    success: Optional[bool] = None
    latency_ms: Optional[float] = None
    model_used: Optional[str] = None
    tokens: Optional[int] = None
    tools_used: List[str] = field(default_factory=list)
    retries: int = 0
    rollbacks: int = 0
    memory_writes: int = 0
    contradictions_detected: int = 0
    confidence: Optional[float] = None
    loop_warnings: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ts_iso"] = datetime.fromtimestamp(self.ts, tz=timezone.utc).isoformat()
        return d


# ---------------------------------------------------------------------------
# Telemetry sink
# ---------------------------------------------------------------------------

class TelemetrySink:
    """
    Thread-safe telemetry collector.

    - In-memory ring buffer (configurable, default 1000 events)
    - Async JSONL file append (non-blocking)
    - Simple report generation
    """

    DEFAULT_RING_SIZE = 1000
    DEFAULT_LOG_DIR   = "logs/telemetry"

    def __init__(
        self,
        ring_size: int = DEFAULT_RING_SIZE,
        log_dir: Optional[str] = None,
    ) -> None:
        self._ring: Deque[TelemetryEvent] = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        log_dir = log_dir or os.getenv("AURA_TELEMETRY_DIR", self.DEFAULT_LOG_DIR)
        self._log_path = Path(log_dir)
        try:
            self._log_path.mkdir(parents=True, exist_ok=True)
            self._file_path = self._log_path / "events.jsonl"
        except OSError as e:
            logger.warning("[Telemetry] Cannot create log dir %s: %s", log_dir, e)
            self._file_path = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, event: TelemetryEvent) -> None:
        """Record an event synchronously to ring buffer, async to file."""
        with self._lock:
            self._ring.append(event)
        self._write_async(event)

    def emit(self, kind: str, **kwargs) -> TelemetryEvent:
        """Convenience: create + record an event in one call."""
        ev = TelemetryEvent(kind=kind, **kwargs)
        self.record(ev)
        return ev

    def recent(self, n: int = 50, kind: Optional[str] = None) -> List[Dict]:
        """Return the N most recent events (filtered by kind if given)."""
        with self._lock:
            events = list(self._ring)
        if kind:
            events = [e for e in events if e.kind == kind]
        return [e.to_dict() for e in events[-n:]]

    def stats(self) -> Dict[str, Any]:
        """Aggregate stats from ring buffer."""
        with self._lock:
            events = list(self._ring)

        total = len(events)
        if not total:
            return {"total": 0}

        success_events  = [e for e in events if e.success is True]
        failure_events  = [e for e in events if e.success is False]
        latencies       = [e.latency_ms for e in events if e.latency_ms is not None]
        memory_writes   = sum(e.memory_writes for e in events)
        contradictions  = sum(e.contradictions_detected for e in events)
        loop_warnings   = sum(e.loop_warnings for e in events)
        retries         = sum(e.retries for e in events)

        by_kind: Dict[str, int] = {}
        by_model: Dict[str, int] = {}
        for e in events:
            by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
            if e.model_used:
                by_model[e.model_used] = by_model.get(e.model_used, 0) + 1

        return {
            "total": total,
            "successes": len(success_events),
            "failures": len(failure_events),
            "success_rate": round(len(success_events) / total, 3) if total else None,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "memory_writes": memory_writes,
            "contradictions_detected": contradictions,
            "loop_warnings": loop_warnings,
            "retries": retries,
            "by_kind": by_kind,
            "by_model": by_model,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_async(self, event: TelemetryEvent) -> None:
        if not self._file_path:
            return
        try:
            rotate_jsonl_if_needed(self._file_path)
            # Fire and forget — append is fast enough to not bother with a thread
            with open(self._file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        except OSError:
            pass  # Non-critical — ring buffer has the data


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_sink_instance: Optional[TelemetrySink] = None
_sink_lock = threading.Lock()


def get_telemetry() -> TelemetrySink:
    global _sink_instance
    if _sink_instance is None:
        with _sink_lock:
            if _sink_instance is None:
                _sink_instance = TelemetrySink()
    return _sink_instance


def emit(kind: str, **kwargs) -> TelemetryEvent:
    """Module-level convenience wrapper."""
    return get_telemetry().emit(kind, **kwargs)


__all__ = [
    "TelemetryEvent",
    "TelemetryKind",
    "TelemetrySink",
    "emit",
    "get_telemetry",
]

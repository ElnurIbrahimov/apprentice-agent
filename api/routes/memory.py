"""Memory system API endpoints - Memory recall tracking for AURA."""

import asyncio
import functools
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import deque
from threading import Lock

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory", tags=["memory"], dependencies=[Depends(require_api_key)])

# ============================================================================
# Memory Recall Tracking System
# ============================================================================

class MemoryRecallEvent:
    """Represents a memory recall event."""
    def __init__(
        self,
        source: str,
        count: int,
        query: str,
        memories: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.source = source  # 'amem', 'rag', 'kg', 'retriever'
        self.count = count
        self.query = query[:100]  # Truncate for privacy
        self.memories = memories[:5]  # Keep top 5
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.id = f"recall_{self.timestamp.timestamp()}"


class MemoryRecallTracker:
    """Tracks memory recall events for UI visualization."""

    def __init__(self, max_events: int = 50):
        self._events: deque = deque(maxlen=max_events)
        self._lock = Lock()
        self._last_recall_time: Optional[datetime] = None
        self._stats = {
            "total_recalls": 0,
            "amem_recalls": 0,
            "rag_recalls": 0,
            "kg_recalls": 0,
            "total_memories_retrieved": 0,
        }

    def record_recall(
        self,
        source: str,
        count: int,
        query: str,
        memories: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a memory recall event."""
        with self._lock:
            event = MemoryRecallEvent(source, count, query, memories, metadata)
            self._events.append(event)
            self._last_recall_time = event.timestamp

            # Update stats
            self._stats["total_recalls"] += 1
            self._stats["total_memories_retrieved"] += count
            if source == "amem":
                self._stats["amem_recalls"] += 1
            elif source == "rag":
                self._stats["rag_recalls"] += 1
            elif source == "kg":
                self._stats["kg_recalls"] += 1

            logger.debug(f"[MemoryRecall] Recorded {source} recall: {count} memories for '{query[:30]}...'")

    def get_recent_recalls(self, limit: int = 10, since_seconds: Optional[float] = None) -> List[Dict]:
        """Get recent memory recall events."""
        with self._lock:
            events = list(self._events)

            # Filter by time if specified
            if since_seconds is not None:
                cutoff = datetime.now().timestamp() - since_seconds
                events = [e for e in events if e.timestamp.timestamp() > cutoff]

            # Return most recent first
            events = sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]

            return [
                {
                    "id": e.id,
                    "source": e.source,
                    "count": e.count,
                    "query": e.query,
                    "memories": e.memories,
                    "timestamp": e.timestamp.isoformat(),
                    "metadata": e.metadata,
                }
                for e in events
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get memory recall statistics."""
        with self._lock:
            return {
                **self._stats,
                "last_recall": self._last_recall_time.isoformat() if self._last_recall_time else None,
                "recent_count": len(self._events),
            }

    def is_active(self, within_seconds: float = 5.0) -> bool:
        """Check if a memory recall happened recently."""
        with self._lock:
            if not self._last_recall_time:
                return False
            elapsed = (datetime.now() - self._last_recall_time).total_seconds()
            return elapsed < within_seconds

    def clear(self):
        """Clear recall history."""
        with self._lock:
            self._events.clear()
            self._last_recall_time = None


# Per-session tracker instances
_trackers: dict[str, MemoryRecallTracker] = {}
_tracker_lock = threading.Lock()


_MAX_TRACKERS = 50
_SESSION_ID_RE = __import__("re").compile(r'^[a-zA-Z0-9_\-]{1,128}$')

def _get_tracker(session_id: str) -> MemoryRecallTracker:
    if not _SESSION_ID_RE.match(session_id):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    with _tracker_lock:
        if session_id not in _trackers:
            # Evict oldest if at capacity
            if len(_trackers) >= _MAX_TRACKERS:
                oldest_key = next(iter(_trackers))
                del _trackers[oldest_key]
            _trackers[session_id] = MemoryRecallTracker()
        return _trackers[session_id]


def get_tracker() -> MemoryRecallTracker:
    """Get the default memory recall tracker (backward compat)."""
    return _get_tracker("default")


# ============================================================================
# API Response Models
# ============================================================================

class RecallEventResponse(BaseModel):
    """Single memory recall event."""
    id: str
    source: str
    count: int
    query: str
    memories: List[str]
    timestamp: str
    metadata: Dict[str, Any] = {}


class RecallStatusResponse(BaseModel):
    """Memory recall status for UI."""
    is_active: bool
    last_recall: Optional[str] = None
    recent_count: int
    recent_events: List[RecallEventResponse]


class RecallStatsResponse(BaseModel):
    """Memory recall statistics."""
    total_recalls: int
    amem_recalls: int
    rag_recalls: int
    kg_recalls: int
    total_memories_retrieved: int
    last_recall: Optional[str] = None
    recent_count: int


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/recalls/status", response_model=RecallStatusResponse)
async def get_recall_status(session_id: str = Query(default="default")):
    """
    Get current memory recall status for UI indicator.

    Returns whether memories were recalled recently and recent events.
    Designed for polling by the frontend to show visual indicators.
    """
    tracker = _get_tracker(session_id)
    loop = asyncio.get_running_loop()
    recent = await loop.run_in_executor(
        None, functools.partial(tracker.get_recent_recalls, limit=5, since_seconds=30)
    )
    stats = await loop.run_in_executor(None, tracker.get_stats)

    return RecallStatusResponse(
        is_active=await loop.run_in_executor(None, functools.partial(tracker.is_active, within_seconds=3)),
        last_recall=stats["last_recall"],
        recent_count=len(recent),
        recent_events=[RecallEventResponse(**e) for e in recent]
    )


@router.get("/recalls/recent")
async def get_recent_recalls(
    limit: int = 10,
    since_seconds: Optional[float] = None,
    session_id: str = Query(default="default")
):
    """
    Get recent memory recall events.

    Args:
        limit: Maximum events to return (default 10)
        since_seconds: Only return events from the last N seconds
    """
    tracker = _get_tracker(session_id)
    loop = asyncio.get_running_loop()
    events = await loop.run_in_executor(
        None, functools.partial(tracker.get_recent_recalls, limit=min(limit, 50), since_seconds=since_seconds)
    )

    return {
        "count": len(events),
        "events": events
    }


@router.get("/recalls/stats", response_model=RecallStatsResponse)
async def get_recall_stats(session_id: str = Query(default="default")):
    """Get memory recall statistics."""
    tracker = _get_tracker(session_id)
    loop = asyncio.get_running_loop()
    stats = await loop.run_in_executor(None, tracker.get_stats)

    return RecallStatsResponse(**stats)


@router.post("/recalls/record")
async def record_recall(
    source: str,
    count: int,
    query: str,
    memories: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: str = Query(default="default")
):
    """
    Record a memory recall event (called internally by agent).

    This endpoint is used by the agent to notify the tracker when
    memories are retrieved during response generation.
    """
    if memories is None:
        memories = []
    if metadata is None:
        metadata = {}
    tracker = _get_tracker(session_id)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, functools.partial(tracker.record_recall, source, count, query, memories, metadata)
    )

    return {
        "status": "recorded",
        "source": source,
        "count": count
    }


@router.post("/recalls/clear")
async def clear_recalls(session_id: str = Query(default="default")):
    """Clear memory recall history."""
    tracker = _get_tracker(session_id)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, tracker.clear)

    return {"status": "cleared"}


# ============================================================================
# Web UI Memory Endpoints (for WisebasePanel)
# ============================================================================

@router.get("/recent")
async def get_recent_memories(limit: int = Query(default=20, le=100)):
    """Return recent memories from unified memory store."""
    try:
        from aura.memory.unified_memory import get_unified_memory
        um = get_unified_memory()
        memories = um.retrieve("", limit=limit)
        return {"memories": [{"content": m.get("content", ""), "timestamp": m.get("timestamp", ""), "category": m.get("category", "general"), "relevance": m.get("relevance", 0)} for m in (memories if memories else [])]}
    except Exception as e:
        logger.debug(f"[Memory] recent endpoint: {e}")
        return {"memories": []}


@router.get("/search")
async def search_memories(q: str = Query(..., min_length=1, max_length=500)):
    """Search memories by query text."""
    try:
        from aura.memory.unified_memory import get_unified_memory
        um = get_unified_memory()
        results = um.retrieve(q, limit=20)
        return {"results": [{"content": m.get("content", ""), "timestamp": m.get("timestamp", ""), "category": m.get("category", "general"), "relevance": m.get("relevance", 0)} for m in (results if results else [])]}
    except Exception as e:
        logger.debug(f"[Memory] search endpoint: {e}")
        return {"results": []}


class AddMemoryBody(BaseModel):
    content: str
    category: str = "general"


@router.post("/add")
async def add_memory(body: AddMemoryBody):
    """Store a new memory entry."""
    try:
        from aura.memory.unified_memory import get_unified_memory
        um = get_unified_memory()
        um.store_gated(body.content, category=body.category)
        return {"ok": True, "message": "Memory stored"}
    except Exception as e:
        logger.warning(f"[Memory] add endpoint failed: {e}")
        return {"ok": False, "message": str(e)}


# ============================================================================
# Integration Helper for Agent
# ============================================================================

def record_memory_recall(source: str, count: int, query: str, memories: List[str] = None, metadata: Dict = None):
    """
    Helper function to record memory recalls from agent code.

    Usage in agent.py:
        from api.routes.memory import record_memory_recall
        record_memory_recall("amem", len(memories), message, [m.content for m in memories])
    """
    if memories is None:
        memories = []
    if metadata is None:
        metadata = {}

    tracker = get_tracker()
    tracker.record_recall(source, count, query, memories, metadata)

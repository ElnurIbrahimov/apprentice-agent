"""Memory system API endpoints - Memory recall tracking for AURA."""

import asyncio
import functools
import logging
import re
import threading
import time
from collections import deque
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

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
_tracker_access: dict[str, float] = {}
_tracker_lock = threading.Lock()


_MAX_TRACKERS = 50
_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")

def _get_tracker(session_id: str) -> MemoryRecallTracker:
    if not _SESSION_ID_RE.match(session_id):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    with _tracker_lock:
        _tracker_access[session_id] = time.time()
        if session_id not in _trackers:
            # Evict LRU if at capacity
            if len(_trackers) >= _MAX_TRACKERS:
                lru_key = min(_tracker_access, key=_tracker_access.get)
                _trackers.pop(lru_key, None)
                _tracker_access.pop(lru_key, None)
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
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    source: str = Query(..., min_length=1, max_length=64, pattern=r'^[a-zA-Z0-9_\-]+$'),
    count: int = Query(..., ge=0, le=10000),
    query: str = Query(..., min_length=1, max_length=2000),
    memories: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: str = Query(default="default"),
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
    loop = asyncio.get_running_loop()

    def _fetch():
        from aura.memory.unified_memory import get_unified_memory
        return get_unified_memory().list_recent(offset=0, limit=limit)

    try:
        rows = await loop.run_in_executor(None, _fetch)
        return {
            "memories": [
                {
                    "id": m.get("id", ""),
                    "content": m.get("content", ""),
                    "timestamp": m.get("created_at", ""),
                    "source": m.get("source", ""),
                    "category": m.get("category", ""),
                    "importance": m.get("importance", 0.0),
                    "tags": m.get("tags", []),
                }
                for m in (rows or [])
            ]
        }
    except Exception as e:
        logger.warning(f"[Memory] recent endpoint: {e}")
        raise HTTPException(status_code=503, detail="Memory store unavailable") from None


@router.get("/search")
async def search_memories(q: str = Query(..., min_length=1, max_length=500)):
    """Search memories via UnifiedMemory.query (BM25 + semantic)."""
    loop = asyncio.get_running_loop()

    def _search():
        from aura.memory.unified_memory import get_unified_memory
        return get_unified_memory().query(q, k=20)

    try:
        results = await loop.run_in_executor(None, _search)
        return {
            "results": [
                {
                    "id": getattr(r, "source_id", "") or "",
                    "content": getattr(r, "content", "") or "",
                    "source": getattr(r, "source", "") or "",
                    "score": float(getattr(r, "score", 0.0) or 0.0),
                    "importance": float(getattr(r, "importance", 0.0) or 0.0),
                    "relevance": float(getattr(r, "relevance", 0.0) or 0.0),
                }
                for r in (results or [])
            ]
        }
    except Exception as e:
        logger.warning(f"[Memory] search endpoint: {e}")
        raise HTTPException(status_code=503, detail="Memory store unavailable") from None


# ============================================================================
# Memory Browser CRUD (backing the Mini App "Brain" tab)
# ============================================================================

class MemoryItem(BaseModel):
    id: str
    content: str
    title: str = ""
    source: str = ""
    memory_type: str = ""
    importance: float = 0.0
    tags: List[str] = Field(default_factory=list)
    pinned: bool = False
    category: str = ""
    lifecycle_state: str = ""
    access_count: int = 0
    strength: float = 0.0
    created_at: str = ""
    updated_at: str = ""


class MemoryBrowseResponse(BaseModel):
    items: List[MemoryItem]
    total: int
    offset: int
    limit: int
    source: Optional[str] = None


class MemoryPatchBody(BaseModel):
    content: Optional[str] = Field(default=None, max_length=50000)
    tags: Optional[List[str]] = None
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)


def _get_um():
    from aura.memory.unified_memory import get_unified_memory
    return get_unified_memory()


@router.get("/browse", response_model=MemoryBrowseResponse)
async def browse_memories(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    source: Optional[str] = Query(default=None, max_length=64),
):
    """Paginated timeline of memories. Backs the Mini App Brain tab."""
    loop = asyncio.get_running_loop()

    def _fetch():
        um = _get_um()
        items = um.list_recent(offset=offset, limit=limit, source_filter=source)
        total = um.count_memories(source_filter=source)
        return items, total

    try:
        items, total = await loop.run_in_executor(None, _fetch)
        return MemoryBrowseResponse(
            items=[MemoryItem(**it) for it in items],
            total=total,
            offset=offset,
            limit=limit,
            source=source,
        )
    except Exception as e:
        logger.warning(f"[Memory] browse endpoint: {e}")
        raise HTTPException(status_code=503, detail="Memory store unavailable") from None


@router.get("/item/{memory_id}", response_model=MemoryItem)
async def get_memory_item(memory_id: str):
    """Fetch a single memory by ID."""
    loop = asyncio.get_running_loop()

    def _fetch():
        return _get_um().get_memory(memory_id)

    row = await loop.run_in_executor(None, _fetch)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
    return MemoryItem(**row)


@router.patch("/item/{memory_id}")
async def patch_memory_item(memory_id: str, body: MemoryPatchBody):
    """Update content, tags, or importance on a memory."""
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    loop = asyncio.get_running_loop()

    def _update():
        return _get_um().update_memory(memory_id, **fields)

    ok = await loop.run_in_executor(None, _update)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
    return {"id": memory_id, "status": "updated"}


@router.delete("/item/{memory_id}")
async def delete_memory_item(memory_id: str):
    """Hard-delete a memory by ID."""
    loop = asyncio.get_running_loop()

    def _delete():
        return _get_um().delete_memory(memory_id)

    ok = await loop.run_in_executor(None, _delete)
    return {"deleted": bool(ok), "id": memory_id}


@router.post("/item/{memory_id}/pin")
async def pin_memory_item(memory_id: str):
    """Add the 'pinned' tag to a memory."""
    loop = asyncio.get_running_loop()
    ok = await loop.run_in_executor(None, lambda: _get_um().set_pinned(memory_id, True))
    if not ok:
        raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
    return {"id": memory_id, "pinned": True}


@router.delete("/item/{memory_id}/pin")
async def unpin_memory_item(memory_id: str):
    """Remove the 'pinned' tag from a memory."""
    loop = asyncio.get_running_loop()
    ok = await loop.run_in_executor(None, lambda: _get_um().set_pinned(memory_id, False))
    if not ok:
        raise HTTPException(status_code=404, detail=f"Memory not found: {memory_id}")
    return {"id": memory_id, "pinned": False}


@router.get("/stats")
async def memory_stats():
    """Lightweight stats for the Brain tab header."""
    loop = asyncio.get_running_loop()

    def _fetch():
        um = _get_um()
        stats = um.get_stats() or {}
        stats["sources"] = um.list_sources()
        stats["total_count"] = um.count_memories()
        return stats

    try:
        return await loop.run_in_executor(None, _fetch)
    except Exception as e:
        logger.warning(f"[Memory] stats endpoint: {e}")
        raise HTTPException(status_code=503, detail="Memory store unavailable") from None


@router.get("/kg/top")
async def memory_kg_top(limit: int = Query(default=20, ge=1, le=100)):
    """Top knowledge graph entities ranked by access_count."""
    loop = asyncio.get_running_loop()

    def _fetch():
        try:
            from aura.services.agent_service import agent_service
            agent = getattr(agent_service, "agent", None)
            if agent is None:
                return []
            kg = getattr(agent, "kg_brain", None)
            if kg is None:
                return []
            stats_fn = getattr(kg, "get_statistics", None)
            if stats_fn is None:
                return []
            stats = stats_fn() or {}
            dist = stats.get("entity_type_distribution", {}) or {}
            total_entities = int(stats.get("total_entities", 0) or 0)
            total_relationships = int(stats.get("total_relationships", 0) or 0)
            return {
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "type_distribution": dist,
            }
        except Exception as exc:
            logger.debug(f"[Memory] kg/top error: {exc}")
            return {"total_entities": 0, "total_relationships": 0, "type_distribution": {}}

    result = await loop.run_in_executor(None, _fetch)
    # Pull the top entries from the main knowledge_graph tool if available
    def _top_nodes():
        try:
            from api.services.agent_service import agent_service
            agent = getattr(agent_service, "agent", None)
            if not agent:
                return []
            kg_tool = (getattr(agent, "tools", None) or {}).get("knowledge_graph")
            if not kg_tool:
                return []
            nodes = []
            graph = getattr(kg_tool, "_graph", None) or getattr(kg_tool, "graph", None)
            raw_nodes = getattr(graph, "nodes", None) if graph else None
            if raw_nodes:
                items = list(raw_nodes.items() if hasattr(raw_nodes, "items") else raw_nodes)
                for entry in items[:limit * 3]:
                    if isinstance(entry, tuple) and len(entry) == 2:
                        nid, ndata = entry
                    else:
                        nid, ndata = entry, {}
                    if not isinstance(ndata, dict):
                        ndata = {"label": str(ndata)}
                    nodes.append({
                        "id": str(nid),
                        "label": str(ndata.get("label") or nid),
                        "type": str(ndata.get("type") or "entity"),
                        "confidence": float(ndata.get("confidence", 0.0) or 0.0),
                        "access_count": int(ndata.get("access_count", 0) or 0),
                    })
            nodes.sort(key=lambda n: (n["access_count"], n["confidence"]), reverse=True)
            return nodes[:limit]
        except Exception as exc:
            logger.debug(f"[Memory] kg top_nodes error: {exc}")
            return []

    nodes = await loop.run_in_executor(None, _top_nodes)
    if isinstance(result, dict):
        result["nodes"] = nodes
    else:
        result = {"nodes": nodes}
    return result


class AddMemoryBody(BaseModel):
    content: str = Field(..., max_length=50000)
    category: str = Field("general", max_length=100)


@router.post("/add")
async def add_memory(body: AddMemoryBody):
    """Store a new memory entry."""
    try:
        import asyncio
        loop = asyncio.get_running_loop()

        def _store():
            from aura.memory.unified_memory import get_unified_memory
            um = get_unified_memory()
            um.store_gated(body.content, category=body.category)

        await loop.run_in_executor(None, _store)
        return {"ok": True, "message": "Memory stored"}
    except Exception as e:
        logger.warning(f"[Memory] add endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to store memory") from None


# ============================================================================
# Integration Helper for Agent
# ============================================================================

def record_memory_recall(source: str, count: int, query: str, memories: List[str] | None = None, metadata: Dict | None = None):
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

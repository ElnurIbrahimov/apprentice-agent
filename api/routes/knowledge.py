"""
Knowledge clip API -- save and search web selections from the AURA Chrome extension.

Rewired to UnifiedMemory (2026-03-22) after aura_episodic_memory was deleted
during the memory consolidation.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"], dependencies=[Depends(require_api_key)])


def _get_unified_memory():
    """Get the UnifiedMemory singleton, or None if unavailable."""
    try:
        from aura.memory.unified_memory import get_unified_memory
        return get_unified_memory()
    except Exception as e:
        logger.warning("[Knowledge] UnifiedMemory unavailable: %s", e)
        return None


# -- Models ---------------------------------------------------------------

class SaveRequest(BaseModel):
    text: str = Field(..., max_length=50_000)
    url: Optional[str] = Field("", max_length=2048)
    title: Optional[str] = Field("", max_length=500)
    tags: Optional[List[str]] = []
    importance: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    source_type: Optional[str] = "selection"


class SaveResponse(BaseModel):
    episode_id: str
    status: str
    message: str


class KnowledgeResult(BaseModel):
    episode_id: str
    content: str
    title: str
    url: str
    score: float
    saved_at: str


class SearchResponse(BaseModel):
    query: str
    results: List[KnowledgeResult]
    count: int


# -- Endpoints -------------------------------------------------------------

from api.utils import EndpointRateLimiter

_knowledge_save_limiter = EndpointRateLimiter(max_per_minute=30)

@router.post("/save", response_model=SaveResponse)
async def save_knowledge(body: SaveRequest):
    """Save a web selection into AURA's unified memory."""
    _knowledge_save_limiter.check()
    mem = _get_unified_memory()
    if mem is None:
        raise HTTPException(status_code=503, detail="Memory store unavailable.")

    try:
        tags = list(body.tags or [])
        if body.source_type:
            tags.append(f"source:{body.source_type}")

        content = body.text.strip()
        if body.title:
            content = f"[{body.title}] {content}"

        ids = mem.store(
            content=content,
            source="knowledge_clip",
            importance=body.importance or 0.7,
            tags=tags,
            episode_type="learning",
        )
        episode_id = ids.get("store", "unknown")
        logger.info("[Knowledge] Saved clip %s from %s", episode_id, body.url)

        return SaveResponse(
            episode_id=str(episode_id),
            status="saved",
            message=f"Saved '{body.title or body.url or 'clip'}' to AURA memory.",
        )
    except Exception as e:
        logger.error("[Knowledge] Save failed: %s", e)
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.get("/list")
async def list_knowledge(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0, le=1000),
):
    """List saved knowledge clips with pagination."""
    mem = _get_unified_memory()
    if mem is None:
        raise HTTPException(503, "Memory store unavailable.")

    try:
        # Query UnifiedMemory; filter to knowledge_clip source post-query
        all_results = mem.query("knowledge clip", k=(limit + offset) * 3)
        results = [r for r in all_results if r.source == "knowledge_clip"]

        # Apply offset/limit
        paged = results[offset:][:limit]

        items = []
        for r in paged:
            meta = r.metadata or {}
            tags_raw = meta.get("tags", "")
            tags = tags_raw.split(",") if isinstance(tags_raw, str) and tags_raw else (tags_raw if isinstance(tags_raw, list) else [])
            items.append({
                "episode_id": r.source_id or r.content_hash,
                "content": r.content,
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "saved_at": meta.get("created_at", ""),
                "tags": tags,
            })

        return {"items": items, "has_more": len(results) > offset + limit}

    except Exception as e:
        logger.error("[Knowledge] List failed: %s", e)
        raise HTTPException(500, safe_error_detail(e))


@router.delete("/{episode_id}")
async def delete_knowledge(episode_id: str):
    """Delete a saved knowledge clip by ID."""
    mem = _get_unified_memory()
    if mem is None:
        raise HTTPException(503, "Memory store unavailable.")

    try:
        store = getattr(mem, '_store', None)
        if store is None:
            mem._ensure_store()
            store = getattr(mem, '_store', None)
        if store and hasattr(store, 'delete'):
            ok = store.delete(episode_id)
            if not ok:
                raise HTTPException(404, f"Clip '{episode_id}' not found.")
            return {"deleted": episode_id}
        raise HTTPException(503, "Delete not supported on current memory backend.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Knowledge] Delete failed: %s", e)
        raise HTTPException(500, safe_error_detail(e))


@router.get("/search", response_model=SearchResponse)
async def search_knowledge(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50),
):
    """Semantic search over saved web clips via UnifiedMemory."""
    mem = _get_unified_memory()
    if mem is None:
        raise HTTPException(status_code=503, detail="Memory store unavailable.")

    try:
        all_results = mem.query(q, k=limit * 3)
        raw_results = [r for r in all_results if r.source == "knowledge_clip"][:limit]

        results = []
        for r in raw_results:
            meta = r.metadata or {}
            results.append(KnowledgeResult(
                episode_id=r.source_id or r.content_hash,
                content=r.content,
                title=meta.get("title", ""),
                url=meta.get("url", ""),
                score=round(r.score, 4),
                saved_at=meta.get("created_at", ""),
            ))

        return SearchResponse(query=q, results=results, count=len(results))
    except Exception as e:
        logger.error("[Knowledge] Search failed: %s", e)
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

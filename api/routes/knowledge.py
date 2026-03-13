"""
Knowledge clip API — save and search web selections from the AURA Chrome extension.
No auth required (localhost-only, extension origin).
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"], dependencies=[Depends(require_api_key)])

# Lazy-load the episodic store so the API boots even if Qdrant is unavailable
def _get_store():
    """Reuse the agent's existing EpisodicMemoryStore to avoid Qdrant lock conflicts."""
    try:
        from api.services.agent_service import agent_service
        agent = getattr(agent_service, 'agent', None)
        if agent and hasattr(agent, 'episodic_memory') and agent.episodic_memory is not None:
            store = agent.episodic_memory
            if getattr(store, '_available', False):
                return store
            logger.warning("[Knowledge] Agent episodic store exists but is unavailable")
            return None
    except Exception as e:
        logger.warning("[Knowledge] Could not get agent episodic store: %s", e)
    return None


# ── Models ────────────────────────────────────────────────────────────────────

class SaveRequest(BaseModel):
    text: str = Field(..., max_length=50_000)
    url: Optional[str] = Field("", max_length=2048)
    title: Optional[str] = Field("", max_length=500)
    tags: Optional[List[str]] = []
    importance: Optional[float] = 0.7
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/save", response_model=SaveResponse)
async def save_knowledge(body: SaveRequest):
    """Save a web selection into AURA's episodic memory."""
    store = _get_store()
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Episodic memory store unavailable. Is Qdrant running?"
        )

    try:
        from aura_episodic_memory import Episode, EpisodeType, TemporalContext

        episode = Episode(
            content=body.text.strip(),
            episode_type=EpisodeType.LEARNING,
            temporal_context=TemporalContext(timestamp=datetime.now()),
            importance=body.importance,
            metadata={
                "url": body.url,
                "title": body.title,
                "tags": body.tags,
                "source_type": body.source_type,
            },
        )
        episode_id = store.store_episode(episode)
        logger.info("[Knowledge] Saved episode %s from %s", episode_id, body.url)

        return SaveResponse(
            episode_id=episode_id,
            status="saved",
            message=f"Saved '{body.title or body.url or 'clip'}' to AURA memory.",
        )
    except Exception as e:
        logger.error("[Knowledge] Save failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_knowledge(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List saved knowledge clips (LEARNING episodes) with pagination."""
    store = _get_store()
    if store is None:
        raise HTTPException(503, "Episodic memory store unavailable.")

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        with store._lock:
            results, next_page_offset = store.client.scroll(
                collection_name="episodic_memory",
                scroll_filter=Filter(
                    must=[FieldCondition(key="episode_type", match=MatchValue(value="learning"))]
                ),
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

        items = []
        for r in results:
            p = r.payload or {}
            meta = p.get("metadata", {}) or {}
            items.append(
                {
                    "episode_id": p.get("id", str(r.id)),
                    "content": p.get("content", ""),
                    "title": meta.get("title", ""),
                    "url": meta.get("url", ""),
                    "saved_at": p.get("timestamp", ""),
                    "tags": meta.get("tags", []),
                }
            )

        return {"items": items, "has_more": next_page_offset is not None}

    except Exception as e:
        logger.error("[Knowledge] List failed: %s", e)
        raise HTTPException(500, str(e))


@router.delete("/{episode_id}")
async def delete_knowledge(episode_id: str):
    """Delete a saved knowledge clip by episode ID."""
    store = _get_store()
    if store is None:
        raise HTTPException(503, "Episodic memory store unavailable.")

    try:
        ok = store.delete_episode(episode_id)
        if not ok:
            raise HTTPException(404, f"Episode '{episode_id}' not found.")
        return {"deleted": episode_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Knowledge] Delete failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/search", response_model=SearchResponse)
async def search_knowledge(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50),
):
    """Semantic search over saved web clips in episodic memory."""
    store = _get_store()
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Episodic memory store unavailable."
        )

    try:
        from aura_episodic_memory import EpisodeQuery, EpisodeType

        query = EpisodeQuery(
            query_text=q,
            episode_types=[EpisodeType.LEARNING],
            limit=limit,
        )
        raw_results = store.search(query)

        results = []
        for r in raw_results:
            ep = r.episode
            meta = getattr(ep, "metadata", {}) or {}
            results.append(KnowledgeResult(
                episode_id=ep.id,
                content=ep.content,
                title=meta.get("title", ""),
                url=meta.get("url", ""),
                score=round(r.score, 4),
                saved_at=ep.temporal_context.timestamp.isoformat()
                         if ep.temporal_context and ep.temporal_context.timestamp
                         else "",
            ))

        return SearchResponse(query=q, results=results, count=len(results))
    except Exception as e:
        logger.error("[Knowledge] Search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

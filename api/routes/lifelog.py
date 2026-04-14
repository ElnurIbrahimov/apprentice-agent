"""Browsing lifelog ingestion endpoint.

Receives batched page-visit events from the Chrome extension's
`lifelog.ts` content script. Each event is a (url, title, dwell_ms,
scroll_max_pct, optional selection) tuple. Events are persisted into the
existing UnifiedMemory store tagged `source=lifelog` so:

  * Chat queries like "what did I read about X this week?" can use
    `UnifiedMemory.query(q)` to find them.
  * The existing Dream consolidation pass clusters recent memories and
    calls `update_profile_from_memories` — this automatically picks up
    lifelog entries and promotes recurring topics into the UserProfile's
    `active_goals` / `key_facts`, which is injected into every system
    prompt via `UserProfile.to_system_prompt()`.

No new database, no new scheduler. Reuses existing infrastructure end to end.

Deployment:
  1. git pull on Hetzner
  2. systemctl restart aura-api
  3. No env vars or migrations

Added as part of SOTA Round 2 (Cluster 3).
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/lifelog", tags=["lifelog"], dependencies=[Depends(require_api_key)])

MAX_EVENTS_PER_BATCH = 50
MAX_URL_LEN = 2048
MAX_TITLE_LEN = 512
MAX_SELECTION_LEN = 500

# Backend-side denylist: last-ditch guard even if the extension forgot to skip.
DENYLIST_PATTERNS = [
    re.compile(r"^https?://[^/]*(bank|payment|checkout|password)[^/]*/", re.IGNORECASE),
    re.compile(r"^https?://accounts\.google\.com/", re.IGNORECASE),
    re.compile(r"^https?://login\.microsoftonline\.com/", re.IGNORECASE),
    re.compile(r"^https?://[^/]*1password\.com/", re.IGNORECASE),
    re.compile(r"^https?://[^/]*lastpass\.com/", re.IGNORECASE),
    re.compile(r"^https?://[^/]*bitwarden\.com/", re.IGNORECASE),
]


class LifelogEvent(BaseModel):
    url: str = Field(..., max_length=MAX_URL_LEN)
    title: str = Field(default="", max_length=MAX_TITLE_LEN)
    dwell_ms: int = Field(default=0, ge=0, le=24 * 3600 * 1000)
    scroll_max_pct: int = Field(default=0, ge=0, le=100)
    selection: Optional[str] = Field(default=None, max_length=MAX_SELECTION_LEN)
    timestamp: int = Field(default=0, ge=0)


class LifelogBatch(BaseModel):
    events: List[LifelogEvent] = Field(..., max_length=MAX_EVENTS_PER_BATCH)


class LifelogStoreResponse(BaseModel):
    stored: int
    skipped: int


def _is_allowed(url: str) -> bool:
    for pat in DENYLIST_PATTERNS:
        if pat.match(url):
            return False
    return True


def _format_event(ev: LifelogEvent) -> str:
    """Build the text that goes into UnifiedMemory."""
    dwell_s = ev.dwell_ms // 1000
    parts = [f"Visited {ev.title or '(untitled)'}"]
    parts.append(f"at {ev.url}")
    parts.append(f"— dwelled {dwell_s}s, scrolled to {ev.scroll_max_pct}%")
    if ev.selection:
        parts.append(f'. Highlighted: "{ev.selection.strip()[:400]}"')
    return " ".join(parts)


@router.post("/events", response_model=LifelogStoreResponse)
async def ingest_events(batch: LifelogBatch) -> LifelogStoreResponse:
    """Store a batch of browsing events into UnifiedMemory."""
    try:
        from aura.memory.unified_memory import get_unified_memory
    except Exception as e:
        logger.warning("[Lifelog] UnifiedMemory unavailable: %s", e)
        raise HTTPException(503, detail="UnifiedMemory not available")

    memory = get_unified_memory()
    stored = 0
    skipped = 0

    for ev in batch.events:
        if not _is_allowed(ev.url):
            skipped += 1
            continue
        try:
            content = _format_event(ev)
            metadata = {
                "source": "lifelog",
                "url": ev.url,
                "title": ev.title,
                "dwell_ms": ev.dwell_ms,
                "scroll_max_pct": ev.scroll_max_pct,
                "ts_ms": ev.timestamp,
            }
            # UnifiedMemory.store(content, metadata=...) — tagged so Dream
            # consolidation can find lifelog entries by source.
            try:
                memory.store(content=content, metadata=metadata)  # type: ignore[attr-defined]
            except TypeError:
                # Older signature fallback
                memory.store(content, metadata)  # type: ignore[misc]
            stored += 1
        except Exception as e:
            logger.debug("[Lifelog] store failed for %s: %s", ev.url[:80], e)
            skipped += 1

    logger.info("[Lifelog] ingested %d events (%d skipped)", stored, skipped)
    return LifelogStoreResponse(stored=stored, skipped=skipped)


@router.get("/recent")
async def recent(limit: int = Query(default=20, ge=1, le=200)) -> dict:
    """Return the most recent lifelog entries for a sidebar 'today' strip."""
    try:
        from aura.memory.unified_memory import get_unified_memory
        memory = get_unified_memory()
        # Query with an empty string to get recency-sorted results, filtered to lifelog source.
        try:
            results = memory.query("", limit=limit, source_filter="lifelog")  # type: ignore[attr-defined]
        except TypeError:
            results = memory.query("lifelog recent browsing", limit=limit)  # type: ignore[attr-defined]
        items = []
        for r in (results or []):
            md = getattr(r, "metadata", {}) or {}
            if md.get("source") != "lifelog":
                continue
            items.append({
                "url": md.get("url", ""),
                "title": md.get("title", ""),
                "dwell_ms": md.get("dwell_ms", 0),
                "ts_ms": md.get("ts_ms", 0),
                "snippet": getattr(r, "content", "")[:200],
            })
        return {"items": items, "count": len(items)}
    except Exception as e:
        logger.debug("[Lifelog] recent failed: %s", e)
        return {"items": [], "count": 0, "error": safe_error_detail(e)}


@router.get("/search")
async def search(q: str = Query(..., min_length=1, max_length=300), limit: int = Query(default=10, ge=1, le=50)) -> dict:
    """Search within the lifelog corpus only."""
    try:
        from aura.memory.unified_memory import get_unified_memory
        memory = get_unified_memory()
        try:
            results = memory.query(q, limit=limit, source_filter="lifelog")  # type: ignore[attr-defined]
        except TypeError:
            results = memory.query(q, limit=limit)  # type: ignore[attr-defined]
        items = []
        for r in (results or []):
            md = getattr(r, "metadata", {}) or {}
            if md.get("source") != "lifelog":
                continue
            items.append({
                "url": md.get("url", ""),
                "title": md.get("title", ""),
                "snippet": getattr(r, "content", "")[:300],
                "score": float(getattr(r, "score", 0)),
            })
        return {"items": items, "count": len(items), "query": q}
    except Exception as e:
        logger.debug("[Lifelog] search failed: %s", e)
        return {"items": [], "count": 0, "query": q, "error": safe_error_detail(e)}

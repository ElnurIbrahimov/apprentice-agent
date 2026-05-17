"""
Real web search with sources via fallback chain (Tavily -> Brave -> SearXNG).
"""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"], dependencies=[Depends(require_api_key)])


async def _run_search(q: str, limit: int) -> dict:
    """Run web search using fallback chain. Raises HTTPException on failure."""
    try:
        from aura.tools.search_fallback import web_search_with_fallback
    except ImportError:
        raise HTTPException(503, "Search fallback module not available") from None

    try:
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, lambda: web_search_with_fallback(query=q, max_results=limit)
        )
    except Exception as e:
        logger.error("[Search] Search failed: %s", e)
        raise HTTPException(500, safe_error_detail(e, "Search failed")) from e

    if not resp.get("success", True) and resp.get("error"):
        raise HTTPException(502, resp["error"])

    return resp


@router.get("")
async def web_search(q: str = Query(..., max_length=500), limit: int = Query(5, ge=1, le=10), model: str = Query(None)):
    """Search the web and return answer + source cards."""
    resp = await _run_search(q, limit)

    return {
        "query": q,
        "answer": resp.get("answer", ""),
        "source": resp.get("source", "web"),
        "sources": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", r.get("content", ""))[:200],
                "score": r.get("score", 0),
            }
            for r in resp.get("results", [])
        ],
    }


@router.get("/results")
async def search_results(q: str = Query(..., max_length=500), limit: int = Query(8, ge=1, le=10)):
    """Return raw search results for frontend panels to inject into LLM prompts.

    Returns a list of result objects with title, url, and full snippet text
    (up to 400 chars each) suitable for building a context block.
    """
    resp = await _run_search(q, limit)

    return {
        "query": q,
        "source": resp.get("source", "web"),
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", r.get("content", ""))[:400],
            }
            for r in resp.get("results", [])
        ],
    }

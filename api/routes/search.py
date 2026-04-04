"""
Real web search with sources via Tavily API.
"""

import os
import asyncio
import logging
from fastapi import APIRouter, Query, HTTPException, Depends

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"], dependencies=[Depends(require_api_key)])


async def _run_tavily_search(q: str, limit: int) -> dict:
    """Run Tavily search and return raw response. Raises HTTPException on failure."""
    try:
        from tavily import TavilyClient
    except ImportError:
        raise HTTPException(503, "tavily-python not installed. Run: pip install tavily-python")

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise HTTPException(503, "TAVILY_API_KEY not set in environment")

    client = TavilyClient(api_key=api_key)
    try:
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, lambda: client.search(q, search_depth="basic", max_results=limit, include_answer=True)
        )
    except Exception as e:
        logger.error("[Search] Tavily search failed: %s", e)
        raise HTTPException(500, safe_error_detail(e, "Search failed"))

    return resp


@router.get("")
async def web_search(q: str = Query(..., max_length=500), limit: int = Query(5, ge=1, le=10), model: str = Query(None)):
    """Search the web via Tavily and return answer + source cards."""
    resp = await _run_tavily_search(q, limit)

    return {
        "query": q,
        "answer": resp.get("answer", ""),
        "sources": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:200],
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
    resp = await _run_tavily_search(q, limit)

    return {
        "query": q,
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:400],
            }
            for r in resp.get("results", [])
        ],
    }

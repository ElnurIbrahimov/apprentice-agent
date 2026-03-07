"""
Real web search with sources via Tavily API.
"""

import os
import asyncio
import logging
from fastapi import APIRouter, Query, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("")
async def web_search(q: str = Query(..., max_length=500), limit: int = Query(5, ge=1, le=10), model: str = Query(None)):
    """Search the web via Tavily and return answer + source cards."""
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
        raise HTTPException(500, f"Search failed: {e}")

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

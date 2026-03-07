"""
Deep Research endpoint — multi-source web research with LLM synthesis.
Streams progress updates, then delivers a structured markdown report.
"""

import os
import json
import logging
import asyncio
from urllib.parse import urlparse
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research", tags=["research"])

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate"
DEFAULT_MODEL = "qwen3.5:397b-cloud"

DEPTH_CONFIG = {
    "quick":    {"max_results": 3, "num_queries": 1},
    "standard": {"max_results": 5, "num_queries": 2},
    "deep":     {"max_results": 5, "num_queries": 4},
}


class ResearchRequest(BaseModel):
    query: str
    depth: str = "standard"
    model: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_tavily():
    try:
        from tavily import TavilyClient
    except ImportError:
        raise HTTPException(503, "tavily-python not installed. Run: pip install tavily-python")
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise HTTPException(503, "TAVILY_API_KEY not set in environment")
    return TavilyClient(api_key=api_key)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lstrip("www.")
    except Exception:
        return url


def _format_sources(sources: list[dict]) -> str:
    parts = []
    for i, s in enumerate(sources, 1):
        snippet = s.get("content", "")[:400]
        parts.append(f"[{i}] {s['title']}\nURL: {s['url']}\n{snippet}")
    return "\n\n".join(parts)


async def _run_search(client: "TavilyClient", query: str, max_results: int) -> list[dict]:
    """Run a single Tavily search in a thread (TavilyClient is sync)."""
    loop = asyncio.get_event_loop()
    def _search():
        return client.search(
            query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
        )
    try:
        resp = await loop.run_in_executor(None, _search)
        return resp.get("results", [])
    except Exception as e:
        logger.warning("[Research] Search failed for %r: %s", query, e)
        return []


async def _generate_followup_queries(query: str, n: int, model: str) -> list[str]:
    """Ask Ollama for n follow-up search angles as a JSON array."""
    prompt = (
        f"Given the research topic: \"{query}\"\n"
        f"Suggest {n} different search angles that would give complementary information. "
        f"Reply ONLY with a JSON array of {n} short search query strings. No explanation."
    )
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            resp = await c.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False})
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            # Extract JSON array from response (model may wrap it in markdown)
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start != -1 and end > start:
                queries = json.loads(raw[start:end])
                if isinstance(queries, list):
                    return [str(q) for q in queries[:n]]
    except Exception as e:
        logger.warning("[Research] Follow-up query generation failed: %s", e)
    # Fallback: basic variants
    fallbacks = [
        f"{query} overview",
        f"{query} recent developments",
        f"{query} analysis",
        f"{query} examples",
    ]
    return fallbacks[:n]


async def _synthesize(query: str, sources: list[dict], model: str) -> str:
    """Call Ollama to write the research report."""
    formatted = _format_sources(sources)
    prompt = (
        f"You are a thorough research analyst. Research topic: {query}\n\n"
        f"Sources:\n{formatted}\n\n"
        "Write a comprehensive research report with these sections:\n"
        "## Executive Summary\n"
        "## Key Findings\n"
        "(5-7 bullet points)\n"
        "## Detailed Analysis\n"
        "(3-4 focused paragraphs)\n"
        "## Conclusions\n\n"
        "Use inline citations like [1], [2] referencing the source numbers above. "
        "Be specific and cite sources frequently. Write in clear, professional prose."
    )
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            resp = await c.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False})
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except httpx.TimeoutException:
        return "*Report generation timed out. Sources are listed below.*"
    except Exception as e:
        logger.error("[Research] Synthesis failed: %s", e)
        return f"*Report generation failed: {e}*"


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post("")
async def deep_research(req: ResearchRequest):
    depth = req.depth if req.depth in DEPTH_CONFIG else "standard"
    cfg = DEPTH_CONFIG[depth]
    model = req.model or DEFAULT_MODEL
    query = req.query.strip()

    if not query:
        raise HTTPException(400, "query must not be empty")

    tavily = _get_tavily()

    async def generate():
        # ── Phase 1: Search ──────────────────────────────────────────────────
        yield json.dumps({"status": "searching", "message": "Searching the web..."}) + "\n"

        # Primary search
        primary_results = await _run_search(tavily, query, cfg["max_results"])

        extra_results: list[dict] = []
        if cfg["num_queries"] > 1:
            n_followup = cfg["num_queries"] - 1
            yield json.dumps({"status": "searching", "message": f"Generating {n_followup} follow-up queries..."}) + "\n"
            followup_queries = await _generate_followup_queries(query, n_followup, model)

            yield json.dumps({"status": "searching", "message": f"Running {n_followup} additional searches..."}) + "\n"
            followup_tasks = [
                _run_search(tavily, fq, cfg["max_results"])
                for fq in followup_queries
            ]
            followup_batches = await asyncio.gather(*followup_tasks)
            for batch in followup_batches:
                extra_results.extend(batch)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        all_raw: list[dict] = []
        for r in primary_results + extra_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_raw.append(r)

        # Cap at desired total
        total_cap = cfg["max_results"] * cfg["num_queries"]
        all_raw = all_raw[:total_cap]

        # Build clean source list
        sources = [
            {
                "index": i + 1,
                "title": r.get("title", "Untitled"),
                "url": r.get("url", ""),
                "domain": _domain(r.get("url", "")),
                "content": r.get("content", ""),
                "snippet": r.get("content", "")[:200],
            }
            for i, r in enumerate(all_raw)
        ]

        # ── Phase 2: Analyze ─────────────────────────────────────────────────
        yield json.dumps({"status": "analyzing", "message": f"Analyzing {len(sources)} sources..."}) + "\n"

        # ── Phase 3: Write ───────────────────────────────────────────────────
        yield json.dumps({"status": "writing", "message": "Writing research report..."}) + "\n"

        report = await _synthesize(query, sources, model)

        # Build citations list (sources actually referenced in report)
        citations = []
        for s in sources:
            if f"[{s['index']}]" in report:
                citations.append({"index": s["index"], "title": s["title"], "url": s["url"]})

        # If none were cited inline, list all sources as citations
        if not citations:
            citations = [{"index": s["index"], "title": s["title"], "url": s["url"]} for s in sources]

        # ── Done ─────────────────────────────────────────────────────────────
        yield json.dumps({
            "status": "done",
            "query": query,
            "depth": depth,
            "report": report,
            "sources": [{"index": s["index"], "title": s["title"], "url": s["url"], "domain": s["domain"], "snippet": s["snippet"]} for s in sources],
            "citations": citations,
        }) + "\n"

    return StreamingResponse(generate(), media_type="text/plain")

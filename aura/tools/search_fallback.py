"""Hybrid web search pipeline with parallel provider fan-out.

Pipeline:
  1. Optional LLM query decomposition (3-5 sub-queries for complex asks).
  2. Parallel fan-out across Tavily (advanced depth, raw content),
     Brave Search, and Firecrawl.search — every provider gets every
     sub-query at the same time.
  3. Dedupe by URL + BM25 rerank against the original query.
  4. Parallel Firecrawl.scrape on the top N URLs that don't already
     have full content, so the LLM reads pages instead of snippets.

This replaces the previous sequential Tavily → Brave → Firecrawl
fallback, which returned whichever provider responded first even
if the result was thin. The new pipeline always calls all three,
always merges, and always reranks — same pattern ChatGPT / Claude /
Perplexity use.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-level query cache (5-minute TTL)
# ---------------------------------------------------------------------------
_search_cache: Dict[Tuple[str, int], Tuple[float, dict]] = {}
_CACHE_TTL = 300


def _cache_get(query: str, max_results: int):
    key = (query.lower().strip(), max_results)
    if key in _search_cache:
        ts, result = _search_cache[key]
        if time.time() - ts < _CACHE_TTL:
            logger.debug("[SearchFallback] Cache hit for '%s…'", key[0][:40])
            return result
        del _search_cache[key]
    return None


def _cache_put(query: str, max_results: int, result: dict):
    key = (query.lower().strip(), max_results)
    _search_cache[key] = (time.time(), result)


# ---------------------------------------------------------------------------
# Module-level LLM wiring (optional — agent sets this at startup)
# ---------------------------------------------------------------------------
_default_llm_func: Optional[Callable] = None
_llm_lock = threading.Lock()


def set_search_llm(llm_func: Optional[Callable]) -> None:
    """Register an LLM function used for query decomposition.

    The agent calls this at startup with ``brain.think`` so the search
    pipeline can fan out multi-angle queries without every caller
    having to thread the LLM through.
    """
    global _default_llm_func
    with _llm_lock:
        _default_llm_func = llm_func


# ---------------------------------------------------------------------------
# Lazy provider singletons
# ---------------------------------------------------------------------------
_tavily = None
_brave = None
_firecrawl = None
_singleton_lock = threading.Lock()


def _get_tavily():
    global _tavily
    if _tavily is None:
        with _singleton_lock:
            if _tavily is None:
                try:
                    from aura.tools.tavily_tool import TavilyTool
                    _tavily = TavilyTool()
                except Exception as e:
                    logger.debug("[SearchFallback] Tavily init failed: %s", e)
                    _tavily = False
    return _tavily or None


def _get_brave():
    global _brave
    if _brave is None:
        with _singleton_lock:
            if _brave is None:
                try:
                    from aura.tools.brave_search import BraveSearchTool
                    _brave = BraveSearchTool()
                except Exception as e:
                    logger.debug("[SearchFallback] Brave init failed: %s", e)
                    _brave = False
    return _brave or None


def _get_firecrawl():
    global _firecrawl
    if _firecrawl is None:
        with _singleton_lock:
            if _firecrawl is None:
                try:
                    from aura.tools.firecrawl_tool import FirecrawlTool
                    _firecrawl = FirecrawlTool()
                except Exception as e:
                    logger.debug("[SearchFallback] Firecrawl init failed: %s", e)
                    _firecrawl = False
    return _firecrawl or None


# ---------------------------------------------------------------------------
# Normalization — every provider maps to a uniform record shape
# ---------------------------------------------------------------------------
def _normalize_tavily(result: dict, query: str) -> List[dict]:
    if not isinstance(result, dict):
        return []
    out = []
    for r in result.get("results", []):
        if not isinstance(r, dict):
            continue
        content = r.get("raw_content") or r.get("content") or ""
        out.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (content[:500] if content else "") or r.get("content", "")[:500],
            "content": content,
            "source": "tavily",
            "score": float(r.get("score", 0.0) or 0.0),
            "query": query,
        })
    return out


def _normalize_brave(result: dict, query: str) -> List[dict]:
    if not isinstance(result, dict):
        return []
    out = []
    for r in result.get("results", []):
        if not isinstance(r, dict):
            continue
        desc = r.get("description") or r.get("snippet") or ""
        out.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": desc[:500] if desc else "",
            "content": desc,
            "source": "brave",
            "score": 0.0,
            "query": query,
        })
    return out


def _normalize_firecrawl(result: dict, query: str) -> List[dict]:
    if not isinstance(result, dict):
        return []
    out = []
    for r in result.get("results", []):
        if not isinstance(r, dict):
            continue
        md = r.get("markdown", "") or ""
        out.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": md[:500] if md else "",
            "content": md,
            "source": "firecrawl",
            "score": 0.0,
            "query": query,
        })
    return out


def _deduplicate(results: List[dict]) -> List[dict]:
    """Remove duplicate URLs, keeping the first (highest-priority) occurrence.

    When a duplicate URL is seen, prefer keeping the version with more
    content — a Brave snippet gets replaced by a Firecrawl scrape for
    the same URL, not the other way around.
    """
    by_url: Dict[str, dict] = {}
    order: List[str] = []
    keyless: List[dict] = []

    for r in results:
        url = (r.get("url") or "").rstrip("/").lower()
        if not url:
            keyless.append(r)
            continue
        if url not in by_url:
            by_url[url] = r
            order.append(url)
        else:
            existing = by_url[url]
            new_content = len(r.get("content", "") or "")
            old_content = len(existing.get("content", "") or "")
            if new_content > old_content:
                by_url[url] = r

    return [by_url[u] for u in order] + keyless


# ---------------------------------------------------------------------------
# Provider callers — each returns list[dict], empty on failure
# ---------------------------------------------------------------------------
def _call_tavily(query: str, max_results: int) -> List[dict]:
    tool = _get_tavily()
    if not tool:
        return []
    try:
        result = tool.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_raw_content=True,
            include_answer=False,
        )
        return _normalize_tavily(result, query)
    except Exception as e:
        logger.debug("[SearchFallback] Tavily call failed: %s", e)
        return []


def _call_brave(query: str, max_results: int) -> List[dict]:
    tool = _get_brave()
    if not tool:
        return []
    try:
        result = tool.run(query=query, count=max_results)
        return _normalize_brave(result, query)
    except Exception as e:
        logger.debug("[SearchFallback] Brave call failed: %s", e)
        return []


def _call_firecrawl(query: str, max_results: int) -> List[dict]:
    tool = _get_firecrawl()
    if not tool:
        return []
    try:
        result = tool.search(query=query, limit=max_results)
        return _normalize_firecrawl(result, query)
    except Exception as e:
        logger.debug("[SearchFallback] Firecrawl call failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Top-N scrape — fill full markdown for results that came back thin
# ---------------------------------------------------------------------------
def _scrape_top_urls(results: List[dict], n: int = 3) -> List[dict]:
    tool = _get_firecrawl()
    if not tool or n <= 0 or not results:
        return results

    needing_scrape: List[Tuple[int, str]] = []
    for idx in range(min(len(results), n * 2)):
        url = results[idx].get("url", "")
        content = results[idx].get("content", "") or ""
        if url and url.startswith(("http://", "https://")) and len(content) < 800:
            needing_scrape.append((idx, url))
        if len(needing_scrape) >= n:
            break

    if not needing_scrape:
        return results

    def _do_scrape(job: Tuple[int, str]) -> Tuple[int, str]:
        idx, url = job
        try:
            data = tool.scrape(url)
            if isinstance(data, dict):
                return idx, data.get("markdown", "") or ""
            return idx, ""
        except Exception as e:
            logger.debug("[SearchFallback] scrape %s failed: %s", url, e)
            return idx, ""

    with ThreadPoolExecutor(max_workers=min(len(needing_scrape), 3)) as pool:
        futures = [pool.submit(_do_scrape, job) for job in needing_scrape]
        for fut in as_completed(futures, timeout=25):
            try:
                idx, md = fut.result(timeout=0)
            except Exception:
                continue
            if md and 0 <= idx < len(results):
                results[idx]["content"] = md
                results[idx]["snippet"] = md[:500]
                results[idx]["scraped"] = True
    return results


# ---------------------------------------------------------------------------
# BM25 rerank — reuses the scorer from web_search.py
# ---------------------------------------------------------------------------
def _rerank(results: List[dict], query: str) -> List[dict]:
    try:
        from aura.tools.web_search import _rerank_results
        return _rerank_results(results, query)
    except Exception as e:
        logger.debug("[SearchFallback] rerank failed: %s", e)
        return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def web_search_with_fallback(
    query: str,
    max_results: int = 8,
    tool_registry: Optional[dict] = None,  # kept for API compatibility, ignored
    llm_func: Optional[Callable] = None,
    scrape_top_n: int = 3,
    decompose: bool = True,
) -> dict:
    """Hybrid parallel web search with optional query decomposition.

    Args:
        query: The user query.
        max_results: Final merged result count.
        tool_registry: Unused; retained for backwards compatibility.
        llm_func: LLM callable for query decomposition. Falls back to
            the module-level default registered via ``set_search_llm``.
        scrape_top_n: Firecrawl-scrape this many top URLs to get full
            markdown content. Set to 0 to disable scraping for speed.
        decompose: When True, decompose complex queries into sub-queries
            and fan out in parallel. Skipped automatically for short
            or already-specific queries.

    Returns:
        dict with keys: success, query, sub_queries, num_results,
        total_raw, results, source. Each result has title, url,
        snippet, content, source, score, scraped.
    """
    if not query or not query.strip():
        return {"success": False, "error": "Empty query", "results": [], "query": ""}
    query = query.strip()

    cached = _cache_get(query, max_results)
    if cached is not None:
        return cached

    # ------------------------------------------------------------
    # Phase 1: decompose (optional)
    # ------------------------------------------------------------
    sub_queries: List[str] = [query]
    if decompose:
        active_llm = llm_func or _default_llm_func
        try:
            from aura.tools.search_planner import decompose_query
            sub_queries = decompose_query(query, llm_func=active_llm)
        except Exception as e:
            logger.debug("[SearchFallback] decompose failed: %s", e)
            sub_queries = [query]
        if not sub_queries:
            sub_queries = [query]

    per_query_results = max(4, (max_results // max(len(sub_queries), 1)) + 2)

    # ------------------------------------------------------------
    # Phase 2: parallel provider fan-out
    # ------------------------------------------------------------
    provider_calls = [
        ("tavily", _call_tavily),
        ("brave", _call_brave),
        ("firecrawl", _call_firecrawl),
    ]

    all_results: List[dict] = []
    max_parallel = min(len(sub_queries) * len(provider_calls), 9)

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        future_map = {}
        for sq in sub_queries:
            for name, fn in provider_calls:
                fut = pool.submit(fn, sq, per_query_results)
                future_map[fut] = (name, sq)

        for fut in as_completed(future_map, timeout=25):
            name, sq = future_map[fut]
            try:
                batch = fut.result(timeout=0)
                if batch:
                    all_results.extend(batch)
            except Exception as e:
                logger.debug("[SearchFallback] %s[%s] failed: %s", name, sq[:40], e)

    if not all_results:
        return {
            "success": False,
            "error": "All search providers failed or returned nothing",
            "results": [],
            "query": query,
            "sub_queries": sub_queries if len(sub_queries) > 1 else None,
        }

    # ------------------------------------------------------------
    # Phase 3: dedupe + rerank
    # ------------------------------------------------------------
    deduped = _deduplicate(all_results)
    ranked = _rerank(deduped, query)
    top = ranked[: max_results]

    # ------------------------------------------------------------
    # Phase 4: scrape top N for full content
    # ------------------------------------------------------------
    top = _scrape_top_urls(top, n=scrape_top_n)

    out = {
        "success": True,
        "query": query,
        "sub_queries": sub_queries if len(sub_queries) > 1 else None,
        "num_results": len(top),
        "total_raw": len(all_results),
        "results": top,
        "source": "hybrid",
    }
    _cache_put(query, max_results, out)
    return out


__all__ = ["web_search_with_fallback", "set_search_llm"]

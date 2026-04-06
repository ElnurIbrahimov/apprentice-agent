"""Shared web search fallback chain.

Provides a single function that tries Tavily → Brave → Firecrawl in order,
with retry logic, result normalization, and deduplication.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session-level query cache (5-minute TTL)
# ---------------------------------------------------------------------------
_search_cache: Dict[Tuple[str, int], Tuple[float, dict]] = {}
_CACHE_TTL = 300  # 5 minutes


def _cache_get(query: str, max_results: int):
    key = (query.lower().strip(), max_results)
    if key in _search_cache:
        ts, result = _search_cache[key]
        if time.time() - ts < _CACHE_TTL:
            logger.debug(f"[SearchFallback] Cache hit for '{key[0][:40]}…'")
            return result
        del _search_cache[key]
    return None


def _cache_put(query: str, max_results: int, result: dict):
    key = (query.lower().strip(), max_results)
    _search_cache[key] = (time.time(), result)


def _is_good_result(result) -> bool:
    """Check if a search result is valid and usable."""
    if not isinstance(result, dict):
        return False
    # Explicit success flag
    if result.get("success") is False:
        return False
    # Explicit error
    if "error" in result and result["error"]:
        return False
    # Has actual results
    results = result.get("results", [])
    if isinstance(results, list) and len(results) > 0:
        return True
    # Tavily returns "answer" sometimes
    if result.get("answer"):
        return True
    return False


def _normalize_results(result: dict, source: str) -> dict:
    """Normalize results to a consistent format: {success, query, source, results[{title, url, snippet}]}."""
    raw_results = result.get("results", [])
    if isinstance(raw_results, str):
        # Some tools return results as a string
        return {"success": True, "source": source, "results": [], "raw_text": raw_results}

    normalized = []
    for r in raw_results:
        if isinstance(r, dict):
            normalized.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet") or r.get("content") or r.get("description") or r.get("markdown", "")[:300] or "",
                "source": source,
            })

    return {
        "success": True,
        "query": result.get("query", ""),
        "source": source,
        "results": normalized,
        "num_results": len(normalized),
        # Preserve extra fields from Tavily
        "answer": result.get("answer"),
    }


def _deduplicate(results: List[dict]) -> List[dict]:
    """Remove duplicate results by URL."""
    seen_urls = set()
    deduped = []
    for r in results:
        url = r.get("url", "").rstrip("/").lower()
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(r)
        elif not url:
            deduped.append(r)  # Keep results without URLs
    return deduped


def _retry_call(fn, retries: int = 1, delay: float = 1.0):
    """Simple retry with exponential backoff."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(delay * (2 ** attempt))
                logger.debug(f"[SearchFallback] Retry {attempt + 1}/{retries} after: {e}")
    raise last_err  # type: ignore


# Lazy-initialized tool instances (module-level singletons)
_tavily = None
_brave = None
_firecrawl = None


def web_search_with_fallback(
    query: str,
    max_results: int = 8,
    tool_registry: Optional[dict] = None,
) -> dict:
    """Execute a web search with Tavily → Brave → Firecrawl fallback chain.

    Args:
        query: Search query string.
        max_results: Maximum results to return.
        tool_registry: Optional agent tool registry. If provided, uses
            registered tool instances instead of creating new ones.

    Returns:
        dict with normalized search results, or {"error": "..."} on total failure.
    """
    global _tavily, _brave, _firecrawl

    # --- Check cache first ---
    cached = _cache_get(query, max_results)
    if cached is not None:
        return cached

    errors: list[str] = []

    # --- Tavily (best quality, AI-optimized) ---
    try:
        if tool_registry and "tavily_search" in tool_registry:
            result = tool_registry["tavily_search"].execute(f"search {query}")
        else:
            if _tavily is None:
                from aura.tools.tavily_tool import TavilyTool
                _tavily = TavilyTool()
            result = _retry_call(lambda: _tavily.search(query=query, max_results=max_results))

        if _is_good_result(result):
            normalized = _normalize_results(result, "tavily")
            normalized["results"] = _deduplicate(normalized["results"])
            _cache_put(query, max_results, normalized)
            logger.debug(f"[SearchFallback] Tavily returned {len(normalized['results'])} results")
            return normalized
        errors.append(f"Tavily: {result.get('error', 'no results') if isinstance(result, dict) else 'bad response'}")
    except Exception as e:
        errors.append(f"Tavily: {e}")
        logger.debug(f"[SearchFallback] Tavily failed: {e}")

    # --- Brave (fresh results, good for news) ---
    try:
        if tool_registry and "brave_search" in tool_registry:
            result = tool_registry["brave_search"].execute(f"search {query}")
        else:
            if _brave is None:
                from aura.tools.brave_search import BraveSearchTool
                _brave = BraveSearchTool()
            result = _retry_call(lambda: _brave.run(query=query, count=max_results))

        if _is_good_result(result):
            normalized = _normalize_results(result, "brave")
            normalized["results"] = _deduplicate(normalized["results"])
            _cache_put(query, max_results, normalized)
            logger.debug(f"[SearchFallback] Brave returned {len(normalized['results'])} results")
            return normalized
        errors.append(f"Brave: {result.get('error', 'no results') if isinstance(result, dict) else 'bad response'}")
    except Exception as e:
        errors.append(f"Brave: {e}")
        logger.debug(f"[SearchFallback] Brave failed: {e}")

    # --- Firecrawl (search + scrape, last resort) ---
    try:
        if tool_registry and "firecrawl" in tool_registry:
            fc = tool_registry["firecrawl"]
            result = fc.search(query, limit=max_results) if hasattr(fc, 'search') else fc.execute(query)
        else:
            if _firecrawl is None:
                from aura.tools.firecrawl_tool import FirecrawlTool
                _firecrawl = FirecrawlTool()
            result = _retry_call(lambda: _firecrawl.search(query=query, limit=max_results))

        if _is_good_result(result):
            normalized = _normalize_results(result, "firecrawl")
            normalized["results"] = _deduplicate(normalized["results"])
            _cache_put(query, max_results, normalized)
            logger.debug(f"[SearchFallback] Firecrawl returned {len(normalized['results'])} results")
            return normalized
        errors.append(f"Firecrawl: {result.get('error', 'no results') if isinstance(result, dict) else 'bad response'}")
    except Exception as e:
        errors.append(f"Firecrawl: {e}")
        logger.debug(f"[SearchFallback] Firecrawl failed: {e}")

    # --- All providers failed ---
    error_summary = "; ".join(errors) if errors else "Unknown error"
    logger.error(f"[SearchFallback] All search providers failed: {error_summary}")
    return {"success": False, "error": f"All search providers failed: {error_summary}", "results": []}

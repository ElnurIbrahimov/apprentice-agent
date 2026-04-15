"""Query decomposition planner for web search.

Splits a complex, multi-faceted query into 3-5 focused sub-queries
so the downstream search pipeline can fan out in parallel and cover
different angles (same pattern ChatGPT/Claude/Perplexity use).

Skips decomposition for short/specific queries where it would add
latency without improving recall.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """You are a search query planner. Break the user's query into 3-5 focused sub-queries that cover different angles, aspects, or facets.

Rules:
- Each sub-query must be a concrete search engine query (no natural language).
- Cover distinct angles: definitions, comparisons, recent news, technical details, examples.
- Keep sub-queries short (3-8 words each).
- Do NOT add commentary or explanation.
- Return ONLY a JSON array of strings, nothing else.

Examples:

User query: best practices for microservices
JSON array: ["microservices architecture patterns", "microservices vs monolith tradeoffs", "microservices deployment kubernetes", "microservices testing strategies", "microservices observability tools"]

User query: compare postgres vs mysql for high-write workloads
JSON array: ["postgres high write performance benchmarks", "mysql write throughput tuning", "postgres vs mysql replication lag", "postgres mysql concurrent writes lock contention", "postgres mysql durability fsync settings"]

User query: how does retrieval augmented generation work
JSON array: ["RAG retrieval augmented generation overview", "RAG embedding models chunking strategies", "RAG vs fine-tuning comparison", "RAG hallucination reduction techniques", "RAG production deployment latency"]

User query: {query}

JSON array:"""


# Query classification -------------------------------------------------

_FACTUAL_STARTS = (
    "what is ", "what are ", "what's ", "who is ", "who was ", "who are ",
    "when did ", "when was ", "when is ", "where is ", "where was ",
    "define ", "meaning of ", "capital of ",
)

_COMPARISON_MARKERS = (" vs ", " versus ", " compared to ", " difference between ",
                      " or ", " better than ")
_LIST_STARTS = ("list of ", "top ", "best ", "examples of ")
_HOWTO_STARTS = ("how to ", "how do i ", "how can i ", "how does ")


def _classify_query(query: str) -> str:
    """Coarse query-type classifier — factual, comparison, list, howto, exploratory.

    Factual queries are answered by the first hit and don't benefit from fan-out,
    so the planner returns them unchanged without calling an LLM.
    """
    if not query:
        return "exploratory"
    q = query.strip().lower()
    if any(q.startswith(p) for p in _FACTUAL_STARTS):
        if len(q.split()) <= 12:
            return "factual"
    if any(m in q for m in _COMPARISON_MARKERS):
        return "comparison"
    if any(q.startswith(p) for p in _LIST_STARTS):
        return "list"
    if any(q.startswith(p) for p in _HOWTO_STARTS):
        return "howto"
    return "exploratory"


# Decomposition cache --------------------------------------------------

_CACHE_MAX_SIZE = 256
_cache: "OrderedDict[str, List[str]]" = OrderedDict()
_cache_lock = threading.Lock()
_cache_stats = {"hits": 0, "misses": 0}


def _cache_key(query: str) -> str:
    return hashlib_md5(query.strip().lower())


def hashlib_md5(s: str) -> str:
    import hashlib
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _cache_get(query: str) -> Optional[List[str]]:
    key = _cache_key(query)
    with _cache_lock:
        if key not in _cache:
            _cache_stats["misses"] += 1
            return None
        # LRU touch
        value = _cache.pop(key)
        _cache[key] = value
        _cache_stats["hits"] += 1
        return list(value)


def _cache_put(query: str, sub_queries: List[str]) -> None:
    key = _cache_key(query)
    with _cache_lock:
        _cache[key] = list(sub_queries)
        while len(_cache) > _CACHE_MAX_SIZE:
            _cache.popitem(last=False)


def get_planner_stats() -> dict:
    """Hit/miss counters for the decomposition cache."""
    total = _cache_stats["hits"] + _cache_stats["misses"]
    rate = _cache_stats["hits"] / total if total else 0.0
    return {**_cache_stats, "total": total, "hit_rate": round(rate, 3), "size": len(_cache)}


def clear_planner_cache() -> None:
    """Drop the LRU cache — exposed for tests and long-running processes."""
    with _cache_lock:
        _cache.clear()


# Original helpers -----------------------------------------------------


def _looks_specific(query: str) -> bool:
    """True when decomposition should be skipped — query is already focused."""
    q = query.strip()
    if not q:
        return True
    words = q.split()
    if len(words) < 6:
        return True
    if '"' in q:
        return True
    q_lower = q.lower()
    simple_starts = (
        "what is ", "who is ", "when did ", "when was ",
        "where is ", "how do i ", "define ", "what's ",
    )
    if any(q_lower.startswith(s) for s in simple_starts) and len(words) < 11:
        return True
    return False


def _extract_json_array(text: str) -> Optional[list]:
    """Pull the first JSON array of strings out of a messy LLM response."""
    if not text:
        return None
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text, flags=re.MULTILINE)
    match = re.search(r"\[[^\[\]]*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


def decompose_query(
    query: str,
    llm_func: Optional[Callable[..., str]] = None,
    max_subqueries: int = 5,
) -> List[str]:
    """Return a list of sub-queries for parallel search fan-out.

    Returns ``[query]`` unchanged when:
    - query is short/specific / classified as factual (no LLM call made)
    - no llm_func provided
    - LLM call or JSON parse fails (graceful fallback)

    The first element is always the original query so the fan-out
    covers the literal user intent even if the LLM rewrite is weird.
    Results are cached in-process (LRU, size 256) so repeat queries
    across sessions in a long-running agent skip the LLM entirely.
    """
    if not query or not query.strip():
        return []
    query = query.strip()

    # Fast-path: short/specific or factual queries don't benefit from fan-out.
    if _looks_specific(query):
        return [query]
    if _classify_query(query) == "factual":
        logger.debug("[SearchPlanner] factual query, skipping decomposition: %r", query)
        return [query]
    if llm_func is None:
        return [query]

    # Cache: stable per (trim+lower) query hash.
    cached = _cache_get(query)
    if cached is not None:
        logger.debug("[SearchPlanner] cache hit for %r", query)
        return cached

    prompt = _DECOMPOSE_PROMPT.format(query=query)
    raw: Optional[str] = None
    for kwargs in ({"use_history": False}, {}):
        try:
            raw = llm_func(prompt, **kwargs)
            break
        except TypeError:
            continue
        except Exception as exc:
            logger.debug("[SearchPlanner] LLM call failed: %s", exc)
            return [query]
    if not raw:
        return [query]

    parsed = _extract_json_array(str(raw))
    if not parsed:
        logger.debug("[SearchPlanner] no JSON array in LLM response")
        return [query]

    cleaned: List[str] = []
    for item in parsed:
        if isinstance(item, str) and item.strip():
            s = item.strip()
            if s not in cleaned:
                cleaned.append(s)

    if not cleaned:
        return [query]

    if query not in cleaned:
        cleaned.insert(0, query)

    result = cleaned[: max(1, max_subqueries)]
    _cache_put(query, result)
    return result


__all__ = ["decompose_query", "get_planner_stats", "clear_planner_cache"]

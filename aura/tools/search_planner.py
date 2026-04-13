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
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

_DECOMPOSE_PROMPT = """You are a search query planner. Break the user's query into 3-5 focused sub-queries that cover different angles, aspects, or facets.

Rules:
- Each sub-query must be a concrete search engine query (no natural language).
- Cover distinct angles: definitions, comparisons, recent news, technical details, examples.
- Keep sub-queries short (3-8 words each).
- Do NOT add commentary or explanation.
- Return ONLY a JSON array of strings, nothing else.

User query: {query}

JSON array:"""


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
    - query is short/specific (no LLM call made)
    - no llm_func provided
    - LLM call or JSON parse fails (graceful fallback)

    The first element is always the original query so the fan-out
    covers the literal user intent even if the LLM rewrite is weird.
    """
    if not query or not query.strip():
        return []
    query = query.strip()

    if _looks_specific(query):
        return [query]
    if llm_func is None:
        return [query]

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

    return cleaned[: max(1, max_subqueries)]


__all__ = ["decompose_query"]

"""Shared web search fallback chain.

Provides a single function that tries Tavily → Brave → SearXNG in order.
Used by both agent.py (tool dispatch) and agentic_loop.py (direct execution).
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-initialized tool instances (module-level singletons)
_tavily = None
_brave = None


def web_search_with_fallback(
    query: str,
    max_results: int = 8,
    tool_registry: Optional[dict] = None,
) -> dict:
    """Execute a web search with Tavily → Brave → SearXNG fallback chain.

    Args:
        query: Search query string.
        max_results: Maximum results to return.
        tool_registry: Optional agent tool registry. If provided, uses
            registered tool instances instead of creating new ones.

    Returns:
        dict with search results, or {"error": "..."} on total failure.
    """
    global _tavily, _brave

    # --- Strategy 1: Use tool registry if available ---
    if tool_registry:
        for tool_name in ("tavily_search", "brave_search", "web_search"):
            if tool_name in tool_registry:
                try:
                    result = tool_registry[tool_name].execute(f"search {query}")
                    if isinstance(result, dict) and "error" not in result:
                        return result
                    # Some tools return error as a string
                    if isinstance(result, str) and "error" not in result.lower()[:50]:
                        return {"results": result}
                except Exception as e:
                    logger.debug(f"[SearchFallback] {tool_name} failed: {e}")
                    continue
        # Fall through to direct instantiation if registry tools all failed

    # --- Strategy 2: Direct instantiation ---
    # Tavily
    try:
        if _tavily is None:
            from aura.tools.tavily_tool import TavilyTool
            _tavily = TavilyTool()
        result = _tavily.search(query=query, max_results=max_results)
        if isinstance(result, dict) and "error" not in result:
            return result
        logger.debug(f"[SearchFallback] Tavily error: {result.get('error', '?')}")
    except Exception as e:
        logger.debug(f"[SearchFallback] Tavily exception: {e}")

    # Brave
    try:
        if _brave is None:
            from aura.tools.brave_search import BraveSearchTool
            _brave = BraveSearchTool()
        result = _brave.run(query=query, count=max_results)
        if isinstance(result, dict) and "error" not in result:
            return result
        if isinstance(result, dict):
            logger.debug(f"[SearchFallback] Brave error: {result.get('error', '?')}")
    except Exception as e:
        logger.debug(f"[SearchFallback] Brave exception: {e}")

    # SearXNG (final fallback)
    try:
        from aura.tools.web_search import WebSearchTool
        ws = WebSearchTool()
        return ws.search(query=query, num_results=max_results)
    except Exception as e:
        return {"error": f"All search providers failed: {e}"}

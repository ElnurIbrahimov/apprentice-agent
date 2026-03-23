"""tool_search — discover available deferred tools by name or keyword.

Follows the standard Aura tool pattern (name, description, execute).
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ToolSearchTool:
    """Search for available tools that aren't loaded by default.

    Query modes:
      ``select:name1,name2`` — exact lookup by tool name
      ``+keyword other terms`` — name must contain keyword, ranked by terms
      ``keyword query`` — regex search across name + description
    """

    name = "tool_search"
    description = (
        "Search for available tools by name or keyword. "
        "Use 'select:name1,name2' for exact lookup, '+keyword terms' to filter by name, "
        "or free-text keywords to search."
    )

    def execute(self, action: str, **kwargs) -> Dict:
        """Execute a tool search query.

        Args:
            action: The search query string.

        Returns:
            dict with 'results' list and 'count'.
        """
        from aura.tools.deferred_registry import deferred_registry

        query = action.strip() if action else ""

        if not query or query.lower() in ("list", "list_all", "all"):
            results = deferred_registry.list_all()
            return {
                "status": "success",
                "action": "list_all",
                "count": len(results),
                "results": results,
                "hint": "Use 'select:tool_name' to get details, or ask me to use any tool by name.",
            }

        results = deferred_registry.search(query)

        if not results:
            return {
                "status": "no_results",
                "query": query,
                "count": 0,
                "results": [],
                "hint": f"No deferred tools matched '{query}'. Try broader keywords or 'list' to see all.",
            }

        return {
            "status": "success",
            "query": query,
            "count": len(results),
            "results": results,
        }

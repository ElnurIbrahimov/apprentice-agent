"""Web search tool using DuckDuckGo."""

from typing import Optional
from ddgs import DDGS


class WebSearchTool:
    """Tool for web searching using DuckDuckGo."""

    name = "web_search"
    description = "Search the web for information using DuckDuckGo"

    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query: str, max_results: int = 5) -> dict:
        """Perform a web search and return results."""
        try:
            results = list(self.ddgs.text(query, max_results=max_results))
            return {
                "success": True,
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    }
                    for r in results
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    def news(self, query: str, max_results: int = 5) -> dict:
        """Search for news articles."""
        try:
            results = list(self.ddgs.news(query, max_results=max_results))
            return {
                "success": True,
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("body", ""),
                        "date": r.get("date", ""),
                        "source": r.get("source", "")
                    }
                    for r in results
                ]
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    def instant_answer(self, query: str) -> dict:
        """Get an instant answer if available."""
        try:
            results = list(self.ddgs.answers(query))
            if results:
                return {
                    "success": True,
                    "query": query,
                    "answer": results[0].get("text", ""),
                    "source": results[0].get("url", "")
                }
            return {
                "success": True,
                "query": query,
                "answer": None,
                "message": "No instant answer available"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a web search action by name."""
        actions = {
            "search": self.search,
            "news": self.news,
            "answer": self.instant_answer
        }
        if action not in actions:
            return {"success": False, "error": f"Unknown action: {action}"}
        return actions[action](**kwargs)

"""Web search using SearXNG."""

import requests
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Web search using SearXNG."""

    name = "web_search"
    description = "Search the web using SearXNG"

    # Primary instance (known working)
    PRIMARY_INSTANCE = "https://serxng-deployment-production.up.railway.app"

    # Fallback instances
    FALLBACK_INSTANCES = [
        "https://searx.be",
        "https://search.sapti.me",
    ]

    def __init__(self):
        self.timeout = 15

    def search(self, query: str, num_results: int = 10, categories: str = "general") -> Dict:
        """
        Search using SearXNG.

        Args:
            query: Search query
            num_results: Number of results to return
            categories: Search categories (general, news, images)

        Returns:
            Dict with success status and results
        """
        logger.info(f"[SEARXNG] Searching: {query}")

        # Try primary instance first, then fallbacks
        instances = [self.PRIMARY_INSTANCE] + self.FALLBACK_INSTANCES

        for instance in instances:
            try:
                response = requests.get(
                    f"{instance}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "categories": categories,
                    },
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "application/json",
                    },
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])[:num_results]

                    formatted = [{
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", ""),
                        "engine": r.get("engine", "searxng"),
                    } for r in results]

                    if formatted:
                        logger.info(f"[SEARXNG] Found {len(formatted)} results from {instance}")
                        return {
                            "success": True,
                            "query": query,
                            "source": instance,
                            "results": formatted,
                            "num_results": len(formatted),
                        }

                logger.warning(f"[SEARXNG] {instance} returned {response.status_code}")

            except requests.Timeout:
                logger.warning(f"[SEARXNG] {instance} timed out")
            except requests.RequestException as e:
                logger.warning(f"[SEARXNG] {instance} error: {e}")

        return {
            "success": False,
            "error": "All SearXNG instances failed.",
            "query": query,
        }

    def news(self, query: str, num_results: int = 10) -> Dict:
        """Search news."""
        return self.search(query, num_results, categories="news")

    def images(self, query: str, num_results: int = 10) -> Dict:
        """Search images."""
        return self.search(query, num_results, categories="images")

    def instant_answer(self, query: str) -> Dict:
        """Get instant answer."""
        result = self.search(query, num_results=3)
        if result.get("success") and result.get("results"):
            first = result["results"][0]
            return {
                "success": True,
                "query": query,
                "answer": first.get("snippet", ""),
                "source": first.get("url", ""),
            }
        return result

    def run(self, query: str) -> Dict:
        """Main entry point."""
        return self.search(query)


def web_search(query: str, num_results: int = 10) -> Dict:
    """Search the web using SearXNG."""
    tool = WebSearchTool()
    return tool.search(query, num_results)

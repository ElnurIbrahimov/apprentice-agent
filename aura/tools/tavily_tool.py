"""Tavily Search & Extract tool for AURA — AI-optimized search with full content."""

import logging
import os
import requests
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


class TavilyTool:
    name = "tavily_search"
    description = (
        "Search the web using Tavily (AI-optimized). "
        "Use search_depth='advanced' for deep research. "
        "Can also extract clean content from a URL."
    )

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("TAVILY_API_KEY", "")

    def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 8,
        include_answer: bool = True,
        include_raw_content: bool = False,
        topic: str = "general",
    ) -> Dict:
        """Search with Tavily. search_depth: 'basic' (fast) or 'advanced' (thorough)."""
        if not self._api_key:
            return {"error": "TAVILY_API_KEY not set"}

        payload = {
            "api_key": self._api_key,
            "query": query.strip(),
            "search_depth": search_depth,
            "max_results": min(max_results, 20),
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "topic": topic,
        }

        try:
            resp = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"[Tavily] Search failed: {e}")
            return {"error": str(e)}

        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
            })

        return {
            "query": query,
            "answer": data.get("answer", ""),
            "results": results,
            "total": len(results),
        }

    def extract(self, urls: List[str]) -> Dict:
        """Extract clean content from one or more URLs."""
        if not self._api_key:
            return {"error": "TAVILY_API_KEY not set"}

        payload = {
            "api_key": self._api_key,
            "urls": urls[:5],  # API limit
        }

        try:
            resp = requests.post(TAVILY_EXTRACT_URL, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"[Tavily] Extract failed: {e}")
            return {"error": str(e)}

    def run(self, query: str, deep: bool = False) -> Dict:
        return self.search(query, search_depth="advanced" if deep else "basic")

    def __call__(self, query: str, deep: bool = False) -> Dict:
        return self.run(query, deep)

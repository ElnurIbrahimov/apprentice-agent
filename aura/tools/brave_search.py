"""Brave Search tool for AURA — uses Brave Search API."""

import logging
import os
import requests
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchTool:
    name = "brave_search"
    description = "Search the web using Brave Search. Returns fresh, unfiltered results."

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("BRAVE_API_KEY", "")

    def run(self, query: str, count: int = 10, country: str = "US") -> Dict:
        if not self._api_key:
            return {"error": "BRAVE_API_KEY not set"}
        if not query or not query.strip():
            return {"error": "Empty query"}

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params = {
            "q": query.strip(),
            "count": min(count, 20),
            "country": country,
            "search_lang": "en",
            "safesearch": "moderate",
        }

        try:
            resp = requests.get(BRAVE_API_URL, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"[BraveSearch] Request failed: {e}")
            return {"error": str(e)}

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
            })

        return {
            "query": query,
            "results": results,
            "total": len(results),
        }

    def __call__(self, query: str, count: int = 10) -> Dict:
        return self.run(query, count)

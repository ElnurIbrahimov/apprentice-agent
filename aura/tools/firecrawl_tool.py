"""Firecrawl tool for AURA — scrape, crawl, and search the web with clean markdown output."""

import logging
import os
from typing import Dict, List, Optional

import requests

from aura.security.ssrf_guard import validate_url_safe

logger = logging.getLogger(__name__)

FIRECRAWL_BASE = "https://api.firecrawl.dev/v1"


class FirecrawlTool:
    name = "firecrawl"
    description = (
        "Scrape a URL or search/crawl the web using Firecrawl. "
        "Returns clean markdown content. Great for reading articles, docs, or any webpage."
    )

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "")
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def scrape(self, url: str, formats: Optional[List[str]] = None) -> Dict:
        """Scrape a single URL and return clean markdown."""
        if not self._api_key:
            return {"error": "FIRECRAWL_API_KEY not set"}

        # SSRF guard — refuse private IPs, loopback, link-local, DNS-rebinding,
        # and non-http(s) schemes before we hand the URL to Firecrawl. Even
        # though Firecrawl fetches server-side, a redirect bug on their side
        # could otherwise become our SSRF.
        try:
            validate_url_safe(url)
        except ValueError as e:
            logger.warning(f"[Firecrawl] scrape refused for {url}: {e}")
            return {"error": f"URL rejected by SSRF guard: {e}"}

        payload = {
            "url": url,
            "formats": formats or ["markdown"],
        }

        try:
            resp = requests.post(
                f"{FIRECRAWL_BASE}/scrape",
                headers=self._headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"[Firecrawl] Scrape failed for {url}: {e}")
            return {"error": str(e)}

        result = data.get("data", {})
        return {
            "url": url,
            "markdown": result.get("markdown", ""),
            "title": result.get("metadata", {}).get("title", ""),
            "description": result.get("metadata", {}).get("description", ""),
        }

    def search(self, query: str, limit: int = 5) -> Dict:
        """Search the web and return scraped content for each result."""
        if not self._api_key:
            return {"error": "FIRECRAWL_API_KEY not set"}

        payload = {
            "query": query.strip(),
            "limit": min(limit, 10),
        }

        try:
            resp = requests.post(
                f"{FIRECRAWL_BASE}/search",
                headers=self._headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"[Firecrawl] Search failed: {e}")
            return {"error": str(e)}

        results = []
        for item in data.get("data", []):
            results.append({
                "title": item.get("metadata", {}).get("title", ""),
                "url": item.get("url", ""),
                "markdown": item.get("markdown", "")[:2000],  # trim long pages
            })

        return {
            "query": query,
            "results": results,
            "total": len(results),
        }

    def run(self, query_or_url: str) -> Dict:
        """Auto-detect: if it looks like a URL, scrape it. Otherwise search."""
        if query_or_url.startswith("http://") or query_or_url.startswith("https://"):
            return self.scrape(query_or_url)
        return self.search(query_or_url)

    def __call__(self, query_or_url: str) -> Dict:
        return self.run(query_or_url)

"""Web search using SearXNG with rate limiting and validation.

SECURITY: Rate limited to prevent abuse and DDoS.
"""

import os
import requests
import logging
import threading
import time
from typing import Dict, List
from functools import wraps

logger = logging.getLogger(__name__)

# Try to import tenacity for retry logic
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logger.warning("[WEB_SEARCH] tenacity not available, using basic retry")

# Try to import validation
try:
    from .validation import validate_query, validate_int, sanitize_for_log, MAX_QUERY_LENGTH
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    MAX_QUERY_LENGTH = 1000


# ============================================================================
#                    RATE LIMITING
# ============================================================================

class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute  # seconds between calls
        self.last_call_time = 0.0
        self._lock = threading.Lock()

    def wait_if_needed(self):
        """Block until rate limit allows another call."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug(f"[RATE_LIMIT] Sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            self.last_call_time = time.time()


# Global rate limiter (30 calls/minute = 1 call every 2 seconds)
_rate_limiter = RateLimiter(calls_per_minute=30)


def rate_limited(func):
    """Decorator to apply rate limiting."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        _rate_limiter.wait_if_needed()
        return func(*args, **kwargs)
    return wrapper


# ============================================================================
#                    RETRY LOGIC
# ============================================================================

def basic_retry(func, max_attempts=3, base_delay=1.0):
    """Basic retry with exponential backoff (fallback if tenacity unavailable)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except (requests.Timeout, requests.ConnectionError) as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"[RETRY] Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
        raise last_exception
    return wrapper


class WebSearchTool:
    """Web search using SearXNG with rate limiting and validation."""

    name = "web_search"
    description = "Search the web using SearXNG"

    # Primary instance - local SearXNG (fastest, no rate limits)
    # Configurable via SEARXNG_URL env var or Config.SEARXNG_URL
    PRIMARY_INSTANCE = os.environ.get("SEARXNG_URL", "http://localhost:8888")

    # Fallback instances (if local is down)
    FALLBACK_INSTANCES = [
        "https://serxng-deployment-production.up.railway.app",
        "https://searx.be",
        "https://search.sapti.me",
    ]

    def __init__(self):
        self.timeout = 15
        self.max_query_length = MAX_QUERY_LENGTH
        self.max_results = 100

    def search(self, query: str, num_results: int = 10, categories: str = "general") -> Dict:
        """
        Search using SearXNG with rate limiting and validation.

        Args:
            query: Search query (max 1000 chars)
            num_results: Number of results to return (1-100)
            categories: Search categories (general, news, images)

        Returns:
            Dict with success status and results
        """
        # === INPUT VALIDATION ===
        try:
            if VALIDATION_AVAILABLE:
                query = validate_query(query)
                num_results = validate_int(num_results, "num_results", min_val=1, max_val=self.max_results)
            else:
                # Basic validation fallback
                if not query or not isinstance(query, str):
                    return {"success": False, "error": "Query must be a non-empty string"}
                query = query.strip()[:self.max_query_length]
                num_results = max(1, min(int(num_results), self.max_results))
        except ValueError as e:
            return {"success": False, "error": f"Validation error: {e}", "blocked_by": "input_validation"}

        # Validate categories
        valid_categories = {"general", "news", "images", "videos", "music", "files", "it", "science", "social media"}
        if categories not in valid_categories:
            categories = "general"

        # Log sanitized query
        if VALIDATION_AVAILABLE:
            logger.info(f"[SEARXNG] Searching: {sanitize_for_log(query, 50)}")
        else:
            logger.info(f"[SEARXNG] Searching: {query[:50]}...")

        # Try primary instance first, then fallbacks
        instances = [self.PRIMARY_INSTANCE] + self.FALLBACK_INSTANCES

        for instance in instances:
            try:
                result = self._search_instance(instance, query, num_results, categories)
                if result.get("success"):
                    return result
            except Exception as e:
                logger.warning(f"[SEARXNG] {instance} error: {e}")
                continue

        return {
            "success": False,
            "error": "All SearXNG instances failed.",
            "query": query,
        }

    @rate_limited
    def _search_instance(self, instance: str, query: str, num_results: int, categories: str) -> Dict:
        """Search a single instance with rate limiting."""
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

        if response.status_code == 429:
            # Rate limited by server
            logger.warning(f"[SEARXNG] {instance} rate limited (429)")
            raise requests.RequestException("Rate limited")

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
        return {"success": False}

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

"""Web search using SearXNG with rate limiting and validation.

SECURITY: Rate limited to prevent abuse and DDoS.
"""

import logging
import math
import os
import re
import threading
import time
from functools import wraps
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

# Try to import tenacity for retry logic
try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    logger.warning("[WEB_SEARCH] tenacity not available, using basic retry")

# Try to import validation
try:
    from .validation import MAX_QUERY_LENGTH, sanitize_for_log, validate_int, validate_query
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
        sleep_time = 0.0
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                # Reserve the slot now so other threads compute their wait correctly
                self.last_call_time = now + sleep_time
            else:
                self.last_call_time = now
        # Sleep OUTSIDE the lock so other threads aren't blocked
        if sleep_time > 0:
            logger.debug(f"[RATE_LIMIT] Sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)


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


# ============================================================================
#                    BM25 RERANKING
# ============================================================================

_BM25_K1 = 1.2
_BM25_B = 0.75
_SPLIT_RE = re.compile(r'\W+')


def _rerank_results(results: List[Dict], query: str) -> List[Dict]:
    """Rerank search results using BM25 scoring against the query.

    Scores each result's title + snippet against the query terms.
    IDF is computed from the result set itself (N = len(results)).
    """
    if not results or not query:
        return results

    query_terms = [t.lower() for t in _SPLIT_RE.split(query) if t]
    if not query_terms:
        return results

    N = len(results)

    # Build documents: title + snippet for each result
    docs = []
    for r in results:
        text = (r.get("title", "") + " " + r.get("snippet", "")).lower()
        tokens = [t for t in _SPLIT_RE.split(text) if t]
        docs.append(tokens)

    # Average document length
    avg_dl = sum(len(d) for d in docs) / max(N, 1)

    # Document frequency for each query term
    df = {}
    for term in query_terms:
        df[term] = sum(1 for d in docs if term in d)

    # Score each document
    scored = []
    for i, doc_tokens in enumerate(docs):
        dl = len(doc_tokens)
        score = 0.0
        for term in query_terms:
            tf = doc_tokens.count(term)
            if tf == 0:
                continue
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            n = df.get(term, 0)
            idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
            # TF saturation
            tf_sat = (tf * (_BM25_K1 + 1)) / (tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(avg_dl, 1)))
            score += idf * tf_sat
        scored.append((score, i))

    # Sort by score descending, stable (preserves original order for ties)
    scored.sort(key=lambda x: -x[0])
    return [results[i] for _, i in scored]


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

        # SECURITY: Validate primary SearXNG URL at init time to prevent SSRF
        # via SEARXNG_URL env var pointing to internal services
        try:
            from aura.security.ssrf_guard import validate_url_safe
            validate_url_safe(self.PRIMARY_INSTANCE)
        except (ValueError, ImportError) as e:
            logger.warning(f"[SEARXNG] Primary instance failed SSRF validation ({e}), using fallbacks only")
            self.PRIMARY_INSTANCE = None  # Will skip to fallback instances

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

        # Try primary instance first, then fallbacks (skip None if SSRF-blocked)
        instances = ([self.PRIMARY_INSTANCE] if self.PRIMARY_INSTANCE else []) + self.FALLBACK_INSTANCES

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

            # Rerank by BM25 relevance to query
            formatted = _rerank_results(formatted, query)

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

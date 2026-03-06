"""Deep research tool - searches multiple queries, reads pages, synthesizes report."""

import logging
import time
from typing import Dict, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)

# Timeout constants
OVERALL_TIMEOUT = 60  # Total time allowed for research
PAGE_FETCH_TIMEOUT = 10  # Time allowed per page fetch
SEARCH_TIMEOUT = 15  # Time allowed per search query


class DeepResearchTool:
    """
    Deep research: searches multiple queries, reads full pages, synthesizes report.
    Has timeout protection to prevent hanging.
    """

    name = "deep_research"
    description = "Conduct deep research on a topic using multiple searches and page reads"

    def __init__(self):
        self.timeout = PAGE_FETCH_TIMEOUT
        self.overall_timeout = OVERALL_TIMEOUT
        self.searcher = None
        self.browser = None
        self._progress_callback: Optional[Callable[[str], None]] = None

        try:
            from .web_search import WebSearchTool
            self.searcher = WebSearchTool()
        except ImportError as e:
            logger.warning(f"WebSearchTool not available: {e}")

        try:
            from .browser import BrowserTool
            self.browser = BrowserTool()
        except ImportError as e:
            logger.warning(f"BrowserTool not available: {e}")

    def set_progress_callback(self, callback: Callable[[str], None]):
        """Set callback for progress updates (e.g., send to Telegram)."""
        self._progress_callback = callback

    def _log_progress(self, message: str):
        """Log progress and call callback if set."""
        logger.debug(f"[RESEARCH] {message}")
        logger.info(f"[RESEARCH] {message}")
        if self._progress_callback:
            try:
                self._progress_callback(message)
            except:
                pass

    def _generate_queries(self, topic: str) -> List[str]:
        """Generate multiple search angles."""
        return [
            topic,
            f"{topic} latest 2025",
            f"{topic} explained",
            f"{topic} analysis",
            f"what is {topic}",
        ]

    def _fetch_page(self, url: str) -> Dict:
        """Fetch full page content with timeout."""
        try:
            if self.browser and hasattr(self.browser, 'open'):
                # Navigate to the URL first, then extract text
                nav_result = self.browser.open(url)
                if not nav_result.get("success"):
                    return {"url": url, "content": "", "success": False}
                text_result = self.browser.get_text()
                if isinstance(text_result, dict):
                    content = text_result.get("text", "")
                else:
                    content = str(text_result) if text_result else ""
                return {"url": url, "content": content[:5000] if content else "", "success": bool(content)}
            elif self.browser and hasattr(self.browser, 'fetch'):
                result = self.browser.fetch(url)
                content = result.get("text", "") if isinstance(result, dict) else str(result)
                return {"url": url, "content": content[:5000], "success": True}
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
        return {"url": url, "content": "", "success": False}

    def research(self, topic: str, depth: str = "standard") -> Dict:
        """
        Conduct deep research with timeout protection.

        Args:
            topic: What to research
            depth: "quick" (5 pages), "standard" (10), "deep" (20)

        Returns:
            Dict with results (partial if timeout)
        """
        start_time = time.time()
        self._log_progress(f"Starting research: {topic}")

        if not self.searcher:
            return {"success": False, "error": "Search not available"}

        # Set depth
        max_pages = {"quick": 5, "standard": 10, "deep": 20}.get(depth, 10)

        # Track partial results
        all_urls = []
        pages = []
        queries_completed = 0
        timed_out = False

        # Step 1: Multiple searches (with timeout check)
        queries = self._generate_queries(topic)
        seen = set()

        for i, query in enumerate(queries):
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.overall_timeout:
                self._log_progress(f"Timeout during search phase ({elapsed:.1f}s)")
                timed_out = True
                break

            self._log_progress(f"Search {i+1}/{len(queries)}: {query[:40]}...")

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.searcher.search, query, 10)
                    try:
                        result = future.result(timeout=SEARCH_TIMEOUT)
                        if result.get("success"):
                            for r in result.get("results", []):
                                url = r.get("url", "")
                                if url and url not in seen:
                                    seen.add(url)
                                    all_urls.append({
                                        "url": url,
                                        "title": r.get("title", ""),
                                        "snippet": r.get("snippet", "")
                                    })
                        queries_completed += 1
                    except FuturesTimeoutError:
                        self._log_progress(f"Search {i+1} timed out, skipping")
            except Exception as e:
                self._log_progress(f"Search {i+1} failed: {e}")

        self._log_progress(f"Found {len(all_urls)} unique URLs from {queries_completed} searches")

        # Check timeout before page fetching
        elapsed = time.time() - start_time
        if elapsed > self.overall_timeout:
            timed_out = True

        # Step 2: Fetch pages in parallel (if browser available and time remaining)
        if self.browser and not timed_out and all_urls:
            remaining_time = self.overall_timeout - elapsed
            urls_to_fetch = all_urls[:max_pages]

            self._log_progress(f"Fetching {len(urls_to_fetch)} pages ({remaining_time:.0f}s remaining)...")

            try:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = {executor.submit(self._fetch_page, u["url"]): u for u in urls_to_fetch}

                    for i, future in enumerate(as_completed(futures, timeout=remaining_time)):
                        # Check overall timeout
                        if time.time() - start_time > self.overall_timeout:
                            self._log_progress("Timeout during page fetch, returning partial results")
                            timed_out = True
                            break

                        try:
                            result = future.result(timeout=PAGE_FETCH_TIMEOUT)
                            if result.get("success"):
                                pages.append(result)
                                self._log_progress(f"Fetched page {len(pages)}/{len(urls_to_fetch)}")
                        except FuturesTimeoutError:
                            self._log_progress(f"Page fetch timed out")
                        except Exception as e:
                            logger.warning(f"Page fetch error: {e}")

            except FuturesTimeoutError:
                self._log_progress("Page fetching timed out, using partial results")
                timed_out = True
            except Exception as e:
                self._log_progress(f"Page fetching error: {e}")

        # Step 3: Build report
        elapsed = time.time() - start_time
        self._log_progress(f"Research complete in {elapsed:.1f}s")

        # Compile content
        if pages:
            content_summary = "\n\n".join([f"## {p['url']}\n{p['content'][:1000]}" for p in pages[:5]])
        else:
            # Use snippets if no pages fetched
            snippets = [f"- {u['title']}: {u['snippet']}" for u in all_urls[:max_pages] if u.get('snippet')]
            content_summary = "\n".join(snippets)

        return {
            "success": True,
            "topic": topic,
            "depth": depth,
            "queries_run": queries_completed,
            "urls_found": len(all_urls),
            "pages_read": len(pages),
            "time_seconds": round(elapsed, 1),
            "timed_out": timed_out,
            "sources": [{"url": u["url"], "title": u["title"], "snippet": u["snippet"][:200]} for u in all_urls[:max_pages]],
            "content": content_summary,
            "summary": f"Researched '{topic}': {queries_completed} searches, {len(all_urls)} sources, {len(pages)} pages read in {elapsed:.1f}s" + (" (partial - timed out)" if timed_out else "")
        }

    def run(self, query: str) -> Dict:
        """Main entry point with timeout protection."""
        start_time = time.time()

        # Check for depth hints
        query_lower = query.lower()
        if "deep" in query_lower or "thorough" in query_lower:
            depth = "deep"
        elif "quick" in query_lower or "fast" in query_lower:
            depth = "quick"
        else:
            depth = "standard"

        # Clean topic
        topic = query_lower.replace("deep research", "").replace("research", "").strip()
        if not topic:
            topic = query

        # Run with overall timeout protection
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.research, topic, depth)
                try:
                    return future.result(timeout=OVERALL_TIMEOUT + 5)  # Small buffer
                except FuturesTimeoutError:
                    elapsed = time.time() - start_time
                    return {
                        "success": False,
                        "error": f"Research timed out after {elapsed:.1f}s",
                        "topic": topic,
                        "timed_out": True
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "topic": topic
            }


# Convenience function
def deep_research(topic: str, depth: str = "standard") -> Dict:
    """Conduct deep research on a topic."""
    tool = DeepResearchTool()
    return tool.research(topic, depth)

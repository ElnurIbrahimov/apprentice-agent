"""Deep research tool — multi-phase search, rank, fetch, summarize pipeline.

SOTA upgrades over original:
- 4-phase architecture (broad search → gap analysis → targeted fill → synthesis)
- ResultRanker: BM25-inspired + domain authority + snippet quality scoring
- ResearchCache: SQLite-backed caching for search results and page content
- HierarchicalSummarizer: per-page LLM summaries → cross-source synthesis
- Search backend priority: Tavily (if available) → SearXNG fallback
- Optional LLM integration with full graceful degradation
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Timeout constants
# ---------------------------------------------------------------------------
OVERALL_TIMEOUT = 60   # Total time allowed for research
PAGE_FETCH_TIMEOUT = 10  # Time allowed per page fetch
SEARCH_TIMEOUT = 15    # Time allowed per search query

# Phase budget fractions (of OVERALL_TIMEOUT)
PHASE_BUDGET = {
    "broad":     0.40,
    "gap":       0.20,
    "targeted":  0.30,
    "synthesis": 0.10,
}


# ============================================================================
#  ResultRanker — scores search results before page fetching
# ============================================================================

class ResultRanker:
    """Scores and ranks search results by relevance before fetching pages."""

    AUTHORITY_TIERS = {
        # Tier 1: Academic / authoritative (1.0)
        'arxiv.org': 1.0, 'wikipedia.org': 1.0, 'nature.com': 1.0,
        'science.org': 1.0, 'ieee.org': 1.0, 'acm.org': 1.0,
        'nih.gov': 1.0, 'pubmed.ncbi.nlm.nih.gov': 1.0,
        'scholar.google.com': 0.95, 'semanticscholar.org': 0.95,
        'gov': 0.9, 'edu': 0.9,
        # Tier 2: Tech / quality (0.7-0.8)
        'github.com': 0.7, 'stackoverflow.com': 0.7, 'docs.python.org': 0.8,
        'developer.mozilla.org': 0.8, 'huggingface.co': 0.7,
        'pytorch.org': 0.75, 'tensorflow.org': 0.75,
        # Tier 3: General (0.4-0.5)
        'medium.com': 0.5, 'towardsdatascience.com': 0.5, 'dev.to': 0.5,
        'blog': 0.4,
        # Tier 4: Social / UGC (0.2-0.3)
        'reddit.com': 0.3, 'quora.com': 0.3, 'twitter.com': 0.2,
        'x.com': 0.2,
    }

    # ---- scoring components ----

    def _bm25_score(self, text: str, query: str) -> float:
        """Simple BM25-inspired keyword overlap score (no external deps)."""
        if not text or not query:
            return 0.0
        text_lower = text.lower()
        query_terms = query.lower().split()
        if not query_terms:
            return 0.0

        k1 = 1.2
        score = 0.0
        text_tokens = text_lower.split()
        text_len = len(text_tokens) + 1  # avoid div-by-zero

        for term in query_terms:
            tf = text_lower.count(term)
            # BM25 saturation: tf / (tf + k1)
            score += tf / (tf + k1) if tf > 0 else 0.0

        return min(score / len(query_terms), 1.0)

    def _snippet_quality(self, snippet: str) -> float:
        """Score snippet by length and specificity."""
        if not snippet:
            return 0.0
        length = len(snippet.strip())
        # Length score — caps at 200 chars
        length_score = min(length / 200.0, 1.0)
        if length < 20:
            length_score *= 0.3  # heavy penalty for very short

        # Specificity bonus: numbers, dates, technical terms
        specificity = 0.0
        if re.search(r'\d{4}', snippet):  # year-like
            specificity += 0.1
        if re.search(r'\d+\.?\d*%', snippet):  # percentage
            specificity += 0.1
        if re.search(r'\d+\.\d+', snippet):  # decimal number
            specificity += 0.05
        if any(w in snippet.lower() for w in ['study', 'research', 'paper', 'found', 'results', 'analysis']):
            specificity += 0.1

        return min(length_score + specificity, 1.0)

    def _domain_authority(self, url: str) -> float:
        """Tiered domain scoring."""
        if not url:
            return 0.4
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
        except Exception:
            return 0.4

        # Direct hostname match
        for domain, score in self.AUTHORITY_TIERS.items():
            if domain in hostname:
                return score

        # TLD fallback
        tld = hostname.rsplit('.', 1)[-1] if '.' in hostname else ''
        if tld in ('edu', 'gov', 'ac'):
            return self.AUTHORITY_TIERS.get(tld, 0.9)

        return 0.4  # unknown default

    # ---- main entry ----

    def rank_results(self, results: List[Dict], topic: str) -> List[Dict]:
        """
        Composite ranking: 0.4*bm25 + 0.3*snippet_quality + 0.3*domain_authority.
        Returns sorted list (highest first) with '_rank_score' attached.
        """
        scored = []
        for r in results:
            text = f"{r.get('title', '')} {r.get('snippet', '')}"
            bm25 = self._bm25_score(text, topic)
            snip = self._snippet_quality(r.get('snippet', ''))
            auth = self._domain_authority(r.get('url', ''))
            composite = 0.4 * bm25 + 0.3 * snip + 0.3 * auth
            entry = dict(r)
            entry['_rank_score'] = round(composite, 4)
            scored.append(entry)

        scored.sort(key=lambda x: x['_rank_score'], reverse=True)
        return scored


# ============================================================================
#  ResearchCache — SQLite cache for search results and page content
# ============================================================================

class ResearchCache:
    """SQLite cache for search results and page content.  Best-effort — all
    operations are wrapped in try/except so a DB failure never breaks research."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or str(Path("data") / "research_cache.db")
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        try:
            os.makedirs(os.path.dirname(self._db_path) or '.', exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    results_json TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS page_cache (
                    url_hash TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            self._conn.commit()
        except Exception as e:
            logger.debug(f"[ResearchCache] init failed (non-critical): {e}")
            self._conn = None

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()

    # ---- search cache ----

    def get_search(self, query: str) -> Optional[List[Dict]]:
        if not self._conn:
            return None
        try:
            row = self._conn.execute(
                "SELECT results_json FROM search_cache WHERE query_hash = ? AND expires_at > ?",
                (self._hash(query), time.time())
            ).fetchone()
            return json.loads(row[0]) if row else None
        except Exception:
            return None

    def set_search(self, query: str, results: List[Dict], ttl_hours: int = 24):
        if not self._conn:
            return
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO search_cache (query_hash, results_json, expires_at) VALUES (?, ?, ?)",
                (self._hash(query), json.dumps(results, default=str), time.time() + ttl_hours * 3600)
            )
            self._conn.commit()
        except Exception:
            pass

    # ---- page cache ----

    def get_page(self, url: str) -> Optional[str]:
        if not self._conn:
            return None
        try:
            row = self._conn.execute(
                "SELECT content FROM page_cache WHERE url_hash = ? AND expires_at > ?",
                (self._hash(url), time.time())
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def set_page(self, url: str, content: str, ttl_days: int = 7):
        if not self._conn:
            return
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO page_cache (url_hash, content, expires_at) VALUES (?, ?, ?)",
                (self._hash(url), content, time.time() + ttl_days * 86400)
            )
            self._conn.commit()
        except Exception:
            pass

    # ---- maintenance ----

    def cleanup(self):
        if not self._conn:
            return
        try:
            now = time.time()
            self._conn.execute("DELETE FROM search_cache WHERE expires_at < ?", (now,))
            self._conn.execute("DELETE FROM page_cache WHERE expires_at < ?", (now,))
            self._conn.commit()
        except Exception:
            pass


# ============================================================================
#  HierarchicalSummarizer — multi-stage summarization with optional LLM
# ============================================================================

class HierarchicalSummarizer:
    """Multi-stage summarization.  Uses LLM when available, degrades to
    simple truncation / concatenation otherwise."""

    def __init__(self, llm_func: Optional[Callable] = None):
        self.llm = llm_func

    def _call_llm(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """Safe LLM call — returns None on any failure."""
        if not self.llm:
            return None
        try:
            return self.llm(prompt, system_prompt=system_prompt)
        except Exception as e:
            logger.debug(f"[Summarizer] LLM call failed: {e}")
            return None

    # ---- per-page ----

    def summarize_page(self, url: str, content: str, topic: str) -> str:
        """Per-page summary (~200 words).  Falls back to truncation if no LLM."""
        if not content:
            return ""

        result = self._call_llm(
            prompt=(
                f"Summarize the following web page content in ~200 words, "
                f"focusing on information relevant to: {topic}\n\n"
                f"URL: {url}\n"
                f"Content:\n{content[:6000]}"
            ),
            system_prompt="You are a research assistant. Be concise, factual, and specific."
        )
        if result:
            return result.strip()

        # Fallback: first 500 chars, trimmed to last sentence boundary
        fallback = content[:500]
        last_period = fallback.rfind('.')
        if last_period > 100:
            fallback = fallback[:last_period + 1]
        return fallback

    # ---- cross-page synthesis ----

    def synthesize(self, page_summaries: List[Dict], topic: str) -> str:
        """Cross-page report: Key Findings, Consensus, Conflicts, Gaps."""
        if not page_summaries:
            return ""

        combined = "\n\n".join(
            f"[{s.get('url', 'unknown')}]\n{s.get('summary', '')}"
            for s in page_summaries if s.get('summary')
        )

        result = self._call_llm(
            prompt=(
                f"You have research summaries from multiple sources about: {topic}\n\n"
                f"{combined[:8000]}\n\n"
                "Synthesize a report with these sections:\n"
                "1. **Key Findings** — the most important facts/insights\n"
                "2. **Consensus** — what multiple sources agree on\n"
                "3. **Conflicts** — where sources disagree\n"
                "4. **Gaps** — what is NOT well covered\n\n"
                "Be concise and specific. Cite sources by URL where relevant."
            ),
            system_prompt="You are a senior research analyst. Produce clear, structured synthesis."
        )
        if result:
            return result.strip()

        # Fallback: just concatenate
        return combined[:3000]

    # ---- gap identification ----

    def identify_gaps(self, page_summaries: List[Dict], topic: str) -> List[str]:
        """What's missing — feeds back into iterative deepening."""
        if not page_summaries:
            return []

        combined = "\n".join(
            s.get('summary', '')[:300] for s in page_summaries if s.get('summary')
        )

        result = self._call_llm(
            prompt=(
                f"Based on these research summaries about '{topic}':\n\n"
                f"{combined[:4000]}\n\n"
                "What important aspects of this topic are NOT covered? "
                "List 2-4 specific knowledge gaps as search queries, one per line. "
                "Return ONLY the queries, nothing else."
            ),
            system_prompt="You are a research gap analyst."
        )
        if result:
            lines = [line.strip().lstrip('0123456789.-) ') for line in result.strip().split('\n')]
            return [line for line in lines if line and len(line) > 5]

        return []  # No LLM → no gap analysis

    # ---- final report ----

    def build_final_report(self, synthesis: str, sources: List[Dict], metadata: Dict) -> str:
        """Formatted output with sources list."""
        parts = []

        # Header
        topic = metadata.get('topic', 'Unknown')
        parts.append(f"# Deep Research Report: {topic}")
        parts.append("")

        # Metadata line
        phases = metadata.get('phases_completed', 0)
        elapsed = metadata.get('time_seconds', 0)
        pages = metadata.get('pages_read', 0)
        parts.append(f"*{pages} sources analyzed across {phases} phases in {elapsed}s*")
        parts.append("")

        # Synthesis
        if synthesis:
            parts.append(synthesis)
            parts.append("")

        # Sources
        if sources:
            parts.append("## Sources")
            for i, src in enumerate(sources[:20], 1):
                title = src.get('title', src.get('url', 'Unknown'))
                url = src.get('url', '')
                parts.append(f"{i}. [{title}]({url})")
            parts.append("")

        return "\n".join(parts)


# ============================================================================
#  DeepResearchTool — main orchestrator
# ============================================================================

class DeepResearchTool:
    """Deep research: multi-phase search, rank, fetch, summarize pipeline.

    4-Phase architecture:
      Phase 1 (40% budget): Broad search — LLM-generated or hardcoded queries
      Phase 2 (20% budget): Gap analysis — LLM identifies missing coverage
      Phase 3 (30% budget): Targeted searches + page fetching to fill gaps
      Phase 4 (10% budget): Hierarchical summarization → final report

    Depth mapping:
      quick    = Phase 1 only
      standard = Phases 1–3
      deep     = All 4 phases
    """

    name = "deep_research"
    description = "Conduct deep research on a topic using multiple searches and page reads"

    def __init__(self, llm_func: Optional[Callable] = None):
        self.timeout = PAGE_FETCH_TIMEOUT
        self.overall_timeout = OVERALL_TIMEOUT
        self.searcher = None
        self.tavily = None
        self.browser = None
        self.llm = llm_func
        self._progress_callback: Optional[Callable[[str], None]] = None

        # Internal components
        self.ranker = ResultRanker()
        self.cache = ResearchCache()
        self.summarizer = HierarchicalSummarizer(llm_func)

        # --- Search backend priority ---
        # 1. Tavily (if available and API key set)
        try:
            from .tavily_tool import TavilyTool
            if os.getenv("TAVILY_API_KEY"):
                self.tavily = TavilyTool()
                logger.info("[DeepResearch] Using Tavily search backend")
        except ImportError:
            pass

        # 2. SearXNG / WebSearchTool (fallback)
        try:
            from .web_search import WebSearchTool
            self.searcher = WebSearchTool()
        except ImportError as e:
            logger.warning(f"WebSearchTool not available: {e}")

        # 3. Browser for page fetching
        try:
            from .browser import BrowserTool
            self.browser = BrowserTool()
        except ImportError as e:
            logger.warning(f"BrowserTool not available: {e}")

    def set_llm(self, llm_func: Callable):
        """Set or update the LLM function post-init (called from agent.py)."""
        self.llm = llm_func
        self.summarizer.llm = llm_func

    def set_progress_callback(self, callback: Callable[[str], None]):
        """Set callback for progress updates (e.g., send to Telegram)."""
        self._progress_callback = callback

    def _log_progress(self, message: str):
        """Log progress and call callback if set."""
        logger.info(f"[RESEARCH] {message}")
        if self._progress_callback:
            try:
                self._progress_callback(message)
            except Exception as e:
                logger.debug(f"[DeepResearch] non-critical: {e}")

    # ------------------------------------------------------------------
    #  Query generation
    # ------------------------------------------------------------------

    def _generate_queries(self, topic: str) -> List[str]:
        """Generate search queries — LLM-powered when available, hardcoded fallback."""
        # Try LLM generation first
        if self.llm:
            try:
                result = self.llm(
                    f"Generate 5-7 diverse search queries to thoroughly research: {topic}\n\n"
                    "Include different angles: overview, recent developments, technical details, "
                    "expert opinions, contrasting views.\n"
                    "Return ONLY the queries, one per line. No numbering or bullets.",
                    system_prompt="You are a research query strategist."
                )
                if result:
                    lines = result.strip().split('\n')
                    queries = []
                    for line in lines:
                        cleaned = line.strip().lstrip('0123456789.-) •*')
                        cleaned = cleaned.strip('"\'').strip()
                        if cleaned and len(cleaned) > 5:
                            queries.append(cleaned)
                    if len(queries) >= 3:
                        logger.debug(f"[DeepResearch] LLM generated {len(queries)} queries")
                        return queries[:7]
            except Exception as e:
                logger.debug(f"[DeepResearch] LLM query gen failed: {e}")

        # Hardcoded fallback
        return [
            topic,
            f"{topic} latest 2025",
            f"{topic} explained",
            f"{topic} analysis",
            f"what is {topic}",
        ]

    # ------------------------------------------------------------------
    #  Search execution (Tavily → SearXNG)
    # ------------------------------------------------------------------

    def _search(self, query: str, num_results: int = 10) -> Dict:
        """Execute a search using the best available backend.

        Returns dict with 'success' and 'results' keys.
        Results are list of dicts with url, title, snippet.
        """
        # Check cache first
        cached = self.cache.get_search(query)
        if cached is not None:
            return {"success": True, "results": cached, "cached": True}

        result = None

        # Try Tavily first
        if self.tavily:
            try:
                tavily_result = self.tavily.search(query, max_results=num_results)
                if not tavily_result.get("error"):
                    results = []
                    for item in tavily_result.get("results", []):
                        results.append({
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "snippet": item.get("content", ""),
                        })
                    if results:
                        result = {"success": True, "results": results, "cached": False}
            except Exception as e:
                logger.debug(f"[DeepResearch] Tavily search failed: {e}")

        # Fall back to SearXNG
        if result is None and self.searcher:
            try:
                searx_result = self.searcher.search(query, num_results)
                if searx_result.get("success"):
                    results = []
                    for item in searx_result.get("results", []):
                        results.append({
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                        })
                    result = {"success": True, "results": results, "cached": False}
            except Exception as e:
                logger.debug(f"[DeepResearch] SearXNG search failed: {e}")

        if result and result.get("success"):
            # Cache for next time
            self.cache.set_search(query, result["results"])
            return result

        return {"success": False, "results": [], "cached": False}

    # ------------------------------------------------------------------
    #  Page fetching
    # ------------------------------------------------------------------

    def _fetch_page(self, url: str) -> Dict:
        """Fetch full page content with cache support."""
        # Check cache
        cached_content = self.cache.get_page(url)
        if cached_content is not None:
            return {"url": url, "content": cached_content, "success": True, "cached": True}

        try:
            if self.browser and hasattr(self.browser, 'open'):
                nav_result = self.browser.open(url)
                if not nav_result.get("success"):
                    return {"url": url, "content": "", "success": False, "cached": False}
                text_result = self.browser.get_text()
                if isinstance(text_result, dict):
                    content = text_result.get("text", "")
                else:
                    content = str(text_result) if text_result else ""
                content = content[:5000] if content else ""
                if content:
                    self.cache.set_page(url, content)
                return {"url": url, "content": content, "success": bool(content), "cached": False}
            elif self.browser and hasattr(self.browser, 'fetch'):
                result = self.browser.fetch(url)
                content = result.get("text", "") if isinstance(result, dict) else str(result)
                content = content[:5000]
                if content:
                    self.cache.set_page(url, content)
                return {"url": url, "content": content, "success": True, "cached": False}
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")

        return {"url": url, "content": "", "success": False, "cached": False}

    # ------------------------------------------------------------------
    #  Phase executors
    # ------------------------------------------------------------------

    def _phase1_broad_search(self, topic: str, deadline: float) -> tuple:
        """Phase 1: Broad search with multiple queries.
        Returns (all_urls, queries_completed, cached_count).
        """
        queries = self._generate_queries(topic)
        all_urls = []
        seen = set()
        queries_completed = 0
        cached_count = 0

        for i, query in enumerate(queries):
            if time.time() > deadline:
                self._log_progress(f"Phase 1 timeout after {queries_completed} queries")
                break

            self._log_progress(f"Phase 1 — Search {i+1}/{len(queries)}: {query[:50]}...")

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._search, query, 10)
                    try:
                        result = future.result(timeout=SEARCH_TIMEOUT)
                        if result.get("success"):
                            if result.get("cached"):
                                cached_count += len(result.get("results", []))
                            for r in result.get("results", []):
                                url = r.get("url", "")
                                if url and url not in seen:
                                    seen.add(url)
                                    all_urls.append({
                                        "url": url,
                                        "title": r.get("title", ""),
                                        "snippet": r.get("snippet", ""),
                                    })
                            queries_completed += 1
                    except FuturesTimeoutError:
                        self._log_progress(f"Search {i+1} timed out, skipping")
            except Exception as e:
                self._log_progress(f"Search {i+1} failed: {e}")

        # Rank results
        if all_urls:
            all_urls = self.ranker.rank_results(all_urls, topic)

        self._log_progress(f"Phase 1 complete: {len(all_urls)} URLs from {queries_completed} searches")
        return all_urls, queries_completed, cached_count

    def _phase2_gap_analysis(self, page_summaries: List[Dict], topic: str, deadline: float) -> List[str]:
        """Phase 2: Identify knowledge gaps from Phase 1 results."""
        if time.time() > deadline:
            return []

        self._log_progress("Phase 2 — Identifying knowledge gaps...")
        gaps = self.summarizer.identify_gaps(page_summaries, topic)

        if gaps:
            self._log_progress(f"Phase 2 complete: {len(gaps)} gaps identified")
        else:
            self._log_progress("Phase 2 complete: no gaps identified (no LLM or sufficient coverage)")

        return gaps

    def _phase3_targeted_search(self, gaps: List[str], topic: str, existing_urls: set,
                                 deadline: float) -> tuple:
        """Phase 3: Targeted searches to fill gaps, then fetch pages.
        Returns (new_urls, pages_fetched, cached_count).
        """
        new_urls = []
        seen = set(existing_urls)
        cached_count = 0

        # Search for each gap
        for i, gap_query in enumerate(gaps):
            if time.time() > deadline:
                break

            self._log_progress(f"Phase 3 — Gap search {i+1}/{len(gaps)}: {gap_query[:50]}...")

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self._search, gap_query, 5)
                    try:
                        result = future.result(timeout=SEARCH_TIMEOUT)
                        if result.get("success"):
                            if result.get("cached"):
                                cached_count += len(result.get("results", []))
                            for r in result.get("results", []):
                                url = r.get("url", "")
                                if url and url not in seen:
                                    seen.add(url)
                                    new_urls.append({
                                        "url": url,
                                        "title": r.get("title", ""),
                                        "snippet": r.get("snippet", ""),
                                    })
                    except FuturesTimeoutError:
                        pass
            except Exception:
                pass

        # Rank new results
        if new_urls:
            new_urls = self.ranker.rank_results(new_urls, topic)

        self._log_progress(f"Phase 3 search: {len(new_urls)} new URLs from {len(gaps)} gap queries")
        return new_urls, cached_count

    def _fetch_pages_parallel(self, urls: List[Dict], max_pages: int,
                               deadline: float) -> tuple:
        """Fetch pages in parallel with timeout protection.
        Returns (pages, cached_count).
        """
        if not self.browser or not urls:
            return [], 0

        remaining = deadline - time.time()
        if remaining <= 0:
            return [], 0

        urls_to_fetch = urls[:max_pages]
        pages = []
        cached_count = 0

        self._log_progress(f"Fetching {len(urls_to_fetch)} pages ({remaining:.0f}s remaining)...")

        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self._fetch_page, u["url"]): u for u in urls_to_fetch}

                for future in as_completed(futures, timeout=remaining):
                    if time.time() > deadline:
                        self._log_progress("Page fetch timeout, returning partial results")
                        break

                    try:
                        result = future.result(timeout=PAGE_FETCH_TIMEOUT)
                        if result.get("success"):
                            if result.get("cached"):
                                cached_count += 1
                            pages.append(result)
                            self._log_progress(f"Fetched page {len(pages)}/{len(urls_to_fetch)}")
                    except FuturesTimeoutError:
                        self._log_progress("Page fetch timed out")
                    except Exception as e:
                        logger.warning(f"Page fetch error: {e}")

        except FuturesTimeoutError:
            self._log_progress("Page fetching timed out, using partial results")
        except Exception as e:
            self._log_progress(f"Page fetching error: {e}")

        return pages, cached_count

    def _summarize_pages(self, pages: List[Dict], topic: str) -> List[Dict]:
        """Summarize each fetched page. Returns list of {url, summary} dicts."""
        summaries = []
        for page in pages:
            url = page.get("url", "")
            content = page.get("content", "")
            if not content:
                continue
            summary = self.summarizer.summarize_page(url, content, topic)
            if summary:
                summaries.append({"url": url, "summary": summary})
        return summaries

    # ------------------------------------------------------------------
    #  Main research orchestrator
    # ------------------------------------------------------------------

    def research(self, topic: str, depth: str = "standard") -> Dict:
        """
        Conduct deep research with 4-phase pipeline and timeout protection.

        Args:
            topic: What to research
            depth: "quick" (Phase 1 only), "standard" (Phases 1-3), "deep" (all 4)

        Returns:
            Dict with results (partial if timeout)
        """
        start_time = time.time()
        overall_deadline = start_time + self.overall_timeout
        self._log_progress(f"Starting research: {topic} (depth={depth})")

        if not self.searcher and not self.tavily:
            return {
                "success": False, "error": "No search backend available",
                "topic": topic, "depth": depth,
                "queries_run": 0, "urls_found": 0, "pages_read": 0,
                "time_seconds": 0, "timed_out": False,
                "sources": [], "content": "", "summary": "",
                "synthesis": "", "page_summaries": [], "knowledge_gaps": [],
                "phases_completed": 0, "cached_results": 0,
            }

        # Depth config
        max_pages = {"quick": 5, "standard": 10, "deep": 20}.get(depth, 10)
        max_phase = {"quick": 1, "standard": 3, "deep": 4}.get(depth, 3)

        # Phase deadlines
        p1_deadline = start_time + self.overall_timeout * PHASE_BUDGET["broad"]
        p2_deadline = p1_deadline + self.overall_timeout * PHASE_BUDGET["gap"]
        p3_deadline = p2_deadline + self.overall_timeout * PHASE_BUDGET["targeted"]
        # Phase 4 gets whatever is left up to overall_deadline

        timed_out = False
        total_cached = 0
        phases_completed = 0
        all_urls = []
        pages = []
        page_summaries = []
        knowledge_gaps = []
        synthesis = ""

        # ============================================================
        # PHASE 1: Broad search
        # ============================================================
        all_urls, queries_completed, cached = self._phase1_broad_search(
            topic, p1_deadline
        )
        total_cached += cached
        phases_completed = 1

        if time.time() > overall_deadline:
            timed_out = True

        # Fetch pages for Phase 1 results (use remaining Phase 1 + some Phase 2 budget)
        if not timed_out and all_urls:
            fetch_deadline = min(p2_deadline, overall_deadline)
            pages, page_cached = self._fetch_pages_parallel(
                all_urls, max_pages, fetch_deadline
            )
            total_cached += page_cached

            # Summarize fetched pages
            if pages:
                page_summaries = self._summarize_pages(pages, topic)

        # ============================================================
        # PHASE 2: Gap analysis (if depth allows)
        # ============================================================
        if max_phase >= 2 and not timed_out and time.time() < overall_deadline:
            knowledge_gaps = self._phase2_gap_analysis(
                page_summaries, topic, p2_deadline
            )
            phases_completed = 2

            if time.time() > overall_deadline:
                timed_out = True

        # ============================================================
        # PHASE 3: Targeted search to fill gaps
        # ============================================================
        if max_phase >= 3 and not timed_out and knowledge_gaps and time.time() < overall_deadline:
            existing_url_set = {u.get("url", "") for u in all_urls}
            new_urls, gap_cached = self._phase3_targeted_search(
                knowledge_gaps, topic, existing_url_set, p3_deadline
            )
            total_cached += gap_cached

            # Fetch new pages
            if new_urls and time.time() < overall_deadline:
                remaining_page_slots = max(0, max_pages - len(pages))
                if remaining_page_slots > 0:
                    new_pages, new_page_cached = self._fetch_pages_parallel(
                        new_urls, remaining_page_slots, min(p3_deadline, overall_deadline)
                    )
                    total_cached += new_page_cached
                    pages.extend(new_pages)

                    # Summarize new pages
                    if new_pages:
                        new_summaries = self._summarize_pages(new_pages, topic)
                        page_summaries.extend(new_summaries)

                # Merge new URLs into all_urls
                all_urls.extend(new_urls)

            phases_completed = 3

            if time.time() > overall_deadline:
                timed_out = True

        # ============================================================
        # PHASE 4: Hierarchical summarization
        # ============================================================
        if max_phase >= 4 and not timed_out and page_summaries and time.time() < overall_deadline:
            self._log_progress("Phase 4 — Synthesizing final report...")
            synthesis = self.summarizer.synthesize(page_summaries, topic)
            phases_completed = 4

        # ============================================================
        # Build output
        # ============================================================
        elapsed = round(time.time() - start_time, 1)
        self._log_progress(f"Research complete in {elapsed}s ({phases_completed} phases)")

        # Content: either synthesis or page content or snippets
        if synthesis:
            content_summary = synthesis
        elif pages:
            content_summary = "\n\n".join(
                f"## {p['url']}\n{p['content'][:1000]}" for p in pages[:5]
            )
        else:
            snippets = [
                f"- {u['title']}: {u['snippet']}"
                for u in all_urls[:max_pages] if u.get('snippet')
            ]
            content_summary = "\n".join(snippets)

        # Build final formatted report (if we have synthesis)
        final_report = ""
        if synthesis and phases_completed >= 4:
            metadata = {
                "topic": topic, "phases_completed": phases_completed,
                "time_seconds": elapsed, "pages_read": len(pages),
            }
            sources_for_report = [
                {"url": u["url"], "title": u["title"]}
                for u in all_urls[:max_pages]
            ]
            final_report = self.summarizer.build_final_report(
                synthesis, sources_for_report, metadata
            )

        summary_text = (
            f"Researched '{topic}': {queries_completed} searches, "
            f"{len(all_urls)} sources, {len(pages)} pages read, "
            f"{phases_completed} phases in {elapsed}s"
        )
        if timed_out:
            summary_text += " (partial — timed out)"

        # Cleanup expired cache entries (best-effort, non-blocking)
        try:
            self.cache.cleanup()
        except Exception:
            pass

        return {
            # Original keys (backward-compatible)
            "success": True,
            "topic": topic,
            "depth": depth,
            "queries_run": queries_completed,
            "urls_found": len(all_urls),
            "pages_read": len(pages),
            "time_seconds": elapsed,
            "timed_out": timed_out,
            "sources": [
                {"url": u["url"], "title": u["title"], "snippet": u.get("snippet", "")[:200]}
                for u in all_urls[:max_pages]
            ],
            "content": final_report if final_report else content_summary,
            "summary": summary_text,
            # New keys
            "synthesis": synthesis,
            "page_summaries": page_summaries,
            "knowledge_gaps": knowledge_gaps,
            "phases_completed": phases_completed,
            "cached_results": total_cached,
        }

    # ------------------------------------------------------------------
    #  run() entry point
    # ------------------------------------------------------------------

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
                        "depth": depth,
                        "timed_out": True,
                        "queries_run": 0, "urls_found": 0, "pages_read": 0,
                        "time_seconds": round(elapsed, 1),
                        "sources": [], "content": "", "summary": "",
                        "synthesis": "", "page_summaries": [],
                        "knowledge_gaps": [], "phases_completed": 0,
                        "cached_results": 0,
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "topic": topic,
                "depth": depth,
                "timed_out": False,
                "queries_run": 0, "urls_found": 0, "pages_read": 0,
                "time_seconds": round(time.time() - start_time, 1),
                "sources": [], "content": "", "summary": "",
                "synthesis": "", "page_summaries": [],
                "knowledge_gaps": [], "phases_completed": 0,
                "cached_results": 0,
            }


# ============================================================================
#  Convenience function
# ============================================================================

def deep_research(topic: str, depth: str = "standard") -> Dict:
    """Conduct deep research on a topic."""
    tool = DeepResearchTool()
    return tool.research(topic, depth)

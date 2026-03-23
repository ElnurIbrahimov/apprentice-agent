"""Deep research tool — multi-phase search, rank, fetch, summarize pipeline.

SOTA upgrades over original:
- 4-phase architecture (broad search → gap analysis → targeted fill → synthesis)
- ResultRanker: BM25-inspired + domain authority + snippet quality scoring
- ResearchCache: SQLite-backed caching for search results and page content
- HierarchicalSummarizer: per-page LLM summaries → cross-source synthesis
- Search backend priority: Tavily (if available) → SearXNG fallback
- Optional LLM integration with full graceful degradation

v2 STORM-pattern upgrades:
- Outline-first planning: STORM-style perspective-guided sub-query decomposition
- MMR (Maximal Marginal Relevance): diverse source selection with lambda trade-off
- Citation quality scoring: domain authority + TLD + recency decay
- Information saturation: stop iterating when entity novelty < 5%
- Citation anchoring: post-filter verifies every [N] reference maps to a real source
- Contradiction detection: flags where sources disagree on the same claim
"""

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
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
#  MMR — Maximal Marginal Relevance for diverse source selection
# ============================================================================

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Pure-python fallback."""
    try:
        import numpy as np
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        dot = float(np.dot(va, vb))
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    except ImportError:
        # Pure-python fallback (slower but works)
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


def _get_embedding_for_text(text: str) -> Optional[List[float]]:
    """Get embedding vector via shared Ollama helper. Returns None on failure."""
    try:
        from aura.memory.embedding import get_embedding
        return get_embedding(text, timeout=3.0)
    except Exception:
        return None


# ============================================================================
#  CitationScorer — domain authority + recency scoring
# ============================================================================

class CitationScorer:
    """Score a source's citation quality based on domain authority and recency."""

    TLD_SCORES = {'.edu': 1.0, '.gov': 0.95, '.org': 0.8, '.ac': 0.9}

    HIGH_AUTHORITY = {
        'arxiv.org': 0.95, 'nature.com': 0.95, 'science.org': 0.95,
        'ieee.org': 0.9, 'acm.org': 0.9, 'wikipedia.org': 0.85,
        'github.com': 0.8, 'stackoverflow.com': 0.8,
        'nih.gov': 0.95, 'pubmed.ncbi.nlm.nih.gov': 0.95,
        'scholar.google.com': 0.9, 'semanticscholar.org': 0.9,
        'huggingface.co': 0.8, 'pytorch.org': 0.8, 'tensorflow.org': 0.8,
        'docs.python.org': 0.85, 'developer.mozilla.org': 0.85,
    }

    def score(self, url: str, age_days: int = 0) -> float:
        """Score a source's citation quality.

        Returns float in [0, 1]. Higher = more authoritative + recent.
        """
        if not url:
            return 0.3

        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            return 0.3

        base_score = 0.5  # default for unknown domains

        # Check high-authority domains first
        for d, s in self.HIGH_AUTHORITY.items():
            if d in domain:
                base_score = s
                break
        else:
            # TLD fallback
            for tld, s in self.TLD_SCORES.items():
                if domain.endswith(tld):
                    base_score = s
                    break

        # Recency decay — half-life ~70 days
        if age_days > 0:
            recency_factor = math.exp(-0.01 * age_days)
        else:
            recency_factor = 1.0

        return round(base_score * recency_factor, 4)


# ============================================================================
#  SaturationDetector — stop when novelty drops below threshold
# ============================================================================

class SaturationDetector:
    """Track entity discovery rate across research rounds. Stop when saturated."""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self._entities_per_round: List[int] = []
        self._all_entities: Set[str] = set()

    def add_round(self, new_entities: Set[str]) -> int:
        """Register entities discovered in this round. Returns count of truly new ones."""
        novel = new_entities - self._all_entities
        self._all_entities.update(novel)
        self._entities_per_round.append(len(novel))
        return len(novel)

    def is_saturated(self) -> bool:
        """True when new entity discovery drops below threshold."""
        if len(self._entities_per_round) < 2:
            return False
        latest = self._entities_per_round[-1]
        total = sum(self._entities_per_round)
        if total == 0:
            return True
        novelty_rate = latest / total
        return novelty_rate < self.threshold

    @property
    def total_entities(self) -> int:
        return len(self._all_entities)

    @property
    def rounds(self) -> int:
        return len(self._entities_per_round)


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

    # ---- entity extraction (for saturation detection) ----

    def extract_entities(self, text: str) -> Set[str]:
        """Extract key entities/terms from text for saturation tracking.
        Uses LLM when available, falls back to simple noun-phrase heuristic."""
        if not text:
            return set()

        result = self._call_llm(
            prompt=(
                f"Extract the key named entities, technical terms, and concepts from this text. "
                f"Return them as a comma-separated list, nothing else.\n\n{text[:3000]}"
            ),
            system_prompt="You are an entity extraction tool. Be precise."
        )
        if result:
            entities = {e.strip().lower() for e in result.split(',') if e.strip()}
            return entities

        # Fallback: extract capitalized multi-word phrases and technical terms
        entities = set()
        # Capitalized phrases (2-4 words)
        for match in re.finditer(r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})', text):
            entities.add(match.group().lower())
        # Quoted terms
        for match in re.finditer(r'"([^"]{2,50})"', text):
            entities.add(match.group(1).lower())
        return entities

    # ---- section-by-section synthesis (STORM-style) ----

    def synthesize_section(self, section_title: str, relevant_summaries: List[Dict],
                           topic: str, sources: List[Dict]) -> str:
        """Synthesize a single report section from relevant summaries with citations."""
        if not relevant_summaries:
            return ""

        combined = "\n\n".join(
            f"[Source {i+1}: {s.get('url', 'unknown')}]\n{s.get('summary', '')}"
            for i, s in enumerate(relevant_summaries) if s.get('summary')
        )

        source_index = "\n".join(
            f"[{i+1}] {src.get('url', '')}" for i, src in enumerate(sources)
        )

        result = self._call_llm(
            prompt=(
                f"Write the '{section_title}' section of a research report about: {topic}\n\n"
                f"Available sources (use [N] citations):\n{source_index}\n\n"
                f"Source summaries:\n{combined[:6000]}\n\n"
                f"Write 2-4 paragraphs. Every factual claim MUST cite its source as [N]. "
                f"Be specific, include data points and numbers where available."
            ),
            system_prompt="You are a research writer. Always cite sources with [N] notation."
        )
        if result:
            return result.strip()

        # Fallback: concatenate summaries
        return combined[:1500]

    # ---- contradiction detection ----

    def detect_contradictions(self, page_summaries: List[Dict], topic: str) -> List[str]:
        """Detect where sources disagree on claims."""
        if not page_summaries or len(page_summaries) < 2:
            return []

        combined = "\n\n".join(
            f"[{s.get('url', 'unknown')}]: {s.get('summary', '')[:400]}"
            for s in page_summaries if s.get('summary')
        )

        result = self._call_llm(
            prompt=(
                f"Review these research summaries about '{topic}' and identify any "
                f"CONTRADICTIONS — places where sources disagree on facts, numbers, "
                f"claims, or conclusions.\n\n{combined[:6000]}\n\n"
                "List each contradiction as:\n"
                "- CLAIM: <what is disputed>\n"
                "  SOURCE A says: <position> [url]\n"
                "  SOURCE B says: <position> [url]\n\n"
                "If no contradictions are found, return 'NONE'."
            ),
            system_prompt="You are a fact-checking analyst. Be precise about disagreements."
        )
        if result and result.strip().upper() != 'NONE':
            return [result.strip()]

        return []

    # ---- citation anchoring post-filter ----

    @staticmethod
    def anchor_citations(text: str, sources: List[Dict]) -> str:
        """Post-filter: verify every [N] citation maps to an actual source.
        Removes or flags unanchored citations."""
        if not text or not sources:
            return text

        max_valid = len(sources)

        def _replace_citation(match):
            num = int(match.group(1))
            if 1 <= num <= max_valid:
                return match.group(0)  # valid — keep it
            return f"[?{num}]"  # invalid — flag it

        # Replace invalid [N] references
        anchored = re.sub(r'\[(\d+)\]', _replace_citation, text)
        return anchored

    # ---- structured citation extraction ----

    @staticmethod
    def build_structured_citations(text: str, sources: List[Dict]) -> List[Dict]:
        """Extract which [N] markers appear in the text and build structured
        citation objects that the frontend can render as first-class UI elements.

        Returns list of dicts: {id, url, title, snippet, relevance_score}
        Only includes citations that are actually referenced in the text.
        """
        if not text or not sources:
            return []

        # Find all [N] references in the text
        referenced_ids = sorted(set(int(m) for m in re.findall(r'\[(\d+)\]', text)))

        citations = []
        for cid in referenced_ids:
            if 1 <= cid <= len(sources):
                src = sources[cid - 1]
                citations.append({
                    "id": cid,
                    "url": src.get("url", ""),
                    "title": src.get("title", src.get("url", "Unknown")),
                    "snippet": src.get("snippet", src.get("summary", ""))[:200],
                    "relevance_score": src.get("citation_score", 0),
                })

        return citations

    # ---- final report ----

    def build_final_report(self, synthesis: str, sources: List[Dict], metadata: Dict,
                           contradictions: Optional[List[str]] = None,
                           outline: Optional[List[str]] = None) -> Dict:
        """Formatted output with sources list, contradictions, and outline structure.

        Returns a dict with:
          text: the full markdown report string
          citations: structured list of {id, url, title, snippet, relevance_score}
        """
        parts = []

        # Header
        topic = metadata.get('topic', 'Unknown')
        parts.append(f"# Deep Research Report: {topic}")
        parts.append("")

        # Metadata line
        phases = metadata.get('phases_completed', 0)
        elapsed = metadata.get('time_seconds', 0)
        pages = metadata.get('pages_read', 0)
        entities = metadata.get('entities_tracked', 0)
        parts.append(f"*{pages} sources analyzed across {phases} phases in {elapsed}s")
        if entities:
            parts.append(f" | {entities} unique entities tracked")
        parts.append("*")
        parts.append("")

        # Synthesis (section-by-section if outline was used)
        if synthesis:
            parts.append(synthesis)
            parts.append("")

        # Contradictions section
        if contradictions:
            parts.append("## Contradictions & Disagreements")
            for c in contradictions:
                parts.append(c)
            parts.append("")

        # Sources section (still included in text for backwards compat)
        if sources:
            parts.append("## Sources")
            for i, src in enumerate(sources[:30], 1):
                title = src.get('title', src.get('url', 'Unknown'))
                url = src.get('url', '')
                citation_score = src.get('citation_score', '')
                score_str = f" (quality: {citation_score})" if citation_score else ""
                parts.append(f"{i}. [{title}]({url}){score_str}")
            parts.append("")

        report_text = "\n".join(parts)

        # Build structured citations from [N] markers in the synthesis
        structured_citations = self.build_structured_citations(synthesis or "", sources)

        return {
            "text": report_text,
            "citations": structured_citations,
        }


# ============================================================================
#  ResearchProgressEmitter — structured WebSocket progress events
# ============================================================================

class ResearchProgressEmitter:
    """Emits structured research progress events via an optional callback.

    The callback receives a dict like:
        {"type": "research_progress", "stage": "search", "data": {...}}

    If no callback is set, all emit calls are silent no-ops.
    """

    def __init__(self, callback: Optional[Callable[[Dict], None]] = None):
        self._callback = callback

    def set_callback(self, callback: Optional[Callable[[Dict], None]]):
        self._callback = callback

    def _emit(self, stage: str, data: Dict):
        if not self._callback:
            return
        try:
            self._callback({
                "type": "research_progress",
                "stage": stage,
                "data": data,
            })
        except Exception as e:
            logger.debug(f"[ResearchProgressEmitter] emit failed: {e}")

    def emit_plan(self, subtopics: List[str], outline: Optional[List[str]] = None):
        self._emit("plan", {
            "subtopics": subtopics,
            "outline": outline or [],
            "message": f"Research plan: {len(subtopics)} sub-queries",
        })

    def emit_search(self, query: str, step: int = 0, total: int = 0):
        self._emit("search", {
            "query": query,
            "step": step,
            "total": total,
            "message": f"Searching ({step}/{total}): {query[:80]}",
        })

    def emit_source_found(self, url: str, title: str):
        self._emit("source", {
            "url": url,
            "title": title,
            "message": f"Found: {title[:80]}",
        })

    def emit_finding(self, text: str, url: str = ""):
        self._emit("finding", {
            "text": text[:300],
            "url": url,
            "message": f"Finding: {text[:120]}",
        })

    def emit_synthesis_started(self):
        self._emit("synthesis", {
            "message": "Synthesizing final report...",
        })

    def emit_progress(self, step: int, total: int, message: str):
        self._emit("search", {
            "step": step,
            "total": total,
            "message": message,
        })


# ============================================================================
#  DeepResearchTool — main orchestrator
# ============================================================================

class DeepResearchTool:
    """Deep research: STORM-pattern outline-first, MMR-diverse, citation-anchored pipeline.

    Architecture (5 phases):
      Phase 0 (plan):     STORM outline — perspective-guided sub-query decomposition
      Phase 1 (broad):    Search per sub-query, MMR-select diverse sources
      Phase 2 (gap):      LLM identifies missing coverage, saturation check
      Phase 3 (targeted): Fill gaps with targeted searches + page fetching
      Phase 4 (synthesis): Section-by-section synthesis with citation anchoring +
                           contradiction detection

    Depth mapping:
      quick    = Phase 1 only (no planning)
      standard = Phases 0–3
      deep     = All phases (0–4)
    """

    name = "deep_research"
    description = "Conduct deep research on a topic using STORM outline-first planning and MMR diversity"

    def __init__(self, llm_func: Optional[Callable] = None):
        self.timeout = PAGE_FETCH_TIMEOUT
        self.overall_timeout = OVERALL_TIMEOUT
        self.searcher = None
        self.tavily = None
        self.browser = None
        self.llm = llm_func
        self._progress_callback: Optional[Callable[[str], None]] = None
        self._emitter = ResearchProgressEmitter()
        # Shared pool — centralized in aura.pools
        from aura.pools import tool_pool
        self._executor = tool_pool()

        # Internal components
        self.ranker = ResultRanker()
        self.cache = ResearchCache()
        self.summarizer = HierarchicalSummarizer(llm_func)
        self.citation_scorer = CitationScorer()
        self.saturation = SaturationDetector(threshold=0.05)

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

    def close(self):
        """Shut down the shared executor."""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)

    def set_llm(self, llm_func: Callable):
        """Set or update the LLM function post-init (called from agent.py)."""
        self.llm = llm_func
        self.summarizer.llm = llm_func

    def set_progress_callback(self, callback: Callable[[str], None]):
        """Set callback for progress updates (e.g., send to Telegram)."""
        self._progress_callback = callback

    def set_ws_callback(self, callback: Callable[[Dict], None]):
        """Set WebSocket progress callback for real-time streaming events.

        The callback receives dicts with 'type', 'stage', and 'data' keys.
        If not set, research works exactly as before (no WebSocket dependency).
        """
        self._emitter.set_callback(callback)

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
    #  STORM-pattern research planning (Phase 0)
    # ------------------------------------------------------------------

    def _plan_research(self, topic: str) -> Dict:
        """Decompose query into perspective-guided sub-queries (STORM pattern).

        Returns a plan dict with:
          perspectives: list of {name, questions}
          outline: list of section titles
          key_entities: list of expected key entities
          all_queries: flattened list of all sub-questions for search
        """
        default_plan = {
            "perspectives": [],
            "outline": [
                f"Overview of {topic}",
                f"Key details and analysis",
                f"Recent developments",
                f"Expert opinions and debate",
            ],
            "key_entities": [],
            "all_queries": self._generate_queries(topic),
        }

        if not self.llm:
            return default_plan

        try:
            result = self.llm(
                f"""You are a research planner. Decompose this research query into a structured outline.

Query: {topic}

Generate:
1. 3-5 different perspectives to investigate (e.g., "Technical Expert", "Skeptic", "Historian", "Practitioner")
2. For each perspective, 2-3 specific sub-questions to answer
3. A research outline with sections
4. Key entities/terms to track

Return as JSON (no markdown fences):
{{
    "perspectives": [
        {{"name": "Perspective Name", "questions": ["specific question 1", "specific question 2"]}}
    ],
    "outline": ["Section 1: Title", "Section 2: Title"],
    "key_entities": ["entity1", "entity2"]
}}""",
                system_prompt="You are a research planner. Return valid JSON only, no markdown."
            )
            if result:
                # Strip markdown fences if present
                cleaned = result.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r'^```\w*\n?', '', cleaned)
                    cleaned = re.sub(r'\n?```$', '', cleaned)
                    cleaned = cleaned.strip()

                plan = json.loads(cleaned)

                # Validate structure
                perspectives = plan.get("perspectives", [])
                outline = plan.get("outline", [])
                key_entities = plan.get("key_entities", [])

                if not perspectives or not outline:
                    logger.debug("[DeepResearch] STORM plan missing fields, using default")
                    return default_plan

                # Flatten all questions from perspectives into search queries
                all_queries = []
                for p in perspectives:
                    for q in p.get("questions", []):
                        if q and len(q) > 5:
                            all_queries.append(q)

                # Add the base topic as a query too
                all_queries.insert(0, topic)

                plan["all_queries"] = all_queries[:12]  # Cap at 12 sub-queries
                logger.info(
                    f"[DeepResearch] STORM plan: {len(perspectives)} perspectives, "
                    f"{len(outline)} sections, {len(all_queries)} queries"
                )
                return plan

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"[DeepResearch] STORM plan parse failed: {e}")
        except Exception as e:
            logger.debug(f"[DeepResearch] STORM plan failed: {e}")

        return default_plan

    # ------------------------------------------------------------------
    #  MMR — diverse source selection
    # ------------------------------------------------------------------

    def _mmr_select(self, candidates: List[Dict], query_embedding: List[float],
                    selected: Optional[List[Dict]] = None,
                    lambda_param: float = 0.7, k: int = 10) -> List[Dict]:
        """Select k diverse, relevant results using Maximal Marginal Relevance.

        score = lambda * sim(doc, query) - (1 - lambda) * max_sim(doc, selected)

        Each candidate dict should have an 'embedding' key (list of floats).
        Candidates without embeddings are appended at the end (relevance-only fallback).
        """
        if not candidates:
            return []

        # Split into embeddable and non-embeddable
        with_emb = [c for c in candidates if c.get('embedding')]
        without_emb = [c for c in candidates if not c.get('embedding')]

        if not with_emb or not query_embedding:
            # No embeddings available — fall back to rank-order
            return candidates[:k]

        results = list(selected or [])
        remaining = list(with_emb)

        for _ in range(min(k, len(remaining))):
            best_score = -float('inf')
            best_idx = 0

            for i, doc in enumerate(remaining):
                doc_emb = doc['embedding']
                relevance = _cosine_similarity(doc_emb, query_embedding)

                diversity = 0.0
                if results:
                    diversity = max(
                        _cosine_similarity(doc_emb, s['embedding'])
                        for s in results if s.get('embedding')
                    ) if any(s.get('embedding') for s in results) else 0.0

                score = lambda_param * relevance - (1 - lambda_param) * diversity

                if score > best_score:
                    best_score = score
                    best_idx = i

            results.append(remaining.pop(best_idx))

        # Append non-embeddable candidates at the end if we still have room
        remaining_slots = k - len(results)
        if remaining_slots > 0 and without_emb:
            results.extend(without_emb[:remaining_slots])

        return results

    def _embed_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Add embedding vectors to search result candidates (best-effort).

        Embeds `title + snippet` for each candidate. Skips on failure.
        Uses the shared Ollama embedding endpoint.
        """
        for c in candidates:
            if c.get('embedding'):
                continue  # Already has one
            text = f"{c.get('title', '')} {c.get('snippet', '')}".strip()
            if not text:
                continue
            try:
                emb = _get_embedding_for_text(text[:500])
                if emb:
                    c['embedding'] = emb
            except Exception:
                pass  # Best-effort — skip on failure
        return candidates

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

    def _phase1_broad_search(self, topic: str, deadline: float,
                             plan: Optional[Dict] = None) -> tuple:
        """Phase 1: Broad search with plan-derived or generated queries.

        Uses STORM plan queries if available, falls back to _generate_queries.
        Applies MMR diversity selection and citation quality scoring.

        Returns (all_urls, queries_completed, cached_count).
        """
        # Use plan queries if available, else generate
        if plan and plan.get("all_queries"):
            queries = plan["all_queries"]
        else:
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
            self._emitter.emit_search(query, step=i + 1, total=len(queries))

            try:
                future = self._executor.submit(self._search, query, 10)
                try:
                    result = future.result(timeout=SEARCH_TIMEOUT)
                    if result.get("success"):
                        if result.get("cached"):
                            cached_count += len(result.get("results", []))
                        for r in result.get("results", []):
                            url = r.get("url", "")
                            if url and url not in seen:
                                seen.add(url)
                                entry = {
                                    "url": url,
                                    "title": r.get("title", ""),
                                    "snippet": r.get("snippet", ""),
                                    "citation_score": self.citation_scorer.score(url),
                                }
                                all_urls.append(entry)
                                self._emitter.emit_source_found(url, r.get("title", ""))
                        queries_completed += 1
                except FuturesTimeoutError:
                    self._log_progress(f"Search {i+1} timed out, skipping")
            except Exception as e:
                self._log_progress(f"Search {i+1} failed: {e}")

        # Rank results first (composite score)
        if all_urls:
            all_urls = self.ranker.rank_results(all_urls, topic)

        # MMR diversity selection — embed candidates and select diverse top-k
        if len(all_urls) > 15:
            self._log_progress(f"Applying MMR diversity selection on {len(all_urls)} candidates...")
            try:
                query_emb = _get_embedding_for_text(topic)
                if query_emb:
                    all_urls = self._embed_candidates(all_urls)
                    all_urls = self._mmr_select(
                        all_urls, query_emb,
                        lambda_param=0.7, k=min(len(all_urls), 30)
                    )
                    self._log_progress(f"MMR selected {len(all_urls)} diverse sources")
            except Exception as e:
                logger.debug(f"[DeepResearch] MMR selection failed (using rank order): {e}")

        # Track entities for saturation detection
        if all_urls:
            snippet_text = " ".join(u.get("snippet", "") for u in all_urls[:20])
            entities = self.summarizer.extract_entities(snippet_text)
            self.saturation.add_round(entities)

        self._log_progress(f"Phase 1 complete: {len(all_urls)} URLs from {queries_completed} searches")
        return all_urls, queries_completed, cached_count

    def _phase2_gap_analysis(self, page_summaries: List[Dict], topic: str,
                             deadline: float) -> List[str]:
        """Phase 2: Identify knowledge gaps + check saturation.

        Also tracks entities from page summaries and checks if research
        has saturated (novelty < 5%).
        """
        if time.time() > deadline:
            return []

        # Track entities from page summaries for saturation
        summary_text = " ".join(s.get("summary", "") for s in page_summaries)
        if summary_text:
            entities = self.summarizer.extract_entities(summary_text)
            new_count = self.saturation.add_round(entities)
            self._log_progress(
                f"Phase 2 — Entity tracking: {new_count} new entities "
                f"({self.saturation.total_entities} total)"
            )

            if self.saturation.is_saturated():
                self._log_progress("Phase 2 — Research saturated (novelty < 5%), skipping gap fill")
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
        """Phase 3: Targeted searches to fill gaps with MMR diversity + saturation check.
        Returns (new_urls, cached_count).
        """
        new_urls = []
        seen = set(existing_urls)
        cached_count = 0

        # Search for each gap
        for i, gap_query in enumerate(gaps):
            if time.time() > deadline:
                break

            # Check saturation — stop early if we've converged
            if self.saturation.is_saturated():
                self._log_progress("Phase 3 — Saturated, stopping gap searches early")
                break

            self._log_progress(f"Phase 3 — Gap search {i+1}/{len(gaps)}: {gap_query[:50]}...")
            self._emitter.emit_search(gap_query, step=i + 1, total=len(gaps))

            try:
                future = self._executor.submit(self._search, gap_query, 5)
                try:
                    result = future.result(timeout=SEARCH_TIMEOUT)
                    if result.get("success"):
                        if result.get("cached"):
                            cached_count += len(result.get("results", []))
                        round_results = []
                        for r in result.get("results", []):
                            url = r.get("url", "")
                            if url and url not in seen:
                                seen.add(url)
                                entry = {
                                    "url": url,
                                    "title": r.get("title", ""),
                                    "snippet": r.get("snippet", ""),
                                    "citation_score": self.citation_scorer.score(url),
                                }
                                round_results.append(entry)
                                self._emitter.emit_source_found(url, r.get("title", ""))

                        # Track entities from this round for saturation
                        if round_results:
                            round_text = " ".join(r.get("snippet", "") for r in round_results)
                            entities = self.summarizer.extract_entities(round_text)
                            self.saturation.add_round(entities)

                        new_urls.extend(round_results)
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
            futures = {self._executor.submit(self._fetch_page, u["url"]): u for u in urls_to_fetch}

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
                self._emitter.emit_finding(summary, url=url)
        return summaries

    # ------------------------------------------------------------------
    #  Main research orchestrator
    # ------------------------------------------------------------------

    def research(self, topic: str, depth: str = "standard") -> Dict:
        """
        Conduct deep research with STORM outline-first pipeline.

        Flow:
          Phase 0: STORM plan — perspective decomposition + outline (standard/deep)
          Phase 1: Broad search with plan queries, MMR diversity, citation scoring
          Phase 2: Gap analysis + saturation check
          Phase 3: Targeted gap-fill searches
          Phase 4: Section-by-section synthesis + contradiction detection +
                   citation anchoring (deep only)

        Args:
            topic: What to research
            depth: "quick" (Phase 1 only), "standard" (Phases 0-3), "deep" (all phases)

        Returns:
            Dict with results (partial if timeout)
        """
        start_time = time.time()
        overall_deadline = start_time + self.overall_timeout
        self._log_progress(f"Starting research: {topic} (depth={depth})")

        # Reset saturation detector for this run
        self.saturation = SaturationDetector(threshold=0.05)

        if not self.searcher and not self.tavily:
            return {
                "success": False, "error": "No search backend available",
                "topic": topic, "depth": depth,
                "queries_run": 0, "urls_found": 0, "pages_read": 0,
                "time_seconds": 0, "timed_out": False,
                "sources": [], "content": "", "summary": "",
                "synthesis": "", "page_summaries": [], "knowledge_gaps": [],
                "phases_completed": 0, "cached_results": 0,
                "contradictions": [], "research_plan": {},
                "entities_tracked": 0, "saturated": False,
                "citations": [],
            }

        # Depth config
        max_pages = {"quick": 5, "standard": 10, "deep": 20}.get(depth, 10)
        max_phase = {"quick": 1, "standard": 4, "deep": 4}.get(depth, 4)

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
        contradictions = []
        research_plan = {}

        # ============================================================
        # PHASE 0: STORM research planning (standard + deep)
        # ============================================================
        if max_phase >= 2 and time.time() < overall_deadline:
            self._log_progress("Phase 0 — STORM research planning...")
            research_plan = self._plan_research(topic)
            if research_plan.get("perspectives"):
                perspectives_str = ", ".join(
                    p.get("name", "?") for p in research_plan.get("perspectives", [])
                )
                self._log_progress(
                    f"Phase 0 complete: {len(research_plan.get('outline', []))} sections, "
                    f"perspectives: {perspectives_str}"
                )
            # Emit plan to WebSocket (works for both LLM and default plans)
            self._emitter.emit_plan(
                research_plan.get("all_queries", []),
                research_plan.get("outline", []),
            )

        # ============================================================
        # PHASE 1: Broad search (with plan-derived queries + MMR)
        # ============================================================
        all_urls, queries_completed, cached = self._phase1_broad_search(
            topic, p1_deadline, plan=research_plan if research_plan else None
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
        # PHASE 2: Gap analysis + saturation check
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
        # PHASE 4: Synthesis + contradictions + citation anchoring
        # ============================================================
        if max_phase >= 4 and not timed_out and page_summaries and time.time() < overall_deadline:
            self._log_progress("Phase 4 — Synthesizing final report...")
            self._emitter.emit_synthesis_started()

            # Build source index for citations
            sources_for_synthesis = [
                {
                    "url": u.get("url", ""),
                    "title": u.get("title", ""),
                    "citation_score": u.get("citation_score", ""),
                }
                for u in all_urls[:max_pages]
            ]

            # Section-by-section synthesis if we have an outline
            outline = research_plan.get("outline", [])
            if outline and self.llm:
                self._log_progress(f"Phase 4 — Section-by-section synthesis ({len(outline)} sections)...")
                section_parts = []
                for section_title in outline:
                    if time.time() > overall_deadline:
                        break
                    section_text = self.summarizer.synthesize_section(
                        section_title, page_summaries, topic, sources_for_synthesis
                    )
                    if section_text:
                        section_parts.append(f"## {section_title}\n\n{section_text}")

                if section_parts:
                    synthesis = "\n\n".join(section_parts)
                else:
                    # Fallback to bulk synthesis
                    synthesis = self.summarizer.synthesize(page_summaries, topic)
            else:
                # No outline — use original bulk synthesis
                synthesis = self.summarizer.synthesize(page_summaries, topic)

            # Contradiction detection
            if time.time() < overall_deadline:
                self._log_progress("Phase 4 — Detecting contradictions...")
                contradictions = self.summarizer.detect_contradictions(
                    page_summaries, topic
                )
                if contradictions:
                    self._log_progress(f"Found {len(contradictions)} contradiction(s)")

            # Citation anchoring — verify all [N] refs are valid
            if synthesis:
                synthesis = self.summarizer.anchor_citations(synthesis, sources_for_synthesis)

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

        # Build sources list (used for both content and citations)
        sources_list = [
            {
                "url": u.get("url", ""),
                "title": u.get("title", ""),
                "snippet": u.get("snippet", "")[:200],
                "citation_score": u.get("citation_score", ""),
            }
            for u in all_urls[:max_pages]
        ]

        # Build final formatted report (if we have synthesis)
        final_report = ""
        structured_citations = []
        if synthesis and phases_completed >= 4:
            metadata = {
                "topic": topic, "phases_completed": phases_completed,
                "time_seconds": elapsed, "pages_read": len(pages),
                "entities_tracked": self.saturation.total_entities,
            }
            report_result = self.summarizer.build_final_report(
                synthesis, sources_list, metadata,
                contradictions=contradictions,
                outline=research_plan.get("outline"),
            )
            # build_final_report now returns {text, citations}
            if isinstance(report_result, dict):
                final_report = report_result.get("text", "")
                structured_citations = report_result.get("citations", [])
            else:
                final_report = report_result

        # If no structured citations from the report, build from synthesis text
        if not structured_citations and sources_list:
            text_for_citations = synthesis or content_summary
            structured_citations = self.summarizer.build_structured_citations(
                text_for_citations, sources_list
            )

        # If still no structured citations (no [N] markers), create from sources
        if not structured_citations and sources_list:
            structured_citations = [
                {
                    "id": i,
                    "url": s.get("url", ""),
                    "title": s.get("title", s.get("url", "Unknown")),
                    "snippet": s.get("snippet", "")[:200],
                    "relevance_score": s.get("citation_score", 0),
                }
                for i, s in enumerate(sources_list[:15], 1)
            ]

        summary_text = (
            f"Researched '{topic}': {queries_completed} searches, "
            f"{len(all_urls)} sources, {len(pages)} pages read, "
            f"{phases_completed} phases in {elapsed}s"
        )
        if self.saturation.is_saturated():
            summary_text += " (saturated)"
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
            "sources": sources_list,
            "content": final_report if final_report else content_summary,
            "summary": summary_text,
            # New keys
            "synthesis": synthesis,
            "page_summaries": page_summaries,
            "knowledge_gaps": knowledge_gaps,
            "phases_completed": phases_completed,
            "cached_results": total_cached,
            # v2 STORM keys
            "contradictions": contradictions,
            "research_plan": research_plan,
            "entities_tracked": self.saturation.total_entities,
            "saturated": self.saturation.is_saturated(),
            # v3 Structured citations for frontend rendering
            "citations": structured_citations,
        }

    # ------------------------------------------------------------------
    #  Pre-research clarification check
    # ------------------------------------------------------------------

    # Words too generic to form a specific research query on their own
    _COMMON_WORDS = frozenset({
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'must', 'about',
        'what', 'how', 'why', 'when', 'where', 'who', 'which', 'that', 'this',
        'it', 'they', 'we', 'you', 'i', 'me', 'my', 'our', 'your', 'their',
        'some', 'any', 'all', 'most', 'many', 'much', 'more', 'very', 'just',
        'also', 'so', 'too', 'not', 'no', 'or', 'and', 'but', 'if', 'then',
        'of', 'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'up',
        'out', 'into', 'over', 'after', 'before', 'between', 'under', 'above',
        'tell', 'find', 'get', 'know', 'look', 'think', 'want', 'give',
        'use', 'make', 'go', 'see', 'come', 'take', 'good', 'bad', 'new',
        'old', 'big', 'small', 'thing', 'things', 'stuff', 'something',
        'everything', 'nothing', 'research', 'search', 'information', 'info',
        'topic', 'subject', 'learn', 'explain', 'help', 'please',
    })

    # Patterns that signal a specific entity (URL, technical term, proper noun, etc.)
    _SPECIFIC_PATTERNS = re.compile(
        r'https?://'                          # URLs
        r'|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'   # Multi-word proper nouns
        r'|[A-Z]{2,}'                         # Acronyms (CRISPR, NATO, GPT)
        r'|\d{4}'                             # Years (2025)
        r'|v\d+\.\d+'                         # Version numbers
        r'|\S+\.\S+\.\S+'                     # Dotted identifiers (torch.nn.Module)
    )

    # Bypass phrases — user explicitly wants research without clarification
    _FORCE_PHRASES = re.compile(
        r'just\s+research|research\s+it|go\s+ahead|skip\s+clarif|do\s+it|'
        r'just\s+do\s+it|don.t\s+ask|no\s+questions',
        re.IGNORECASE,
    )

    def _needs_clarification(self, query: str, force: bool = False) -> Optional[str]:
        """Check if query is too vague for productive research.

        Returns a clarifying question string if vague, None if specific enough.
        Heuristics first (zero cost), LLM fallback only for borderline cases.
        """
        if force:
            return None

        if self._FORCE_PHRASES.search(query):
            return None

        query_stripped = query.strip()
        words = query_stripped.split()
        has_specific = bool(self._SPECIFIC_PATTERNS.search(query_stripped))

        # Clearly specific: has entities or is long enough with substance
        if has_specific or len(words) >= 12:
            return None

        # Count non-common (substantive) words
        non_common = [w for w in words if w.lower().strip('.,!?') not in self._COMMON_WORDS]

        # Clearly vague: short and all common words, OR single-word queries
        if (len(words) <= 5 and len(non_common) == 0) or len(words) <= 2:
            return (
                f"Your query \"{query_stripped}\" is quite broad. "
                "Could you narrow it down? For example, specify a particular aspect, "
                "time period, technology, person, or angle you're most interested in."
            )

        # Enough substance — let it through
        if len(non_common) >= 2 and len(words) >= 5:
            return None

        # Borderline — ask LLM for a quick verdict (only if available)
        if self.llm and len(non_common) <= 1 and len(words) < 8:
            try:
                verdict = self.llm(
                    f"Is this research query specific enough to get useful results, "
                    f"or is it too vague?\nQuery: \"{query_stripped}\"\n"
                    f"Reply with EXACTLY one line: SPECIFIC or VAGUE: <one clarifying question>",
                    "You classify research queries. Be strict — if the query could mean "
                    "many different things, it's VAGUE. Reply in one line only."
                )
                if verdict and verdict.strip().upper().startswith("VAGUE"):
                    parts = verdict.split(":", 1)
                    if len(parts) > 1 and parts[1].strip():
                        return parts[1].strip()
                    return (
                        f"Could you be more specific about \"{query_stripped}\"? "
                        "What particular aspect are you most interested in?"
                    )
            except Exception as e:
                logger.debug(f"[DeepResearch] Clarification LLM check failed: {e}")

        return None

    # ------------------------------------------------------------------
    #  run() entry point
    # ------------------------------------------------------------------

    def run(self, query: str, force: bool = False) -> Dict:
        """Main entry point with timeout protection.

        Args:
            query: Research query string.
            force: If True, skip the vagueness clarification check.
        """
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

        # --- Pre-research clarification gate ---
        # Use original case so entity-detection patterns (acronyms, proper nouns) work
        topic_original_case = re.sub(r'(?i)deep\s+research|research', '', query).strip() or query
        clarification = self._needs_clarification(topic_original_case, force=force)
        if clarification:
            logger.info(f"[DeepResearch] Query too vague, requesting clarification: {topic}")
            return {
                "needs_clarification": True,
                "question": clarification,
                "original_query": query,
                "success": False,
            }

        # Run with overall timeout protection
        try:
            future = self._executor.submit(self.research, topic, depth)
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
                        "contradictions": [], "research_plan": {},
                        "entities_tracked": 0, "saturated": False,
                        "citations": [],
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
                "contradictions": [], "research_plan": {},
                "entities_tracked": 0, "saturated": False,
                "citations": [],
            }


# ============================================================================
#  Convenience function
# ============================================================================

def deep_research(topic: str, depth: str = "standard") -> Dict:
    """Conduct deep research on a topic."""
    tool = DeepResearchTool()
    return tool.research(topic, depth)

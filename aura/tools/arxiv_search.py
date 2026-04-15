"""arXiv search tool for finding and downloading academic papers."""

import json
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import arxiv

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import ollama as _ollama_check
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import networkx as _nx_check
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from ..config import Config


def _normalize_arxiv_id(raw: Optional[str]) -> Optional[str]:
    """Strip 'arXiv:' prefix and version suffix (v1, v2) for stable equality.

    Examples:
        'arXiv:2301.00001v2' -> '2301.00001'
        '2301.00001'         -> '2301.00001'
        'cs.AI/0001001v1'    -> 'cs.AI/0001001'
    """
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.lower().startswith("arxiv:"):
        cleaned = cleaned[6:]
    cleaned = re.sub(r"v\d+$", "", cleaned)
    return cleaned or None


# ---------------------------------------------------------------------------
# SQLite cache for Semantic Scholar API responses
# ---------------------------------------------------------------------------

_S2_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "arxiv_cache.db"
_s2_conn: Optional[sqlite3.Connection] = None
_s2_cache_stats = {"hits": 0, "misses": 0}


def _s2_get_conn() -> sqlite3.Connection:
    """Lazy-init the SQLite connection (WAL mode, 24-hour TTL cache)."""
    global _s2_conn
    if _s2_conn is None:
        _S2_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _s2_conn = sqlite3.connect(str(_S2_DB_PATH), check_same_thread=False)
        _s2_conn.execute("PRAGMA journal_mode=WAL")
        _s2_conn.execute(
            "CREATE TABLE IF NOT EXISTS s2_cache "
            "(cache_key TEXT PRIMARY KEY, response_json TEXT, cached_at TEXT)"
        )
        _s2_conn.commit()
    return _s2_conn


def _s2_cache_get(key: str) -> Optional[dict]:
    """Return parsed JSON if cached < 24 hours ago, else None."""
    conn = _s2_get_conn()
    row = conn.execute(
        "SELECT response_json, cached_at FROM s2_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    if row is None:
        _s2_cache_stats["misses"] += 1
        return None
    cached_at = datetime.fromisoformat(row[1])
    if datetime.now(timezone.utc) - cached_at > timedelta(hours=24):
        _s2_cache_stats["misses"] += 1
        return None
    _s2_cache_stats["hits"] += 1
    return json.loads(row[0])


def _s2_cache_put(key: str, data: dict) -> None:
    """Store a JSON-serializable dict with an ISO timestamp."""
    conn = _s2_get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO s2_cache (cache_key, response_json, cached_at) VALUES (?, ?, ?)",
        (key, json.dumps(data), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _detect_citation_communities(
    center_id: str,
    center_title: Optional[str],
    citations: List[dict],
    references: List[dict],
) -> List[dict]:
    """Greedy-modularity community detection over the center paper's 1-hop
    citation graph. Returns [] if networkx is missing or the graph is trivial."""
    if not NETWORKX_AVAILABLE:
        return []
    try:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities
    except Exception:
        return []

    graph = nx.Graph()
    graph.add_node(center_id, title=center_title or "")

    for group, kind in [(citations, "cites"), (references, "refs")]:
        for item in group:
            node_id = item.get("arxiv_id") or item.get("title")
            if not node_id or node_id == center_id:
                continue
            graph.add_node(node_id, title=item.get("title") or "")
            graph.add_edge(center_id, node_id, kind=kind)

    if graph.number_of_nodes() < 3 or graph.number_of_edges() < 2:
        return []

    try:
        raw = list(greedy_modularity_communities(graph))
    except Exception:
        return []

    out: List[dict] = []
    for idx, community in enumerate(raw):
        members = list(community)
        top_papers = [
            {"arxiv_id": n, "title": graph.nodes[n].get("title", "")}
            for n in members[:5]
        ]
        out.append({
            "community_id": idx,
            "size": len(members),
            "top_papers": top_papers,
        })
    return out


def get_cache_stats() -> dict:
    """Return hit/miss counters for the Semantic Scholar cache."""
    total = _s2_cache_stats["hits"] + _s2_cache_stats["misses"]
    hit_rate = _s2_cache_stats["hits"] / total if total else 0.0
    return {**_s2_cache_stats, "total": total, "hit_rate": round(hit_rate, 3)}


class ArxivSearchTool:
    """Tool for searching arXiv and downloading papers."""

    name = "arxiv_search"
    description = "Search arXiv for academic papers, download PDFs, and extract abstracts"

    def __init__(self, download_dir: Optional[Path] = None):
        self.download_dir = download_dir or Path.cwd() / "arxiv_papers"
        self.client = arxiv.Client()

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance"
    ) -> dict:
        """
        Search arXiv for papers matching a query.

        Args:
            query: Search query (keywords, author, title, etc.)
            max_results: Maximum number of results to return
            sort_by: Sort order - 'relevance', 'submitted', or 'updated'
        """
        try:
            sort_criterion = {
                "relevance": arxiv.SortCriterion.Relevance,
                "submitted": arxiv.SortCriterion.SubmittedDate,
                "updated": arxiv.SortCriterion.LastUpdatedDate
            }.get(sort_by, arxiv.SortCriterion.Relevance)

            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_criterion
            )

            results = []
            seen_ids: set = set()
            for paper in self.client.results(search):
                short_id = paper.get_short_id()
                normalized = _normalize_arxiv_id(short_id)
                # Dedup: multiple versions of the same paper collapse to the latest entry.
                if normalized and normalized in seen_ids:
                    continue
                if normalized:
                    seen_ids.add(normalized)
                results.append({
                    "id": paper.entry_id,
                    "arxiv_id": normalized or short_id,
                    "arxiv_id_raw": short_id,
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "published": paper.published.isoformat() if paper.published else None,
                    "updated": paper.updated.isoformat() if paper.updated else None,
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                    "primary_category": paper.primary_category
                })

            return {
                "success": True,
                "query": query,
                "count": len(results),
                "results": results
            }
        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    def get_paper(self, arxiv_id: str) -> dict:
        """
        Get details of a specific paper by arXiv ID.

        Args:
            arxiv_id: The arXiv ID (e.g., '2301.00001' or 'cs.AI/0001001')
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(self.client.results(search), None)

            if not paper:
                return {"success": False, "error": f"Paper not found: {arxiv_id}"}

            return {
                "success": True,
                "paper": {
                    "id": paper.entry_id,
                    "arxiv_id": paper.get_short_id(),
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary,
                    "published": paper.published.isoformat() if paper.published else None,
                    "updated": paper.updated.isoformat() if paper.updated else None,
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url,
                    "primary_category": paper.primary_category,
                    "comment": paper.comment,
                    "journal_ref": paper.journal_ref,
                    "doi": paper.doi
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e), "arxiv_id": arxiv_id}

    def get_abstract(self, arxiv_id: str) -> dict:
        """
        Get just the abstract of a paper.

        Args:
            arxiv_id: The arXiv ID
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(self.client.results(search), None)

            if not paper:
                return {"success": False, "error": f"Paper not found: {arxiv_id}"}

            return {
                "success": True,
                "arxiv_id": arxiv_id,
                "title": paper.title,
                "abstract": paper.summary
            }
        except Exception as e:
            return {"success": False, "error": str(e), "arxiv_id": arxiv_id}

    def download_pdf(
        self,
        arxiv_id: str,
        filename: Optional[str] = None,
        directory: Optional[str] = None
    ) -> dict:
        """
        Download the PDF of a paper.

        Args:
            arxiv_id: The arXiv ID
            filename: Optional custom filename (without extension)
            directory: Optional download directory (uses default if not specified)
        """
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(self.client.results(search), None)

            if not paper:
                return {"success": False, "error": f"Paper not found: {arxiv_id}"}

            download_path = Path(directory) if directory else self.download_dir
            download_path.mkdir(parents=True, exist_ok=True)

            if filename:
                filepath = download_path / f"{filename}.pdf"
            else:
                safe_title = "".join(
                    c if c.isalnum() or c in " -_" else "_"
                    for c in paper.title[:50]
                ).strip()
                filepath = download_path / f"{paper.get_short_id()}_{safe_title}.pdf"

            paper.download_pdf(dirpath=str(download_path), filename=filepath.name)

            return {
                "success": True,
                "arxiv_id": arxiv_id,
                "title": paper.title,
                "path": str(filepath),
                "size_bytes": filepath.stat().st_size if filepath.exists() else None
            }
        except Exception as e:
            return {"success": False, "error": str(e), "arxiv_id": arxiv_id}

    def search_by_author(self, author: str, max_results: int = 10) -> dict:
        """
        Search for papers by a specific author.

        Args:
            author: Author name to search for
            max_results: Maximum number of results
        """
        return self.search(f'au:"{author}"', max_results=max_results)

    def search_by_category(
        self,
        category: str,
        query: Optional[str] = None,
        max_results: int = 10
    ) -> dict:
        """
        Search within a specific arXiv category.

        Args:
            category: arXiv category (e.g., 'cs.AI', 'physics.quant-ph')
            query: Optional additional search terms
            max_results: Maximum number of results
        """
        if query:
            full_query = f"cat:{category} AND ({query})"
        else:
            full_query = f"cat:{category}"
        return self.search(full_query, max_results=max_results, sort_by="submitted")

    def get_recent(self, category: str, max_results: int = 10) -> dict:
        """
        Get recent papers from a category.

        Args:
            category: arXiv category
            max_results: Maximum number of results
        """
        return self.search_by_category(category, max_results=max_results)

    def summarize_search(self, query: str, max_results: int = 5) -> dict:
        """
        Search arXiv and generate an AI-powered research summary.

        Searches for papers, collects abstracts, and uses the configured fast model to
        generate a comprehensive markdown summary comparing the papers.

        Args:
            query: Search query (keywords, topic, etc.)
            max_results: Maximum number of papers to include (default 5)

        Returns:
            dict with success status and markdown summary
        """
        try:
            # Search for papers
            search_result = self.search(query, max_results=max_results)
            if not search_result["success"]:
                return search_result

            papers = search_result["results"]
            if not papers:
                return {
                    "success": True,
                    "query": query,
                    "summary": f"# Research Summary: {query}\n\nNo papers found matching your query."
                }

            # Build context for the LLM
            papers_context = []
            for i, paper in enumerate(papers, 1):
                authors = ", ".join(paper["authors"][:3])
                if len(paper["authors"]) > 3:
                    authors += " et al."
                papers_context.append(
                    f"## Paper {i}: {paper['title']}\n"
                    f"**arXiv ID:** {paper['arxiv_id']}\n"
                    f"**Authors:** {authors}\n"
                    f"**Published:** {paper['published'][:10] if paper['published'] else 'N/A'}\n"
                    f"**Categories:** {', '.join(paper['categories'])}\n"
                    f"**Abstract:** {paper['abstract']}\n"
                )

            papers_text = "\n---\n".join(papers_context)

            # Generate summary using the configured fast model
            prompt = f"""You are a research assistant. Analyze the following {len(papers)} academic papers from arXiv about "{query}" and create a comprehensive research summary.

{papers_text}

---

Create a well-structured markdown summary that includes:

1. **Overview**: A brief introduction to the research area and why these papers are relevant
2. **Key Findings**: The main contributions and findings from each paper (be specific)
3. **Methodology**: For each paper, describe the experimental setup, dataset(s), model(s) and training regime — the details a reader would need to replicate the work
4. **Empirical Results**: Concrete numbers from each paper (benchmark scores, ablation deltas, wall-clock comparisons) when present in the abstract
5. **Limitations Acknowledged**: What each paper explicitly says it does NOT solve, or where it underperforms
6. **Comparative Analysis**: How do these papers relate to each other? Common themes, differences in approach, complementary findings
7. **Research Gaps**: Questions that remain unanswered or future directions suggested
8. **Relevance Assessment**: Rate each paper's relevance to the query (High/Medium/Low) with a brief justification

Format the output as clean markdown. Be concise but informative. When a section genuinely cannot be answered from the abstracts alone, write "Not discussed in abstracts" rather than inventing detail."""

            if not OLLAMA_AVAILABLE:
                return {"success": False, "error": "ollama not available for summarization"}
            import ollama

            response = ollama.chat(
                model=Config.get_model("fast"),
                messages=[{"role": "user", "content": prompt}]
            )

            summary_content = response["message"]["content"]

            # Build the final markdown summary
            markdown_summary = f"""# Research Summary: {query}

**Query:** {query}
**Papers Analyzed:** {len(papers)}
**Generated by:** {Config.get_model("fast")}

---

{summary_content}

---

## Papers Included

"""
            for paper in papers:
                markdown_summary += f"- [{paper['title']}]({paper['pdf_url']}) ({paper['arxiv_id']})\n"

            return {
                "success": True,
                "query": query,
                "papers_count": len(papers),
                "summary": markdown_summary,
                "papers": papers
            }

        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    # ------------------------------------------------------------------
    # Semantic Scholar: citation graph
    # ------------------------------------------------------------------

    def citation_graph(self, arxiv_id: str, depth: int = 1) -> dict:
        """
        Fetch citation graph (citations + references) from Semantic Scholar.

        Args:
            arxiv_id: arXiv paper ID (optionally prefixed with 'arXiv:')
            depth: How many levels deep to fetch (max 2)
        """
        if not REQUESTS_AVAILABLE:
            return {"success": False, "error": "requests library not installed"}

        import requests

        arxiv_id = _normalize_arxiv_id(arxiv_id) or arxiv_id
        depth = min(depth, 2)

        cache_key = f"citation_graph:{arxiv_id}:d{depth}"
        cached = _s2_cache_get(cache_key)
        if cached is not None:
            cached_copy = dict(cached)
            cached_copy["cache_hit"] = True
            return cached_copy

        fields = (
            "title,"
            "citations.title,citations.externalIds,"
            "references.title,references.externalIds"
        )
        url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{arxiv_id}?fields={fields}"

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Semantic Scholar returned HTTP {resp.status_code}",
                    "arxiv_id": arxiv_id,
                }
            data = resp.json()
        except Exception as e:
            return {"success": False, "error": str(e), "arxiv_id": arxiv_id}

        def _extract(items):
            out = []
            for item in (items or []):
                ext = item.get("externalIds") or {}
                out.append({
                    "title": item.get("title"),
                    "arxiv_id": _normalize_arxiv_id(ext.get("ArXiv")),
                })
            return out

        citations = _extract(data.get("citations"))
        references = _extract(data.get("references"))

        # Community detection: build a small citation graph of the paper and its
        # immediate references, then run greedy modularity to surface clusters.
        communities = _detect_citation_communities(arxiv_id, data.get("title"), citations, references)

        result = {
            "success": True,
            "paper": {"id": arxiv_id, "title": data.get("title")},
            "citations": citations,
            "references": references,
            "citation_count": len(citations),
            "reference_count": len(references),
            "communities": communities,
            "cache_hit": False,
        }

        # Depth 2: fetch one level deeper for each citation that has an arxiv_id
        if depth > 1:
            deeper = []
            for cit in citations:
                if not cit.get("arxiv_id"):
                    continue
                time.sleep(1)  # rate-limit courtesy
                sub = self.citation_graph(cit["arxiv_id"], depth=1)
                if sub.get("success"):
                    deeper.append(sub)
            result["deeper_citations"] = deeper

        _s2_cache_put(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Semantic Scholar: related papers (recommendations)
    # ------------------------------------------------------------------

    def related_papers(self, arxiv_id: str, max_results: int = 10) -> dict:
        """
        Get related paper recommendations from Semantic Scholar.

        Args:
            arxiv_id: arXiv paper ID
            max_results: Maximum number of recommendations (default 10)
        """
        if not REQUESTS_AVAILABLE:
            return {"success": False, "error": "requests library not installed"}

        import requests

        arxiv_id = _normalize_arxiv_id(arxiv_id) or arxiv_id

        cache_key = f"related_papers:{arxiv_id}:{max_results}"
        cached = _s2_cache_get(cache_key)
        if cached is not None:
            cached_copy = dict(cached)
            cached_copy["cache_hit"] = True
            return cached_copy

        url = (
            f"https://api.semanticscholar.org/recommendations/v1/papers/"
            f"forpaper/ArXiv:{arxiv_id}"
            f"?fields=title,externalIds,abstract,year&limit={max_results}"
        )

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Semantic Scholar returned HTTP {resp.status_code}",
                    "arxiv_id": arxiv_id,
                }
            data = resp.json()
        except Exception as e:
            return {"success": False, "error": str(e), "arxiv_id": arxiv_id}

        papers = []
        for item in data.get("recommendedPapers", []):
            ext = item.get("externalIds") or {}
            papers.append({
                "title": item.get("title"),
                "arxiv_id": _normalize_arxiv_id(ext.get("ArXiv")),
                "abstract": item.get("abstract"),
                "year": item.get("year"),
            })

        result = {
            "success": True,
            "arxiv_id": arxiv_id,
            "count": len(papers),
            "related": papers,
            "cache_hit": False,
        }
        _s2_cache_put(cache_key, result)
        return result

    def get_cache_stats(self) -> dict:
        """Hit/miss counters for the Semantic Scholar cache. Shared across all instances."""
        return get_cache_stats()

    # ------------------------------------------------------------------
    # BibTeX generation
    # ------------------------------------------------------------------

    def bibtex(self, arxiv_id: str) -> dict:
        """
        Generate a BibTeX @article entry for an arXiv paper.

        Args:
            arxiv_id: arXiv paper ID
        """
        arxiv_id = arxiv_id.replace("arXiv:", "").strip()
        paper_result = self.get_paper(arxiv_id)
        if not paper_result.get("success"):
            return paper_result

        p = paper_result["paper"]
        authors = " and ".join(p.get("authors", []))
        # Extract year from published date
        year = p.get("published", "")[:4] if p.get("published") else "n.d."
        # Build a cite key from first author surname + year
        first_author = (p.get("authors") or ["unknown"])[0].split()[-1].lower()
        cite_key = f"{first_author}{year}"

        bib = (
            f"@article{{{cite_key},\n"
            f"  title = {{{p.get('title', '')}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year = {{{year}}},\n"
            f"  journal = {{arXiv preprint arXiv:{p.get('arxiv_id', arxiv_id)}}},\n"
            f"  eprint = {{{p.get('arxiv_id', arxiv_id)}}},\n"
            f"  archivePrefix = {{arXiv}},\n"
            f"  primaryClass = {{{p.get('primary_category', '')}}}\n"
            f"}}"
        )

        return {"success": True, "bibtex": bib}

    # ------------------------------------------------------------------
    # Batch summarize (comparative)
    # ------------------------------------------------------------------

    def batch_summarize(self, arxiv_ids: List[str]) -> dict:
        """
        Fetch multiple papers and produce a comparative AI summary.

        Args:
            arxiv_ids: List of arXiv IDs to compare
        """
        if not OLLAMA_AVAILABLE:
            return {"success": False, "error": "ollama not available for summarization"}

        import ollama

        papers = []
        for aid in arxiv_ids:
            result = self.get_paper(aid.replace("arXiv:", "").strip())
            if result.get("success"):
                papers.append(result["paper"])

        if not papers:
            return {"success": False, "error": "No papers could be fetched"}

        # Build combined prompt
        papers_context = []
        for i, p in enumerate(papers, 1):
            authors = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            papers_context.append(
                f"## Paper {i}: {p['title']}\n"
                f"**arXiv ID:** {p.get('arxiv_id', 'N/A')}\n"
                f"**Authors:** {authors}\n"
                f"**Published:** {(p.get('published') or 'N/A')[:10]}\n"
                f"**Abstract:** {p.get('abstract', 'N/A')}\n"
            )

        papers_text = "\n---\n".join(papers_context)

        prompt = (
            f"You are a research assistant. Compare the following {len(papers)} academic papers "
            f"and produce a concise comparative summary.\n\n"
            f"{papers_text}\n\n---\n\n"
            f"Provide:\n"
            f"1. **Common themes** across the papers\n"
            f"2. **Key differences** in approach or findings\n"
            f"3. **Methodology**: experimental setup, datasets, models, training regime for each paper\n"
            f"4. **Empirical Results**: concrete benchmark numbers, ablations, or measured deltas when present\n"
            f"5. **Limitations Acknowledged**: what each paper explicitly says it does not solve or where it underperforms\n"
            f"6. **Research gaps** and potential future directions\n\n"
            f"Format as clean markdown. When a section cannot be answered from abstracts alone, "
            f"write 'Not discussed in abstracts' rather than inventing detail."
        )

        try:
            response = ollama.chat(
                model=Config.get_model("fast"),
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response["message"]["content"]
        except Exception as e:
            return {"success": False, "error": f"LLM summarization failed: {e}"}

        return {
            "success": True,
            "papers_count": len(papers),
            "summary": summary,
            "papers": papers,
        }

    def execute(self, action: str, **kwargs) -> dict:
        """Execute an arXiv action by name."""
        actions = {
            "search": self.search,
            "get_paper": self.get_paper,
            "get_abstract": self.get_abstract,
            "download": self.download_pdf,
            "by_author": self.search_by_author,
            "by_category": self.search_by_category,
            "recent": self.get_recent,
            "summarize": self.summarize_search,
            "citation_graph": self.citation_graph,
            "citations": self.citation_graph,
            "related": self.related_papers,
            "related_papers": self.related_papers,
            "bibtex": self.bibtex,
            "cite": self.bibtex,
            "batch_summarize": self.batch_summarize,
            "compare_papers": self.batch_summarize,
        }
        if action not in actions:
            return {"success": False, "error": f"Unknown action: {action}"}
        return actions[action](**kwargs)

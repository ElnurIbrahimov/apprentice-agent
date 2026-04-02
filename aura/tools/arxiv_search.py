"""arXiv search tool for finding and downloading academic papers."""

from pathlib import Path
from typing import Optional, List
import arxiv
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone

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

from ..config import Config


# ---------------------------------------------------------------------------
# SQLite cache for Semantic Scholar API responses
# ---------------------------------------------------------------------------

_S2_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "arxiv_cache.db"
_s2_conn: Optional[sqlite3.Connection] = None


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
        return None
    cached_at = datetime.fromisoformat(row[1])
    if datetime.now(timezone.utc) - cached_at > timedelta(hours=24):
        return None
    return json.loads(row[0])


def _s2_cache_put(key: str, data: dict) -> None:
    """Store a JSON-serializable dict with an ISO timestamp."""
    conn = _s2_get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO s2_cache (cache_key, response_json, cached_at) VALUES (?, ?, ?)",
        (key, json.dumps(data), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


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
            for paper in self.client.results(search):
                results.append({
                    "id": paper.entry_id,
                    "arxiv_id": paper.get_short_id(),
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
3. **Comparative Analysis**: How do these papers relate to each other? What are the common themes, differences in approach, or complementary findings?
4. **Research Gaps**: What questions remain unanswered or what future directions are suggested?
5. **Relevance Assessment**: Rate each paper's relevance to the query (High/Medium/Low) with a brief justification

Format the output as clean markdown. Be concise but informative."""

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

        arxiv_id = arxiv_id.replace("arXiv:", "").strip()
        depth = min(depth, 2)

        cache_key = f"citation_graph:{arxiv_id}:d{depth}"
        cached = _s2_cache_get(cache_key)
        if cached is not None:
            return cached

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
                    "arxiv_id": ext.get("ArXiv"),
                })
            return out

        citations = _extract(data.get("citations"))
        references = _extract(data.get("references"))

        result = {
            "success": True,
            "paper": {"id": arxiv_id, "title": data.get("title")},
            "citations": citations,
            "references": references,
            "citation_count": len(citations),
            "reference_count": len(references),
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

        arxiv_id = arxiv_id.replace("arXiv:", "").strip()

        cache_key = f"related_papers:{arxiv_id}:{max_results}"
        cached = _s2_cache_get(cache_key)
        if cached is not None:
            return cached

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
                "arxiv_id": ext.get("ArXiv"),
                "abstract": item.get("abstract"),
                "year": item.get("year"),
            })

        result = {
            "success": True,
            "arxiv_id": arxiv_id,
            "count": len(papers),
            "related": papers,
        }
        _s2_cache_put(cache_key, result)
        return result

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
            f"3. **Strengths and limitations** of each\n"
            f"4. **Research gaps** and potential future directions\n\n"
            f"Format as clean markdown."
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

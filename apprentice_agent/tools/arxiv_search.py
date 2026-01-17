"""arXiv search tool for finding and downloading academic papers."""

from pathlib import Path
from typing import Optional, List
import arxiv


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

    def execute(self, action: str, **kwargs) -> dict:
        """Execute an arXiv action by name."""
        actions = {
            "search": self.search,
            "get_paper": self.get_paper,
            "get_abstract": self.get_abstract,
            "download": self.download_pdf,
            "by_author": self.search_by_author,
            "by_category": self.search_by_category,
            "recent": self.get_recent
        }
        if action not in actions:
            return {"success": False, "error": f"Unknown action: {action}"}
        return actions[action](**kwargs)

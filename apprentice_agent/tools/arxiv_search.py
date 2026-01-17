"""arXiv search tool for finding and downloading academic papers."""

from pathlib import Path
from typing import Optional, List
import arxiv
import ollama


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

        Searches for papers, collects abstracts, and uses llama3:8b to
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

            # Generate summary using llama3:8b
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

            response = ollama.chat(
                model="llama3:8b",
                messages=[{"role": "user", "content": prompt}]
            )

            summary_content = response["message"]["content"]

            # Build the final markdown summary
            markdown_summary = f"""# Research Summary: {query}

**Query:** {query}
**Papers Analyzed:** {len(papers)}
**Generated by:** llama3:8b

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
            "summarize": self.summarize_search
        }
        if action not in actions:
            return {"success": False, "error": f"Unknown action: {action}"}
        return actions[action](**kwargs)

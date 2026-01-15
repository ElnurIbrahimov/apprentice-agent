"""PDF reader tool for extracting and analyzing PDF content."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional, Union
import re


class PDFReaderTool:
    """Tool for reading and analyzing PDF documents."""

    MAX_CHARS = 32000  # Maximum characters to extract

    def __init__(self):
        """Initialize PDF reader tool."""
        pass

    def _parse_page_spec(self, page_spec: str, total_pages: int) -> list[int]:
        """Parse page specification string into list of page numbers (0-indexed).

        Args:
            page_spec: Page specification like "1-3,5", "all", "last", "1,3,5"
            total_pages: Total number of pages in the document

        Returns:
            List of 0-indexed page numbers
        """
        if not page_spec or page_spec.lower() == "all":
            return list(range(total_pages))

        if page_spec.lower() == "last":
            return [total_pages - 1]

        if page_spec.lower() == "first":
            return [0]

        pages = set()
        parts = page_spec.replace(" ", "").split(",")

        for part in parts:
            if "-" in part:
                # Range like "1-3"
                try:
                    start, end = part.split("-")
                    start = int(start) - 1  # Convert to 0-indexed
                    end = int(end) - 1
                    start = max(0, min(start, total_pages - 1))
                    end = max(0, min(end, total_pages - 1))
                    pages.update(range(start, end + 1))
                except ValueError:
                    continue
            else:
                # Single page like "5"
                try:
                    page = int(part) - 1  # Convert to 0-indexed
                    if 0 <= page < total_pages:
                        pages.add(page)
                except ValueError:
                    continue

        return sorted(list(pages))

    def info(self, path: str) -> dict:
        """Get PDF metadata and page count.

        Args:
            path: Path to the PDF file

        Returns:
            dict with success status, page count, and metadata
        """
        pdf_path = Path(path)

        if not pdf_path.exists():
            return {
                "success": False,
                "error": f"PDF not found: {path}"
            }

        if pdf_path.suffix.lower() != ".pdf":
            return {
                "success": False,
                "error": f"Not a PDF file: {path}"
            }

        try:
            doc = fitz.open(str(pdf_path))
            metadata = doc.metadata

            result = {
                "success": True,
                "path": str(pdf_path.absolute()),
                "page_count": len(doc),
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
            }

            doc.close()
            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read PDF info: {str(e)}"
            }

    def extract_text(self, path: str, pages: str = "all") -> dict:
        """Extract text from PDF pages with page markers.

        Args:
            path: Path to the PDF file
            pages: Page specification (e.g., "1-3,5", "all", "last")

        Returns:
            dict with success status and extracted text
        """
        pdf_path = Path(path)

        if not pdf_path.exists():
            return {
                "success": False,
                "error": f"PDF not found: {path}"
            }

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            page_numbers = self._parse_page_spec(pages, total_pages)

            if not page_numbers:
                doc.close()
                return {
                    "success": False,
                    "error": f"Invalid page specification: {pages}"
                }

            extracted_text = []
            total_chars = 0

            for page_num in page_numbers:
                if total_chars >= self.MAX_CHARS:
                    extracted_text.append(f"\n[... truncated at {self.MAX_CHARS} characters ...]")
                    break

                page = doc[page_num]
                text = page.get_text()

                # Add page marker
                page_header = f"\n{'='*40}\n[Page {page_num + 1} of {total_pages}]\n{'='*40}\n"

                # Check if adding this page would exceed limit
                if total_chars + len(page_header) + len(text) > self.MAX_CHARS:
                    remaining = self.MAX_CHARS - total_chars - len(page_header) - 50
                    if remaining > 0:
                        extracted_text.append(page_header)
                        extracted_text.append(text[:remaining])
                        extracted_text.append("\n[... page truncated ...]")
                    break

                extracted_text.append(page_header)
                extracted_text.append(text)
                total_chars += len(page_header) + len(text)

            doc.close()

            full_text = "".join(extracted_text)

            return {
                "success": True,
                "path": str(pdf_path.absolute()),
                "pages_extracted": page_numbers,
                "total_pages": total_pages,
                "char_count": len(full_text),
                "text": full_text
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to extract text: {str(e)}"
            }

    def search(self, path: str, query: str) -> dict:
        """Search for text in PDF and return matching pages.

        Args:
            path: Path to the PDF file
            query: Text to search for (case-insensitive)

        Returns:
            dict with success status and matching pages with context
        """
        pdf_path = Path(path)

        if not pdf_path.exists():
            return {
                "success": False,
                "error": f"PDF not found: {path}"
            }

        if not query:
            return {
                "success": False,
                "error": "Search query cannot be empty"
            }

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            query_lower = query.lower()

            matches = []

            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()

                if query_lower in text.lower():
                    # Find context around the match
                    text_lower = text.lower()
                    idx = text_lower.find(query_lower)
                    start = max(0, idx - 100)
                    end = min(len(text), idx + len(query) + 100)
                    context = text[start:end].strip()

                    # Count occurrences on this page
                    count = text_lower.count(query_lower)

                    matches.append({
                        "page": page_num + 1,
                        "occurrences": count,
                        "context": f"...{context}..."
                    })

            doc.close()

            return {
                "success": True,
                "path": str(pdf_path.absolute()),
                "query": query,
                "total_pages": total_pages,
                "pages_with_matches": len(matches),
                "matches": matches
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to search PDF: {str(e)}"
            }

    def read(self, path: str, pages: str = "all", summarize: bool = False) -> dict:
        """Main entry point for reading PDF content.

        Args:
            path: Path to the PDF file
            pages: Page specification (e.g., "1-3,5", "all", "last")
            summarize: If True, return a brief summary instead of full text

        Returns:
            dict with success status and content
        """
        # First get info
        info_result = self.info(path)
        if not info_result.get("success"):
            return info_result

        # Extract text
        extract_result = self.extract_text(path, pages)
        if not extract_result.get("success"):
            return extract_result

        result = {
            "success": True,
            "path": extract_result["path"],
            "total_pages": extract_result["total_pages"],
            "pages_read": extract_result["pages_extracted"],
            "title": info_result.get("title", ""),
            "author": info_result.get("author", ""),
        }

        if summarize:
            # Return condensed info for summarization
            text = extract_result["text"]
            # Truncate for summarization if needed
            if len(text) > 8000:
                text = text[:8000] + "\n[... content truncated for summarization ...]"
            result["text"] = text
            result["note"] = "Content prepared for summarization"
        else:
            result["text"] = extract_result["text"]

        return result

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a PDF action.

        Args:
            action: Action to perform (info, read, extract, search)
            **kwargs: Additional arguments

        Returns:
            dict with action result
        """
        action_lower = action.lower()

        # Extract path from action or kwargs
        path = kwargs.get("path")
        if not path:
            path = self._extract_path(action)

        if not path:
            return {
                "success": False,
                "error": "No PDF path provided. Specify the path to the PDF file."
            }

        # Determine action type
        if "info" in action_lower or "metadata" in action_lower:
            return self.info(path)

        elif "search" in action_lower or "find" in action_lower:
            query = kwargs.get("query")
            if not query:
                query = self._extract_query(action)
            if not query:
                return {
                    "success": False,
                    "error": "No search query provided."
                }
            return self.search(path, query)

        elif "extract" in action_lower:
            pages = kwargs.get("pages", "all")
            return self.extract_text(path, pages)

        else:
            # Default: read with optional summarization
            pages = kwargs.get("pages", "all")
            summarize = "summar" in action_lower
            return self.read(path, pages, summarize)

    def _extract_path(self, action: str) -> Optional[str]:
        """Extract PDF path from action string."""
        # Look for quoted paths
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            for q in quoted:
                if q.lower().endswith('.pdf'):
                    return q

        # Look for paths with .pdf extension
        path_pattern = r'[\w./\\:-]+\.pdf'
        paths = re.findall(path_pattern, action, re.IGNORECASE)
        if paths:
            return paths[0]

        # Look for Windows paths
        win_paths = re.findall(r'[A-Za-z]:[/\\][\w./\\-]+', action)
        for wp in win_paths:
            if '.pdf' in wp.lower():
                return wp

        return None

    def _extract_query(self, action: str) -> Optional[str]:
        """Extract search query from action string."""
        # Look for quoted strings
        quoted = re.findall(r'["\']([^"\']+)["\']', action)
        if quoted:
            # Return first quoted string that's not a path
            for q in quoted:
                if not q.lower().endswith('.pdf'):
                    return q

        # Look for "for X" or "search X" patterns
        patterns = [
            r'(?:search|find|look)\s+(?:for\s+)?["\']?([^"\']+?)["\']?\s+in',
            r'search\s+["\']?(.+?)["\']?\s*$',
        ]
        for pattern in patterns:
            match = re.search(pattern, action, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None


# Singleton instance for easy import
pdf_reader_tool = PDFReaderTool()

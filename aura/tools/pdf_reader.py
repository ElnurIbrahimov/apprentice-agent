"""PDF reader tool for extracting and analyzing PDF content."""

import os

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    fitz = None
    FITZ_AVAILABLE = False
import re
from typing import Optional


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

        return sorted(pages)

    def info(self, path: str) -> dict:
        """Get PDF metadata and page count.

        Args:
            path: Path to the PDF file

        Returns:
            dict with success status, page count, and metadata
        """
        from pathlib import Path as _Path
        DOCS_DIR = _Path(__file__).parent.parent.parent / "data"
        resolved = _Path(path).resolve()
        docs_dir_resolved = DOCS_DIR.resolve()
        if not (str(resolved).startswith(str(docs_dir_resolved) + os.sep) or str(resolved) == str(docs_dir_resolved)):
            return {"success": False, "error": "Path not within allowed data directory"}

        if not FITZ_AVAILABLE:
            return {"success": False, "error": "PyMuPDF (fitz) not installed. Run: pip install PyMuPDF"}

        # SECURITY: Use the resolved path for all operations to prevent TOCTOU
        pdf_path = resolved

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
                "error": f"Failed to read PDF info: {e!s}"
            }

    def extract_text(self, path: str, pages: str = "all") -> dict:
        """Extract text from PDF pages with page markers.

        Args:
            path: Path to the PDF file
            pages: Page specification (e.g., "1-3,5", "all", "last")

        Returns:
            dict with success status and extracted text
        """
        if not FITZ_AVAILABLE:
            return {"success": False, "error": "PyMuPDF (fitz) not installed. Run: pip install PyMuPDF"}

        from pathlib import Path as _Path
        DOCS_DIR = _Path(__file__).parent.parent.parent / "data"
        resolved = _Path(path).resolve()
        docs_dir_resolved = DOCS_DIR.resolve()
        if not (str(resolved).startswith(str(docs_dir_resolved) + os.sep) or str(resolved) == str(docs_dir_resolved)):
            return {"success": False, "error": "Path not within allowed data directory"}

        # SECURITY: Use the resolved path for all operations to prevent TOCTOU
        pdf_path = resolved

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
                "error": f"Failed to extract text: {e!s}"
            }

    def search(self, path: str, query: str) -> dict:
        """Search for text in PDF and return matching pages.

        Args:
            path: Path to the PDF file
            query: Text to search for (case-insensitive)

        Returns:
            dict with success status and matching pages with context
        """
        if not FITZ_AVAILABLE:
            return {"success": False, "error": "PyMuPDF (fitz) not installed. Run: pip install PyMuPDF"}

        from pathlib import Path as _Path
        DOCS_DIR = _Path(__file__).parent.parent.parent / "data"
        resolved = _Path(path).resolve()
        docs_dir_resolved = DOCS_DIR.resolve()
        if not (str(resolved).startswith(str(docs_dir_resolved) + os.sep) or str(resolved) == str(docs_dir_resolved)):
            return {"success": False, "error": "Path not within allowed data directory"}

        # SECURITY: Use the resolved path for all operations to prevent TOCTOU
        pdf_path = resolved

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
                "error": f"Failed to search PDF: {e!s}"
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

    def extract_tables(self, path: str, pages: str = "all") -> dict:
        """Extract tables from PDF pages.

        Args:
            path: Path to the PDF file
            pages: Page specification (e.g., "1-3,5", "all", "last")

        Returns:
            dict with success status and extracted tables
        """
        if not FITZ_AVAILABLE:
            return {"success": False, "error": "PyMuPDF (fitz) not installed. Run: pip install PyMuPDF"}

        from pathlib import Path as _Path
        DOCS_DIR = _Path(__file__).parent.parent.parent / "data"
        resolved = _Path(path).resolve()
        docs_dir_resolved = DOCS_DIR.resolve()
        if not (str(resolved).startswith(str(docs_dir_resolved) + os.sep) or str(resolved) == str(docs_dir_resolved)):
            return {"success": False, "error": "Path not within allowed data directory"}

        pdf_path = resolved

        if not pdf_path.exists():
            return {"success": False, "error": f"PDF not found: {path}"}

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            page_numbers = self._parse_page_spec(pages, total_pages)

            if not page_numbers:
                doc.close()
                return {"success": False, "error": f"Invalid page specification: {pages}"}

            tables = []

            for page_num in page_numbers:
                page = doc[page_num]

                if not hasattr(page, 'find_tables'):
                    doc.close()
                    return {
                        "success": False,
                        "error": "PyMuPDF version too old for table extraction. Upgrade: pip install --upgrade PyMuPDF"
                    }

                found = page.find_tables()
                for table in found.tables:
                    raw = table.extract()
                    if len(raw) < 2:
                        continue
                    headers = [str(h) if h is not None else "" for h in raw[0]]
                    rows = []
                    for row in raw[1:]:
                        rows.append({headers[i]: (str(cell) if cell is not None else "") for i, cell in enumerate(row) if i < len(headers)})
                    tables.append({
                        "page": page_num + 1,
                        "headers": headers,
                        "rows": rows,
                        "row_count": len(rows)
                    })

            doc.close()

            return {
                "success": True,
                "tables": tables,
                "total_tables": len(tables)
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to extract tables: {e!s}"}

    def extract_structured(self, path: str, pages: str = "all") -> dict:
        """Extract structured content (headers, paragraphs, tables) with font/position info.

        Args:
            path: Path to the PDF file
            pages: Page specification (e.g., "1-3,5", "all", "last")

        Returns:
            dict with success status and structured sections
        """
        if not FITZ_AVAILABLE:
            return {"success": False, "error": "PyMuPDF (fitz) not installed. Run: pip install PyMuPDF"}

        from collections import Counter
        from pathlib import Path as _Path
        DOCS_DIR = _Path(__file__).parent.parent.parent / "data"
        resolved = _Path(path).resolve()
        docs_dir_resolved = DOCS_DIR.resolve()
        if not (str(resolved).startswith(str(docs_dir_resolved) + os.sep) or str(resolved) == str(docs_dir_resolved)):
            return {"success": False, "error": "Path not within allowed data directory"}

        pdf_path = resolved

        if not pdf_path.exists():
            return {"success": False, "error": f"PDF not found: {path}"}

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            page_numbers = self._parse_page_spec(pages, total_pages)

            if not page_numbers:
                doc.close()
                return {"success": False, "error": f"Invalid page specification: {pages}"}

            sections = []

            for page_num in page_numbers:
                page = doc[page_num]
                text_dict = page.get_text("dict")
                blocks = text_dict.get("blocks", [])

                # Collect font sizes from text spans to find body size
                font_sizes = []
                for block in blocks:
                    if block.get("type") == 0:  # text block
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                text = span.get("text", "").strip()
                                if text:
                                    font_sizes.extend([round(span["size"], 1)] * len(text))

                body_size = Counter(font_sizes).most_common(1)[0][0] if font_sizes else 12.0

                # Detect multi-column layout by analyzing x-coordinates
                x_positions = []
                for block in blocks:
                    if block.get("type") == 0:
                        x_positions.append(round(block.get("bbox", [0])[0], 0))
                len({x for x in x_positions if x_positions.count(x) >= 2}) if x_positions else 1

                # Extract inline tables for this page
                page_tables = []
                if hasattr(page, 'find_tables'):
                    found = page.find_tables()
                    for table in found.tables:
                        raw = table.extract()
                        if len(raw) >= 2:
                            headers = [str(h) if h is not None else "" for h in raw[0]]
                            rows = []
                            for row in raw[1:]:
                                rows.append({headers[i]: (str(cell) if cell is not None else "") for i, cell in enumerate(row) if i < len(headers)})
                            page_tables.append({
                                "headers": headers,
                                "rows": rows,
                                "row_count": len(rows)
                            })

                # Process text blocks into sections
                for block in blocks:
                    if block.get("type") == 0:  # text block
                        text_parts = []
                        max_font_size = 0.0
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                t = span.get("text", "")
                                if t.strip():
                                    text_parts.append(t)
                                    if span["size"] > max_font_size:
                                        max_font_size = span["size"]

                        content = " ".join(text_parts).strip()
                        if not content:
                            continue

                        rounded_size = round(max_font_size, 1)
                        if rounded_size > body_size + 1.0:
                            block_type = "header"
                        else:
                            block_type = "paragraph"

                        sections.append({
                            "type": block_type,
                            "content": content,
                            "page": page_num + 1,
                            "font_size": rounded_size
                        })

                # Embed tables inline
                for tbl in page_tables:
                    header_str = " | ".join(tbl["headers"])
                    row_strs = []
                    for row in tbl["rows"]:
                        row_strs.append(" | ".join(row.get(h, "") for h in tbl["headers"]))
                    content = header_str + "\n" + "\n".join(row_strs)
                    sections.append({
                        "type": "table",
                        "content": content,
                        "page": page_num + 1,
                        "font_size": body_size
                    })

            doc.close()

            return {
                "success": True,
                "sections": sections
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to extract structured content: {e!s}"}

    def extract_images(self, path: str, pages: str = "all", output_dir: str | None = None) -> dict:
        """Extract images from PDF pages.

        Args:
            path: Path to the PDF file
            pages: Page specification (e.g., "1-3,5", "all", "last")
            output_dir: Optional directory to save images as PNG (must be within data/)

        Returns:
            dict with success status and image metadata
        """
        if not FITZ_AVAILABLE:
            return {"success": False, "error": "PyMuPDF (fitz) not installed. Run: pip install PyMuPDF"}

        from pathlib import Path as _Path
        DOCS_DIR = _Path(__file__).parent.parent.parent / "data"
        resolved = _Path(path).resolve()
        docs_dir_resolved = DOCS_DIR.resolve()
        if not (str(resolved).startswith(str(docs_dir_resolved) + os.sep) or str(resolved) == str(docs_dir_resolved)):
            return {"success": False, "error": "Path not within allowed data directory"}

        if output_dir:
            out_resolved = _Path(output_dir).resolve()
            if not (str(out_resolved).startswith(str(docs_dir_resolved) + os.sep) or str(out_resolved) == str(docs_dir_resolved)):
                return {"success": False, "error": "output_dir must be within the data/ directory"}
            out_resolved.mkdir(parents=True, exist_ok=True)
        else:
            out_resolved = None

        pdf_path = resolved

        if not pdf_path.exists():
            return {"success": False, "error": f"PDF not found: {path}"}

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            page_numbers = self._parse_page_spec(pages, total_pages)

            if not page_numbers:
                doc.close()
                return {"success": False, "error": f"Invalid page specification: {pages}"}

            images = []
            img_index = 0

            for page_num in page_numbers:
                page = doc[page_num]
                image_list = page.get_images()

                for img_info in image_list:
                    xref = img_info[0]
                    try:
                        pix = fitz.Pixmap(doc, xref)
                        # Convert CMYK to RGB if needed
                        if pix.n - pix.alpha > 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        saved_path = None
                        if out_resolved:
                            img_filename = f"page{page_num + 1}_img{img_index}.png"
                            save_path = out_resolved / img_filename
                            pix.save(str(save_path))
                            saved_path = str(save_path)

                        images.append({
                            "page": page_num + 1,
                            "width": pix.width,
                            "height": pix.height,
                            "path": saved_path
                        })

                        pix = None
                        img_index += 1
                    except Exception:
                        continue

            doc.close()

            return {
                "success": True,
                "images": images,
                "total_images": len(images)
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to extract images: {e!s}"}

    def extract_links(self, path: str) -> dict:
        """Extract URI links from all PDF pages.

        Args:
            path: Path to the PDF file

        Returns:
            dict with success status and extracted links
        """
        if not FITZ_AVAILABLE:
            return {"success": False, "error": "PyMuPDF (fitz) not installed. Run: pip install PyMuPDF"}

        from pathlib import Path as _Path
        DOCS_DIR = _Path(__file__).parent.parent.parent / "data"
        resolved = _Path(path).resolve()
        docs_dir_resolved = DOCS_DIR.resolve()
        if not (str(resolved).startswith(str(docs_dir_resolved) + os.sep) or str(resolved) == str(docs_dir_resolved)):
            return {"success": False, "error": "Path not within allowed data directory"}

        pdf_path = resolved

        if not pdf_path.exists():
            return {"success": False, "error": f"PDF not found: {path}"}

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
            links = []

            for page_num in range(total_pages):
                page = doc[page_num]
                for link in page.get_links():
                    if link.get("kind") == 2:  # URI link
                        uri = link.get("uri", "")
                        if uri:
                            links.append({
                                "page": page_num + 1,
                                "uri": uri
                            })

            doc.close()

            return {
                "success": True,
                "links": links,
                "total_links": len(links)
            }

        except Exception as e:
            return {"success": False, "error": f"Failed to extract links: {e!s}"}

    def page_info(self, path: str, page_num: int) -> dict:
        """Get detailed info about a specific PDF page.

        Args:
            path: Path to the PDF file
            page_num: 1-indexed page number

        Returns:
            dict with success status and page details
        """
        if not FITZ_AVAILABLE:
            return {"success": False, "error": "PyMuPDF (fitz) not installed. Run: pip install PyMuPDF"}

        from pathlib import Path as _Path
        DOCS_DIR = _Path(__file__).parent.parent.parent / "data"
        resolved = _Path(path).resolve()
        docs_dir_resolved = DOCS_DIR.resolve()
        if not (str(resolved).startswith(str(docs_dir_resolved) + os.sep) or str(resolved) == str(docs_dir_resolved)):
            return {"success": False, "error": "Path not within allowed data directory"}

        pdf_path = resolved

        if not pdf_path.exists():
            return {"success": False, "error": f"PDF not found: {path}"}

        try:
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)

            idx = page_num - 1  # Convert to 0-indexed
            if idx < 0 or idx >= total_pages:
                doc.close()
                return {"success": False, "error": f"Page {page_num} out of range (1-{total_pages})"}

            page = doc[idx]
            rect = page.rect

            text = page.get_text()
            image_count = len(page.get_images())
            link_count = len([l for l in page.get_links() if l.get("kind") == 2])

            table_count = 0
            if hasattr(page, 'find_tables'):
                table_count = len(page.find_tables().tables)

            result = {
                "success": True,
                "page_num": page_num,
                "width": round(rect.width, 2),
                "height": round(rect.height, 2),
                "rotation": page.rotation,
                "text_length": len(text),
                "image_count": image_count,
                "table_count": table_count,
                "link_count": link_count
            }

            doc.close()
            return result

        except Exception as e:
            return {"success": False, "error": f"Failed to get page info: {e!s}"}

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

        elif "table" in action_lower:
            pages = kwargs.get("pages", "all")
            return self.extract_tables(path, pages)

        elif "structured" in action_lower or "layout" in action_lower:
            pages = kwargs.get("pages", "all")
            return self.extract_structured(path, pages)

        elif "image" in action_lower:
            pages = kwargs.get("pages", "all")
            output_dir = kwargs.get("output_dir")
            return self.extract_images(path, pages, output_dir)

        elif "link" in action_lower:
            return self.extract_links(path)

        elif "page_info" in action_lower:
            page_num = kwargs.get("page_num")
            if not page_num:
                return {"success": False, "error": "No page_num provided for page_info action."}
            return self.page_info(path, int(page_num))

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

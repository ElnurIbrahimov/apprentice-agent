"""Local RAG (Retrieval Augmented Generation) Tool for AURA.

Index local documents (PDFs, text, code, markdown) and retrieve relevant
context to augment LLM responses with your personal knowledge base.

Features:
- Document loaders for PDF, TXT, MD, code files, DOCX
- Smart chunking with overlap for context continuity
- Vector embeddings via Ollama (nomic-embed-text)
- Semantic search with cosine similarity
- Automatic context injection for LLM queries
"""

import ast
import json
import hashlib
import logging
import re
import tempfile
import os
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import fitz  # PyMuPDF for PDFs
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_CHUNK_SIZE = 512  # tokens (approx 4 chars per token)
DEFAULT_CHUNK_OVERLAP = 64
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 5
SEMANTIC_OVERLAP = 200  # chars of overlap between semantic chunks
MAX_SECTION_CHARS = 2000  # split markdown sections larger than this


# ==================== Semantic Chunking ====================

def _add_overlap(chunks: List[Dict[str, Any]], overlap: int = SEMANTIC_OVERLAP) -> List[Dict[str, Any]]:
    """Add trailing overlap from the next chunk to each chunk for context continuity."""
    if not chunks or overlap <= 0:
        return chunks
    for i in range(len(chunks) - 1):
        next_text = chunks[i + 1]["content"]
        overlap_text = next_text[:overlap]
        if overlap_text:
            chunks[i]["content"] = chunks[i]["content"] + "\n" + overlap_text
    return chunks


def _chunk_code_python(content: str, file_path: str) -> List[Dict[str, Any]]:
    """Chunk Python code using AST — split on functions and classes.
    Falls back to empty list if AST parsing fails (caller will use fallback)."""
    chunks: List[Dict[str, Any]] = []
    lines = content.split("\n")
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return []

    covered_lines: set = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            kind = "code_class"
            name = node.name
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "code_function"
            name = node.name
        else:
            continue

        start = node.lineno  # 1-based
        end = node.end_lineno or start

        snippet = "\n".join(lines[start - 1:end])
        start_offset = sum(len(lines[j]) + 1 for j in range(start - 1))  # byte offset

        covered_lines.update(range(start, end + 1))
        chunks.append({
            "content": snippet,
            "chunk_type": kind,
            "name": name,
            "start_offset": start_offset,
            "line_start": start,
            "line_end": end,
        })

    # Capture module-level code not inside any function/class
    uncovered = []
    current_start = None
    for i in range(1, len(lines) + 1):
        if i not in covered_lines and lines[i - 1].strip():
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                uncovered.append((current_start, i - 1))
                current_start = None
    if current_start is not None:
        uncovered.append((current_start, len(lines)))

    for start, end in uncovered:
        snippet = "\n".join(lines[start - 1:end]).strip()
        if len(snippet) > 30:  # skip trivial fragments (imports-only, etc.)
            start_offset = sum(len(lines[j]) + 1 for j in range(start - 1))
            chunks.append({
                "content": snippet,
                "chunk_type": "code_module",
                "name": "module_level",
                "start_offset": start_offset,
                "line_start": start,
                "line_end": end,
            })

    # Sort by position in file
    chunks.sort(key=lambda c: c["line_start"])
    return chunks


# Regex patterns for JS/TS/Go/Rust/Java/etc top-level symbols
_CODE_PATTERNS = [
    # JS/TS
    (re.compile(r'^(export\s+)?(default\s+)?(async\s+)?function\s+(\w+)', re.MULTILINE), "code_function"),
    (re.compile(r'^(export\s+)?(default\s+)?class\s+(\w+)', re.MULTILINE), "code_class"),
    (re.compile(r'^(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=]*)?=>', re.MULTILINE), "code_function"),
    # Rust
    (re.compile(r'^\s*(pub\s+)?(async\s+)?fn\s+(\w+)', re.MULTILINE), "code_function"),
    (re.compile(r'^\s*(pub\s+)?struct\s+(\w+)', re.MULTILINE), "code_class"),
    (re.compile(r'^\s*(pub\s+)?enum\s+(\w+)', re.MULTILINE), "code_class"),
    (re.compile(r'^\s*(pub\s+)?trait\s+(\w+)', re.MULTILINE), "code_class"),
    # Go
    (re.compile(r'^\s*func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(', re.MULTILINE), "code_function"),
    (re.compile(r'^\s*type\s+(\w+)\s+struct\b', re.MULTILINE), "code_class"),
    # Java/C#
    (re.compile(r'^\s*(public|private|protected)?\s*(static\s+)?(?:class|interface)\s+(\w+)', re.MULTILINE), "code_class"),
    # Python fallback (if AST fails)
    (re.compile(r'^\s*(async\s+)?def\s+(\w+)\s*\(', re.MULTILINE), "code_function"),
    (re.compile(r'^\s*class\s+(\w+)[\s(:]', re.MULTILINE), "code_class"),
]


def _chunk_code_regex(content: str, file_path: str) -> List[Dict[str, Any]]:
    """Chunk code files using regex symbol detection. Works for JS/TS/Go/Rust/etc."""
    lines = content.split("\n")
    # Find all symbol start positions
    symbols: List[Tuple[int, str, str]] = []  # (line_start_0based, kind, name)

    for regex, kind in _CODE_PATTERNS:
        for m in regex.finditer(content):
            line_num = content[:m.start()].count("\n")  # 0-based
            # Extract name: last non-None capturing group that's not a keyword
            keywords = {"export", "default", "async", "const", "let", "var", "pub",
                        "static", "public", "private", "protected", "struct", "interface", "class", "def"}
            name = None
            for g in reversed(m.groups()):
                if g and g.strip() not in keywords:
                    name = g.strip()
                    break
            if name:
                symbols.append((line_num, kind, name))

    if not symbols:
        return []

    # Deduplicate by line
    seen_lines: set = set()
    unique_symbols = []
    for line_num, kind, name in sorted(symbols, key=lambda x: x[0]):
        if line_num not in seen_lines:
            seen_lines.add(line_num)
            unique_symbols.append((line_num, kind, name))

    # Create chunks: each symbol extends to the line before the next symbol (or EOF)
    chunks: List[Dict[str, Any]] = []
    for idx, (line_num, kind, name) in enumerate(unique_symbols):
        if idx + 1 < len(unique_symbols):
            end_line = unique_symbols[idx + 1][0]  # exclusive, 0-based
        else:
            end_line = len(lines)

        snippet = "\n".join(lines[line_num:end_line]).rstrip()
        if not snippet.strip():
            continue

        start_offset = sum(len(lines[j]) + 1 for j in range(line_num))
        chunks.append({
            "content": snippet,
            "chunk_type": kind,
            "name": name,
            "start_offset": start_offset,
            "line_start": line_num + 1,
            "line_end": end_line,
        })

    return chunks


def _chunk_markdown(content: str, file_path: str) -> List[Dict[str, Any]]:
    """Chunk markdown/text by headers, then by paragraph boundaries within large sections."""
    chunks: List[Dict[str, Any]] = []

    # Split on markdown headers (# ## ### etc)
    header_re = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    sections: List[Tuple[str, str, int]] = []  # (title, body, char_offset)

    matches = list(header_re.finditer(content))
    if not matches:
        # No headers — treat entire content as one section, split on paragraphs
        sections = [("", content, 0)]
    else:
        # Content before first header
        if matches[0].start() > 0:
            preamble = content[:matches[0].start()].strip()
            if preamble:
                sections.append(("preamble", preamble, 0))

        for i, m in enumerate(matches):
            title = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            body = content[start:end].strip()
            sections.append((title, f"{m.group(0)}\n{body}" if body else m.group(0), m.start()))

    for title, body, char_offset in sections:
        if len(body) <= MAX_SECTION_CHARS:
            if body.strip():
                chunks.append({
                    "content": body.strip(),
                    "chunk_type": "markdown_section",
                    "name": title or "untitled",
                    "start_offset": char_offset,
                    "line_start": content[:char_offset].count("\n") + 1,
                    "line_end": content[:char_offset + len(body)].count("\n") + 1,
                })
        else:
            # Section too long — split on paragraph boundaries (double newline)
            paragraphs = re.split(r'\n\n+', body)
            current_chunk = ""
            chunk_start = char_offset

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current_chunk) + len(para) + 2 > MAX_SECTION_CHARS and current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "chunk_type": "paragraph",
                        "name": title or "untitled",
                        "start_offset": chunk_start,
                        "line_start": content[:chunk_start].count("\n") + 1,
                        "line_end": content[:chunk_start + len(current_chunk)].count("\n") + 1,
                    })
                    chunk_start = chunk_start + len(current_chunk)
                    current_chunk = para
                else:
                    current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para

            if current_chunk.strip():
                chunks.append({
                    "content": current_chunk.strip(),
                    "chunk_type": "paragraph" if len(chunks) > 0 else "markdown_section",
                    "name": title or "untitled",
                    "start_offset": chunk_start,
                    "line_start": content[:chunk_start].count("\n") + 1,
                    "line_end": content[:chunk_start + len(current_chunk)].count("\n") + 1,
                })

    return chunks


def _chunk_pdf_pages(path: "Path") -> List[Dict[str, Any]]:
    """Chunk PDF by page boundaries, then by paragraphs within large pages.
    Returns chunks with page metadata. Requires PyMuPDF."""
    if not PYMUPDF_AVAILABLE:
        return []

    chunks: List[Dict[str, Any]] = []
    doc = fitz.open(path)
    try:
        offset = 0

        for page_idx, page in enumerate(doc):
            page_text = page.get_text().strip()
            if not page_text:
                continue

            page_num = page_idx + 1

            if len(page_text) <= MAX_SECTION_CHARS:
                chunks.append({
                    "content": page_text,
                    "chunk_type": "page",
                    "name": f"page_{page_num}",
                    "start_offset": offset,
                    "line_start": page_num,  # Repurposed as page number for PDFs
                    "line_end": page_num,
                    "page": page_num,
                })
            else:
                # Split large pages on paragraph boundaries
                paragraphs = re.split(r'\n\n+', page_text)
                current_chunk = ""
                para_idx = 0

                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    if len(current_chunk) + len(para) + 2 > MAX_SECTION_CHARS and current_chunk:
                        chunks.append({
                            "content": current_chunk.strip(),
                            "chunk_type": "paragraph",
                            "name": f"page_{page_num}_part_{para_idx}",
                            "start_offset": offset,
                            "line_start": page_num,
                            "line_end": page_num,
                            "page": page_num,
                        })
                        offset += len(current_chunk)
                        para_idx += 1
                        current_chunk = para
                    else:
                        current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para

                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "chunk_type": "paragraph" if para_idx > 0 else "page",
                        "name": f"page_{page_num}" + (f"_part_{para_idx}" if para_idx > 0 else ""),
                        "start_offset": offset,
                        "line_start": page_num,
                        "line_end": page_num,
                        "page": page_num,
                    })

            offset += len(page_text) + 2  # account for page separator
    finally:
        doc.close()
    return chunks


def _chunk_code_by_lines(content: str, file_path: str, max_chars: int = MAX_SECTION_CHARS) -> List[Dict[str, Any]]:
    """Line-based code chunking fallback: split on blank-line boundaries."""
    chunks: List[Dict[str, Any]] = []
    lines = content.split("\n")
    current_chunk_lines: List[str] = []
    chunk_start = 1  # 1-based

    for i, line in enumerate(lines):
        current_chunk_lines.append(line)
        text_so_far = "\n".join(current_chunk_lines)

        # Split at blank lines when chunk is big enough
        at_blank = (line.strip() == "")
        at_end = (i == len(lines) - 1)

        if (at_blank and len(text_so_far) >= max_chars) or at_end:
            snippet = text_so_far.strip()
            if snippet:
                start_offset = sum(len(lines[j]) + 1 for j in range(chunk_start - 1))
                chunks.append({
                    "content": snippet,
                    "chunk_type": "code_block",
                    "name": f"lines_{chunk_start}_{i + 1}",
                    "start_offset": start_offset,
                    "line_start": chunk_start,
                    "line_end": i + 1,
                })
            current_chunk_lines = []
            chunk_start = i + 2  # next line, 1-based

    return chunks


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata."""
    id: str
    content: str
    source: str  # File path
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """A search result with relevance score."""
    chunk: DocumentChunk
    score: float


class LocalRAG:
    """Local RAG system for AURA.

    Index local documents and retrieve relevant context for LLM queries.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the RAG system.

        Args:
            data_dir: Directory for storing index data
        """
        from ..config import Config

        self.data_dir = data_dir or (Path(os.getenv("AURA_DATA_DIR", "data")) / "rag")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._index_file_path = self.data_dir / "index.json"
        self.embeddings_file = self.data_dir / "embeddings.npy"

        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.indexed_files: Dict[str, str] = {}  # path -> hash
        self._chunk_has_embedding: list[bool] = []

        self._load_index()

        # Check if embedding model is available
        self.embedding_available = self._check_embedding_model()

    def _check_embedding_model(self) -> bool:
        """Check if the embedding model is available."""
        if not OLLAMA_AVAILABLE:
            logger.warning("[RAG] Ollama not available")
            return False

        try:
            client = ollama.Client()
            response = client.list()
            # Handle both old dict format and new ListResponse format
            if hasattr(response, 'models'):
                # New format: ListResponse with models attribute
                model_names = [m.model.split(':')[0] for m in response.models]
            else:
                # Old format: dict with 'models' key
                model_names = [m.get('name', '').split(':')[0] for m in response.get('models', [])]

            if EMBEDDING_MODEL.split(':')[0] in model_names:
                return True
            else:
                logger.warning(f"[RAG] Embedding model {EMBEDDING_MODEL} not found. Run: ollama pull {EMBEDDING_MODEL}")
                return False
        except Exception as e:
            logger.warning(f"[RAG] Could not check embedding model: {e}")
            return False

    def _load_index(self) -> None:
        """Load existing index from disk."""
        try:
            if self._index_file_path.exists():
                data = json.loads(self._index_file_path.read_text(encoding="utf-8"))
                self.chunks = [DocumentChunk(**c) for c in data.get("chunks", [])]
                self.indexed_files = data.get("indexed_files", {})
                logger.info(f"[RAG] Loaded {len(self.chunks)} chunks from index")

            if self.embeddings_file.exists():
                self.embeddings = np.load(self.embeddings_file)
                logger.info(f"[RAG] Loaded embeddings: {self.embeddings.shape}")

            n_emb = len(self.embeddings) if self.embeddings is not None else 0
            self._chunk_has_embedding = [False] * len(self.chunks)
            for i in range(min(n_emb, len(self.chunks))):
                self._chunk_has_embedding[i] = True
        except Exception as e:
            logger.warning(f"[RAG] Could not load index: {e}")
            self.chunks = []
            self.embeddings = None
            self.indexed_files = {}
            self._chunk_has_embedding = []

    def _save_index(self) -> None:
        """Save index to disk (atomic writes to prevent data loss on crash)."""
        try:
            # Save chunks (without embeddings in JSON) -- atomic temp+rename
            chunks_data = []
            for c in self.chunks:
                chunk_dict = asdict(c)
                chunk_dict['embedding'] = None  # Don't store in JSON
                chunks_data.append(chunk_dict)

            data = {
                "chunks": chunks_data,
                "indexed_files": self.indexed_files,
                "updated_at": datetime.now().isoformat()
            }
            content = json.dumps(data, indent=2)
            fd, tmp_path = tempfile.mkstemp(dir=self._index_file_path.parent, suffix='.tmp')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(content)
                os.replace(tmp_path, str(self._index_file_path))
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            # Save embeddings as numpy array -- atomic temp+rename
            if self.embeddings is not None:
                fd, tmp_path = tempfile.mkstemp(dir=self.embeddings_file.parent, suffix='.tmp.npy')
                os.close(fd)
                try:
                    np.save(tmp_path, self.embeddings)
                    os.replace(tmp_path, str(self.embeddings_file))
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise

            logger.info(f"[RAG] Saved index with {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"[RAG] Could not save index: {e}")

    def _get_file_hash(self, path: Path) -> str:
        """Get hash of file content for change detection."""
        return hashlib.md5(path.read_bytes()).hexdigest()

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using Ollama."""
        if not self.embedding_available:
            return None

        try:
            client = ollama.Client()
            response = client.embeddings(model=EMBEDDING_MODEL, prompt=text)
            # Handle both old dict format and new EmbeddingsResponse object format
            if hasattr(response, 'embedding'):
                return response.embedding
            return response.get('embedding')
        except Exception as e:
            logger.error(f"[RAG] Embedding error: {e}")
            return None

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters (~4 chars per token)
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        # Convert token-based sizes to character-based (approx 4 chars per token)
        char_chunk_size = chunk_size * 4
        char_overlap = overlap * 4

        # Clean text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce multiple newlines
        text = text.strip()

        if len(text) <= char_chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + char_chunk_size

            # Try to break at sentence or paragraph boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind('\n\n', start, end)
                if para_break > start + char_chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in ['. ', '.\n', '! ', '? ']:
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + char_chunk_size // 2:
                            end = sent_break + len(sep)
                            break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - char_overlap
            if start >= len(text):
                break

        return chunks

    def _smart_chunk(self, path: Path, content: str) -> List[Dict[str, Any]]:
        """Route to semantic chunking based on file type.

        Returns list of dicts with keys: content, chunk_type, name, start_offset,
        line_start, line_end (and optionally 'page' for PDFs).
        Falls back to naive _chunk_text for unknown types.
        """
        suffix = path.suffix.lower()
        path_str = str(path)
        chunks: List[Dict[str, Any]] = []

        # --- Code files: AST or regex-based ---
        code_extensions = {
            '.py', '.pyi', '.pyw',
            '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.mts', '.cts',
            '.vue', '.svelte',
            '.go', '.rs', '.java', '.c', '.cpp', '.h', '.hpp', '.cs',
            '.rb', '.php', '.swift', '.kt',
        }
        if suffix in code_extensions:
            if suffix in ('.py', '.pyi', '.pyw'):
                chunks = _chunk_code_python(content, path_str)
                if not chunks:
                    # AST failed — try regex
                    chunks = _chunk_code_regex(content, path_str)
            else:
                chunks = _chunk_code_regex(content, path_str)

            if not chunks:
                # Regex found nothing — fall back to line-based splitting
                chunks = _chunk_code_by_lines(content, path_str)

        # --- PDF files: page-based ---
        elif suffix == '.pdf':
            chunks = _chunk_pdf_pages(path)

        # --- Markdown / text ---
        elif suffix in ('.md', '.mdx', '.txt', '.rst', '.adoc'):
            chunks = _chunk_markdown(content, path_str)

        # --- Unknown / other text: try markdown-style first, then naive ---
        if not chunks:
            # Try markdown chunker (works on any text with paragraphs)
            chunks = _chunk_markdown(content, path_str)

        if not chunks:
            # Final fallback: original naive character-split
            text_chunks = self._chunk_text(content)
            for i, text in enumerate(text_chunks):
                chunks.append({
                    "content": text,
                    "chunk_type": "text_block",
                    "name": f"chunk_{i}",
                    "start_offset": 0,
                    "line_start": 0,
                    "line_end": 0,
                })

        # Add overlap between adjacent chunks
        chunks = _add_overlap(chunks)

        return chunks

    # ==================== Document Loaders ====================

    def _load_text_file(self, path: Path) -> str:
        """Load plain text file."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")

    def _load_pdf(self, path: Path) -> str:
        """Load PDF file using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

        text_parts = []
        doc = fitz.open(path)
        try:
            for page in doc:
                text_parts.append(page.get_text())
        finally:
            doc.close()
        return "\n\n".join(text_parts)

    def _load_docx(self, path: Path) -> str:
        """Load Word document."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not installed. Run: pip install python-docx")

        doc = DocxDocument(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    def _load_document(self, path: Path) -> str:
        """Load document based on file type. Returns raw text (no fencing)."""
        suffix = path.suffix.lower()

        if suffix == '.pdf':
            return self._load_pdf(path)
        elif suffix == '.docx':
            return self._load_docx(path)
        else:
            # All text-based files (code, markdown, config, etc.) — load raw
            return self._load_text_file(path)

    # ==================== Indexing ====================

    def index_file(self, path: str | Path, force: bool = False) -> Dict[str, Any]:
        """Index a single file.

        Args:
            path: Path to file
            force: Re-index even if file hasn't changed

        Returns:
            Result dict with chunks_added count
        """
        path = Path(path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        if not path.is_file():
            return {"success": False, "error": f"Not a file: {path}"}

        path_str = str(path.absolute())
        file_hash = self._get_file_hash(path)

        # Check if already indexed and unchanged
        if not force and path_str in self.indexed_files:
            if self.indexed_files[path_str] == file_hash:
                return {"success": True, "message": "File already indexed", "chunks_added": 0}

        try:
            # Load document content (PDF chunking handles its own loading)
            suffix = path.suffix.lower()
            if suffix == '.pdf':
                content = ""  # PDF chunker reads the file directly
            else:
                content = self._load_document(path)

            # Use semantic chunking (falls back to naive splits internally)
            smart_chunks = self._smart_chunk(path, content)

            if not smart_chunks:
                return {"success": False, "error": "No content extracted"}

            # Create chunk objects with embeddings
            new_chunks = []
            new_embeddings = []

            for i, sc in enumerate(smart_chunks):
                text = sc["content"]
                chunk_id = f"{path.stem}_{i}_{hashlib.md5(text.encode()).hexdigest()[:8]}"

                embedding = self._get_embedding(text)

                chunk = DocumentChunk(
                    id=chunk_id,
                    content=text,
                    source=path_str,
                    chunk_index=i,
                    metadata={
                        "filename": path.name,
                        "file_type": path.suffix,
                        "chunk_type": sc.get("chunk_type", "unknown"),
                        "chunk_name": sc.get("name", ""),
                        "start_offset": sc.get("start_offset", 0),
                        "line_start": sc.get("line_start", 0),
                        "line_end": sc.get("line_end", 0),
                        "page": sc.get("page"),
                        "indexed_at": datetime.now().isoformat()
                    },
                    embedding=embedding
                )
                new_chunks.append(chunk)

                if embedding:
                    new_embeddings.append(embedding)

            # Remove old chunks AFTER new ones generated successfully, then add
            self.chunks = [c for c in self.chunks if c.source != path_str]
            self.chunks.extend(new_chunks)
            self.indexed_files[path_str] = file_hash

            # Update embeddings array
            if new_embeddings:
                if self.embeddings is None:
                    # Rebuild embeddings for all chunks
                    all_embeddings = []
                    for c in self.chunks:
                        if c.embedding:
                            all_embeddings.append(c.embedding)
                    if all_embeddings:
                        self.embeddings = np.array(all_embeddings)
                else:
                    # Rebuild from scratch to maintain alignment
                    all_embeddings = [c.embedding for c in self.chunks if c.embedding]
                    self.embeddings = np.array(all_embeddings) if all_embeddings else None

            # Rebuild _chunk_has_embedding to match self.chunks after update
            n_emb = len(self.embeddings) if self.embeddings is not None else 0
            self._chunk_has_embedding = [False] * len(self.chunks)
            for i in range(min(n_emb, len(self.chunks))):
                self._chunk_has_embedding[i] = True

            self._save_index()

            return {
                "success": True,
                "file": path.name,
                "chunks_added": len(new_chunks),
                "has_embeddings": bool(new_embeddings)
            }

        except Exception as e:
            logger.error(f"[RAG] Index error for {path}: {e}")
            return {"success": False, "error": str(e)}

    def index_directory(
        self,
        directory: str | Path,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        force: bool = False
    ) -> Dict[str, Any]:
        """Index all documents in a directory.

        Args:
            directory: Directory path
            extensions: File extensions to include (default: common doc types)
            recursive: Search subdirectories
            force: Re-index all files

        Returns:
            Result dict with files_indexed and total_chunks
        """
        directory = Path(directory)
        if not directory.exists():
            return {"success": False, "error": f"Directory not found: {directory}"}

        if extensions is None:
            extensions = [
                '.txt', '.md', '.pdf', '.docx',
                '.py', '.js', '.ts', '.jsx', '.tsx',
                '.java', '.cpp', '.c', '.h', '.go',
                '.rs', '.rb', '.json', '.yaml', '.yml',
                '.html', '.css', '.sql', '.sh', '.bat'
            ]

        # Normalize extensions
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]

        # Find files
        pattern = '**/*' if recursive else '*'
        files = [f for f in directory.glob(pattern) if f.is_file() and f.suffix.lower() in extensions]

        results = {
            "success": True,
            "directory": str(directory),
            "files_found": len(files),
            "files_indexed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "total_chunks": 0,
            "errors": []
        }

        for file in files:
            result = self.index_file(file, force=force)
            if result.get("success"):
                if result.get("chunks_added", 0) > 0:
                    results["files_indexed"] += 1
                    results["total_chunks"] += result.get("chunks_added", 0)
                else:
                    results["files_skipped"] += 1
            else:
                results["files_failed"] += 1
                results["errors"].append(f"{file.name}: {result.get('error')}")

        return results

    # ==================== Search ====================

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _keyword_search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[SearchResult]:
        """Fallback keyword-based search when embeddings unavailable."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))

        scored = []
        for chunk in self.chunks:
            chunk_words = set(re.findall(r'\b\w+\b', chunk.content.lower()))
            if not query_words or not chunk_words:
                continue

            # Jaccard similarity
            intersection = len(query_words & chunk_words)
            union = len(query_words | chunk_words)
            score = intersection / union if union > 0 else 0

            if score > 0:
                scored.append(SearchResult(chunk=chunk, score=score))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """Search for relevant document chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            List of SearchResult objects
        """
        if not self.chunks:
            return []

        # Try embedding-based search first
        if self.embedding_available and self.embeddings is not None and len(self.embeddings) > 0:
            query_embedding = self._get_embedding(query)

            if query_embedding is not None:
                query_vec = np.array(query_embedding)

                # Calculate similarities
                scores = []
                emb_idx = 0
                for idx, chunk in enumerate(self.chunks):
                    if idx < len(self._chunk_has_embedding) and self._chunk_has_embedding[idx] and emb_idx < len(self.embeddings):
                        score = self._cosine_similarity(query_vec, self.embeddings[emb_idx])
                        scores.append(SearchResult(chunk=chunk, score=score))
                        emb_idx += 1

                # Sort by score
                scores.sort(key=lambda x: x.score, reverse=True)

                # Filter by min_score and return top_k
                results = [r for r in scores if r.score >= min_score][:top_k]
                # === PHASE 1: Track memory recall ===
                try:
                    from api.routes.memory import record_memory_recall
                    if results:
                        record_memory_recall("rag", len(results), query, [r.chunk.content[:80] for r in results[:5]])
                except Exception:
                    pass
                try:
                    from api.routes.context import track_context_from_memory
                    if results:
                        track_context_from_memory([r.chunk.content[:80] for r in results[:5]])
                except Exception:
                    pass
                return results

        # Fallback to keyword search
        logger.info("[RAG] Using keyword search (embeddings unavailable)")
        return self._keyword_search(query, top_k)

    def get_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        max_tokens: int = 2000
    ) -> str:
        """Get formatted context for LLM augmentation.

        Args:
            query: User query
            top_k: Number of chunks to include
            max_tokens: Maximum tokens in context

        Returns:
            Formatted context string
        """
        results = self.search(query, top_k=top_k)

        if not results:
            return ""

        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Approximate

        for r in results:
            if total_chars + len(r.chunk.content) > max_chars:
                break

            source = Path(r.chunk.source).name
            meta = r.chunk.metadata
            chunk_type = meta.get("chunk_type", "")
            chunk_name = meta.get("chunk_name", "")
            label = f"{source}"
            if chunk_type:
                label += f" ({chunk_type}"
                if chunk_name:
                    label += f": {chunk_name}"
                label += ")"
            context_parts.append(
                f"[Source: {label} | Relevance: {r.score:.0%}]\n{r.chunk.content}"
            )
            total_chars += len(r.chunk.content)

        if not context_parts:
            return ""

        return "=== Relevant Context from Your Documents ===\n\n" + "\n\n---\n\n".join(context_parts)

    # ==================== Management ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        files_by_type = {}
        for chunk in self.chunks:
            file_type = chunk.metadata.get("file_type", "unknown")
            files_by_type[file_type] = files_by_type.get(file_type, 0) + 1

        return {
            "total_chunks": len(self.chunks),
            "total_files": len(self.indexed_files),
            "embeddings_available": self.embeddings is not None,
            "embedding_dimensions": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "chunks_by_type": files_by_type,
            "embedding_model": EMBEDDING_MODEL if self.embedding_available else "unavailable"
        }

    def list_indexed_files(self) -> List[Dict[str, Any]]:
        """List all indexed files."""
        files = []
        for path, hash_val in self.indexed_files.items():
            chunk_count = sum(1 for c in self.chunks if c.source == path)
            files.append({
                "path": path,
                "filename": Path(path).name,
                "chunks": chunk_count,
                "hash": hash_val[:8]
            })
        return files

    def remove_file(self, path: str | Path) -> Dict[str, Any]:
        """Remove a file from the index."""
        path_str = str(Path(path).absolute())

        if path_str not in self.indexed_files:
            return {"success": False, "error": "File not in index"}

        # Remove chunks
        old_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.source != path_str]
        removed = old_count - len(self.chunks)

        # Remove from indexed files
        del self.indexed_files[path_str]

        # Rebuild embeddings
        embeddings = [c.embedding for c in self.chunks if c.embedding]
        self.embeddings = np.array(embeddings) if embeddings else None

        self._save_index()

        return {"success": True, "chunks_removed": removed}

    def clear_index(self) -> Dict[str, Any]:
        """Clear the entire index."""
        old_count = len(self.chunks)

        self.chunks = []
        self.embeddings = None
        self.indexed_files = {}

        # Delete files
        if self._index_file_path.exists():
            self._index_file_path.unlink()
        if self.embeddings_file.exists():
            self.embeddings_file.unlink()

        return {"success": True, "chunks_removed": old_count}


class LocalRAGTool:
    """Tool interface for AURA agent."""

    name = "local_rag"
    description = """Index and search your local documents (PDFs, text, code, markdown).

    Actions:
    - index <file_or_directory>: Index a file or directory
    - search <query>: Search indexed documents
    - context <query>: Get relevant context for a query
    - stats: Show index statistics
    - list: List indexed files
    - remove <path>: Remove a file from index
    - clear: Clear entire index
    """

    def __init__(self):
        self.rag = LocalRAG()

    def execute(self, action: str) -> Dict[str, Any]:
        """Execute a RAG action."""
        action = action.strip()

        # Parse action
        if action.startswith("index "):
            path = action[6:].strip().strip('"\'')
            path_obj = Path(path)
            if path_obj.is_dir():
                return self.rag.index_directory(path)
            else:
                return self.rag.index_file(path)

        elif action.startswith("search "):
            query = action[7:].strip()
            results = self.rag.search(query)
            return {
                "success": True,
                "query": query,
                "results": [
                    {
                        "content": r.chunk.content[:500] + "..." if len(r.chunk.content) > 500 else r.chunk.content,
                        "source": Path(r.chunk.source).name,
                        "chunk_type": r.chunk.metadata.get("chunk_type", ""),
                        "chunk_name": r.chunk.metadata.get("chunk_name", ""),
                        "score": f"{r.score:.0%}"
                    }
                    for r in results
                ]
            }

        elif action.startswith("context "):
            query = action[8:].strip()
            context = self.rag.get_context(query)
            return {
                "success": True,
                "query": query,
                "context": context if context else "No relevant context found."
            }

        elif action == "stats":
            return {"success": True, **self.rag.get_stats()}

        elif action == "list":
            return {"success": True, "files": self.rag.list_indexed_files()}

        elif action.startswith("remove "):
            path = action[7:].strip().strip('"\'')
            return self.rag.remove_file(path)

        elif action == "clear":
            return self.rag.clear_index()

        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "help": self.description
            }


# Singleton instance (thread-safe double-checked locking)
_rag_instance: Optional[LocalRAG] = None
_rag_lock = threading.Lock()

def get_local_rag() -> LocalRAG:
    """Get or create the LocalRAG singleton."""
    global _rag_instance
    if _rag_instance is None:
        with _rag_lock:
            if _rag_instance is None:
                _rag_instance = LocalRAG()
    return _rag_instance

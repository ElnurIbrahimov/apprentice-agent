"""World-class Semantic Codebase Index for AURA.

Features:
- Incremental indexing with SHA-256 content hashing (skip unchanged files)
- Content-hash embedding cache: only re-embed chunks whose text actually changed
- mtime fast-path: skip reading files whose modification time hasn't changed
- Smart semantic chunking: tree-sitter AST (8 languages), Python AST, JS/TS regex, section fallback
- Multi-signal ranking: BM25 keyword + semantic similarity + recency + importance
- Project structure awareness: entry points, import graph, metadata, test detection
- Aider-style repo map: compact symbol listing for LLM context windows
- SQLite-backed persistence at data/codebase_index/index.db

Search API:
- search(query, limit=10) — hybrid semantic + keyword search
- find_file(pattern) — glob-based file finding
- get_project_summary() — project structure, entry points, key files
- get_file_outline(path) — symbol outline for a file
- generate_repo_map(max_tokens=2000) — compact file+symbol map for LLMs
"""

import ast
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Tree-sitter (optional, preferred chunking path) ────────────
_TREESITTER_AVAILABLE = False
try:
    from tree_sitter_language_pack import get_parser as _ts_get_parser
    _TREESITTER_AVAILABLE = True
    logger.debug("[CodebaseIndex] tree-sitter-language-pack available")
except ImportError:
    logger.debug("[CodebaseIndex] tree-sitter not installed, using fallback chunkers")

# Extension → tree-sitter language name
_EXT_TO_TS_LANG: Dict[str, str] = {
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
}

# Per-language config: which top-level node types to extract and how to get names.
# Each entry: node_type → (chunk_kind, name_strategy)
# name_strategy: "name" = child_by_field_name('name'),
#                "declarator" = child_by_field_name('declarator') then recurse,
#                "type_spec" = Go type_declaration child type_spec
_TS_NODE_CONFIG: Dict[str, Dict[str, Tuple[str, str]]] = {
    "python": {
        "function_definition": ("function", "name"),
        "class_definition": ("class", "name"),
        "decorated_definition": ("decorated", "unwrap"),
    },
    "javascript": {
        "function_declaration": ("function", "name"),
        "class_declaration": ("class", "name"),
        "lexical_declaration": ("variable", "js_lexical"),
        "variable_declaration": ("variable", "js_lexical"),
        "export_statement": ("export", "unwrap_export"),
    },
    "typescript": {
        "function_declaration": ("function", "name"),
        "class_declaration": ("class", "name"),
        "interface_declaration": ("interface", "name"),
        "type_alias_declaration": ("type", "name"),
        "enum_declaration": ("enum", "name"),
        "lexical_declaration": ("variable", "js_lexical"),
        "variable_declaration": ("variable", "js_lexical"),
        "export_statement": ("export", "unwrap_export"),
    },
    "go": {
        "function_declaration": ("function", "name"),
        "method_declaration": ("method", "name"),
        "type_declaration": ("type", "go_type_decl"),
    },
    "rust": {
        "function_item": ("function", "name"),
        "struct_item": ("struct", "name"),
        "enum_item": ("enum", "name"),
        "trait_item": ("trait", "name"),
        "impl_item": ("impl", "rust_impl"),
    },
    "java": {
        "class_declaration": ("class", "name"),
        "interface_declaration": ("interface", "name"),
        "enum_declaration": ("enum", "name"),
        "method_declaration": ("method", "name"),
    },
    "c": {
        "function_definition": ("function", "c_func"),
        "struct_specifier": ("struct", "name"),
        "enum_specifier": ("enum", "name"),
        "type_definition": ("type", "c_typedef"),
    },
    "cpp": {
        "function_definition": ("function", "c_func"),
        "class_specifier": ("class", "name"),
        "struct_specifier": ("struct", "name"),
        "enum_specifier": ("enum", "name"),
        "namespace_definition": ("namespace", "name"),
    },
}

# ── Embedding config ────────────────────────────────────────────
_EMBED_MODEL = "nomic-embed-text:latest"
try:
    from aura.config import Config as _CbConfig
    _EMBED_URL = _CbConfig.OLLAMA_HOST + "/api/embeddings"
except Exception:
    _EMBED_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/embeddings"

# SECURITY: Validate embedding URL against SSRF at import time.
# Whitelist localhost/127.0.0.1 — the Ollama embedding service runs locally.
try:
    from urllib.parse import urlparse as _urlparse

    from aura.security.ssrf_guard import validate_url_safe
    _parsed_embed = _urlparse(_EMBED_URL)
    _embed_host = (_parsed_embed.hostname or "").lower()
    if _embed_host in ("localhost", "127.0.0.1", "::1"):
        pass  # Trusted local embedding service (Ollama)
    else:
        validate_url_safe(_EMBED_URL)
except ValueError as _e:
    logger.warning(f"[CodebaseIndex] Embed URL failed SSRF validation: {_e}")
    _EMBED_URL = None  # Disable embeddings — callers must check
except ImportError:
    pass  # ssrf_guard not available — allow (local dev)

# ── File classification ─────────────────────────────────────────
# Higher weight = more important in search ranking
_FILE_TYPE_WEIGHTS = {
    # Source code (highest)
    ".py": 1.0, ".pyi": 0.9, ".js": 1.0, ".jsx": 1.0, ".ts": 1.0, ".tsx": 1.0,
    ".mjs": 0.9, ".cjs": 0.9, ".mts": 0.9, ".cts": 0.9,
    ".go": 1.0, ".rs": 1.0, ".java": 1.0, ".c": 1.0, ".cpp": 1.0, ".h": 0.9,
    ".rb": 1.0, ".php": 1.0, ".swift": 1.0, ".kt": 1.0, ".cs": 1.0,
    ".vue": 0.95, ".svelte": 0.95,
    # Config (medium)
    ".json": 0.6, ".jsonc": 0.6, ".yaml": 0.6, ".yml": 0.6, ".toml": 0.6,
    ".ini": 0.5, ".cfg": 0.5, ".env": 0.4,
    # Docs (lower)
    ".md": 0.4, ".mdx": 0.4, ".txt": 0.3, ".rst": 0.3,
    # Shell/scripts
    ".sh": 0.7, ".bash": 0.7, ".zsh": 0.7, ".bat": 0.5, ".ps1": 0.5,
    # SQL
    ".sql": 0.8,
    # Styles
    ".css": 0.5, ".scss": 0.5, ".sass": 0.5, ".less": 0.5,
    # HTML/templates
    ".html": 0.5, ".htm": 0.5,
}

# Files that signal importance (entry points, project roots)
_IMPORTANT_FILENAMES = frozenset({
    "main.py", "app.py", "server.py", "index.py", "cli.py", "manage.py", "wsgi.py", "asgi.py",
    "index.js", "index.ts", "main.js", "main.ts", "app.js", "app.ts", "server.js", "server.ts",
    "index.jsx", "index.tsx", "App.jsx", "App.tsx",
    "main.go", "main.rs", "lib.rs", "mod.rs",
    "package.json", "pyproject.toml", "setup.py", "setup.cfg",
    "Cargo.toml", "go.mod", "pom.xml", "build.gradle",
    "requirements.txt", "Pipfile",
    "Makefile", "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "README.md", "README.rst", "CHANGELOG.md",
    "tsconfig.json", ".eslintrc.json", "jest.config.js", "vite.config.ts",
    "next.config.js", "next.config.mjs", "next.config.ts",
    "tailwind.config.js", "tailwind.config.ts",
})

# Directories to always skip
_SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    ".mypy_cache", ".tox", ".eggs", ".cache", ".aura",
    ".parcel-cache", ".turbo", ".svelte-kit", "target",
    "vendor", ".gradle", ".idea", ".vs", ".vscode",
})

_BINARY_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mkv", ".flac", ".ogg",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".o", ".obj",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".pyc", ".pyo", ".class", ".jar",
    ".db", ".sqlite", ".sqlite3",
    ".lock",
})

_MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB


# ── Helpers ─────────────────────────────────────────────────────

def _embed(text: str) -> Optional[list]:
    """Get embedding from nomic-embed-text via Ollama."""
    try:
        import requests
        r = requests.post(
            _EMBED_URL,
            json={"model": _EMBED_MODEL, "prompt": text[:500]},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json().get("embedding")
    except Exception as e:
        logger.debug("[CodebaseIndex] Embedding failed: %s", e)
    return None


# ── Embedding Cache (content-hash → embedding) ────────────────
# Only re-embeds chunks whose content actually changed, even within
# a modified file.  Cache is a separate SQLite DB so it survives
# full re-indexes and can be shared across projects.

class EmbeddingCache:
    """Content-hash → embedding cache.  Only re-embed changed chunks."""

    def __init__(self, cache_path: str = "data/codebase_index/embedding_cache.db"):
        self._path = Path(cache_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # Protect shared connection across threads
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                content_hash TEXT PRIMARY KEY,
                embedding BLOB,
                file_path TEXT,
                chunk_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()
        self._hits = 0
        self._misses = 0

    def get(self, content: str) -> Optional[list]:
        """Look up cached embedding by content hash."""
        h = hashlib.sha256(content.encode()).hexdigest()
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding FROM chunk_embeddings WHERE content_hash = ?", (h,)
            ).fetchone()
        if row:
            import numpy as np
            self._hits += 1
            return np.frombuffer(row[0], dtype=np.float32).tolist()
        self._misses += 1
        return None

    def put(self, content: str, embedding: list, file_path: str, chunk_type: str):
        """Store embedding keyed by content hash."""
        h = hashlib.sha256(content.encode()).hexdigest()
        import numpy as np
        blob = np.array(embedding, dtype=np.float32).tobytes()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO chunk_embeddings "
                "(content_hash, embedding, file_path, chunk_type) VALUES (?, ?, ?, ?)",
                (h, blob, file_path, chunk_type),
            )
            self._conn.commit()

    def get_or_compute(self, content: str, file_path: str, chunk_type: str) -> Optional[list]:
        """Return cached embedding or compute + cache a new one."""
        cached = self.get(content)
        if cached is not None:
            return cached
        emb = _embed(content)
        if emb:
            self.put(content, emb, file_path, chunk_type)
        return emb

    def get_stats(self) -> dict:
        """Return cache statistics."""
        count = self._conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
        total = self._hits + self._misses
        hit_rate = round(self._hits / total, 3) if total > 0 else 0.0
        return {
            "cached_chunks": count,
            "session_hits": self._hits,
            "session_misses": self._misses,
            "session_hit_rate": hit_rate,
        }

    def prune_stale(self, days: int = 30):
        """Remove embeddings older than N days (garbage collection)."""
        self._conn.execute(
            "DELETE FROM chunk_embeddings WHERE created_at < datetime('now', ?)",
            (f"-{days} days",),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()


def _cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 0.0 else 0.0


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _should_skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or name.startswith(".")


def _should_skip_file(path: Path) -> bool:
    if path.suffix.lower() in _BINARY_EXTS:
        return True
    try:
        if path.stat().st_size > _MAX_FILE_SIZE:
            return True
    except OSError:
        return True
    return False


def _is_test_file(rel_path: str) -> bool:
    """Detect test files by path patterns."""
    p = rel_path.lower().replace("\\", "/")
    parts = p.split("/")
    # Directory-based
    if any(d in ("test", "tests", "__tests__", "spec", "specs", "e2e", "fixtures") for d in parts[:-1]):
        return True
    # Filename-based
    name = parts[-1] if parts else ""
    if name.startswith("test_") or name.endswith("_test.py") or name.endswith(".test.js"):
        return True
    if name.endswith(".test.ts") or name.endswith(".test.tsx") or name.endswith(".test.jsx"):
        return True
    if name.endswith(".spec.js") or name.endswith(".spec.ts") or name.endswith(".spec.tsx"):
        return True
    if name.startswith("test_") or name.endswith("_test.go"):
        return True
    return False


def _walk_files(root: Path) -> List[Path]:
    """Walk directory tree, yielding source files."""
    results = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if not _should_skip_file(fpath):
                    results.append(fpath)
    except PermissionError:
        pass
    return results


# ── BM25 implementation (lightweight, no deps) ─────────────────

class _BM25:
    """Minimal BM25 scorer for in-memory ranking."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score(self, query_terms: List[str], doc_text: str, avg_dl: float, n_docs: int,
              df: Dict[str, int]) -> float:
        """Score a single document against query terms."""
        doc_words = doc_text.lower().split()
        dl = len(doc_words)
        if dl == 0 or avg_dl == 0:
            return 0.0

        tf_map: Dict[str, int] = {}
        for w in doc_words:
            tf_map[w] = tf_map.get(w, 0) + 1

        score = 0.0
        for term in query_terms:
            t = term.lower()
            if t not in tf_map:
                continue
            tf = tf_map[t]
            doc_freq = df.get(t, 0)
            idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / avg_dl))
            score += idf * tf_norm
        return score


_bm25 = _BM25()

# ── Semantic chunking ───────────────────────────────────────────

# ── Tree-sitter AST chunking (preferred path) ──────────────────

def _ts_get_node_name(node, strategy: str, lang: str) -> Optional[str]:
    """Extract a human-readable name from a tree-sitter node using the given strategy."""
    try:
        if strategy == "name":
            n = node.child_by_field_name("name")
            return n.text.decode("utf-8") if n else None

        elif strategy == "c_func":
            # C/C++: function_definition → declarator (function_declarator) → declarator (identifier)
            decl = node.child_by_field_name("declarator")
            if decl and decl.type == "function_declarator":
                inner = decl.child_by_field_name("declarator")
                return inner.text.decode("utf-8") if inner else None
            return decl.text.decode("utf-8") if decl else None

        elif strategy == "c_typedef":
            # C: type_definition → declarator is the alias name
            decl = node.child_by_field_name("declarator")
            return decl.text.decode("utf-8") if decl else None

        elif strategy == "go_type_decl":
            # Go: type_declaration → child type_spec → name field
            for child in node.children:
                if child.type == "type_spec":
                    n = child.child_by_field_name("name")
                    return n.text.decode("utf-8") if n else None
            return None

        elif strategy == "rust_impl":
            # Rust: impl_item → type field gives the type being implemented
            t = node.child_by_field_name("type")
            trait = node.child_by_field_name("trait")
            if trait and t:
                return f"{trait.text.decode('utf-8')} for {t.text.decode('utf-8')}"
            return t.text.decode("utf-8") if t else None

        elif strategy == "js_lexical":
            # JS/TS: lexical_declaration → first variable_declarator → name
            for child in node.children:
                if child.type == "variable_declarator":
                    n = child.child_by_field_name("name")
                    return n.text.decode("utf-8") if n else None
            return None

        elif strategy == "unwrap":
            # Python decorated_definition: the inner node is the actual def/class
            body = node.child_by_field_name("definition")
            if body:
                n = body.child_by_field_name("name")
                return n.text.decode("utf-8") if n else None
            return None

        elif strategy == "unwrap_export":
            # JS/TS export_statement: find the inner declaration
            for child in node.children:
                if child.type in (
                    "function_declaration", "class_declaration",
                    "interface_declaration", "type_alias_declaration",
                    "enum_declaration", "lexical_declaration", "variable_declaration",
                ):
                    inner_cfg = _TS_NODE_CONFIG.get(lang, {}).get(child.type)
                    if inner_cfg:
                        return _ts_get_node_name(child, inner_cfg[1], lang)
            return None

    except Exception:
        return None
    return None


def _ts_get_node_kind(node, kind_raw: str, name_strategy: str, lang: str) -> str:
    """Get the chunk kind, resolving wrappers like export/decorated to the inner kind."""
    if name_strategy == "unwrap_export":
        for child in node.children:
            inner_cfg = _TS_NODE_CONFIG.get(lang, {}).get(child.type)
            if inner_cfg:
                return inner_cfg[0]
        return "variable"
    if name_strategy == "unwrap":
        body = node.child_by_field_name("definition")
        if body:
            if body.type == "class_definition":
                return "class"
            return "function"
        return "function"
    return kind_raw


def _ts_extract_methods(class_node, rel_path: str, class_name: str,
                        content_bytes: bytes, lang: str) -> List[dict]:
    """Extract method-level chunks from inside a class/struct/impl body."""
    methods = []
    # Find the body node (different field names per language)
    body = (
        class_node.child_by_field_name("body") or
        class_node.child_by_field_name("declaration_list")  # Rust impl
    )
    if not body:
        return methods

    # Node types that represent methods inside a class body
    method_types = {
        "python": ("function_definition",),
        "javascript": ("method_definition",),
        "typescript": ("method_definition", "public_field_definition"),
        "java": ("method_declaration", "constructor_declaration"),
        "rust": ("function_item",),
        "cpp": ("function_definition",),
        "go": (),  # Go methods are top-level, not inside struct bodies
        "c": (),
    }
    target_types = method_types.get(lang, ())

    for child in body.children:
        if child.type not in target_types:
            continue
        name_node = child.child_by_field_name("name")
        if not name_node:
            # C/C++ methods use declarator
            decl = child.child_by_field_name("declarator")
            if decl and decl.type == "function_declarator":
                name_node = decl.child_by_field_name("declarator")
            elif decl:
                name_node = decl
        if not name_node:
            continue

        method_name = name_node.text.decode("utf-8")
        start_line = child.start_point[0] + 1  # 1-based
        end_line = child.end_point[0] + 1
        node_text = content_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")

        methods.append({
            "id": f"{rel_path}:{class_name}.{method_name}:{start_line}",
            "file_path": rel_path,
            "name": f"{class_name}.{method_name}",
            "kind": "method",
            "line_start": start_line,
            "line_end": end_line,
            "content": node_text[:800],
            "docstring": "",
            "decorators": "",
            "language": lang,
            "node_type": "method",
            "node_name": method_name,
        })

    return methods


def _chunk_treesitter(content: str, rel_path: str) -> Optional[List[dict]]:
    """AST-aware chunking via tree-sitter. Returns None if unsupported/unavailable."""
    if not _TREESITTER_AVAILABLE:
        return None

    ext = Path(rel_path).suffix.lower()
    lang = _EXT_TO_TS_LANG.get(ext)
    if not lang:
        return None

    node_config = _TS_NODE_CONFIG.get(lang)
    if not node_config:
        return None

    try:
        parser = _ts_get_parser(lang)
        content_bytes = content.encode("utf-8")
        tree = parser.parse(content_bytes)
        root = tree.root_node
    except Exception as e:
        logger.debug("[CodebaseIndex] tree-sitter parse failed for %s: %s", rel_path, e)
        return None

    chunks = []

    for node in root.children:
        node_type = node.type

        # Skip non-declaration nodes (comments, whitespace, semicolons, etc.)
        if node_type not in node_config:
            continue

        kind_raw, name_strategy = node_config[node_type]

        # Get chunk kind (resolve wrappers)
        chunk_kind = _ts_get_node_kind(node, kind_raw, name_strategy, lang)

        # Get the name
        name = _ts_get_node_name(node, name_strategy, lang)
        if not name:
            continue

        start_line = node.start_point[0] + 1  # 1-based
        end_line = node.end_point[0] + 1
        node_text = content_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")

        chunks.append({
            "id": f"{rel_path}:{name}:{start_line}",
            "file_path": rel_path,
            "name": name,
            "kind": chunk_kind,
            "line_start": start_line,
            "line_end": end_line,
            "content": node_text[:800],
            "docstring": "",
            "decorators": "",
            "language": lang,
            "node_type": node.type,
            "node_name": name,
        })

        # For classes/structs/impl blocks, also extract methods as separate chunks
        if chunk_kind in ("class", "struct", "impl", "trait", "interface"):
            method_chunks = _ts_extract_methods(node, rel_path, name, content_bytes, lang)
            chunks.extend(method_chunks)

        # For export_statement / decorated_definition wrappers, extract inner methods too
        if name_strategy == "unwrap_export":
            for child in node.children:
                if child.type in ("class_declaration",):
                    inner_name = _ts_get_node_name(child, "name", lang)
                    if inner_name:
                        method_chunks = _ts_extract_methods(
                            child, rel_path, inner_name, content_bytes, lang
                        )
                        chunks.extend(method_chunks)
        elif name_strategy == "unwrap":
            body = node.child_by_field_name("definition")
            if body and body.type == "class_definition":
                method_chunks = _ts_extract_methods(
                    body, rel_path, name, content_bytes, lang
                )
                chunks.extend(method_chunks)

    if not chunks:
        return None  # Fall back to existing chunkers

    logger.debug(
        "[CodebaseIndex] tree-sitter extracted %d chunks from %s (%s)",
        len(chunks), rel_path, lang
    )
    return chunks


# ── Fallback chunkers ──────────────────────────────────────────

def _chunk_python_ast(content: str, rel_path: str) -> List[dict]:
    """Use Python AST to extract function/class chunks."""
    chunks = []
    lines = content.split("\n")
    try:
        tree = ast.parse(content, filename=rel_path)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function"
            name = node.name
        elif isinstance(node, ast.ClassDef):
            kind = "class"
            name = node.name
        else:
            continue

        start = node.lineno  # 1-based
        end = node.end_lineno or start

        # For classes, include up to first method start or 30 lines
        if kind == "class":
            first_method_line = None
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    first_method_line = child.lineno
                    break
            if first_method_line and first_method_line > start:
                # Class definition + docstring + first few lines before methods
                end = min(first_method_line - 1, start + 30)
            else:
                end = min(end, start + 30)

        # For functions, cap at 40 lines
        if kind == "function":
            end = min(end, start + 40)

        snippet = "\n".join(lines[start - 1:end])

        # Build decorators string
        decorators = ""
        if hasattr(node, "decorator_list") and node.decorator_list:
            dec_names = []
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    dec_names.append(f"@{dec.id}")
                elif isinstance(dec, ast.Attribute):
                    dec_names.append(f"@{ast.dump(dec)[:40]}")
            decorators = " ".join(dec_names)

        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        if len(docstring) > 200:
            docstring = docstring[:200] + "..."

        chunks.append({
            "id": f"{rel_path}:{name}:{start}",
            "file_path": rel_path,
            "name": name,
            "kind": kind,
            "line_start": start,
            "line_end": end,
            "content": snippet[:800],
            "docstring": docstring,
            "decorators": decorators,
        })

    return chunks


# JS/TS regex patterns for extracting top-level symbols
_JS_PATTERNS = [
    # export function / export default function / async function (top-level only: no indent or export)
    (re.compile(r'^(export\s+)?(default\s+)?(async\s+)?function\s+(\w+)\s*[\(<]', re.MULTILINE), "function", 4),
    # export class / class (top-level)
    (re.compile(r'^(export\s+)?(default\s+)?class\s+(\w+)[\s{<]', re.MULTILINE), "class", 3),
    # export const/let/var name = () => (arrow functions, top-level)
    (re.compile(r'^(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*\w[^=]*)?=>', re.MULTILINE), "function", 3),
    # export const/let/var name = function (top-level)
    (re.compile(r'^(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function', re.MULTILINE), "function", 3),
    # export const/let/var (non-function, top-level)
    (re.compile(r'^(export\s+)?(const|let|var)\s+(\w+)\s*[=:]', re.MULTILINE), "variable", 3),
    # export interface (top-level)
    (re.compile(r'^(export\s+)?interface\s+(\w+)[\s{<]', re.MULTILINE), "interface", 2),
    # export type (top-level)
    (re.compile(r'^(export\s+)?type\s+(\w+)\s*[=<]', re.MULTILINE), "type", 2),
    # export enum (top-level)
    (re.compile(r'^(export\s+)?enum\s+(\w+)[\s{]', re.MULTILINE), "enum", 2),
]


def _chunk_js_ts(content: str, rel_path: str) -> List[dict]:
    """Extract JS/TS top-level symbols using regex."""
    chunks = []
    lines = content.split("\n")
    seen_names = set()

    for regex, kind, name_group in _JS_PATTERNS:
        for m in regex.finditer(content):
            name = m.group(name_group)
            if not name or name in seen_names:
                continue
            seen_names.add(name)

            # Calculate line number
            line_start = content[:m.start()].count("\n") + 1

            # Find end of block (look for matching brace or next top-level symbol)
            # Simple heuristic: take up to 40 lines for functions/classes, 15 for others
            max_lines = 40 if kind in ("function", "class") else 15
            line_end = min(line_start + max_lines, len(lines))

            snippet = "\n".join(lines[line_start - 1:line_end])

            chunks.append({
                "id": f"{rel_path}:{name}:{line_start}",
                "file_path": rel_path,
                "name": name,
                "kind": kind,
                "line_start": line_start,
                "line_end": line_end,
                "content": snippet[:800],
                "docstring": "",
                "decorators": "",
            })

    return chunks


# Generic regex patterns for Go, Rust, etc.
_GENERIC_PATTERNS = [
    # Python (fallback if AST fails)
    (re.compile(r'^\s*(async\s+)?def\s+(\w+)\s*\('), "function", -1),
    (re.compile(r'^\s*class\s+(\w+)[\s(:]'), "class", -1),
    # Rust
    (re.compile(r'^\s*(pub\s+)?(async\s+)?fn\s+(\w+)[\s<(]'), "function", -1),
    (re.compile(r'^\s*(pub\s+)?struct\s+(\w+)[\s{<]'), "struct", -1),
    (re.compile(r'^\s*(pub\s+)?enum\s+(\w+)[\s{<]'), "enum", -1),
    (re.compile(r'^\s*(pub\s+)?trait\s+(\w+)[\s{<:]'), "trait", -1),
    # Go
    (re.compile(r'^\s*func\s+(\([^)]*\)\s+)?(\w+)\s*\('), "function", -1),
    (re.compile(r'^\s*type\s+(\w+)\s+(struct|interface)\b'), "struct", -1),
]


def _chunk_generic(content: str, rel_path: str) -> List[dict]:
    """Fallback regex-based chunking for any language."""
    chunks = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        for regex, kind, _ in _GENERIC_PATTERNS:
            m = regex.search(line)
            if m:
                # Get the name from the last non-None group that isn't a keyword
                name = None
                keywords = {"struct", "interface", "class", "function", "def", "pub", "async", "export", "default"}
                for g in reversed(m.groups()):
                    if g and g.strip() not in keywords and g.strip():
                        name = g.strip()
                        break
                if not name or name in keywords:
                    continue

                snippet_end = min(i + 20, len(lines))
                snippet = "\n".join(lines[i:snippet_end])

                chunks.append({
                    "id": f"{rel_path}:{name}:{i+1}",
                    "file_path": rel_path,
                    "name": name,
                    "kind": kind,
                    "line_start": i + 1,
                    "line_end": snippet_end,
                    "content": snippet[:800],
                    "docstring": "",
                    "decorators": "",
                })
                break

    return chunks


def _chunk_by_sections(content: str, rel_path: str) -> List[dict]:
    """Split non-code files by sections (headers, blank line groups)."""
    chunks = []
    lines = content.split("\n")
    current_section = []
    current_start = 1
    current_name = Path(rel_path).stem

    for i, line in enumerate(lines):
        # Detect section boundaries: markdown headers or double blank lines
        is_header = line.startswith("#") or (i > 0 and lines[i-1] == "" and line.startswith("##"))
        is_section_break = (
            is_header or
            (i > 1 and lines[i-1] == "" and lines[i-2] == "" and line.strip())
        )

        if is_section_break and current_section:
            text = "\n".join(current_section).strip()
            if text and len(text) > 20:
                chunks.append({
                    "id": f"{rel_path}:section:{current_start}",
                    "file_path": rel_path,
                    "name": current_name,
                    "kind": "section",
                    "line_start": current_start,
                    "line_end": i,
                    "content": text[:800],
                    "docstring": "",
                    "decorators": "",
                })
            current_section = [line]
            current_start = i + 1
            # Update name from header
            if line.startswith("#"):
                current_name = line.lstrip("#").strip()[:60] or current_name
        else:
            current_section.append(line)

    # Final section
    if current_section:
        text = "\n".join(current_section).strip()
        if text and len(text) > 20:
            chunks.append({
                "id": f"{rel_path}:section:{current_start}",
                "file_path": rel_path,
                "name": current_name,
                "kind": "section",
                "line_start": current_start,
                "line_end": len(lines),
                "content": text[:800],
                "docstring": "",
                "decorators": "",
            })

    return chunks


def _extract_chunks(rel_path: str, content: str) -> List[dict]:
    """Route to the right chunker based on file extension.

    Priority: tree-sitter AST → language-specific fallback → generic regex → sections.
    """
    ext = Path(rel_path).suffix.lower()

    # ── Preferred path: tree-sitter AST chunking ──
    chunks = _chunk_treesitter(content, rel_path)
    if chunks:
        return chunks

    # ── Fallback: language-specific chunkers ──
    chunks = []
    if ext in (".py", ".pyi", ".pyw"):
        chunks = _chunk_python_ast(content, rel_path)
        if not chunks:
            # AST failed (syntax error) — fall back to regex
            chunks = _chunk_generic(content, rel_path)
    elif ext in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts", ".vue", ".svelte"):
        chunks = _chunk_js_ts(content, rel_path)
    elif ext in (".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt"):
        chunks = _chunk_generic(content, rel_path)
    elif ext in (".md", ".mdx", ".txt", ".rst"):
        chunks = _chunk_by_sections(content, rel_path)
    else:
        # For config files, SQL, etc. — try generic, then section-based
        chunks = _chunk_generic(content, rel_path)
        if not chunks:
            chunks = _chunk_by_sections(content, rel_path)

    # If still nothing, store a file-level summary
    if not chunks:
        summary = content[:600].strip()
        if summary:
            chunks = [{
                "id": f"{rel_path}:module:0",
                "file_path": rel_path,
                "name": Path(rel_path).stem,
                "kind": "module",
                "line_start": 1,
                "line_end": min(20, content.count("\n") + 1),
                "content": summary,
                "docstring": "",
                "decorators": "",
            }]

    return chunks


# ── Import graph extraction ─────────────────────────────────────

def _extract_imports_python(content: str) -> List[str]:
    """Extract imported module names from Python source.

    Returns both the full dotted path (e.g. 'aura.tools.browser') and
    the top-level package ('aura'). The full path enables cross-repo
    resolution; the top-level is kept for backward-compatible lookups.
    """
    imports: set = set()
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    imports.add(node.module.split(".")[0])
    except SyntaxError:
        # Regex fallback
        for m in re.finditer(r'^\s*(?:from|import)\s+([\w.]+)', content, re.MULTILINE):
            mod = m.group(1)
            imports.add(mod)
            imports.add(mod.split(".")[0])
    return sorted(imports)


def _extract_imports_js(content: str) -> List[str]:
    """Extract imported module names from JS/TS source."""
    imports = []
    # ES6 imports: import ... from 'module'
    for m in re.finditer(r'''(?:import|export)\s+.*?from\s+['"]([^'"]+)['"]''', content):
        mod = m.group(1)
        if mod.startswith("."):
            # Relative import — normalize
            imports.append(mod)
        else:
            imports.append(mod.split("/")[0])
    # require('module')
    for m in re.finditer(r'''require\s*\(\s*['"]([^'"]+)['"]\s*\)''', content):
        mod = m.group(1)
        if mod.startswith("."):
            imports.append(mod)
        else:
            imports.append(mod.split("/")[0])
    return list(set(imports))


def _extract_imports(rel_path: str, content: str) -> List[str]:
    ext = Path(rel_path).suffix.lower()
    if ext in (".py", ".pyi", ".pyw"):
        return _extract_imports_python(content)
    elif ext in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"):
        return _extract_imports_js(content)
    return []


# ── Project metadata detection ──────────────────────────────────

def _detect_project_metadata(root: Path) -> dict:
    """Detect project metadata from package.json, pyproject.toml, etc."""
    meta = {
        "name": root.name,
        "type": "unknown",
        "language": None,
        "frameworks": [],
        "entry_points": [],
        "package_manager": None,
    }

    # package.json
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
            meta["name"] = pkg.get("name", meta["name"])
            meta["type"] = "node"
            meta["language"] = "javascript"
            deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

            fw_map = {
                "next": "Next.js", "react": "React", "vue": "Vue", "svelte": "Svelte",
                "express": "Express", "fastify": "Fastify", "hono": "Hono",
                "@angular/core": "Angular", "astro": "Astro",
                "tailwindcss": "Tailwind", "prisma": "Prisma", "drizzle-orm": "Drizzle",
            }
            for dep, name in fw_map.items():
                if dep in deps:
                    meta["frameworks"].append(name)

            if "typescript" in deps or (root / "tsconfig.json").exists():
                meta["language"] = "typescript"

            # Entry points
            if pkg.get("main"):
                meta["entry_points"].append(pkg["main"])
            if pkg.get("module"):
                meta["entry_points"].append(pkg["module"])
            for script_name in ("start", "dev", "build"):
                script = pkg.get("scripts", {}).get(script_name, "")
                # Extract filename from script command
                for word in script.split():
                    if word.endswith((".js", ".ts", ".mjs")):
                        meta["entry_points"].append(word)
                        break

            # Package manager
            if (root / "bun.lockb").exists() or (root / "bun.lock").exists():
                meta["package_manager"] = "bun"
            elif (root / "pnpm-lock.yaml").exists():
                meta["package_manager"] = "pnpm"
            elif (root / "yarn.lock").exists():
                meta["package_manager"] = "yarn"
            elif (root / "package-lock.json").exists():
                meta["package_manager"] = "npm"
        except (json.JSONDecodeError, OSError):
            pass

    # pyproject.toml
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        meta["type"] = "python"
        meta["language"] = "python"
        try:
            text = pyproject.read_text(encoding="utf-8")
            # Extract project name
            m = re.search(r'name\s*=\s*"([^"]+)"', text)
            if m:
                meta["name"] = m.group(1)
            # Extract entry points
            for m in re.finditer(r'scripts\]\s*\n((?:\s*\w+\s*=.*\n?)*)', text):
                for line in m.group(1).strip().split("\n"):
                    if "=" in line:
                        parts = line.split("=", 1)
                        meta["entry_points"].append(parts[1].strip().strip('"').split(":")[0])
        except OSError:
            pass

    # requirements.txt
    reqs = root / "requirements.txt"
    if reqs.exists() and meta["language"] is None:
        meta["type"] = "python"
        meta["language"] = "python"
        try:
            text = reqs.read_text(encoding="utf-8").lower()
            py_fw = {"django": "Django", "flask": "Flask", "fastapi": "FastAPI",
                     "streamlit": "Streamlit", "torch": "PyTorch", "transformers": "Transformers"}
            for pkg, name in py_fw.items():
                if pkg in text:
                    meta["frameworks"].append(name)
        except OSError:
            pass

    # Cargo.toml
    if (root / "Cargo.toml").exists():
        meta["type"] = "rust"
        meta["language"] = "rust"
        if (root / "src" / "main.rs").exists():
            meta["entry_points"].append("src/main.rs")
        if (root / "src" / "lib.rs").exists():
            meta["entry_points"].append("src/lib.rs")

    # go.mod
    if (root / "go.mod").exists():
        meta["type"] = "go"
        meta["language"] = "go"
        if (root / "main.go").exists():
            meta["entry_points"].append("main.go")
        if (root / "cmd").exists():
            meta["entry_points"].append("cmd/")

    # Detect common entry point files
    for ep in ("main.py", "app.py", "server.py", "manage.py", "wsgi.py", "asgi.py",
               "index.js", "index.ts", "app.js", "app.ts", "server.js", "server.ts",
               "src/index.js", "src/index.ts", "src/main.js", "src/main.ts",
               "src/App.tsx", "src/App.jsx"):
        if (root / ep).exists() and ep not in meta["entry_points"]:
            meta["entry_points"].append(ep)

    meta["entry_points"] = list(dict.fromkeys(meta["entry_points"]))  # dedupe, keep order
    return meta


# ═══════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════

class CodebaseIndex:
    """SQLite-backed semantic codebase index with hybrid search.

    Index is stored at data/codebase_index/index.db for persistence.
    Supports incremental indexing, multi-signal ranking, and project analysis.
    """

    def __init__(
        self,
        project_path: str,
        db_path: Optional[str] = None,
        workspace_roots: Optional[List[str]] = None,
        _is_sibling: bool = False,
    ):
        self.project_path = Path(project_path).resolve()

        # DB location: explicit path, or data/codebase_index/index.db
        if db_path:
            self._db_path = Path(db_path)
        else:
            self._db_path = Path("data/codebase_index/index.db")

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

        # Content-hash embedding cache (shared, survives re-indexes)
        cache_dir = self._db_path.parent
        self._embedding_cache = EmbeddingCache(
            cache_path=str(cache_dir / "embedding_cache.db")
        )

        # Repo identity
        self._repo_id = self._compute_repo_id(self.project_path)
        self._register_repo(self._repo_id, self.project_path)

        # Sibling indexes for cross-repo resolution. Each sibling owns its
        # own SQLite file so tables stay clean and there's no PK conflict.
        self._siblings: Dict[str, "CodebaseIndex"] = {}
        if workspace_roots and not _is_sibling:
            for root in workspace_roots:
                self.add_workspace_root(root, auto_index=False)

    @staticmethod
    def _compute_repo_id(path: Path) -> str:
        """Short stable id derived from the absolute path."""
        return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]

    def _register_repo(self, repo_id: str, root_path: Path) -> None:
        with self._lock:
            self._get_conn().execute(
                "INSERT OR REPLACE INTO workspace_repos (repo_id, root_path, last_indexed) "
                "VALUES (?, ?, COALESCE((SELECT last_indexed FROM workspace_repos WHERE repo_id=?), 0))",
                (repo_id, str(root_path), repo_id),
            )
            self._get_conn().commit()

    def _sibling_db_path(self, repo_id: str) -> Path:
        """DB file path for a sibling index (lives next to the primary DB)."""
        stem = self._db_path.stem
        return self._db_path.with_name(f"{stem}_repo_{repo_id}.db")

    def add_workspace_root(self, path: str, auto_index: bool = True) -> dict:
        """Register an additional repo root as a sibling index.

        The sibling gets its own SQLite file next to the primary DB, so
        tables stay independent and there's no path collision. Cross-repo
        dependency resolution walks all siblings.
        """
        p = Path(path).resolve()
        if p == self.project_path:
            return {"success": False, "error": "already the primary root"}
        rid = self._compute_repo_id(p)
        if rid in self._siblings:
            return {"success": True, "repo_id": rid, "already_registered": True}

        sibling = CodebaseIndex(
            project_path=str(p),
            db_path=str(self._sibling_db_path(rid)),
            _is_sibling=True,
        )
        self._siblings[rid] = sibling
        self._register_repo(rid, p)

        result = {"success": True, "repo_id": rid, "root": str(p)}
        if auto_index:
            idx_result = sibling.index()
            result["indexed"] = idx_result
        return result

    def index_workspace(self, progress_callback=None, force: bool = False) -> dict:
        """Index the primary repo and every registered sibling, then resolve
        cross-repo imports."""
        primary = self.index(progress_callback=progress_callback, force=force)
        siblings = {}
        for rid, sib in self._siblings.items():
            siblings[rid] = sib.index(progress_callback=progress_callback, force=force)
        resolution = self._resolve_cross_repo_imports()
        return {"primary": primary, "siblings": siblings, "resolution": resolution}

    def _resolve_cross_repo_imports(self) -> dict:
        """Walk imports in every repo (primary + siblings) and try to resolve
        each one against files in the OTHER repos. Stores the resolution on
        the import row via resolved_repo_id / resolved_file."""
        all_repos: Dict[str, "CodebaseIndex"] = {self._repo_id: self}
        all_repos.update(self._siblings)
        if len(all_repos) < 2:
            return {"resolved": 0, "skipped": "need 2+ repos"}

        # For each repo, build a stem/dotted-path → file lookup
        repo_file_index: Dict[str, Dict[str, str]] = {}
        for rid, repo in all_repos.items():
            idx: Dict[str, str] = {}
            with repo._lock:
                rows = repo._get_conn().execute("SELECT rel_path FROM files").fetchall()
            for (fp,) in rows:
                stem = Path(fp).stem
                dotted = fp.replace("/", ".").rsplit(".", 1)[0]
                idx.setdefault(stem, fp)
                idx.setdefault(dotted, fp)
            repo_file_index[rid] = idx

        resolved_total = 0
        for rid, repo in all_repos.items():
            with repo._lock:
                conn = repo._get_conn()
                rows = conn.execute(
                    "SELECT rowid, imports_module FROM imports WHERE resolved_repo_id IS NULL"
                ).fetchall()

                for rowid, module in rows:
                    if not module:
                        continue
                    for other_rid, other_idx in repo_file_index.items():
                        if other_rid == rid:
                            continue
                        target = other_idx.get(module)
                        if not target:
                            stem = module.rsplit(".", 1)[-1]
                            target = other_idx.get(stem)
                        if target:
                            conn.execute(
                                "UPDATE imports SET resolved_repo_id=?, resolved_file=? "
                                "WHERE rowid=?",
                                (other_rid, target, rowid),
                            )
                            resolved_total += 1
                            break
                conn.commit()

        return {"resolved": resolved_total, "repos_scanned": len(all_repos)}

    # ── DB setup ────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-8000")  # 8MB cache
        return self._conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    rel_path TEXT PRIMARY KEY,
                    abs_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    size INTEGER NOT NULL,
                    file_type TEXT,
                    is_test INTEGER DEFAULT 0,
                    is_entry_point INTEGER DEFAULT 0,
                    importance REAL DEFAULT 0.5,
                    indexed_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    name TEXT,
                    kind TEXT,
                    line_start INTEGER,
                    line_end INTEGER,
                    content TEXT,
                    docstring TEXT,
                    decorators TEXT,
                    embedding TEXT,
                    FOREIGN KEY (file_path) REFERENCES files(rel_path)
                );

                CREATE TABLE IF NOT EXISTS imports (
                    file_path TEXT NOT NULL,
                    imports_module TEXT NOT NULL,
                    PRIMARY KEY (file_path, imports_module),
                    FOREIGN KEY (file_path) REFERENCES files(rel_path)
                );

                CREATE TABLE IF NOT EXISTS project_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS workspace_repos (
                    repo_id TEXT PRIMARY KEY,
                    root_path TEXT UNIQUE NOT NULL,
                    last_indexed REAL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
                CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(kind);
                CREATE INDEX IF NOT EXISTS idx_chunks_name ON chunks(name);
                CREATE INDEX IF NOT EXISTS idx_imports_module ON imports(imports_module);
                CREATE INDEX IF NOT EXISTS idx_files_type ON files(file_type);
            """)
            conn.commit()

            # ── Schema migration: add repo_id / resolution columns ──
            # Uses SQLite ALTER TABLE ADD COLUMN (nullable). Safe on re-run.
            def _has_column(table: str, col: str) -> bool:
                cur = conn.execute(f"PRAGMA table_info({table})")
                return any(row[1] == col for row in cur.fetchall())

            migrations = [
                ("files", "repo_id TEXT"),
                ("chunks", "repo_id TEXT"),
                ("imports", "repo_id TEXT"),
                ("imports", "resolved_repo_id TEXT"),
                ("imports", "resolved_file TEXT"),
            ]
            for table, coldef in migrations:
                colname = coldef.split()[0]
                if not _has_column(table, colname):
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef}")
                        logger.info(f"[CodebaseIndex] migrated {table}: +{colname}")
                    except sqlite3.OperationalError as e:
                        logger.warning(f"[CodebaseIndex] migration {table}.{colname} failed: {e}")

            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_imports_repo ON imports(repo_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_imports_resolved ON imports(resolved_repo_id)")
            conn.commit()

    # ── Indexing ────────────────────────────────────────────────

    def index(self, progress_callback=None, force: bool = False) -> dict:
        """Index or re-index the project incrementally.

        Three-tier skip strategy (fastest to slowest):
        1. mtime unchanged → skip without reading file (instant)
        2. mtime changed but SHA-256 identical → skip (cheap read)
        3. SHA-256 changed → re-chunk, but check embedding cache per-chunk

        Args:
            progress_callback: Optional callable(current, total, file_path)
            force: If True, re-index everything regardless of mtime/hash

        Returns:
            {indexed, skipped_mtime, skipped_hash, removed, total_chunks,
             embedding_cache_stats, elapsed}
        """
        t0 = time.time()

        # Get existing file records for incremental checks
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT rel_path, sha256, mtime FROM files"
            ).fetchall()
        existing = {r[0]: {"sha256": r[1], "mtime": r[2]} for r in rows}

        # Detect project metadata
        meta = _detect_project_metadata(self.project_path)
        entry_points_set = set(meta.get("entry_points", []))

        # Walk all files
        files = _walk_files(self.project_path)
        indexed = 0
        skipped_mtime = 0
        skipped_hash = 0
        total_chunks = 0
        cache_hits = 0
        cache_misses = 0
        current_paths = set()

        for fi, fpath in enumerate(files):
            try:
                rel_path = str(fpath.relative_to(self.project_path)).replace("\\", "/")
            except ValueError:
                continue
            current_paths.add(rel_path)

            # ── Tier 1: mtime fast-path (no file read needed) ──
            if not force and rel_path in existing:
                try:
                    disk_mtime = fpath.stat().st_mtime
                except OSError:
                    continue
                if existing[rel_path]["mtime"] == disk_mtime:
                    skipped_mtime += 1
                    continue

            # ── Tier 2: content hash check (need to read file) ──
            try:
                raw = fpath.read_bytes()
            except (OSError, PermissionError):
                continue

            content_hash = _sha256(raw)

            if not force and rel_path in existing and existing[rel_path]["sha256"] == content_hash:
                # mtime changed (e.g. touch, git checkout) but content is the same
                # Update mtime in DB so next run hits tier-1
                try:
                    stat = fpath.stat()
                    with self._lock:
                        self._get_conn().execute(
                            "UPDATE files SET mtime = ? WHERE rel_path = ?",
                            (stat.st_mtime, rel_path)
                        )
                        self._get_conn().commit()
                except OSError:
                    pass
                skipped_hash += 1
                continue

            # ── Tier 3: file actually changed — re-index ──
            if progress_callback:
                progress_callback(fi + 1, len(files), rel_path)

            try:
                content = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue

            # Extract chunks
            chunks = _extract_chunks(rel_path, content)

            # Extract imports
            imports = _extract_imports(rel_path, content)

            # Compute file metadata
            ext = fpath.suffix.lower()
            is_test = _is_test_file(rel_path)
            is_entry = (rel_path in entry_points_set or fpath.name in _IMPORTANT_FILENAMES)
            type_weight = _FILE_TYPE_WEIGHTS.get(ext, 0.5)
            importance = type_weight
            if is_entry:
                importance = min(1.0, importance + 0.3)
            if is_test:
                importance = max(0.1, importance - 0.3)

            stat = fpath.stat()

            # Write to DB
            with self._lock:
                conn = self._get_conn()
                # Upsert file record
                conn.execute("""
                    INSERT OR REPLACE INTO files (rel_path, abs_path, sha256, mtime, size,
                        file_type, is_test, is_entry_point, importance, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (rel_path, str(fpath), content_hash, stat.st_mtime, stat.st_size,
                      ext, int(is_test), int(is_entry), importance, time.time()))

                # Replace chunks — use embedding cache per-chunk
                conn.execute("DELETE FROM chunks WHERE file_path = ?", (rel_path,))
                for chunk in chunks:
                    embed_text = f"{chunk['kind']} {chunk['name']}: {chunk['content']}"
                    # Try embedding cache first (content-hash level)
                    hits_before = self._embedding_cache._hits
                    emb = self._embedding_cache.get_or_compute(
                        embed_text, rel_path, chunk["kind"]
                    )
                    if self._embedding_cache._hits > hits_before:
                        cache_hits += 1
                    else:
                        cache_misses += 1

                    conn.execute("""
                        INSERT INTO chunks (id, file_path, name, kind, line_start, line_end,
                            content, docstring, decorators, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk["id"], chunk["file_path"], chunk["name"], chunk["kind"],
                        chunk["line_start"], chunk.get("line_end"), chunk["content"],
                        chunk.get("docstring", ""), chunk.get("decorators", ""),
                        json.dumps(emb) if emb else None,
                    ))

                # Replace imports
                conn.execute("DELETE FROM imports WHERE file_path = ?", (rel_path,))
                for mod in imports:
                    conn.execute(
                        "INSERT OR IGNORE INTO imports (file_path, imports_module) VALUES (?, ?)",
                        (rel_path, mod)
                    )

                conn.commit()

            indexed += 1
            total_chunks += len(chunks)

        # Remove files that no longer exist on disk
        removed = 0
        stale = set(existing.keys()) - current_paths
        if stale:
            with self._lock:
                conn = self._get_conn()
                for sp in stale:
                    conn.execute("DELETE FROM chunks WHERE file_path = ?", (sp,))
                    conn.execute("DELETE FROM imports WHERE file_path = ?", (sp,))
                    conn.execute("DELETE FROM files WHERE rel_path = ?", (sp,))
                conn.commit()
            removed = len(stale)

        # Store project metadata
        with self._lock:
            conn = self._get_conn()
            for k, v in meta.items():
                conn.execute(
                    "INSERT OR REPLACE INTO project_meta (key, value) VALUES (?, ?)",
                    (k, json.dumps(v) if isinstance(v, (list, dict)) else str(v))
                )
            conn.commit()

        elapsed = round(time.time() - t0, 1)
        emb_stats = self._embedding_cache.get_stats()
        total_skipped = skipped_mtime + skipped_hash
        logger.info(
            f"[CodebaseIndex] Indexed {indexed} files ({total_chunks} chunks) in {elapsed}s, "
            f"skipped {total_skipped} (mtime:{skipped_mtime} hash:{skipped_hash}), "
            f"removed {removed}, cache hits:{emb_stats['session_hits']}"
        )
        return {
            "indexed": indexed,
            "skipped": total_skipped,
            "skipped_mtime": skipped_mtime,
            "skipped_hash": skipped_hash,
            "removed": removed,
            "total_chunks": total_chunks,
            "embedding_cache_stats": emb_stats,
            "elapsed": elapsed,
        }

    # ── Search (hybrid BM25 + semantic + recency + importance) ──

    def search(self, query: str, limit: int = 10, top_k: int | None = None,
               scope: str = "repo") -> list:
        """Hybrid search: BM25 + semantic + recency + file importance.

        Args:
            query: Natural language or keyword query
            limit: Number of results (also accepts top_k for backward compat)
            scope: "repo" (default, primary only) or "workspace"
                   (primary + all registered sibling repos, merged by score)

        Returns:
            List of {file_path, name, kind, line_start, line_end, content,
                     score, score_breakdown, repo_id (when scope=workspace)}
        """
        if top_k is not None:
            limit = top_k

        # Lazy-start the FS watcher so subsequent edits keep the index fresh.
        if not getattr(self, "_watcher_started", False):
            try:
                from aura.tools.codebase_index_watcher import start_watcher
                if start_watcher(self, str(self.project_path)):
                    self._watcher_started = True
            except Exception:
                logger.debug("codebase_index: watcher start failed", exc_info=True)

        if scope == "workspace" and self._siblings:
            primary = self._search_single(query, limit * 2)
            for r in primary:
                r["repo_id"] = self._repo_id
                r["repo_root"] = str(self.project_path)
            for rid, sib in self._siblings.items():
                sib_results = sib._search_single(query, limit * 2)
                for r in sib_results:
                    r["repo_id"] = rid
                    r["repo_root"] = str(sib.project_path)
                primary.extend(sib_results)
            primary.sort(key=lambda x: x["score"], reverse=True)
            return primary[:limit]

        return self._search_single(query, limit)

    def _search_single(self, query: str, limit: int) -> list:
        """Original single-repo hybrid search. Factored out so scope='workspace'
        can fan out across siblings."""

        query_vec = _embed(query)
        query_terms = re.findall(r'\b\w+\b', query.lower())

        with self._lock:
            conn = self._get_conn()
            rows = conn.execute("""
                SELECT c.id, c.file_path, c.name, c.kind, c.line_start, c.line_end,
                       c.content, c.docstring, c.embedding,
                       f.mtime, f.importance, f.is_test
                FROM chunks c
                LEFT JOIN files f ON c.file_path = f.rel_path
            """).fetchall()

        if not rows:
            return []

        # Compute BM25 stats
        all_texts = []
        doc_freqs: Dict[str, int] = defaultdict(int)
        for row in rows:
            text = (row[6] or "") + " " + (row[2] or "") + " " + (row[7] or "")
            all_texts.append(text)
            seen_terms = set(text.lower().split())
            for t in seen_terms:
                doc_freqs[t] += 1

        total_words = sum(len(t.split()) for t in all_texts)
        avg_dl = total_words / max(len(all_texts), 1)
        n_docs = len(rows)

        # Compute time range for recency normalization
        mtimes = [r[9] for r in rows if r[9]]
        if mtimes:
            max_mtime = max(mtimes)
            min_mtime = min(mtimes)
            mtime_range = max(max_mtime - min_mtime, 1.0)
        else:
            max_mtime = time.time()
            mtime_range = 1.0

        # Score all chunks
        scored = []
        for i, row in enumerate(rows):
            _row_id, fpath, name, kind, line_start, line_end, content, _docstring, emb_str, mtime, importance, _is_test = row

            # 1) BM25 keyword score
            search_text = all_texts[i]
            bm25_score = _bm25.score(query_terms, search_text, avg_dl, n_docs, doc_freqs)

            # 2) Semantic score
            semantic_score = 0.0
            if query_vec and emb_str:
                try:
                    emb = json.loads(emb_str)
                    semantic_score = _cosine(query_vec, emb)
                except (json.JSONDecodeError, ValueError):
                    pass

            # 3) Recency score (0-1, newer = higher)
            recency_score = 0.5
            if mtime:
                recency_score = (mtime - (max_mtime - mtime_range)) / mtime_range

            # 4) Importance score (from file metadata)
            imp_score = importance if importance is not None else 0.5

            # 5) Name match boost
            name_boost = 0.0
            if name:
                name_lower = name.lower()
                for qt in query_terms:
                    if qt == name_lower:
                        name_boost = 0.5
                        break
                    elif qt in name_lower or name_lower in qt:
                        name_boost = max(name_boost, 0.25)

            # Weighted combination
            # Semantic and BM25 are primary signals, recency/importance are secondary
            # Normalize BM25 roughly to 0-1 range (cap at 10)
            bm25_norm = min(bm25_score / 10.0, 1.0) if bm25_score > 0 else 0.0

            final_score = (
                0.35 * semantic_score +
                0.30 * bm25_norm +
                0.10 * recency_score +
                0.10 * imp_score +
                0.15 * name_boost
            )

            scored.append({
                "file_path": fpath,
                "name": name,
                "kind": kind,
                "line_start": line_start,
                "line_end": line_end,
                "content": content,
                "score": round(final_score, 4),
                "score_breakdown": {
                    "semantic": round(semantic_score, 4),
                    "bm25": round(bm25_norm, 4),
                    "recency": round(recency_score, 4),
                    "importance": round(imp_score, 4),
                    "name_match": round(name_boost, 4),
                },
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # ── find_file ───────────────────────────────────────────────

    def find_file(self, pattern: str, limit: int = 50) -> list:
        """Find files by glob pattern.

        Args:
            pattern: Glob pattern (e.g. '*.py', 'src/**/*.ts', 'test_*')

        Returns:
            List of {rel_path, size, mtime, importance, is_test}
        """
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT rel_path, size, mtime, importance, is_test FROM files ORDER BY mtime DESC"
            ).fetchall()

        results = []
        for rel_path, size, mtime, importance, is_test in rows:
            # Match against filename or full relative path
            filename = Path(rel_path).name
            if fnmatch(filename, pattern) or fnmatch(rel_path, pattern):
                results.append({
                    "rel_path": rel_path,
                    "size": size,
                    "mtime": mtime,
                    "importance": importance,
                    "is_test": bool(is_test),
                })
                if len(results) >= limit:
                    break

        return results

    # ── get_project_summary ─────────────────────────────────────

    def get_project_summary(self) -> dict:
        """Return project structure overview.

        Returns:
            {name, type, language, frameworks, entry_points, file_stats,
             top_files, import_graph_summary}
        """
        with self._lock:
            conn = self._get_conn()

            # Project metadata
            meta_rows = conn.execute("SELECT key, value FROM project_meta").fetchall()
            meta = {}
            for k, v in meta_rows:
                try:
                    meta[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    meta[k] = v

            # File stats
            total_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

            # By file type
            type_counts = conn.execute(
                "SELECT file_type, COUNT(*) FROM files GROUP BY file_type ORDER BY COUNT(*) DESC"
            ).fetchall()

            # By kind
            kind_counts = conn.execute(
                "SELECT kind, COUNT(*) FROM chunks GROUP BY kind ORDER BY COUNT(*) DESC"
            ).fetchall()

            # Entry points
            entry_points = conn.execute(
                "SELECT rel_path FROM files WHERE is_entry_point = 1 ORDER BY importance DESC"
            ).fetchall()

            # Top files by importance
            top_files = conn.execute(
                "SELECT rel_path, importance, file_type FROM files ORDER BY importance DESC LIMIT 20"
            ).fetchall()

            # Test file count
            test_count = conn.execute("SELECT COUNT(*) FROM files WHERE is_test = 1").fetchone()[0]

            # Most-imported modules (internal)
            top_imports = conn.execute("""
                SELECT imports_module, COUNT(*) as cnt
                FROM imports
                GROUP BY imports_module
                ORDER BY cnt DESC
                LIMIT 15
            """).fetchall()

        return {
            "name": meta.get("name", self.project_path.name),
            "type": meta.get("type", "unknown"),
            "language": meta.get("language"),
            "frameworks": meta.get("frameworks", []),
            "package_manager": meta.get("package_manager"),
            "entry_points": [r[0] for r in entry_points],
            "file_stats": {
                "total_files": total_files,
                "total_chunks": total_chunks,
                "test_files": test_count,
                "source_files": total_files - test_count,
                "by_type": dict(type_counts),
                "by_kind": dict(kind_counts),
            },
            "top_files": [{"path": r[0], "importance": r[1], "type": r[2]} for r in top_files],
            "most_imported": [{"module": r[0], "count": r[1]} for r in top_imports],
        }

    # ── get_file_outline ────────────────────────────────────────

    def get_file_outline(self, path: str) -> dict:
        """Get symbol outline for a file.

        Args:
            path: Relative or absolute path to the file

        Returns:
            {file_path, symbols: [{name, kind, line_start, line_end, docstring, decorators}],
             imports: [str], is_test, importance}
        """
        # Normalize path
        try:
            p = Path(path)
            if p.is_absolute():
                rel = str(p.relative_to(self.project_path)).replace("\\", "/")
            else:
                rel = str(p).replace("\\", "/")
        except ValueError:
            rel = str(path).replace("\\", "/")

        with self._lock:
            conn = self._get_conn()

            # Get file info
            file_row = conn.execute(
                "SELECT rel_path, is_test, importance FROM files WHERE rel_path = ?", (rel,)
            ).fetchone()

            # Get chunks (symbols)
            chunks = conn.execute(
                "SELECT name, kind, line_start, line_end, docstring, decorators "
                "FROM chunks WHERE file_path = ? ORDER BY line_start",
                (rel,)
            ).fetchall()

            # Get imports
            imports = conn.execute(
                "SELECT imports_module FROM imports WHERE file_path = ?", (rel,)
            ).fetchall()

        if not file_row and not chunks:
            return {"error": f"File not found in index: {rel}"}

        return {
            "file_path": rel,
            "is_test": bool(file_row[1]) if file_row else False,
            "importance": file_row[2] if file_row else 0.5,
            "symbols": [
                {
                    "name": c[0],
                    "kind": c[1],
                    "line_start": c[2],
                    "line_end": c[3],
                    "docstring": c[4] or "",
                    "decorators": c[5] or "",
                }
                for c in chunks
            ],
            "imports": [r[0] for r in imports],
        }

    # ── Get dependents/dependencies ─────────────────────────────

    def get_dependencies(self, path: str) -> dict:
        """Get what a file imports and what imports it.

        Args:
            path: Relative path to the file

        Returns:
            {file_path, imports: [str], imported_by: [str],
             cross_repo: [{module, repo_id, file}] }
        """
        rel = str(Path(path)).replace("\\", "/")

        with self._lock:
            conn = self._get_conn()

            imports = conn.execute(
                "SELECT imports_module, resolved_repo_id, resolved_file FROM imports "
                "WHERE file_path = ?", (rel,)
            ).fetchall()

            stem = Path(rel).stem
            imported_by = conn.execute(
                "SELECT DISTINCT file_path FROM imports WHERE imports_module = ? OR imports_module LIKE ?",
                (stem, f"%.{stem}")
            ).fetchall()

        in_repo = [r[0] for r in imports]
        cross_repo = [
            {"module": r[0], "repo_id": r[1], "file": r[2]}
            for r in imports if r[1] and r[2]
        ]

        # Also check siblings: does any of them import this file's stem?
        cross_imported_by: List[dict] = []
        for rid, sib in self._siblings.items():
            with sib._lock:
                rows = sib._get_conn().execute(
                    "SELECT DISTINCT file_path FROM imports "
                    "WHERE imports_module = ? OR imports_module LIKE ?",
                    (stem, f"%.{stem}"),
                ).fetchall()
            for (fp,) in rows:
                cross_imported_by.append({"repo_id": rid, "file": fp})

        return {
            "file_path": rel,
            "imports": in_repo,
            "imported_by": [r[0] for r in imported_by if r[0] != rel],
            "cross_repo": cross_repo,
            "cross_imported_by": cross_imported_by,
        }

    # ── Repo Map (Aider-style compact symbol listing) ──────────

    def generate_repo_map(self, max_tokens: int = 2000) -> str:
        """Generate a compact repo map showing file structure + key symbols.

        Designed for LLM context windows: one line per file with its top symbols.
        Prioritizes important/entry-point files first, then alphabetical.
        Respects token budget (rough 4 chars/token estimate).

        Args:
            max_tokens: Approximate token budget for the map

        Returns:
            Multi-line string like:
                aura/brain.py: OllamaBrain, _quick_generate, _SHARED_EXECUTOR
                aura/tools/code_search.py: CodeSearchTool, grep, glob, find_definition
        """
        with self._lock:
            conn = self._get_conn()

            # Get all files ordered by importance (entry points first)
            file_rows = conn.execute(
                "SELECT rel_path, importance, is_entry_point FROM files "
                "ORDER BY importance DESC, rel_path ASC"
            ).fetchall()

            # Get all chunks grouped by file
            chunk_rows = conn.execute(
                "SELECT file_path, name, kind FROM chunks "
                "WHERE kind IN ('function', 'class', 'method', 'interface', 'type', 'struct', 'trait', 'enum') "
                "ORDER BY file_path, line_start"
            ).fetchall()

        # Build file→symbols map
        file_symbols: Dict[str, List[str]] = defaultdict(list)
        for fpath, name, _kind in chunk_rows:
            if name and name not in file_symbols[fpath]:
                file_symbols[fpath].append(name)

        # Build repo map lines
        lines = []
        for rel_path, _importance, _is_entry in file_rows:
            symbols = file_symbols.get(rel_path, [])
            if symbols:
                # Cap at 10 symbols per file, prioritize classes/functions
                symbol_str = ", ".join(symbols[:10])
                if len(symbols) > 10:
                    symbol_str += f" (+{len(symbols) - 10} more)"
                lines.append(f"{rel_path}: {symbol_str}")
            else:
                lines.append(rel_path)

        # Truncate to fit token budget (rough: 4 chars per token)
        max_chars = max_tokens * 4
        map_text = "\n".join(lines)
        if len(map_text) > max_chars:
            # Truncate line-by-line to stay under budget
            truncated_lines = []
            char_count = 0
            for line in lines:
                if char_count + len(line) + 1 > max_chars - 30:  # reserve room for truncation notice
                    break
                truncated_lines.append(line)
                char_count += len(line) + 1
            remaining = len(lines) - len(truncated_lines)
            truncated_lines.append(f"... ({remaining} more files)")
            map_text = "\n".join(truncated_lines)

        return map_text

    # ── Stats (backward compatible) ─────────────────────────────

    def stats(self) -> dict:
        """Return index statistics."""
        with self._lock:
            conn = self._get_conn()
            total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            total_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            kinds = conn.execute("SELECT kind, COUNT(*) FROM chunks GROUP BY kind").fetchall()
            with_embeddings = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        return {
            "total_chunks": total_chunks,
            "files_indexed": total_files,
            "by_kind": dict(kinds),
            "chunks_with_embeddings": with_embeddings,
            "embedding_cache": self._embedding_cache.get_stats(),
            "db_path": str(self._db_path),
        }

    # ── Close (backward compatible) ─────────────────────────────

    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
        self._embedding_cache.close()

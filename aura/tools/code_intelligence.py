"""Code Intelligence Tool: project-wide AST indexing and repo map generation.

Uses tree-sitter to parse source files, extract symbols, and build a
semantic repo map compressed to a token budget — like Aider's repo-map.

Optional deps: tree-sitter, tree-sitter-python, tree-sitter-javascript, networkx
Falls back gracefully when not available.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Optional imports
try:
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
}

EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}


class CodeIntelligenceTool:
    """Project-wide AST indexing and repo map generation."""

    name = "code_intelligence"
    description = "Index a project's source code and generate a compressed repo map for LLM context"

    def __init__(self):
        self._symbols: List[Dict] = []          # [{file, name, kind, line, sig}]
        self._graph = None                       # networkx DiGraph
        self._indexed_root: Optional[str] = None

        self._py_lang = None
        self._js_lang = None
        self._py_parser = None
        self._js_parser = None

        if TREE_SITTER_AVAILABLE:
            try:
                self._py_lang = Language(tspython.language())
                self._js_lang = Language(tsjavascript.language())
                self._py_parser = Parser(self._py_lang)
                self._js_parser = Parser(self._js_lang)
            except Exception as e:
                logger.warning(f"[CodeIntel] tree-sitter init failed: {e}")
                self._py_lang = None
                self._js_lang = None
                self._py_parser = None
                self._js_parser = None

    def index_project(self, root_path: str, languages: List[str] = None) -> dict:
        """Parse all source files → extract symbols → build dependency graph.

        Args:
            root_path: Project root directory
            languages: Languages to index (default: python, javascript)

        Returns:
            {files_indexed, symbols_found, time_ms}
        """
        import time
        start = time.time()

        if not TREE_SITTER_AVAILABLE:
            return self._index_fallback(root_path)

        if languages is None:
            languages = ["python", "javascript"]

        root = Path(root_path).resolve()
        if not root.exists():
            return {"success": False, "error": f"Path not found: {root_path}"}

        # Block indexing system directories to prevent accidental traversal
        _BLOCKED_ROOTS = {
            "/", "/bin", "/sbin", "/usr", "/etc", "/var", "/proc", "/sys",
            "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
        }
        root_str = str(root)
        for blocked in _BLOCKED_ROOTS:
            if root_str == blocked or root_str.lower() == blocked.lower():
                return {"success": False, "error": f"Indexing system directory '{root_str}' is not allowed"}

        self._symbols = []
        self._indexed_root = str(root)

        if NETWORKX_AVAILABLE:
            self._graph = nx.DiGraph()

        files_indexed = 0

        for fpath in self._walk_source_files(root, languages):
            try:
                symbols = self._extract_symbols(fpath)
                self._symbols.extend(symbols)
                if self._graph is not None:
                    rel = str(fpath.relative_to(root))
                    self._graph.add_node(rel, kind="file")
                    for sym in symbols:
                        self._graph.add_node(f"{rel}::{sym['name']}", kind=sym["kind"], file=rel)
                        self._graph.add_edge(rel, f"{rel}::{sym['name']}")
                files_indexed += 1
            except Exception as e:
                logger.debug(f"[CodeIntel] Skipping {fpath}: {e}")

        elapsed = int((time.time() - start) * 1000)
        logger.info(f"[CodeIntel] Indexed {files_indexed} files, {len(self._symbols)} symbols in {elapsed}ms")

        return {
            "success": True,
            "files_indexed": files_indexed,
            "symbols_found": len(self._symbols),
            "time_ms": elapsed,
        }

    def _index_fallback(self, root_path: str) -> dict:
        """Simple regex-based fallback when tree-sitter isn't available."""
        import re, time
        start = time.time()
        root = Path(root_path).resolve()
        self._symbols = []
        self._indexed_root = str(root)
        files_indexed = 0

        py_pattern = re.compile(r'^(class|def|async def)\s+(\w+)', re.MULTILINE)

        for fpath in root.rglob("*.py"):
            if any(p in fpath.parts for p in SKIP_DIRS):
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                for m in py_pattern.finditer(text):
                    kind = "class" if m.group(1) == "class" else "function"
                    line = text[:m.start()].count("\n") + 1
                    self._symbols.append({
                        "file": str(fpath.relative_to(root)),
                        "name": m.group(2),
                        "kind": kind,
                        "line": line,
                        "sig": m.group(0),
                    })
                files_indexed += 1
            except Exception:
                pass

        return {
            "success": True,
            "files_indexed": files_indexed,
            "symbols_found": len(self._symbols),
            "time_ms": int((time.time() - start) * 1000),
            "method": "regex_fallback",
        }

    def _walk_source_files(self, root: Path, languages: List[str]):
        """Yield source files matching the requested languages."""
        target_exts = {ext for ext, lang in EXT_LANG.items() if lang in languages}
        for fpath in root.rglob("*"):
            if fpath.is_file() and fpath.suffix in target_exts:
                if not any(part in SKIP_DIRS for part in fpath.parts):
                    yield fpath

    def _extract_symbols(self, fpath: Path) -> List[Dict]:
        """Extract function/class definitions from a file using tree-sitter."""
        if not TREE_SITTER_AVAILABLE or not self._py_lang:
            return []

        ext = fpath.suffix
        lang = EXT_LANG.get(ext)
        if lang == "python":
            parser = self._py_parser
        elif lang == "javascript":
            parser = self._js_parser
        else:
            return []

        if parser is None:
            return []

        text = fpath.read_text(encoding="utf-8", errors="ignore").encode()
        tree = parser.parse(text)

        symbols = []
        root = Path(self._indexed_root) if self._indexed_root else fpath.parent
        rel = str(fpath.relative_to(root))

        def walk(node):
            if node.type in ("function_definition", "async_function_definition", "class_definition",
                              "function_declaration", "arrow_function", "method_definition"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    kind = "class" if "class" in node.type else "function"
                    sig = text[node.start_byte:node.start_byte + 120].decode(errors="ignore").split("\n")[0]
                    symbols.append({
                        "file": rel,
                        "name": name_node.text.decode(errors="ignore"),
                        "kind": kind,
                        "line": node.start_point[0] + 1,
                        "sig": sig.strip(),
                    })
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return symbols

    def get_repo_map(self, active_files: Optional[List[str]] = None, max_tokens: int = 5000) -> str:
        """Generate a compressed repo map for LLM context.

        Uses file-symbol tree format:
          path/file.py:
          │ class Foo:
          │   def bar(self, x):

        Args:
            active_files: Files to prioritize (PageRank bias)
            max_tokens: Approximate token budget (4 chars per token)

        Returns:
            Formatted repo map string
        """
        if not self._symbols:
            return ""

        max_tokens = max(1, max_tokens)
        max_chars = max_tokens * 4

        # Group symbols by file
        by_file: Dict[str, List[Dict]] = {}
        for sym in self._symbols:
            by_file.setdefault(sym["file"], []).append(sym)

        # Score files: active_files first, then alphabetical
        active_set = set(active_files or [])

        def score(fname):
            return (0 if fname in active_set else 1, fname)

        lines = ["# Repo Map\n"]
        chars = len(lines[0])

        for fname in sorted(by_file.keys(), key=score):
            file_lines = [f"\n{fname}:\n"]
            for sym in sorted(by_file[fname], key=lambda s: s["line"]):
                indent = "│   " if sym["kind"] == "function" else "│ "
                file_lines.append(f"{indent}{sym['sig']}\n")

            chunk = "".join(file_lines)
            if chars + len(chunk) > max_chars:
                break
            lines.append(chunk)
            chars += len(chunk)

        return "".join(lines)

    def semantic_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Simple text-based symbol search (embedding search requires Qdrant)."""
        if not self._symbols:
            return []

        query_lower = query.lower()
        scored = []

        for sym in self._symbols:
            score = 0
            if query_lower in sym["name"].lower():
                score += 3
            if query_lower in sym["sig"].lower():
                score += 2
            if query_lower in sym["file"].lower():
                score += 1
            if score > 0:
                scored.append((score, sym))

        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:limit]]

    def find_callers(self, function_name: str) -> List[str]:
        """Find all files that reference this function name (text search)."""
        if not self._indexed_root:
            return []

        callers = []
        search_text = function_name
        root = Path(self._indexed_root)

        for fpath in root.rglob("*.py"):
            if any(p in fpath.parts for p in SKIP_DIRS):
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
                if search_text in text:
                    callers.append(str(fpath.relative_to(root)))
            except Exception:
                pass

        return callers

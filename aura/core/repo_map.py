"""Symbol extraction for repo map — tree-sitter AST with regex fallback.

Provides a compact map of classes, functions, and key symbols in the project
for the system prompt, so the LLM knows what exists without wasting tool calls.
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# --- Tree-sitter support (optional) ---
_TS_AVAILABLE = False
try:
    import tree_sitter_language_pack as tslp
    _TS_AVAILABLE = True
except ImportError:
    pass

# Cache: extension -> (parser, query, kind_map) or None
_ts_cache: dict = {}

# Tree-sitter queries per language
_TS_QUERIES: dict[str, tuple[str, str, dict[str, str]]] = {
    ".py": (
        "python",
        """
        (class_definition name: (identifier) @class_name) @class_def
        (function_definition name: (identifier) @func_name) @func_def
        """,
        {"class_name": "class", "func_name": "function"},
    ),
    ".js": (
        "javascript",
        """
        (class_declaration name: (identifier) @class_name)
        (function_declaration name: (identifier) @func_name)
        (export_statement declaration: (function_declaration name: (identifier) @export_func))
        (variable_declarator name: (identifier) @const_name)
        """,
        {"class_name": "class", "func_name": "function", "export_func": "function", "const_name": "const"},
    ),
    ".ts": (
        "typescript",
        """
        (class_declaration name: (type_identifier) @class_name)
        (interface_declaration name: (type_identifier) @iface_name)
        (type_alias_declaration name: (type_identifier) @type_name)
        (function_declaration name: (identifier) @func_name)
        (variable_declarator name: (identifier) @const_name)
        """,
        {"class_name": "class", "iface_name": "interface", "type_name": "type", "func_name": "function", "const_name": "const"},
    ),
    ".tsx": (
        "tsx",
        """
        (class_declaration name: (type_identifier) @class_name)
        (interface_declaration name: (type_identifier) @iface_name)
        (type_alias_declaration name: (type_identifier) @type_name)
        (function_declaration name: (identifier) @func_name)
        (variable_declarator name: (identifier) @const_name)
        """,
        {"class_name": "class", "iface_name": "interface", "type_name": "type", "func_name": "function", "const_name": "const"},
    ),
    ".go": (
        "go",
        """
        (function_declaration name: (identifier) @func_name)
        (method_declaration name: (field_identifier) @method_name)
        (type_declaration (type_spec name: (type_identifier) @type_name))
        """,
        {"func_name": "func", "method_name": "method", "type_name": "struct"},
    ),
    ".rs": (
        "rust",
        """
        (function_item name: (identifier) @func_name)
        (struct_item name: (type_identifier) @struct_name)
        (enum_item name: (type_identifier) @enum_name)
        (trait_item name: (type_identifier) @trait_name)
        (impl_item type: (type_identifier) @impl_name)
        """,
        {"func_name": "fn", "struct_name": "struct", "enum_name": "enum", "trait_name": "trait", "impl_name": "impl"},
    ),
    ".java": (
        "java",
        """
        (class_declaration name: (identifier) @class_name)
        (interface_declaration name: (identifier) @iface_name)
        (method_declaration name: (identifier) @method_name)
        """,
        {"class_name": "class", "iface_name": "interface", "method_name": "method"},
    ),
}
# Alias extensions
_TS_QUERIES[".jsx"] = (
    "javascript",
    _TS_QUERIES[".js"][1],
    _TS_QUERIES[".js"][2],
)
_TS_QUERIES[".mjs"] = _TS_QUERIES[".js"]
_TS_QUERIES[".cjs"] = _TS_QUERIES[".js"]


def _get_ts_parser_and_query(ext: str):
    """Get cached tree-sitter parser + query for extension, or None."""
    if not _TS_AVAILABLE:
        return None
    if ext in _ts_cache:
        return _ts_cache[ext]

    spec = _TS_QUERIES.get(ext)
    if not spec:
        _ts_cache[ext] = None
        return None

    lang_name, query_src, kind_map = spec
    try:
        import tree_sitter as ts
        lang = tslp.get_language(lang_name)
        parser = tslp.get_parser(lang_name)
        query = ts.Query(lang, query_src) if hasattr(ts, "Query") else lang.query(query_src)
        _ts_cache[ext] = (parser, query, kind_map)
        return _ts_cache[ext]
    except Exception as e:
        logger.debug(f"[RepoMap] tree-sitter init failed for {ext}: {e}")
        _ts_cache[ext] = None
        return None


def _extract_symbols_treesitter(file_path: str) -> Optional[list[dict]]:
    """Extract symbols using tree-sitter. Returns list or None on failure."""
    ext = os.path.splitext(file_path)[1].lower()
    cached = _get_ts_parser_and_query(ext)
    if not cached:
        return None

    parser, query, kind_map = cached
    try:
        with open(file_path, "rb") as f:
            source = f.read()
        tree = parser.parse(source)

        symbols = []
        captures = query.captures(tree.root_node)

        # captures is dict: capture_name -> [node, ...]
        if isinstance(captures, dict):
            for capture_name, nodes in captures.items():
                kind = kind_map.get(capture_name)
                if not kind:
                    continue
                for node in nodes:
                    name = node.text.decode("utf-8") if node.text else ""
                    if not name or name.startswith("_"):
                        continue
                    line = node.start_point[0] + 1

                    # Python: detect methods (function inside class body)
                    actual_kind = kind
                    if ext == ".py" and kind == "function":
                        parent = node.parent
                        while parent:
                            if parent.type == "class_definition":
                                actual_kind = "method"
                                break
                            if parent.type in ("module", "function_definition"):
                                break
                            parent = parent.parent

                    symbols.append({"name": name, "kind": actual_kind, "line": line})
        else:
            # Fallback: list of (node, capture_name) tuples
            for node, capture_name in captures:
                kind = kind_map.get(capture_name)
                if not kind:
                    continue
                name = node.text.decode("utf-8") if node.text else ""
                if not name or name.startswith("_"):
                    continue
                line = node.start_point[0] + 1

                actual_kind = kind
                if ext == ".py" and kind == "function":
                    parent = node.parent
                    while parent:
                        if parent.type == "class_definition":
                            actual_kind = "method"
                            break
                        if parent.type in ("module", "function_definition"):
                            break
                        parent = parent.parent

                symbols.append({"name": name, "kind": actual_kind, "line": line})

        # Sort by line number
        symbols.sort(key=lambda s: s["line"])
        return symbols if symbols else None

    except Exception as e:
        logger.debug(f"[RepoMap] tree-sitter parse failed for {file_path}: {e}")
        return None

# Regex patterns per file extension
LANG_PATTERNS = {
    ".py": [
        (re.compile(r"^class\s+(\w+)"), "class"),
        (re.compile(r"^def\s+(\w+)"), "function"),
        (re.compile(r"^    def\s+(\w+)"), "method"),
    ],
    ".js": [
        (re.compile(r"^(?:export\s+)?(?:default\s+)?class\s+(\w+)"), "class"),
        (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)"), "function"),
        (re.compile(r"^(?:export\s+)?const\s+(\w+)\s*="), "const"),
    ],
    ".ts": [
        (re.compile(r"^(?:export\s+)?interface\s+(\w+)"), "interface"),
        (re.compile(r"^(?:export\s+)?type\s+(\w+)\s*="), "type"),
        (re.compile(r"^(?:export\s+)?(?:default\s+)?class\s+(\w+)"), "class"),
        (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)"), "function"),
        (re.compile(r"^(?:export\s+)?const\s+(\w+)\s*="), "const"),
    ],
    ".tsx": None,  # Use .ts patterns
    ".jsx": None,  # Use .js patterns
    ".go": [
        (re.compile(r"^func\s+(?:\([^)]+\)\s+)?(\w+)"), "func"),
        (re.compile(r"^type\s+(\w+)\s+struct"), "struct"),
        (re.compile(r"^type\s+(\w+)\s+interface"), "interface"),
    ],
    ".rs": [
        (re.compile(r"^pub\s+(?:async\s+)?fn\s+(\w+)"), "fn"),
        (re.compile(r"^pub\s+struct\s+(\w+)"), "struct"),
        (re.compile(r"^pub\s+enum\s+(\w+)"), "enum"),
        (re.compile(r"^pub\s+trait\s+(\w+)"), "trait"),
    ],
    ".java": [
        (re.compile(r"^\s*(?:public|private|protected)?\s*class\s+(\w+)"), "class"),
        (re.compile(r"^\s*(?:public|private|protected)?\s*interface\s+(\w+)"), "interface"),
    ],
}

# Alias extensions
LANG_PATTERNS[".tsx"] = LANG_PATTERNS[".ts"]
LANG_PATTERNS[".jsx"] = LANG_PATTERNS[".js"]
LANG_PATTERNS[".mjs"] = LANG_PATTERNS[".js"]
LANG_PATTERNS[".cjs"] = LANG_PATTERNS[".js"]

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "target", "out",
    ".tox", ".mypy_cache", ".pytest_cache", "egg-info",
    ".aura", "data",
}

SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Cargo.lock", "go.sum",
}

ENTRY_POINTS = {
    "main.py", "app.py", "index.py", "manage.py", "wsgi.py", "asgi.py",
    "index.ts", "index.js", "app.ts", "app.js", "server.ts", "server.js",
    "main.go", "main.rs", "lib.rs",
}

MAX_FILES = 100
MAX_MAP_TOKENS = 4000


def extract_symbols(file_path: str) -> list[dict]:
    """Extract symbols from a single file. Returns [{name, kind, line}].

    Tries tree-sitter AST parsing first (more accurate), falls back to regex.
    """
    # Try tree-sitter first
    if _TS_AVAILABLE:
        ts_result = _extract_symbols_treesitter(file_path)
        if ts_result is not None:
            return ts_result

    # Regex fallback
    ext = os.path.splitext(file_path)[1].lower()
    patterns = LANG_PATTERNS.get(ext)
    if not patterns:
        return []

    symbols = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                for pattern, kind in patterns:
                    m = pattern.match(line)
                    if m:
                        name = m.group(1)
                        # Skip private/dunder methods in map
                        if kind == "method" and name.startswith("_"):
                            continue
                        symbols.append({"name": name, "kind": kind, "line": line_num})
                        break  # One match per line
    except (OSError, PermissionError):
        pass

    return symbols


def build_repo_map(project_root: str, max_tokens: int = MAX_MAP_TOKENS) -> str:
    """Build compact repo map for system prompt.

    Output format:
      src/auth/login.py
        class LoginController
          authenticate, validate_token
        hash_password
      src/models/user.py
        class User
        class UserRole
    """
    files = _collect_files(project_root)
    if not files:
        return ""

    ranked = _rank_files(project_root, files)

    # Extract symbols for each file
    file_symbols = {}
    for fpath in ranked:
        syms = extract_symbols(os.path.join(project_root, fpath))
        if syms:
            file_symbols[fpath] = syms

    if not file_symbols:
        return ""

    return _format_map(file_symbols, max_tokens)


def _collect_files(project_root: str) -> list[str]:
    """Collect source files respecting .gitignore via git ls-files, with os.walk fallback."""
    # Try git ls-files first (respects .gitignore)
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=10, cwd=project_root,
        )
        if result.returncode == 0 and result.stdout.strip():
            files = []
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                fname = os.path.basename(line)
                if fname in SKIP_FILES:
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in LANG_PATTERNS:
                    continue
                files.append(line.replace("\\", "/"))
                if len(files) >= MAX_FILES * 2:
                    break
            return files
    except Exception as e:
        logger.debug(f"[RepoMap] non-critical: {e}")
    # Fallback to os.walk (non-git repos)
    files = []
    root = Path(project_root)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]

        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""

        for fname in filenames:
            if fname in SKIP_FILES:
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in LANG_PATTERNS:
                continue

            rel_path = os.path.join(rel_dir, fname) if rel_dir else fname
            rel_path = rel_path.replace("\\", "/")
            files.append(rel_path)

            if len(files) >= MAX_FILES * 2:
                break
        if len(files) >= MAX_FILES * 2:
            break

    return files


def _rank_files(project_root: str, files: list[str]) -> list[str]:
    """Rank by: git-dirty > entry points > symbol count."""
    dirty_files = set()
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=project_root,
        )
        if result.returncode == 0:
            dirty_files = set(result.stdout.strip().split("\n"))
        # Also include untracked
        result2 = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=5,
            cwd=project_root,
        )
        if result2.returncode == 0:
            dirty_files.update(result2.stdout.strip().split("\n"))
    except Exception as e:
        logger.debug(f"[RepoMap] non-critical: {e}")
    def score(fpath: str) -> int:
        s = 0
        fname = os.path.basename(fpath)
        if fpath in dirty_files:
            s += 100
        if fname in ENTRY_POINTS:
            s += 50
        # Boost core/src directories (most important code)
        if "/core/" in fpath or "/src/" in fpath:
            s += 30
        # Shorter paths = more likely top-level/important
        depth = fpath.count("/")
        s += max(0, 10 - depth * 2)
        return s

    ranked = sorted(files, key=score, reverse=True)
    return ranked[:MAX_FILES]


def _format_map(file_symbols: dict, max_tokens: int) -> str:
    """Format as compact tree, truncate to fit token budget."""
    lines = []
    chars_budget = int(max_tokens * 3)  # ~3 chars per token

    total_chars = 0
    for fpath, symbols in file_symbols.items():
        file_line = fpath
        total_chars += len(file_line) + 1

        if total_chars > chars_budget:
            lines.append(f"... ({len(file_symbols) - len(lines)} more files)")
            break

        lines.append(file_line)

        # Group symbols: classes with their methods, standalone functions
        current_class = None
        class_methods = []

        for sym in symbols:
            if sym["kind"] == "class":
                # Flush previous class methods
                if current_class and class_methods:
                    methods_str = ", ".join(class_methods)
                    entry = f"    {methods_str}"
                    total_chars += len(entry) + 1
                    lines.append(entry)
                    class_methods = []
                current_class = sym["name"]
                entry = f"  class {sym['name']}"
                total_chars += len(entry) + 1
                lines.append(entry)
            elif sym["kind"] == "method" and current_class:
                class_methods.append(sym["name"])
            elif sym["kind"] in ("interface", "type", "struct", "enum", "trait"):
                if current_class and class_methods:
                    methods_str = ", ".join(class_methods)
                    entry = f"    {methods_str}"
                    total_chars += len(entry) + 1
                    lines.append(entry)
                    class_methods = []
                    current_class = None
                entry = f"  {sym['kind']} {sym['name']}"
                total_chars += len(entry) + 1
                lines.append(entry)
            else:
                if current_class and class_methods:
                    methods_str = ", ".join(class_methods)
                    entry = f"    {methods_str}"
                    total_chars += len(entry) + 1
                    lines.append(entry)
                    class_methods = []
                    current_class = None
                entry = f"  {sym['name']}"
                total_chars += len(entry) + 1
                lines.append(entry)

            if total_chars > chars_budget:
                break

        # Flush remaining class methods
        if current_class and class_methods:
            methods_str = ", ".join(class_methods)
            lines.append(f"    {methods_str}")

        if total_chars > chars_budget:
            break

    return "\n".join(lines)

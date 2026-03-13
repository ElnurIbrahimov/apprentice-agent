"""Regex-based symbol extraction for repo map. No tree-sitter dependency.

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
    """Extract symbols from a single file. Returns [{name, kind, line}]."""
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
    """Walk project, skip ignored dirs, return up to MAX_FILES relative paths."""
    files = []
    root = Path(project_root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out skipped directories in-place
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

            if len(files) >= MAX_FILES * 2:  # Collect extra for ranking
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
    except Exception:
        pass

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

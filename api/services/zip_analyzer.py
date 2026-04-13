"""ZIP project analyzer — extracts a code project zip and builds LLM context."""

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Skip rules ────────────────────────────────────────────────────────────────
SKIP_DIRS = {
    "node_modules", "__pycache__", ".git", ".svn", ".hg",
    ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "out",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "coverage", "htmlcoverage",
    ".idea", ".vscode",
    "vendor",
}

SKIP_EXTENSIONS = {
    # Compiled / binary
    ".pyc", ".pyo", ".class", ".o", ".obj", ".so", ".dll", ".dylib", ".exe",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    # Media
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
    ".mp4", ".mp3", ".wav", ".avi", ".mov",
    # Fonts
    ".ttf", ".woff", ".woff2", ".eot",
    # Large blobs
    ".pdf", ".db", ".sqlite", ".sqlite3",
    ".lock",       # package-lock.json, yarn.lock, Cargo.lock — huge, low signal
    ".map",        # Source maps
}

# Per-file char limit — prevents one giant file eating the whole budget
PER_FILE_LIMIT = 8_000

# Total context budget (leaves room for system prompt + user message)
TOTAL_BUDGET = 40_000

# Files matching these names (lower-cased) get read first
PRIORITY_NAMES = [
    "readme.md", "readme.txt", "readme.rst", "readme",
    "main.py", "app.py", "index.py", "server.py", "run.py", "__init__.py",
    "main.ts", "main.js", "index.ts", "index.js", "app.ts", "app.js",
    "main.go", "main.rs", "main.java",
    "package.json", "pyproject.toml", "cargo.toml", "go.mod",
    "requirements.txt", "setup.py", "setup.cfg",
    ".env.example", "docker-compose.yml", "dockerfile",
    "makefile",
]

EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "jsx", ".html": "html", ".css": "css",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".java": "java", ".cpp": "cpp", ".c": "c", ".sh": "bash",
    ".sql": "sql", ".md": "markdown", ".xml": "xml", ".php": "php",
    ".txt": "", ".env": "", ".gitignore": "",
}


def analyze_zip(zip_path: str) -> str:
    """Extract a zip, build a context string with file tree + file contents.

    Always cleans up the temp directory before returning (even on error).
    """
    tmp_dir = tempfile.mkdtemp(prefix="aura_zip_")
    try:
        return _analyze_impl(zip_path, tmp_dir)
    except Exception as e:
        logger.error(f"[ZipAnalyzer] Failed to analyze {zip_path}: {e}")
        return f"[Archive analysis failed: {e}]"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _analyze_impl(zip_path: str, tmp_dir: str) -> str:
    zip_name = Path(zip_path).name
    root = Path(tmp_dir)

    # ── 1. Extract with zip-slip + symlink protection ─────────────────────────
    resolved_root = root.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Skip symlinks — on Unix zipfile.extract() creates real symlinks
            # which can escape the sandbox on later reads via path traversal.
            mode = (member.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                logger.warning(f"[ZipAnalyzer] Skipping symlink member: {member.filename}")
                continue
            target = (root / member.filename).resolve()
            try:
                target.relative_to(resolved_root)
            except ValueError:
                logger.warning(f"[ZipAnalyzer] Skipping unsafe path: {member.filename}")
                continue
            zf.extract(member, tmp_dir)

    # ── 2. Walk tree, collect eligible files ──────────────────────────────────
    all_files = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        # Skip if any ancestor dir is in the skip list
        rel = f.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if _should_skip(rel):
            continue
        all_files.append(rel)

    if not all_files:
        return f"[Archive: {zip_name}]\nNo readable files found in this archive."

    # ── 3. Sort: priority files first, then alphabetical ─────────────────────
    def sort_key(p: Path):
        name = p.name.lower()
        try:
            return (0, PRIORITY_NAMES.index(name), str(p))
        except ValueError:
            return (1, 0, str(p).lower())

    all_files.sort(key=sort_key)

    # ── 4. Build file tree (always included) ──────────────────────────────────
    tree_lines = [f"PROJECT FILE TREE ({len(all_files)} readable files):"]
    for rel in all_files:
        tree_lines.append(f"  {rel}")
    tree_str = "\n".join(tree_lines)
    if len(tree_str) > 3_000:
        tree_str = tree_str[:3_000] + "\n  ... (tree truncated)"

    # ── 5. Read files within budget ───────────────────────────────────────────
    budget = TOTAL_BUDGET - len(tree_str) - 200
    file_sections = []
    skipped = 0

    for rel in all_files:
        if budget <= 200:
            skipped = len(all_files) - len(file_sections)
            break
        try:
            content = (root / rel).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue  # binary that slipped through

        truncated = False
        if len(content) > PER_FILE_LIMIT:
            content = content[:PER_FILE_LIMIT]
            truncated = True

        lang = EXT_TO_LANG.get(rel.suffix.lower(), "")
        note = " [truncated]" if truncated else ""
        section = f"\n--- {rel}{note} ---\n```{lang}\n{content}\n```"

        if len(section) > budget:
            section = section[:budget] + "\n... (file cut for budget)"
            budget = 0
        else:
            budget -= len(section)

        file_sections.append(section)

    # ── 6. Assemble context ───────────────────────────────────────────────────
    parts = [
        f"=== ZIP PROJECT: {zip_name} ===",
        tree_str,
        "\n=== FILE CONTENTS ===",
        *file_sections,
    ]
    if skipped > 0:
        parts.append(f"\n[{skipped} more files not shown — context budget reached]")
    parts.append("=== END ZIP PROJECT ===")

    context = "\n\n".join(parts)
    logger.info(f"[ZipAnalyzer] Built context for {zip_name}: {len(file_sections)} files, {len(context)} chars")
    return context


def _should_skip(rel: Path) -> bool:
    name = rel.name.lower()
    ext = rel.suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return True
    if ".min." in name:
        return True
    # Skip hidden files except a small allowlist
    if name.startswith(".") and name not in {".env.example", ".gitignore", ".dockerignore", ".editorconfig"}:
        return True
    return False

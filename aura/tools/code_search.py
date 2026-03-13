"""Code Search Tool — grep, glob, and definition finder for codebases.

This is the #1 tool that makes coding agents effective. 60%+ of agent time
is spent searching for code, not writing it. Fast, accurate search = faster everything.

Inspired by Claude Code's Grep/Glob tools, ripgrep, and tree-sitter.
"""

import logging
import os
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Directories to always skip during search
SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
    ".mypy_cache", ".tox", ".eggs", "*.egg-info", ".cache",
    ".parcel-cache", ".turbo", ".svelte-kit", "target",  # Rust
    "vendor",  # Go
    ".gradle", ".idea", ".vs", ".vscode",
})

# Binary file extensions to skip
BINARY_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mkv", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".o", ".obj",
    ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".pyc", ".pyo", ".class", ".jar",
    ".db", ".sqlite", ".sqlite3",
    ".lock",  # lockfiles are huge and rarely useful to search
})

# File type to extension mapping (like ripgrep --type)
TYPE_MAP = {
    "py": [".py", ".pyi", ".pyw"],
    "python": [".py", ".pyi", ".pyw"],
    "js": [".js", ".jsx", ".mjs", ".cjs"],
    "javascript": [".js", ".jsx", ".mjs", ".cjs"],
    "ts": [".ts", ".tsx", ".mts", ".cts"],
    "typescript": [".ts", ".tsx", ".mts", ".cts"],
    "rust": [".rs"],
    "go": [".go"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h"],
    "css": [".css", ".scss", ".sass", ".less"],
    "html": [".html", ".htm", ".xhtml"],
    "json": [".json", ".jsonc", ".json5"],
    "yaml": [".yaml", ".yml"],
    "toml": [".toml"],
    "md": [".md", ".mdx"],
    "markdown": [".md", ".mdx"],
    "sql": [".sql"],
    "sh": [".sh", ".bash", ".zsh"],
    "shell": [".sh", ".bash", ".zsh"],
    "ruby": [".rb"],
    "php": [".php"],
    "swift": [".swift"],
    "kotlin": [".kt", ".kts"],
    "dart": [".dart"],
    "vue": [".vue"],
    "svelte": [".svelte"],
}

# Maximum file size to search (skip huge files)
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB

# Maximum results to return
MAX_RESULTS = 200


def _should_skip_dir(name: str) -> bool:
    """Check if directory should be skipped."""
    return name in SKIP_DIRS or name.startswith(".")


def _should_skip_file(path: Path) -> bool:
    """Check if file should be skipped (binary, too large)."""
    if path.suffix.lower() in BINARY_EXTS:
        return True
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return True
    except OSError:
        return True
    return False


def _matches_type(path: Path, file_type: Optional[str]) -> bool:
    """Check if file matches the requested type filter."""
    if not file_type:
        return True
    exts = TYPE_MAP.get(file_type.lower())
    if not exts:
        return True  # Unknown type, don't filter
    return path.suffix.lower() in exts


def _matches_glob_filter(path: Path, glob_filter: Optional[str]) -> bool:
    """Check if file matches a glob filter like '*.py' or '**/*.tsx'."""
    if not glob_filter:
        return True
    return fnmatch(path.name, glob_filter)


def _walk_files(root: Path, file_type: Optional[str] = None,
                glob_filter: Optional[str] = None) -> List[Path]:
    """Walk directory tree, yielding source files that pass all filters."""
    results = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            # Filter out skip directories in-place (modifies os.walk behavior)
            dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]

            for fname in filenames:
                fpath = Path(dirpath) / fname
                if _should_skip_file(fpath):
                    continue
                if not _matches_type(fpath, file_type):
                    continue
                if not _matches_glob_filter(fpath, glob_filter):
                    continue
                results.append(fpath)
    except PermissionError:
        pass
    return results


class CodeSearchTool:
    """Fast code search: grep (content), glob (files), and definition finder.

    The most important tool for any coding agent. Replaces the need to
    read entire files when you just need to find specific code.
    """

    name = "code_search"
    description = "Search code: grep for content patterns, glob for file patterns, find definitions"

    def grep(self, pattern: str, path: str = ".",
             file_type: Optional[str] = None,
             glob_filter: Optional[str] = None,
             case_insensitive: bool = False,
             context_lines: int = 0,
             before_context: int = 0,
             after_context: int = 0,
             output_mode: str = "content",
             max_results: int = MAX_RESULTS) -> dict:
        """Search file contents using regex pattern.

        Args:
            pattern: Regex pattern to search for
            path: Directory or file to search in
            file_type: Filter by file type ('py', 'js', 'ts', etc.)
            glob_filter: Filter by glob pattern ('*.py', '*.tsx')
            case_insensitive: Case-insensitive search
            context_lines: Lines of context around matches (like grep -C)
            before_context: Lines before match (like grep -B)
            after_context: Lines after match (like grep -A)
            output_mode: 'content' (matching lines), 'files' (file paths only), 'count'
            max_results: Maximum number of results

        Returns:
            {success, matches, total_matches, files_searched}
        """
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return {"success": False, "error": f"Invalid regex: {e}"}

            search_path = Path(path).resolve()
            if not search_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            # Single file search
            if search_path.is_file():
                files = [search_path]
            else:
                files = _walk_files(search_path, file_type, glob_filter)

            # Determine context
            ctx_before = before_context or context_lines
            ctx_after = after_context or context_lines

            matches = []
            file_match_counts: Dict[str, int] = {}
            files_searched = 0
            total_matches = 0

            for fpath in files:
                files_searched += 1
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                except (OSError, PermissionError):
                    continue

                lines = content.split("\n")
                file_matches = []

                for i, line in enumerate(lines):
                    if regex.search(line):
                        total_matches += 1
                        if total_matches > max_results:
                            continue  # Keep counting but stop collecting

                        rel_path = str(fpath.relative_to(search_path)) if search_path.is_dir() else fpath.name

                        if output_mode == "files":
                            if rel_path not in file_match_counts:
                                file_match_counts[rel_path] = 0
                            file_match_counts[rel_path] += 1
                        elif output_mode == "count":
                            if rel_path not in file_match_counts:
                                file_match_counts[rel_path] = 0
                            file_match_counts[rel_path] += 1
                        else:
                            # content mode
                            match_entry = {
                                "file": rel_path,
                                "line": i + 1,
                                "text": line.rstrip(),
                            }

                            # Add context lines
                            if ctx_before > 0 or ctx_after > 0:
                                before = []
                                after = []
                                for j in range(max(0, i - ctx_before), i):
                                    before.append(f"{j + 1}: {lines[j].rstrip()}")
                                for j in range(i + 1, min(len(lines), i + 1 + ctx_after)):
                                    after.append(f"{j + 1}: {lines[j].rstrip()}")
                                if before:
                                    match_entry["before"] = before
                                if after:
                                    match_entry["after"] = after

                            file_matches.append(match_entry)

                if file_matches:
                    matches.extend(file_matches)

            # Build result based on output mode
            if output_mode == "files":
                return {
                    "success": True,
                    "files": list(file_match_counts.keys()),
                    "file_counts": file_match_counts,
                    "total_matches": total_matches,
                    "files_searched": files_searched,
                }
            elif output_mode == "count":
                return {
                    "success": True,
                    "counts": file_match_counts,
                    "total_matches": total_matches,
                    "files_searched": files_searched,
                }
            else:
                truncated = total_matches > max_results
                return {
                    "success": True,
                    "matches": matches[:max_results],
                    "total_matches": total_matches,
                    "files_searched": files_searched,
                    "truncated": truncated,
                }

        except Exception as e:
            logger.error(f"[CodeSearch] grep error: {e}")
            return {"success": False, "error": str(e)}

    def glob(self, pattern: str, path: str = ".",
             max_results: int = MAX_RESULTS) -> dict:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern ('**/*.py', 'src/**/*.ts', '*.json')
            path: Directory to search in
            max_results: Maximum files to return

        Returns:
            {success, files, total}
        """
        try:
            search_path = Path(path).resolve()
            if not search_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            results = []
            for fpath in search_path.rglob(pattern):
                # Skip hidden/build directories
                parts = fpath.relative_to(search_path).parts
                if any(_should_skip_dir(p) for p in parts[:-1]):
                    continue
                if fpath.is_file():
                    try:
                        stat = fpath.stat()
                        results.append({
                            "path": str(fpath.relative_to(search_path)),
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                        })
                    except OSError:
                        results.append({
                            "path": str(fpath.relative_to(search_path)),
                        })

            # Sort by modification time (most recent first)
            results.sort(key=lambda x: x.get("modified", 0), reverse=True)

            truncated = len(results) > max_results
            return {
                "success": True,
                "files": results[:max_results],
                "total": len(results),
                "truncated": truncated,
            }

        except Exception as e:
            logger.error(f"[CodeSearch] glob error: {e}")
            return {"success": False, "error": str(e)}

    def find_definition(self, name: str, path: str = ".",
                        file_type: Optional[str] = None) -> dict:
        """Find where a class, function, or variable is defined.

        Uses language-aware regex patterns to find definitions, not just
        references. Much more targeted than plain grep.

        Args:
            name: Symbol name to find (class, function, variable, etc.)
            path: Directory to search in
            file_type: Optional file type filter

        Returns:
            {success, definitions: [{file, line, kind, text}]}
        """
        # Build definition patterns for common languages
        patterns = [
            # Python
            (r'^\s*(async\s+)?def\s+' + re.escape(name) + r'\s*\(', "function"),
            (r'^\s*class\s+' + re.escape(name) + r'[\s(:]', "class"),
            (r'^' + re.escape(name) + r'\s*=', "variable"),
            # JavaScript/TypeScript
            (r'^\s*(export\s+)?(default\s+)?(async\s+)?function\s+' + re.escape(name) + r'\s*[\(<]', "function"),
            (r'^\s*(export\s+)?(default\s+)?class\s+' + re.escape(name) + r'[\s{<]', "class"),
            (r'^\s*(export\s+)?(const|let|var)\s+' + re.escape(name) + r'\s*[=:]', "variable"),
            (r'^\s*(export\s+)?interface\s+' + re.escape(name) + r'[\s{<]', "interface"),
            (r'^\s*(export\s+)?type\s+' + re.escape(name) + r'\s*[=<]', "type"),
            (r'^\s*(export\s+)?enum\s+' + re.escape(name) + r'[\s{]', "enum"),
            # Rust
            (r'^\s*(pub\s+)?(async\s+)?fn\s+' + re.escape(name) + r'[\s<(]', "function"),
            (r'^\s*(pub\s+)?struct\s+' + re.escape(name) + r'[\s{<]', "struct"),
            (r'^\s*(pub\s+)?enum\s+' + re.escape(name) + r'[\s{<]', "enum"),
            (r'^\s*(pub\s+)?trait\s+' + re.escape(name) + r'[\s{<:]', "trait"),
            # Go
            (r'^\s*func\s+(\([^)]*\)\s+)?' + re.escape(name) + r'\s*\(', "function"),
            (r'^\s*type\s+' + re.escape(name) + r'\s+struct\b', "struct"),
            (r'^\s*type\s+' + re.escape(name) + r'\s+interface\b', "interface"),
        ]

        try:
            search_path = Path(path).resolve()
            if not search_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            files = _walk_files(search_path, file_type)
            definitions = []

            compiled = [(re.compile(p), kind) for p, kind in patterns]

            for fpath in files:
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                except (OSError, PermissionError):
                    continue

                lines = content.split("\n")
                rel_path = str(fpath.relative_to(search_path))

                for i, line in enumerate(lines):
                    for regex, kind in compiled:
                        if regex.search(line):
                            definitions.append({
                                "file": rel_path,
                                "line": i + 1,
                                "kind": kind,
                                "text": line.rstrip(),
                            })
                            break  # One match per line is enough

            return {
                "success": True,
                "definitions": definitions,
                "total": len(definitions),
                "name": name,
            }

        except Exception as e:
            logger.error(f"[CodeSearch] find_definition error: {e}")
            return {"success": False, "error": str(e)}

    def find_references(self, name: str, path: str = ".",
                        file_type: Optional[str] = None,
                        max_results: int = 50) -> dict:
        """Find all references to a symbol (not just definitions).

        Args:
            name: Symbol name to find references of
            path: Directory to search in
            file_type: Optional file type filter
            max_results: Maximum results

        Returns:
            {success, references: [{file, line, text}], total}
        """
        # Use word-boundary grep for the name
        pattern = r'\b' + re.escape(name) + r'\b'
        result = self.grep(
            pattern=pattern,
            path=path,
            file_type=file_type,
            max_results=max_results,
        )
        if result.get("success"):
            result["references"] = result.pop("matches", [])
        return result

    def project_structure(self, path: str = ".", max_depth: int = 3) -> dict:
        """Get a tree-like project structure overview.

        Args:
            path: Project root
            max_depth: Maximum directory depth to show

        Returns:
            {success, tree: str, stats: {files, dirs, languages}}
        """
        try:
            root = Path(path).resolve()
            if not root.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            lines = []
            stats = {"files": 0, "dirs": 0, "languages": {}}

            def walk(dir_path: Path, prefix: str, depth: int):
                if depth > max_depth:
                    return

                try:
                    entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
                except PermissionError:
                    return

                dirs = [e for e in entries if e.is_dir() and not _should_skip_dir(e.name)]
                files = [e for e in entries if e.is_file() and e.suffix.lower() not in BINARY_EXTS]

                for i, d in enumerate(dirs):
                    is_last = (i == len(dirs) - 1 and not files)
                    connector = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{connector}{d.name}/")
                    stats["dirs"] += 1
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    walk(d, child_prefix, depth + 1)

                for i, f in enumerate(files):
                    is_last = (i == len(files) - 1)
                    connector = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{connector}{f.name}")
                    stats["files"] += 1
                    ext = f.suffix.lower()
                    for lang, exts in TYPE_MAP.items():
                        if ext in exts and lang == ext.lstrip("."):
                            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
                            break

            lines.append(f"{root.name}/")
            walk(root, "", 0)

            return {
                "success": True,
                "tree": "\n".join(lines[:500]),  # Cap at 500 lines
                "stats": stats,
                "truncated": len(lines) > 500,
            }

        except Exception as e:
            logger.error(f"[CodeSearch] project_structure error: {e}")
            return {"success": False, "error": str(e)}

    def detect_project_type(self, path: str = ".") -> dict:
        """Detect the project type, stack, and key files.

        Checks for common project markers (package.json, requirements.txt,
        Cargo.toml, etc.) and returns structured project metadata.

        Args:
            path: Project root directory

        Returns:
            {success, project_type, stack, key_files, package_manager}
        """
        try:
            root = Path(path).resolve()
            if not root.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            detected = {
                "project_type": "unknown",
                "stack": [],
                "frameworks": [],
                "key_files": [],
                "package_manager": None,
                "language": None,
            }

            # Check for project markers
            markers = {
                "package.json": ("node", "javascript"),
                "tsconfig.json": ("node", "typescript"),
                "requirements.txt": ("python", "python"),
                "pyproject.toml": ("python", "python"),
                "setup.py": ("python", "python"),
                "Pipfile": ("python", "python"),
                "Cargo.toml": ("rust", "rust"),
                "go.mod": ("go", "go"),
                "pom.xml": ("java", "java"),
                "build.gradle": ("java", "java"),
                "Gemfile": ("ruby", "ruby"),
                "composer.json": ("php", "php"),
                "pubspec.yaml": ("dart", "dart"),
                "Package.swift": ("swift", "swift"),
                "Makefile": ("make", None),
                "CMakeLists.txt": ("cmake", "c/c++"),
                "docker-compose.yml": ("docker", None),
                "docker-compose.yaml": ("docker", None),
                "Dockerfile": ("docker", None),
            }

            for marker, (proj_type, lang) in markers.items():
                if (root / marker).exists():
                    detected["key_files"].append(marker)
                    if detected["project_type"] == "unknown":
                        detected["project_type"] = proj_type
                    if lang and lang not in detected["stack"]:
                        detected["stack"].append(lang)

            # Detect frameworks from package.json
            pkg_json = root / "package.json"
            if pkg_json.exists():
                try:
                    import json
                    pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
                    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

                    framework_markers = {
                        "next": "Next.js",
                        "react": "React",
                        "vue": "Vue",
                        "svelte": "Svelte",
                        "@sveltejs/kit": "SvelteKit",
                        "nuxt": "Nuxt",
                        "express": "Express",
                        "fastify": "Fastify",
                        "hono": "Hono",
                        "@angular/core": "Angular",
                        "astro": "Astro",
                        "remix": "Remix",
                        "electron": "Electron",
                        "tailwindcss": "Tailwind CSS",
                        "prisma": "Prisma",
                        "drizzle-orm": "Drizzle",
                        "@supabase/supabase-js": "Supabase",
                    }

                    for dep, fw_name in framework_markers.items():
                        if dep in deps:
                            detected["frameworks"].append(fw_name)

                    # Detect package manager
                    if (root / "bun.lockb").exists() or (root / "bun.lock").exists():
                        detected["package_manager"] = "bun"
                    elif (root / "pnpm-lock.yaml").exists():
                        detected["package_manager"] = "pnpm"
                    elif (root / "yarn.lock").exists():
                        detected["package_manager"] = "yarn"
                    elif (root / "package-lock.json").exists():
                        detected["package_manager"] = "npm"

                    if "typescript" in deps or (root / "tsconfig.json").exists():
                        if "typescript" not in detected["stack"]:
                            detected["stack"].append("typescript")

                except (json.JSONDecodeError, OSError):
                    pass

            # Detect Python frameworks
            reqs = root / "requirements.txt"
            pyproject = root / "pyproject.toml"
            if reqs.exists():
                try:
                    text = reqs.read_text(encoding="utf-8").lower()
                    py_frameworks = {
                        "django": "Django",
                        "flask": "Flask",
                        "fastapi": "FastAPI",
                        "streamlit": "Streamlit",
                        "pytorch": "PyTorch",
                        "torch": "PyTorch",
                        "tensorflow": "TensorFlow",
                        "transformers": "Transformers",
                        "langchain": "LangChain",
                    }
                    for pkg, fw_name in py_frameworks.items():
                        if pkg in text:
                            detected["frameworks"].append(fw_name)
                except OSError:
                    pass

            # Set primary language
            if detected["stack"]:
                detected["language"] = detected["stack"][0]

            # Check for AURA.md
            if (root / "AURA.md").exists():
                detected["key_files"].append("AURA.md")

            detected["success"] = True
            return detected

        except Exception as e:
            logger.error(f"[CodeSearch] detect_project_type error: {e}")
            return {"success": False, "error": str(e)}

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a code search action by name."""
        action_lower = action.lower().strip()

        if action_lower.startswith("grep ") or "search content" in action_lower:
            pattern = kwargs.get("pattern") or action.split(None, 1)[1] if " " in action else ""
            return self.grep(
                pattern=pattern,
                path=kwargs.get("path", "."),
                file_type=kwargs.get("file_type"),
                glob_filter=kwargs.get("glob_filter"),
                case_insensitive=kwargs.get("case_insensitive", False),
                context_lines=kwargs.get("context_lines", 0),
            )
        elif action_lower.startswith("glob ") or "find files" in action_lower:
            pattern = kwargs.get("pattern") or action.split(None, 1)[1] if " " in action else "*"
            return self.glob(
                pattern=pattern,
                path=kwargs.get("path", "."),
            )
        elif action_lower.startswith("def ") or "definition" in action_lower:
            name = kwargs.get("name") or action.split(None, 1)[1] if " " in action else ""
            return self.find_definition(
                name=name,
                path=kwargs.get("path", "."),
                file_type=kwargs.get("file_type"),
            )
        elif action_lower.startswith("ref ") or "references" in action_lower:
            name = kwargs.get("name") or action.split(None, 1)[1] if " " in action else ""
            return self.find_references(
                name=name,
                path=kwargs.get("path", "."),
                file_type=kwargs.get("file_type"),
            )
        elif "structure" in action_lower or "tree" in action_lower:
            return self.project_structure(
                path=kwargs.get("path", "."),
            )
        elif "detect" in action_lower or "project type" in action_lower:
            return self.detect_project_type(
                path=kwargs.get("path", "."),
            )
        else:
            # Default: treat as grep pattern
            return self.grep(
                pattern=action,
                path=kwargs.get("path", "."),
            )

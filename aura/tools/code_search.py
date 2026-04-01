"""Code Search Tool — grep, glob, and definition finder for codebases.

This is the #1 tool that makes coding agents effective. 60%+ of agent time
is spent searching for code, not writing it. Fast, accurate search = faster everything.

Inspired by Claude Code's Grep/Glob tools, ripgrep, and tree-sitter.
"""

import json as _json
import logging
import os
import re
import shutil
import subprocess
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
    return fnmatch(str(path), glob_filter) or fnmatch(path.name, glob_filter)


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

    # Cache ripgrep availability (None = not checked yet)
    _rg_path: Optional[str] = None
    _rg_checked: bool = False

    @classmethod
    def _rg_available(cls) -> Optional[str]:
        """Check if ripgrep (rg) is on PATH. Returns path to rg or None."""
        if cls._rg_checked:
            return cls._rg_path
        cls._rg_checked = True
        cls._rg_path = shutil.which("rg")
        if cls._rg_path:
            logger.info(f"[CodeSearch] ripgrep found at {cls._rg_path}")
        else:
            logger.debug("[CodeSearch] ripgrep not found, using Python fallback")
        return cls._rg_path

    def _grep_ripgrep(
        self,
        pattern: str,
        path: str,
        file_type: Optional[str],
        glob_filter: Optional[str],
        case_insensitive: bool,
        context_lines: int,
        before_context: int,
        after_context: int,
        output_mode: str,
        max_results: int,
    ) -> Optional[dict]:
        """Run grep via ripgrep subprocess. Returns result dict or None on failure."""
        rg = self._rg_available()
        if not rg:
            return None

        search_path = Path(path).resolve()
        if not search_path.exists():
            return None  # Let the Python fallback handle the error message

        cmd = [rg, "--json", f"--max-count={max_results}"]

        # Case sensitivity
        if case_insensitive:
            cmd.append("-i")

        # Context lines
        ctx_before = before_context or context_lines
        ctx_after = after_context or context_lines
        if ctx_before > 0:
            cmd.extend(["-B", str(ctx_before)])
        if ctx_after > 0:
            cmd.extend(["-A", str(ctx_after)])

        # File type filter — use rg's built-in types when possible
        rg_native_types = {
            "py", "python", "js", "javascript", "ts", "typescript",
            "rust", "go", "java", "c", "cpp", "css", "html", "json",
            "yaml", "toml", "md", "markdown", "sql", "sh", "shell",
            "ruby", "php", "swift", "kotlin", "dart",
        }
        if file_type:
            ft_lower = file_type.lower()
            # Map our names to rg's type names
            rg_type_map = {
                "python": "py", "javascript": "js", "typescript": "ts",
                "shell": "sh", "markdown": "md",
            }
            rg_type = rg_type_map.get(ft_lower, ft_lower)

            # Check if rg knows this type natively
            if rg_type in rg_native_types:
                cmd.extend(["--type", rg_type])
            else:
                # Fall back to glob patterns from our TYPE_MAP
                exts = TYPE_MAP.get(ft_lower)
                if exts:
                    for ext in exts:
                        cmd.extend(["--glob", f"*{ext}"])

        # Glob filter
        if glob_filter:
            cmd.extend(["--glob", glob_filter])

        # Pattern and path
        cmd.append(pattern)
        cmd.append(str(search_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            logger.warning("[CodeSearch] ripgrep timed out, falling back to Python")
            return None
        except (OSError, FileNotFoundError):
            logger.warning("[CodeSearch] ripgrep execution failed, falling back to Python")
            return None

        # rg returns exit code 1 for "no matches" (not an error)
        if result.returncode not in (0, 1):
            logger.warning(f"[CodeSearch] ripgrep error (code {result.returncode}): {result.stderr[:200]}")
            return None

        # Parse ripgrep JSON output
        return self._parse_rg_json(
            result.stdout, search_path, output_mode, max_results
        )

    def _parse_rg_json(
        self,
        stdout: str,
        search_path: Path,
        output_mode: str,
        max_results: int,
    ) -> dict:
        """Parse ripgrep --json output into our standard result format."""
        matches = []
        file_match_counts: Dict[str, int] = {}
        files_seen: set = set()
        total_matches = 0

        # Collect context lines keyed by (file, match_line_number)
        # rg JSON emits "context" messages for -B/-C/-A lines
        context_buffer: Dict[str, List] = {}  # key: f"{file}:{line}" -> list of context msgs
        pending_before: List = []  # context lines before the next match

        for raw_line in stdout.splitlines():
            if not raw_line.strip():
                continue
            try:
                msg = _json.loads(raw_line)
            except _json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "begin":
                # New file started
                file_data = msg.get("data", {}).get("path", {})
                fname = file_data.get("text", "")
                if fname:
                    files_seen.add(fname)
                pending_before = []

            elif msg_type == "context":
                # Context line (before or after a match)
                pending_before.append(msg)

            elif msg_type == "match":
                data = msg.get("data", {})
                file_data = data.get("path", {})
                fname = file_data.get("text", "")
                line_number = data.get("line_number", 0)
                lines_data = data.get("lines", {})
                text = lines_data.get("text", "").rstrip("\n").rstrip("\r")

                total_matches += 1

                # Compute relative path
                try:
                    rel_path = str(Path(fname).relative_to(search_path))
                except ValueError:
                    rel_path = fname

                if output_mode in ("files", "count"):
                    file_match_counts[rel_path] = file_match_counts.get(rel_path, 0) + 1
                else:
                    if total_matches <= max_results:
                        match_entry: Dict[str, Any] = {
                            "file": rel_path,
                            "line": line_number,
                            "text": text,
                        }

                        # Attach before-context from pending_before
                        if pending_before:
                            before = []
                            for ctx_msg in pending_before:
                                ctx_data = ctx_msg.get("data", {})
                                ctx_line = ctx_data.get("line_number", 0)
                                ctx_text = ctx_data.get("lines", {}).get("text", "").rstrip("\n").rstrip("\r")
                                before.append(f"{ctx_line}: {ctx_text}")
                            if before:
                                match_entry["before"] = before

                        matches.append(match_entry)
                        # Store reference for after-context attachment
                        context_buffer[f"{rel_path}:{line_number}"] = match_entry

                pending_before = []

            elif msg_type == "end":
                pending_before = []

        # Attach after-context: rg emits context lines after a match before the next match/end.
        # We already handled before-context above. For after-context, we need a second pass
        # through the JSON. Instead, let's do it inline by re-parsing with state tracking.
        # Actually, the simpler approach: re-parse and track after-context properly.
        self._attach_after_context(stdout, search_path, matches, max_results)

        # Build result
        if output_mode == "files":
            return {
                "success": True,
                "files": list(file_match_counts.keys()),
                "file_counts": file_match_counts,
                "total_matches": total_matches,
                "files_searched": len(files_seen),
            }
        elif output_mode == "count":
            return {
                "success": True,
                "counts": file_match_counts,
                "total_matches": total_matches,
                "files_searched": len(files_seen),
            }
        else:
            truncated = total_matches > max_results
            return {
                "success": True,
                "matches": matches[:max_results],
                "total_matches": total_matches,
                "files_searched": len(files_seen),
                "truncated": truncated,
            }

    @staticmethod
    def _attach_after_context(
        stdout: str,
        search_path: Path,
        matches: list,
        max_results: int,
    ) -> None:
        """Second pass: attach after-context lines to match entries."""
        if not matches:
            return

        # Build a lookup from (rel_path, line_number) -> match_entry index
        match_lookup: Dict[tuple, int] = {}
        for idx, m in enumerate(matches):
            match_lookup[(m["file"], m["line"])] = idx

        current_match_idx: Optional[int] = None
        after_lines: List[str] = []

        for raw_line in stdout.splitlines():
            if not raw_line.strip():
                continue
            try:
                msg = _json.loads(raw_line)
            except _json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "match":
                # Flush any pending after-context to previous match
                if current_match_idx is not None and after_lines:
                    if current_match_idx < len(matches):
                        matches[current_match_idx]["after"] = after_lines
                after_lines = []

                # Find this match in our list
                data = msg.get("data", {})
                fname = data.get("path", {}).get("text", "")
                line_number = data.get("line_number", 0)
                try:
                    rel_path = str(Path(fname).relative_to(search_path))
                except ValueError:
                    rel_path = fname
                current_match_idx = match_lookup.get((rel_path, line_number))

            elif msg_type == "context" and current_match_idx is not None:
                ctx_data = msg.get("data", {})
                ctx_line = ctx_data.get("line_number", 0)
                ctx_text = ctx_data.get("lines", {}).get("text", "").rstrip("\n").rstrip("\r")
                # Only add as "after" if this context line comes after the match
                if current_match_idx < len(matches):
                    match_line = matches[current_match_idx]["line"]
                    if ctx_line > match_line:
                        after_lines.append(f"{ctx_line}: {ctx_text}")

            elif msg_type in ("end", "begin"):
                # Flush after-context
                if current_match_idx is not None and after_lines:
                    if current_match_idx < len(matches):
                        matches[current_match_idx]["after"] = after_lines
                after_lines = []
                current_match_idx = None

        # Final flush
        if current_match_idx is not None and after_lines:
            if current_match_idx < len(matches):
                matches[current_match_idx]["after"] = after_lines

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
        # Try ripgrep first — orders of magnitude faster than Python os.walk
        rg_result = self._grep_ripgrep(
            pattern=pattern,
            path=path,
            file_type=file_type,
            glob_filter=glob_filter,
            case_insensitive=case_insensitive,
            context_lines=context_lines,
            before_context=before_context,
            after_context=after_context,
            output_mode=output_mode,
            max_results=max_results,
        )
        if rg_result is not None:
            return rg_result

        # Fallback: pure Python implementation
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

"""Code Edit Tool — surgical find-replace edits on project files.

Instead of rewriting entire files (error-prone, wastes tokens), this tool
performs exact string replacement — the same approach used by Claude Code,
Cursor, and Codex. The key insight: edits should be diffs, not rewrites.

Features:
- Exact string match (primary) with fuzzy fallback (85% threshold)
- Multi-edit batching (apply several edits to one file atomically)
- Unified diff output for review
- Automatic backup with rollback support
- Line-numbered file reading for precise edits
- Auto-verify after edit (syntax check, lint, test runner)
- Automatic rollback on syntax errors
- Aider-style SEARCH/REPLACE block parser
"""

import ast
import difflib
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Directories where edits are blocked for safety
BLOCKED_DIRS = frozenset({
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next",
})


def _is_safe_path(path: Path) -> Tuple[bool, str]:
    """Check if path is safe to edit."""
    resolved = path.resolve()
    parts = resolved.parts

    # Block system directories
    resolved_str = str(resolved).lower()
    blocked_roots = [
        "c:\\windows", "c:\\program files", "c:\\programdata",
        "/etc", "/usr", "/bin", "/sbin", "/var", "/sys", "/proc",
    ]
    for root in blocked_roots:
        if resolved_str.startswith(root):
            return False, f"System path blocked: {root}"

    # Block build/dependency directories
    for part in parts:
        if part in BLOCKED_DIRS:
            return False, f"Cannot edit files in {part}/"

    return True, ""


class CodeEditTool:
    """Surgical code editing via exact string replacement.

    This is NOT sandboxed — it operates on real project files.
    Use for coding agent tasks where you need to modify source code.
    """

    name = "code_edit"
    description = "Edit code files using surgical find-replace (not full rewrites)"

    def __init__(self, backup_enabled: bool = True, auto_verify: bool = True):
        self.backup_enabled = backup_enabled
        self.auto_verify_enabled = auto_verify
        self._last_backups: Dict[str, str] = {}  # path -> backup_path

    # =====================================================================
    #  Read
    # =====================================================================

    def read_file(self, path: str, offset: int = 0, limit: int = 0) -> dict:
        """Read a file with line numbers for precise editing.

        Args:
            path: File path (absolute or relative to cwd)
            offset: Start reading from this line number (1-based, 0 = start)
            limit: Maximum lines to read (0 = all)

        Returns:
            {success, content, lines, total_lines, path}
        """
        try:
            file_path = Path(path).resolve()
            safe, reason = _is_safe_path(file_path)
            if not safe:
                return {"success": False, "error": reason}

            if not file_path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            if not file_path.is_file():
                return {"success": False, "error": f"Not a file: {path}"}

            try:
                raw = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return {"success": False, "error": "Binary file — cannot read as text"}

            all_lines = raw.split("\n")
            total = len(all_lines)

            # Apply offset and limit
            start = max(0, offset - 1) if offset > 0 else 0
            end = start + limit if limit > 0 else total
            selected = all_lines[start:end]

            # Format with line numbers (like cat -n)
            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"{i:>6}\t{line}")

            return {
                "success": True,
                "content": "\n".join(numbered),
                "raw_content": "\n".join(selected),
                "total_lines": total,
                "showing": f"lines {start + 1}-{min(end, total)} of {total}",
                "path": str(file_path),
            }

        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =====================================================================
    #  Auto-verify pipeline
    # =====================================================================

    def _auto_verify_edit(self, file_path: str, edit_result: dict) -> dict:
        """Run verification pipeline after an edit. Returns verification result.

        Three layers:
        1. Syntax check (AST parse) — blocks further checks if broken
        2. Lint check (ruff if available) — advisory
        3. Test runner (pytest on matching test file) — advisory
        """
        verification = {
            "syntax_ok": False,
            "lint_ok": None,
            "tests_ok": None,
            "errors": [],
        }

        # Only verify Python files
        if not file_path.endswith(".py"):
            verification["syntax_ok"] = True  # Skip for non-Python
            return verification

        # Layer 1: Syntax check (AST parse)
        try:
            with open(file_path, encoding="utf-8") as f:
                ast.parse(f.read())
            verification["syntax_ok"] = True
        except SyntaxError as e:
            verification["errors"].append(f"Syntax error at line {e.lineno}: {e.msg}")
            return verification  # Don't proceed if syntax is broken

        # Layer 2: Lint check (ruff if available)
        try:
            result = subprocess.run(
                ["ruff", "check", file_path, "--no-fix", "--quiet"],
                capture_output=True, text=True, timeout=10,
            )
            verification["lint_ok"] = result.returncode == 0
            if not verification["lint_ok"]:
                # Only include first 500 chars of lint output
                lint_out = result.stdout.strip() or result.stderr.strip()
                if lint_out:
                    verification["errors"].append(f"Lint issues:\n{lint_out[:500]}")
        except FileNotFoundError:
            pass  # ruff not installed, skip
        except subprocess.TimeoutExpired:
            pass  # ruff hung, skip

        # Layer 3: Find and run relevant tests
        test_file = self._find_test_file(file_path)
        if test_file:
            try:
                result = subprocess.run(
                    ["python", "-m", "pytest", test_file, "-x", "--tb=short", "-q"],
                    capture_output=True, text=True, timeout=60,
                    cwd=os.path.dirname(file_path),
                )
                verification["tests_ok"] = result.returncode == 0
                if not verification["tests_ok"]:
                    test_out = result.stdout.strip() or result.stderr.strip()
                    verification["errors"].append(
                        f"Test failures ({os.path.basename(test_file)}):\n{test_out[-500:]}"
                    )
            except FileNotFoundError:
                verification["errors"].append("pytest not found — skipping test run")
            except subprocess.TimeoutExpired:
                verification["errors"].append("Test run timed out (60s)")

        return verification

    def _find_test_file(self, source_file: str) -> Optional[str]:
        """Find the test file corresponding to a source file."""
        base = os.path.basename(source_file)
        name = os.path.splitext(base)[0]
        directory = os.path.dirname(source_file)

        # Common patterns: test_X.py, X_test.py, tests/test_X.py
        candidates = [
            os.path.join(directory, f"test_{name}.py"),
            os.path.join(directory, f"{name}_test.py"),
            os.path.join(directory, "tests", f"test_{name}.py"),
            os.path.join(os.path.dirname(directory), "tests", f"test_{name}.py"),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    # =====================================================================
    #  Rollback-on-failure wrapper
    # =====================================================================

    def _apply_with_rollback(self, file_path: str, edit_fn) -> dict:
        """Apply edit with automatic rollback on syntax error.

        Args:
            file_path: Resolved absolute file path
            edit_fn: Callable that performs the edit and returns a result dict.
                     The file must already be written when edit_fn returns.

        Returns:
            Result dict with 'verification' key added.
        """
        # Save original content for rollback
        try:
            with open(file_path, encoding="utf-8") as f:
                original_content = f.read()
        except Exception:
            original_content = None

        # Apply the edit
        result = edit_fn()
        if not result.get("success"):
            return result

        # Verify (only if auto-verify is enabled and we have original content)
        if self.auto_verify_enabled and original_content is not None:
            verification = self._auto_verify_edit(file_path, result)
            result["verification"] = verification

            # Rollback if syntax is broken
            if not verification["syntax_ok"]:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(original_content)
                    result["rolled_back"] = True
                    result["success"] = False
                    result["error"] = (
                        f"Edit caused syntax error — rolled back. "
                        f"{'; '.join(verification['errors'])}"
                    )
                    logger.warning(f"[CodeEdit] Rolled back {file_path}: syntax error after edit")
                except Exception as wb_err:
                    result["rollback_error"] = str(wb_err)
                    logger.error(f"[CodeEdit] Rollback write failed: {wb_err}")

        return result

    # =====================================================================
    #  Core edit
    # =====================================================================

    def edit(self, path: str, old_string: str, new_string: str,
             replace_all: bool = False, dry_run: bool = False) -> dict:
        """Apply a surgical edit: replace old_string with new_string.

        The edit FAILS if old_string is not found in the file (exact match).
        Falls back to fuzzy matching (85% threshold) if exact match fails.

        After a successful (non-dry-run) edit, runs auto-verification:
        syntax check -> lint -> tests. Rolls back automatically on syntax errors.

        Args:
            path: File path
            old_string: Exact text to find and replace
            dry_run: If True, return diff without writing (for preview)
            new_string: Replacement text
            replace_all: Replace all occurrences (default: first only)

        Returns:
            {success, diff, path, backup_path, verification} or
            {success, diff, path, preview} if dry_run
        """
        try:
            # Check instance flag set by agent's preview system
            dry_run = dry_run or getattr(self, '_dry_run_next', False)

            file_path = Path(path).resolve()
            safe, reason = _is_safe_path(file_path)
            if not safe:
                return {"success": False, "error": reason}

            if not file_path.exists():
                return {"success": False, "error": f"File not found: {path}"}

            if old_string == new_string:
                return {"success": False, "error": "old_string and new_string are identical"}

            original = file_path.read_text(encoding="utf-8")

            # === Compute updated content ===
            updated = self._compute_replacement(original, old_string, new_string, replace_all)
            if updated is None:
                # _compute_replacement returns None with error details
                return self._fuzzy_match_error(original, old_string)

            # === Dry run: return diff without writing ===
            if dry_run:
                diff = "".join(difflib.unified_diff(
                    original.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    fromfile=f"a/{file_path.name}",
                    tofile=f"b/{file_path.name}",
                ))
                return {
                    "success": True,
                    "diff": diff,
                    "path": str(file_path),
                    "preview": True,
                }

            # === Apply with rollback protection ===
            resolved_path = str(file_path)

            def _do_edit():
                # Create backup
                backup_path = None
                if self.backup_enabled:
                    backup_path = resolved_path + ".bak"
                    shutil.copy2(resolved_path, backup_path)
                    self._last_backups[resolved_path] = backup_path

                # Write the edit
                file_path.write_text(updated, encoding="utf-8")

                # Generate unified diff
                diff = "".join(difflib.unified_diff(
                    original.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    fromfile=f"a/{file_path.name}",
                    tofile=f"b/{file_path.name}",
                ))

                return {
                    "success": True,
                    "diff": diff,
                    "path": resolved_path,
                    "backup_path": backup_path,
                }

            return self._apply_with_rollback(resolved_path, _do_edit)

        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _compute_replacement(self, original: str, old_string: str,
                             new_string: str, replace_all: bool) -> Optional[str]:
        """Compute the updated file content. Returns None if no match found."""
        # === Exact match (primary) ===
        if old_string in original:
            if replace_all:
                return original.replace(old_string, new_string)
            else:
                if original.count(old_string) > 1:
                    logger.warning(
                        f"[CodeEdit] old_string has {original.count(old_string)} "
                        f"matches — replacing first only"
                    )
                return original.replace(old_string, new_string, 1)

        # === Fuzzy fallback ===
        lines = original.split("\n")
        search_lines = old_string.split("\n")
        best_ratio = 0.0
        best_start = -1

        for i in range(len(lines) - len(search_lines) + 1):
            chunk = "\n".join(lines[i:i + len(search_lines)])
            ratio = difflib.SequenceMatcher(None, chunk, old_string).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        if best_ratio >= 0.85:
            lines[best_start:best_start + len(search_lines)] = new_string.split("\n")
            return "\n".join(lines)

        return None  # No match found

    def _fuzzy_match_error(self, original: str, old_string: str) -> dict:
        """Generate a helpful error when no match is found."""
        lines = original.split("\n")
        search_lines = old_string.split("\n")
        best_ratio = 0.0
        best_start = -1

        for i in range(len(lines) - len(search_lines) + 1):
            chunk = "\n".join(lines[i:i + len(search_lines)])
            ratio = difflib.SequenceMatcher(None, chunk, old_string).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        hint = ""
        if best_ratio > 0.5 and best_start >= 0:
            nearby = "\n".join(lines[best_start:best_start + len(search_lines)])
            hint = f"\n\nClosest match ({best_ratio:.0%}):\n{nearby[:300]}"

        return {
            "success": False,
            "error": f"old_string not found in file (best fuzzy: {best_ratio:.0%}).{hint}",
        }

    # =====================================================================
    #  SEARCH/REPLACE block parser (Aider format)
    # =====================================================================

    def _parse_search_replace_blocks(self, text: str) -> List[dict]:
        """Parse Aider-style SEARCH/REPLACE blocks from LLM output.

        Format:
        <<<<<<< SEARCH
        old code here
        =======
        new code here
        >>>>>>> REPLACE
        """
        blocks = []
        pattern = r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE'
        for match in re.finditer(pattern, text, re.DOTALL):
            blocks.append({
                "search": match.group(1),
                "replace": match.group(2),
            })
        return blocks

    def apply_search_replace_blocks(self, file_path: str, blocks: List[dict]) -> dict:
        """Apply multiple SEARCH/REPLACE blocks to a file.

        Each block has {"search": "...", "replace": "..."}.
        Applies with exact match first, then whitespace-flexible fallback.

        Args:
            file_path: Absolute path to the file
            blocks: List of {search, replace} dicts

        Returns:
            {success, applied, failed, total, verification}
        """
        resolved = Path(file_path).resolve()
        safe, reason = _is_safe_path(resolved)
        if not safe:
            return {"success": False, "error": reason}

        if not resolved.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        resolved_str = str(resolved)

        def _do_apply():
            with open(resolved_str, encoding="utf-8") as f:
                content = f.read()

            original = content
            applied = 0
            failed = []

            for i, block in enumerate(blocks):
                search = block["search"]
                replace = block["replace"]

                if search in content:
                    # Exact match — replace first occurrence
                    content = content.replace(search, replace, 1)
                    applied += 1
                else:
                    # Whitespace-flexible fallback: normalize both and try to locate
                    match_result = self._flexible_search_replace(content, search, replace)
                    if match_result is not None:
                        content = match_result
                        applied += 1
                    else:
                        failed.append({
                            "block": i,
                            "search_preview": search[:100],
                        })

            if applied > 0:
                # Backup before writing
                backup_path = None
                if self.backup_enabled:
                    backup_path = resolved_str + ".bak"
                    shutil.copy2(resolved_str, backup_path)
                    self._last_backups[resolved_str] = backup_path

                with open(resolved_str, "w", encoding="utf-8") as f:
                    f.write(content)

                # Generate diff
                diff = "".join(difflib.unified_diff(
                    original.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f"a/{resolved.name}",
                    tofile=f"b/{resolved.name}",
                ))

                return {
                    "success": True,
                    "applied": applied,
                    "failed": failed,
                    "total": len(blocks),
                    "diff": diff,
                    "path": resolved_str,
                    "backup_path": backup_path,
                }
            else:
                return {
                    "success": False,
                    "applied": 0,
                    "failed": failed,
                    "total": len(blocks),
                    "error": "No blocks could be applied",
                }

        return self._apply_with_rollback(resolved_str, _do_apply)

    def apply_search_replace_text(self, file_path: str, text: str) -> dict:
        """Parse SEARCH/REPLACE blocks from text and apply them to a file.

        Convenience method: parses the text first, then applies.

        Args:
            file_path: Absolute path to the file
            text: Text containing <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks

        Returns:
            Same as apply_search_replace_blocks
        """
        blocks = self._parse_search_replace_blocks(text)
        if not blocks:
            return {
                "success": False,
                "error": "No SEARCH/REPLACE blocks found in text",
                "total": 0,
            }
        return self.apply_search_replace_blocks(file_path, blocks)

    def _flexible_search_replace(self, content: str, search: str,
                                 replace: str) -> Optional[str]:
        """Try whitespace-flexible matching when exact match fails.

        Normalizes whitespace in both content and search string, finds the
        match position, then replaces the corresponding span in the original.

        Returns the updated content, or None if no match.
        """
        # Split into lines for line-based matching (preserves structure better)
        content_lines = content.split("\n")
        search_lines = search.split("\n")

        if not search_lines:
            return None

        # Try to find a line-range where stripped lines match
        search_stripped = [line.strip() for line in search_lines]

        for i in range(len(content_lines) - len(search_lines) + 1):
            candidate = [content_lines[j].strip() for j in range(i, i + len(search_lines))]
            if candidate == search_stripped:
                # Found a whitespace-flexible match — replace those lines
                replace_lines = replace.split("\n")
                content_lines[i:i + len(search_lines)] = replace_lines
                return "\n".join(content_lines)

        return None

    # =====================================================================
    #  Create file
    # =====================================================================

    def create_file(self, path: str, content: str) -> dict:
        """Create a new file. Fails if file already exists.

        Args:
            path: File path to create
            content: File content

        Returns:
            {success, path, bytes_written}
        """
        try:
            file_path = Path(path).resolve()
            safe, reason = _is_safe_path(file_path)
            if not safe:
                return {"success": False, "error": reason}

            if file_path.exists():
                return {"success": False, "error": f"File already exists: {path}. Use edit() to modify."}

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            return {
                "success": True,
                "path": str(file_path),
                "bytes_written": len(content.encode("utf-8")),
            }

        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =====================================================================
    #  Multi-edit (atomic batch)
    # =====================================================================

    def multi_edit(self, path: str, edits: List[Dict[str, str]]) -> dict:
        """Apply multiple edits to a single file atomically.

        All edits must succeed or none are applied. After successful application,
        runs auto-verification with rollback on syntax errors.

        Args:
            path: File path
            edits: List of {old_string, new_string} dicts

        Returns:
            {success, diff, edits_applied, path, verification}
        """
        try:
            file_path = Path(path).resolve()
            safe, reason = _is_safe_path(file_path)
            if not safe:
                return {"success": False, "error": reason}

            if not file_path.exists():
                return {"success": False, "error": f"File not found: {path}"}

            original = file_path.read_text(encoding="utf-8")
            resolved_str = str(file_path)

            # Pre-compute all edits before writing
            current = original
            applied = 0

            for i, edit_item in enumerate(edits):
                old = edit_item.get("old_string", "")
                new = edit_item.get("new_string", "")
                if not old or old == new:
                    continue

                if old in current:
                    current = current.replace(old, new, 1)
                    applied += 1
                else:
                    return {
                        "success": False,
                        "error": f"Edit {i + 1} failed: old_string not found. Rolled back all edits.",
                        "edits_applied": 0,
                    }

            if applied == 0:
                return {"success": False, "error": "No edits to apply"}

            updated = current

            def _do_multi_edit():
                # Backup
                backup_path = None
                if self.backup_enabled:
                    backup_path = resolved_str + ".bak"
                    shutil.copy2(resolved_str, backup_path)
                    self._last_backups[resolved_str] = backup_path

                # Write
                file_path.write_text(updated, encoding="utf-8")

                # Diff
                diff = "".join(difflib.unified_diff(
                    original.splitlines(keepends=True),
                    updated.splitlines(keepends=True),
                    fromfile=f"a/{file_path.name}",
                    tofile=f"b/{file_path.name}",
                ))

                return {
                    "success": True,
                    "diff": diff,
                    "edits_applied": applied,
                    "path": resolved_str,
                    "backup_path": backup_path,
                }

            return self._apply_with_rollback(resolved_str, _do_multi_edit)

        except Exception as e:
            return {"success": False, "error": str(e)}

    # =====================================================================
    #  Rollback
    # =====================================================================

    def rollback(self, path: str) -> dict:
        """Rollback the last edit to a file from its .bak backup.

        Args:
            path: File path to rollback

        Returns:
            {success, restored}
        """
        try:
            file_path = Path(path).resolve()
            backup_path = self._last_backups.get(str(file_path))

            if not backup_path:
                # Try the default .bak path
                backup_path = str(file_path) + ".bak"

            bak = Path(backup_path)
            if not bak.exists():
                return {"success": False, "error": f"No backup found for: {path}"}

            shutil.copy2(str(bak), str(file_path))
            bak.unlink()
            self._last_backups.pop(str(file_path), None)

            return {"success": True, "restored": str(file_path)}

        except Exception as e:
            return {"success": False, "error": str(e)}

    # =====================================================================
    #  Execute dispatcher
    # =====================================================================

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a code edit action."""
        action_lower = action.lower().strip()

        if "read" in action_lower:
            path = kwargs.get("path", "")
            if not path:
                # Try to extract path from action
                parts = action.split()
                path = parts[-1] if len(parts) > 1 else ""
            return self.read_file(
                path=path,
                offset=kwargs.get("offset", 0),
                limit=kwargs.get("limit", 0),
            )
        elif "rollback" in action_lower or "undo" in action_lower:
            path = kwargs.get("path", "")
            return self.rollback(path)
        elif "create" in action_lower or "new file" in action_lower:
            return self.create_file(
                path=kwargs.get("path", ""),
                content=kwargs.get("content", ""),
            )
        elif "search_replace" in action_lower or "search/replace" in action_lower:
            # Apply Aider-style SEARCH/REPLACE blocks
            path = kwargs.get("path", "")
            text = kwargs.get("text", "")
            blocks = kwargs.get("blocks", None)
            if blocks:
                return self.apply_search_replace_blocks(path, blocks)
            elif text:
                return self.apply_search_replace_text(path, text)
            else:
                return {"success": False, "error": "Provide 'text' or 'blocks' for search/replace"}
        else:
            # Default: edit
            return self.edit(
                path=kwargs.get("path", ""),
                old_string=kwargs.get("old_string", ""),
                new_string=kwargs.get("new_string", ""),
                replace_all=kwargs.get("replace_all", False),
            )

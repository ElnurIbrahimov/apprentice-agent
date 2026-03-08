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
"""

import difflib
import logging
import shutil
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

    def __init__(self, backup_enabled: bool = True):
        self.backup_enabled = backup_enabled
        self._last_backups: Dict[str, str] = {}  # path -> backup_path

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

    def edit(self, path: str, old_string: str, new_string: str,
             replace_all: bool = False, dry_run: bool = False) -> dict:
        """Apply a surgical edit: replace old_string with new_string.

        The edit FAILS if old_string is not found in the file (exact match).
        Falls back to fuzzy matching (85% threshold) if exact match fails.

        Args:
            path: File path
            old_string: Exact text to find and replace
            dry_run: If True, return diff without writing (for preview)
            new_string: Replacement text
            replace_all: Replace all occurrences (default: first only)

        Returns:
            {success, diff, path, backup_path} or {success, diff, path, preview} if dry_run
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

            # === Exact match (primary) ===
            if old_string in original:
                if replace_all:
                    updated = original.replace(old_string, new_string)
                else:
                    updated = original.replace(old_string, new_string, 1)

                # Check uniqueness for single replace
                if not replace_all and original.count(old_string) > 1:
                    logger.warning(f"[CodeEdit] old_string has {original.count(old_string)} matches — replacing first only")

            else:
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

                if best_ratio < 0.85:
                    # Help the user by showing nearby content
                    hint = ""
                    if best_ratio > 0.5 and best_start >= 0:
                        nearby = "\n".join(lines[best_start:best_start + len(search_lines)])
                        hint = f"\n\nClosest match ({best_ratio:.0%}):\n{nearby[:300]}"
                    return {
                        "success": False,
                        "error": f"old_string not found in file (best fuzzy: {best_ratio:.0%}).{hint}",
                    }

                lines[best_start:best_start + len(search_lines)] = new_string.split("\n")
                updated = "\n".join(lines)

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

            # === Create backup ===
            backup_path = None
            if self.backup_enabled:
                backup_path = str(file_path) + ".bak"
                shutil.copy2(str(file_path), backup_path)
                self._last_backups[str(file_path)] = backup_path

            # === Write the edit ===
            file_path.write_text(updated, encoding="utf-8")

            # === Generate unified diff ===
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
                "backup_path": backup_path,
            }

        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

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

    def multi_edit(self, path: str, edits: List[Dict[str, str]]) -> dict:
        """Apply multiple edits to a single file atomically.

        Args:
            path: File path
            edits: List of {old_string, new_string} dicts

        Returns:
            {success, diff, edits_applied, path}
        """
        try:
            file_path = Path(path).resolve()
            safe, reason = _is_safe_path(file_path)
            if not safe:
                return {"success": False, "error": reason}

            if not file_path.exists():
                return {"success": False, "error": f"File not found: {path}"}

            original = file_path.read_text(encoding="utf-8")
            current = original
            applied = 0

            for i, edit in enumerate(edits):
                old = edit.get("old_string", "")
                new = edit.get("new_string", "")
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

            # Backup
            backup_path = None
            if self.backup_enabled:
                backup_path = str(file_path) + ".bak"
                shutil.copy2(str(file_path), backup_path)
                self._last_backups[str(file_path)] = backup_path

            # Write
            file_path.write_text(current, encoding="utf-8")

            # Diff
            diff = "".join(difflib.unified_diff(
                original.splitlines(keepends=True),
                current.splitlines(keepends=True),
                fromfile=f"a/{file_path.name}",
                tofile=f"b/{file_path.name}",
            ))

            return {
                "success": True,
                "diff": diff,
                "edits_applied": applied,
                "path": str(file_path),
                "backup_path": backup_path,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

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
        else:
            # Default: edit
            return self.edit(
                path=kwargs.get("path", ""),
                old_string=kwargs.get("old_string", ""),
                new_string=kwargs.get("new_string", ""),
                replace_all=kwargs.get("replace_all", False),
            )

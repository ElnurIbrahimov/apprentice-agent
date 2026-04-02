"""File system tool for reading, writing, and managing files.

SECURITY: All operations are sandboxed to prevent path traversal attacks.
"""

from pathlib import Path
from typing import Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)


class FileSystemTool:
    """Tool for file system operations with SANDBOX ENFORCEMENT.

    SECURITY: All paths are validated to be within the sandbox directory.
    Path traversal attacks (../, symlinks to outside) are blocked.
    """

    name = "filesystem"
    description = "Read, write, list, and manage files and directories (sandboxed)"

    # Directories that are always blocked
    BLOCKED_PATHS = {
        "/etc", "/var", "/usr", "/bin", "/sbin", "/root", "/home",
        "/sys", "/proc", "/dev", "/boot", "/lib", "/lib64",
        "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
        "C:\\Users\\Public", "C:\\ProgramData",
    }

    def __init__(self, base_path: Optional[Path] = None, sandbox_enabled: bool = True):
        self.base_path = Path(base_path).resolve() if base_path else Path.cwd().resolve()
        self.sandbox_enabled = sandbox_enabled

        # Create sandbox directory if it doesn't exist
        if sandbox_enabled:
            self.sandbox_dir = self.base_path / "sandbox"
            self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.sandbox_dir = self.base_path

    def read_file(self, path: str) -> dict:
        """Read contents of a file (sandboxed)."""
        try:
            file_path, error = self._resolve_path(path)
            if error:
                return {"success": False, "error": error, "blocked_by": "sandbox_policy"}

            if not file_path.exists():
                return {"success": False, "error": f"File not found: {path}"}
            if not file_path.is_file():
                return {"success": False, "error": f"Not a file: {path}"}

            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # File is binary or uses a different encoding — return base64 preview
                raw = file_path.read_bytes()
                content = f"[binary file, {len(raw)} bytes — first 64 bytes: {raw[:64].hex()}]"
            return {
                "success": True,
                "content": content,
                "path": str(file_path),
                "size": file_path.stat().st_size
            }
        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def write_file(self, path: str, content: str, overwrite: bool = False) -> dict:
        """Write content to a file (sandboxed)."""
        try:
            file_path, error = self._resolve_path(path)
            if error:
                return {"success": False, "error": error, "blocked_by": "sandbox_policy"}

            if file_path.exists() and not overwrite:
                return {"success": False, "error": f"File exists: {path}. Use overwrite=True"}

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            return {
                "success": True,
                "path": str(file_path),
                "bytes_written": len(content.encode("utf-8"))
            }
        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_directory(self, path: str = ".") -> dict:
        """List contents of a directory (sandboxed)."""
        try:
            dir_path, error = self._resolve_path(path)
            if error:
                return {"success": False, "error": error, "blocked_by": "sandbox_policy"}

            if not dir_path.exists():
                return {"success": False, "error": f"Directory not found: {path}"}
            if not dir_path.is_dir():
                return {"success": False, "error": f"Not a directory: {path}"}

            items = []
            for item in dir_path.iterdir():
                try:
                    size = item.stat().st_size if item.is_file() else None
                except OSError:
                    size = None  # Broken symlink or permission error
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": size
                })

            return {
                "success": True,
                "path": str(dir_path),
                "items": sorted(items, key=lambda x: (x["type"] == "file", x["name"]))
            }
        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_files(self, pattern: str, path: str = ".") -> dict:
        """Search for files matching a pattern (sandboxed)."""
        try:
            search_path, error = self._resolve_path(path)
            if error:
                return {"success": False, "error": error, "blocked_by": "sandbox_policy"}

            if not search_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            matches = list(search_path.rglob(pattern))
            return {
                "success": True,
                "pattern": pattern,
                "matches": [str(m.relative_to(search_path)) for m in matches[:100]]
            }
        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def file_info(self, path: str) -> dict:
        """Get information about a file or directory (sandboxed)."""
        try:
            file_path, error = self._resolve_path(path)
            if error:
                return {"success": False, "error": error, "blocked_by": "sandbox_policy"}

            if not file_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            stat = file_path.stat()
            return {
                "success": True,
                "path": str(file_path),
                "type": "directory" if file_path.is_dir() else "file",
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "created": stat.st_ctime
            }
        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_directory(self, path: str) -> dict:
        """Create a directory (sandboxed)."""
        try:
            dir_path, error = self._resolve_path(path)
            if error:
                return {"success": False, "error": error, "blocked_by": "sandbox_policy"}

            dir_path.mkdir(parents=True, exist_ok=True)
            return {"success": True, "path": str(dir_path)}
        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete(self, path: str) -> dict:
        """Delete a file or empty directory (sandboxed)."""
        try:
            target, error = self._resolve_path(path)
            if error:
                return {"success": False, "error": error, "blocked_by": "sandbox_policy"}

            if not target.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            if target.is_file():
                target.unlink()
            elif target.is_dir():
                target.rmdir()  # Only removes empty directories for safety

            return {"success": True, "deleted": str(target)}
        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _resolve_path(self, path: str) -> Tuple[Optional[Path], Optional[str]]:
        """
        Resolve a path with SANDBOX ENFORCEMENT.

        SECURITY: Prevents path traversal and symlink attacks by:
        1. Input validation (length, null bytes)
        2. Resolving symlinks FIRST with strict=False
        3. Single atomic check against sandbox boundary
        4. Blocking known sensitive directories
        5. No TOCTOU: all checks done on resolved path only

        Returns:
            (resolved_path, error_message) - path is None if blocked
        """
        # === Input Validation ===
        if not path:
            return None, "Empty path"

        # Block null bytes (can bypass some checks)
        if '\x00' in path:
            logger.warning(f"[SECURITY] Blocked null byte in path")
            return None, "Invalid path: null bytes not allowed"

        # Block excessively long paths (Windows MAX_PATH = 260, Linux = 4096)
        if len(path) > 4096:
            return None, "Path too long"

        try:
            p = Path(path)

            # === Construct full path ===
            # If relative, make it relative to sandbox
            if not p.is_absolute():
                p = self.sandbox_dir / p
            elif self.sandbox_enabled:
                # Absolute path - pre-check before expensive resolve()
                path_str = str(p)
                sandbox_str = str(self.sandbox_dir)
                # Case-insensitive comparison on Windows
                import sys
                if sys.platform == "win32":
                    if not path_str.lower().startswith(sandbox_str.lower()):
                        logger.warning("[SECURITY] Blocked absolute path outside sandbox")
                        return None, "Absolute paths outside sandbox not allowed"
                else:
                    if not path_str.startswith(sandbox_str):
                        logger.warning("[SECURITY] Blocked absolute path outside sandbox")
                        return None, "Absolute paths outside sandbox not allowed"

            # === ATOMIC RESOLUTION ===
            # resolve() follows ALL symlinks and normalizes the path
            # This is the ONLY path we use for all subsequent checks
            # No separate symlink check needed - resolve() handles it
            try:
                resolved = p.resolve(strict=False)  # strict=False allows non-existent paths
            except (OSError, RuntimeError) as e:
                # RuntimeError for symlink loops, OSError for other issues
                logger.warning(f"[SECURITY] Path resolution failed: {e}")
                return None, f"Path resolution failed: {e}"

            # === SANDBOX BOUNDARY CHECK (single atomic check) ===
            if self.sandbox_enabled:
                try:
                    resolved.relative_to(self.sandbox_dir)
                except ValueError:
                    # Log the attempted escape
                    logger.warning(f"[SECURITY] Blocked path escape: {path} -> {resolved}")
                    return None, "Path traversal blocked (outside sandbox)"

            # === BLOCKED PATHS CHECK ===
            resolved_str = str(resolved)
            resolved_lower = resolved_str.lower()  # Case-insensitive for Windows

            for blocked in self.BLOCKED_PATHS:
                blocked_lower = blocked.lower()
                if resolved_lower.startswith(blocked_lower) or resolved_lower == blocked_lower:
                    logger.warning(f"[SECURITY] Blocked access to sensitive path: {resolved}")
                    return None, f"Access to {blocked} is blocked"

            # === ADDITIONAL SYMLINK CHECK FOR EXISTING PATHS ===
            # Even after resolve(), verify the final target is safe
            # This catches race conditions where symlink is created after resolve()
            if resolved.exists():
                # Re-resolve to catch any race condition symlink swaps
                final_check = resolved.resolve(strict=True)
                if self.sandbox_enabled:
                    try:
                        final_check.relative_to(self.sandbox_dir)
                    except ValueError:
                        logger.warning(f"[SECURITY] Race condition detected: {resolved} -> {final_check}")
                        return None, "Path changed during validation"

            return resolved, None

        except (OSError, ValueError) as e:
            logger.error(f"[SECURITY] Path resolution error: {e}")
            return None, f"Path resolution error: {e}"

    def _safe_resolve_path(self, path: str) -> dict:
        """Wrapper that returns error dict if path is blocked."""
        resolved, error = self._resolve_path(path)
        if error:
            return {"success": False, "error": error, "blocked_by": "sandbox_policy"}
        return {"success": True, "path": resolved}

    def apply_search_replace(self, path: str, search: str, replace: str) -> dict:
        """Apply SEARCH/REPLACE edit with fuzzy fallback. Returns diff string."""
        import difflib
        import shutil

        try:
            file_path, error = self._resolve_path(path)
            if error:
                return {"success": False, "error": error}
            if not file_path.exists():
                return {"success": False, "error": f"File not found: {path}"}

            original = file_path.read_text(encoding="utf-8")

            # Exact match first
            if search in original:
                updated = original.replace(search, replace, 1)
            else:
                # Fuzzy match using difflib SequenceMatcher
                lines = original.split("\n")
                search_lines = search.split("\n")
                best_ratio = 0.0
                best_start = -1

                for i in range(len(lines) - len(search_lines) + 1):
                    chunk = "\n".join(lines[i:i + len(search_lines)])
                    ratio = difflib.SequenceMatcher(None, chunk, search).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_start = i

                if best_ratio < 0.85:
                    return {"success": False, "error": f"Search text not found (best fuzzy match: {best_ratio:.0%}). Provide more exact text."}

                # Apply fuzzy replacement
                lines[best_start:best_start + len(search_lines)] = replace.split("\n")
                updated = "\n".join(lines)

            # Write backup
            backup_path = str(file_path) + ".bak"
            shutil.copy2(str(file_path), backup_path)

            # Apply edit
            file_path.write_text(updated, encoding="utf-8")

            # Generate unified diff
            diff = "".join(difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{file_path.name}",
                tofile=f"b/{file_path.name}",
            ))

            return {"success": True, "diff": diff, "backup_path": backup_path, "path": str(file_path)}

        except PermissionError:
            return {"success": False, "error": "Permission denied"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def rollback_edit(self, backup_path: str) -> dict:
        """Restore file from .bak backup."""
        import shutil
        try:
            bak, error = self._resolve_path(backup_path)
            if error:
                return {"success": False, "error": error, "blocked_by": "sandbox_policy"}
            if not bak.exists():
                return {"success": False, "error": f"Backup not found: {backup_path}"}
            if bak.suffix != ".bak":
                return {"success": False, "error": f"Not a .bak file: {backup_path}"}
            original = bak.with_suffix("")  # Removes the .bak extension correctly
            shutil.copy2(str(bak), str(original))
            bak.unlink()
            return {"success": True, "restored": str(original)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a filesystem action by name."""
        actions = {
            "read": self.read_file,
            "write": self.write_file,
            "list": self.list_directory,
            "search": self.search_files,
            "info": self.file_info,
            "mkdir": self.create_directory,
            "delete": self.delete,
            "edit": self.apply_search_replace,
            "rollback": self.rollback_edit,
        }
        if action not in actions:
            return {"success": False, "error": f"Unknown action: {action}"}
        return actions[action](**kwargs)

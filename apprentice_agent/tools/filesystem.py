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

            content = file_path.read_text(encoding="utf-8")
            return {
                "success": True,
                "content": content,
                "path": str(file_path),
                "size": len(content)
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
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
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

        SECURITY: Prevents path traversal attacks by:
        1. Blocking absolute paths outside sandbox
        2. Resolving symlinks and checking final destination
        3. Blocking .. traversal that escapes sandbox
        4. Blocking known sensitive directories

        Returns:
            (resolved_path, error_message) - path is None if blocked
        """
        if not path:
            return None, "Empty path"

        try:
            p = Path(path)

            # Block absolute paths when sandboxed
            if self.sandbox_enabled and p.is_absolute():
                # Check if it's within our sandbox
                try:
                    p.resolve().relative_to(self.sandbox_dir)
                except ValueError:
                    logger.warning(f"[SECURITY] Blocked absolute path outside sandbox: {path}")
                    return None, f"Absolute paths outside sandbox not allowed"

            # Resolve relative to sandbox
            if not p.is_absolute():
                p = self.sandbox_dir / p

            # Resolve symlinks and get final path
            resolved = p.resolve()

            # Check for symlink attacks
            if p.exists() and p.is_symlink():
                link_target = p.readlink()
                if link_target.is_absolute():
                    target_resolved = link_target.resolve()
                else:
                    target_resolved = (p.parent / link_target).resolve()

                try:
                    target_resolved.relative_to(self.sandbox_dir)
                except ValueError:
                    logger.warning(f"[SECURITY] Blocked symlink escaping sandbox: {path} -> {target_resolved}")
                    return None, "Symlink points outside sandbox"

            # Verify resolved path is within sandbox
            if self.sandbox_enabled:
                try:
                    resolved.relative_to(self.sandbox_dir)
                except ValueError:
                    logger.warning(f"[SECURITY] Blocked path traversal: {path} -> {resolved}")
                    return None, "Path traversal blocked (outside sandbox)"

            # Check against blocked system paths
            resolved_str = str(resolved)
            for blocked in self.BLOCKED_PATHS:
                if resolved_str.startswith(blocked):
                    logger.warning(f"[SECURITY] Blocked access to sensitive path: {resolved}")
                    return None, f"Access to {blocked} is blocked"

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

    def execute(self, action: str, **kwargs) -> dict:
        """Execute a filesystem action by name."""
        actions = {
            "read": self.read_file,
            "write": self.write_file,
            "list": self.list_directory,
            "search": self.search_files,
            "info": self.file_info,
            "mkdir": self.create_directory,
            "delete": self.delete
        }
        if action not in actions:
            return {"success": False, "error": f"Unknown action: {action}"}
        return actions[action](**kwargs)

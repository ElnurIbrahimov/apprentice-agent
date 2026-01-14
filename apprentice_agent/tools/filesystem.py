"""File system tool for reading, writing, and managing files."""

from pathlib import Path
from typing import Optional
import os


class FileSystemTool:
    """Tool for file system operations."""

    name = "filesystem"
    description = "Read, write, list, and manage files and directories"

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()

    def read_file(self, path: str) -> dict:
        """Read contents of a file."""
        try:
            file_path = self._resolve_path(path)
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
        except Exception as e:
            return {"success": False, "error": str(e)}

    def write_file(self, path: str, content: str, overwrite: bool = False) -> dict:
        """Write content to a file."""
        try:
            file_path = self._resolve_path(path)

            if file_path.exists() and not overwrite:
                return {"success": False, "error": f"File exists: {path}. Use overwrite=True"}

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            return {
                "success": True,
                "path": str(file_path),
                "bytes_written": len(content.encode("utf-8"))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_directory(self, path: str = ".") -> dict:
        """List contents of a directory."""
        try:
            dir_path = self._resolve_path(path)
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
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_files(self, pattern: str, path: str = ".") -> dict:
        """Search for files matching a pattern."""
        try:
            search_path = self._resolve_path(path)
            if not search_path.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            matches = list(search_path.rglob(pattern))
            return {
                "success": True,
                "pattern": pattern,
                "matches": [str(m.relative_to(search_path)) for m in matches[:100]]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def file_info(self, path: str) -> dict:
        """Get information about a file or directory."""
        try:
            file_path = self._resolve_path(path)
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
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_directory(self, path: str) -> dict:
        """Create a directory."""
        try:
            dir_path = self._resolve_path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return {"success": True, "path": str(dir_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete(self, path: str) -> dict:
        """Delete a file or empty directory."""
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return {"success": False, "error": f"Path not found: {path}"}

            if target.is_file():
                target.unlink()
            elif target.is_dir():
                target.rmdir()  # Only removes empty directories for safety

            return {"success": True, "deleted": str(target)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to base_path."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_path / p

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

"""MCP server for Aura — JSON-RPC 2.0 over stdio.

Exposes Aura's tools via MCP protocol so IDEs (VS Code, Cursor, Windsurf)
can use Aura as a tool provider.

Usage:
  python -m aura.core.mcp_server
  # or: aura mcp-serve

VS Code settings.json:
  "mcp.servers": {"aura": {"command": "python", "args": ["-m", "aura.core.mcp_server"]}}
"""

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "aura"
SERVER_VERSION = "1.1.0"

# Always-excluded tools (even in write mode these require explicit per-call approval
# or are simply never safe to expose to arbitrary MCP clients).
_ALWAYS_EXCLUDED = {
    "spawn_agent",
    "shell", "run_shell", "shell_executor",
    "git_push", "git_commit",
    "deploy",
}

# Write tools — opt-in via AURA_MCP_ALLOW_WRITES=true
_WRITE_TOOLS = {
    "write_file", "edit_file", "delete_file",
}


def _allow_writes() -> bool:
    return os.getenv("AURA_MCP_ALLOW_WRITES", "").strip().lower() in ("1", "true", "yes", "on")


def _write_allowlist() -> list[Path]:
    raw = os.getenv("AURA_MCP_WRITE_ALLOWLIST", "").strip()
    if not raw:
        return []
    return [Path(p.strip()).resolve() for p in raw.split(",") if p.strip()]


def _path_allowed(target: str, allowlist: list[Path]) -> bool:
    """True iff the target path is inside at least one allowlisted directory."""
    if not allowlist:
        return True
    try:
        abs_target = Path(target).resolve()
    except Exception:
        return False
    return any(str(abs_target).startswith(str(allowed)) for allowed in allowlist)


def _current_excluded_tools() -> set[str]:
    excluded = set(_ALWAYS_EXCLUDED)
    if not _allow_writes():
        excluded.update(_WRITE_TOOLS)
    return excluded


# Backwards-compat: expose as module-level constant (dynamic accessor preferred)
EXCLUDED_TOOLS = _current_excluded_tools()


def _aura_to_mcp_schema(tool: dict) -> dict:
    """Convert Aura tool schema to MCP format.

    Aura: {type: "function", function: {name, description, parameters}}
    MCP:  {name, description, inputSchema}
    """
    func = tool.get("function", {})
    return {
        "name": func.get("name", ""),
        "description": func.get("description", ""),
        "inputSchema": func.get("parameters", {"type": "object", "properties": {}}),
    }


class AuraMCPServer:
    """Stdio-based MCP server implementing JSON-RPC 2.0."""

    def __init__(self, project_root: str = "."):
        from .agentic_loop import ToolExecutor
        from .tool_schemas import AGENTIC_TOOLS

        self.project_root = os.path.abspath(project_root)
        self.executor = ToolExecutor(self.project_root)
        # Re-evaluate excluded tools at construction time so AURA_MCP_ALLOW_WRITES
        # changes between server runs are picked up without a module reload.
        excluded = _current_excluded_tools()
        self.tools = [
            _aura_to_mcp_schema(t) for t in AGENTIC_TOOLS
            if t["function"]["name"] not in excluded
        ]
        self._write_allowlist = _write_allowlist()
        self._initialized = False

    def run(self):
        """Read JSON-RPC from stdin line-by-line, dispatch, write to stdout."""
        # Set stdio to binary mode for clean JSON-RPC
        if hasattr(sys.stdin, "buffer"):
            input_stream = sys.stdin.buffer
        else:
            input_stream = sys.stdin

        for line in input_stream:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                self._write_error(None, -32700, f"Parse error: {e}")
                continue

            self._handle(request)

    def _handle(self, request: dict):
        """Route JSON-RPC request to handler."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        # Notifications (no id) — acknowledge silently
        if req_id is None and method.startswith("notifications/"):
            return

        if method == "initialize":
            self._handle_initialize(req_id, params)
        elif method == "tools/list":
            self._handle_tools_list(req_id)
        elif method == "tools/call":
            self._handle_tools_call(req_id, params)
        elif method == "resources/list":
            self._handle_resources_list(req_id, params)
        elif method == "resources/read":
            self._handle_resources_read(req_id, params)
        elif method == "ping":
            self._write_result(req_id, {})
        else:
            self._write_error(req_id, -32601, f"Method not found: {method}")

    def _handle_initialize(self, req_id, params: dict):
        """Return protocol version + capabilities."""
        self._initialized = True
        capabilities = {
            "tools": {"listChanged": False},
            "resources": {"subscribe": False, "listChanged": False},
        }
        self._write_result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": capabilities,
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        })

    def _handle_tools_list(self, req_id):
        """Return MCP-formatted tool list."""
        self._write_result(req_id, {"tools": self.tools})

    def _handle_tools_call(self, req_id, params: dict):
        """Execute tool via ToolExecutor, return MCP-formatted result."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if not tool_name:
            self._write_error(req_id, -32602, "Missing tool name")
            return

        known = {t["name"] for t in self.tools}
        if tool_name not in known:
            self._write_error(req_id, -32602, f"Unknown tool: {tool_name}")
            return

        # Write-path allowlist: when allow_writes is on, still restrict paths
        if tool_name in _WRITE_TOOLS and self._write_allowlist:
            path_arg = arguments.get("path") or arguments.get("file_path") or ""
            if path_arg and not _path_allowed(path_arg, self._write_allowlist):
                self._write_error(req_id, -32602,
                    f"Path {path_arg!r} not in AURA_MCP_WRITE_ALLOWLIST")
                return

        try:
            result = self.executor.execute(tool_name, arguments)
            self._write_result(req_id, {
                "content": [{"type": "text", "text": result}],
            })
        except Exception as e:
            self._write_result(req_id, {
                "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                "isError": True,
            })

    # ── Resources (MCP spec) ────────────────────────────────────────────

    def _handle_resources_list(self, req_id, params: dict):
        """List project files as MCP resources.

        Respects .mcpignore (one glob per line) and a hardcoded ignore list
        for obvious noise directories.
        """
        import fnmatch

        project_root = Path(self.project_root)
        if not project_root.exists():
            self._write_result(req_id, {"resources": []})
            return

        ignore_patterns = [
            ".git/*", "node_modules/*", "__pycache__/*",
            ".venv/*", "venv/*", "dist/*", "build/*",
            "*.pyc", "*.pyo", ".pytest_cache/*",
        ]
        mcpignore = project_root / ".mcpignore"
        if mcpignore.exists():
            try:
                ignore_patterns.extend(
                    line.strip()
                    for line in mcpignore.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#")
                )
            except Exception:
                pass

        def _ignored(rel_path: str) -> bool:
            return any(fnmatch.fnmatch(rel_path, pat) for pat in ignore_patterns)

        resources = []
        try:
            for path in project_root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    rel = str(path.relative_to(project_root)).replace("\\", "/")
                except ValueError:
                    continue
                if _ignored(rel):
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size > 512 * 1024:  # skip >512KB files in listing
                    continue
                resources.append({
                    "uri": f"file://{path.as_posix()}",
                    "name": rel,
                    "mimeType": "text/plain",
                })
                if len(resources) >= 1000:
                    break
        except Exception as exc:
            self._write_error(req_id, -32603, f"Resource list error: {exc}")
            return

        self._write_result(req_id, {"resources": resources})

    def _handle_resources_read(self, req_id, params: dict):
        """Read a resource's content."""
        uri = params.get("uri", "")
        if not uri.startswith("file://"):
            self._write_error(req_id, -32602, "Only file:// URIs supported")
            return

        path_str = uri[len("file://"):]
        try:
            path = Path(path_str).resolve()
        except Exception:
            self._write_error(req_id, -32602, "Invalid path")
            return

        project_root = Path(self.project_root).resolve()
        if not str(path).startswith(str(project_root)):
            self._write_error(req_id, -32602, "Path outside project root")
            return

        if not path.is_file():
            self._write_error(req_id, -32602, "Not a file")
            return

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            self._write_error(req_id, -32603, f"Read failed: {exc}")
            return

        self._write_result(req_id, {
            "contents": [{
                "uri": uri,
                "mimeType": "text/plain",
                "text": text,
            }],
        })

    def _write_result(self, req_id, result: dict):
        """Write JSON-RPC success response."""
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        }
        self._write(response)

    def _write_error(self, req_id, code: int, message: str):
        """Write JSON-RPC error response."""
        response = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }
        self._write(response)

    def _write(self, data: dict):
        """Write a JSON line to stdout."""
        line = json.dumps(data, ensure_ascii=False) + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()


def main():
    """Entry point for MCP server."""
    # Suppress logging to stdout (would break JSON-RPC)
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
    )
    server = AuraMCPServer(os.getcwd())
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()

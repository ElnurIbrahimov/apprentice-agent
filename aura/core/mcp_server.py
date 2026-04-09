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

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "aura"
SERVER_VERSION = "1.0.0"

# Tools to exclude from MCP exposure — privileged or dangerous tools
# that should not be callable by external MCP clients without explicit gating
EXCLUDED_TOOLS = {
    "spawn_agent",
    "shell", "run_shell", "shell_executor",  # arbitrary command execution
    "write_file", "edit_file", "delete_file",  # filesystem mutations
    "git_push", "git_commit",  # repository mutations
    "deploy",  # deployment actions
}


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
        self.tools = [
            _aura_to_mcp_schema(t) for t in AGENTIC_TOOLS
            if t["function"]["name"] not in EXCLUDED_TOOLS
        ]
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
        elif method == "ping":
            self._write_result(req_id, {})
        else:
            self._write_error(req_id, -32601, f"Method not found: {method}")

    def _handle_initialize(self, req_id, params: dict):
        """Return protocol version + capabilities."""
        self._initialized = True
        self._write_result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
            },
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

        # Check tool exists
        known = {t["name"] for t in self.tools}
        if tool_name not in known:
            self._write_error(req_id, -32602, f"Unknown tool: {tool_name}")
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

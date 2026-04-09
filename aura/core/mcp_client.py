"""MCP Client — connect to external MCP servers via subprocess stdio.

No dependency on mcp SDK — implements the minimal JSON-RPC 2.0 protocol
directly over subprocess stdin/stdout.
"""

import json
import logging
import os
import subprocess
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class MCPClientConnection:
    """Single connection to an MCP server process."""

    def __init__(self, name: str, command: list[str], env: dict | None = None):
        self.name = name
        self.command = command
        self.env = env
        self.process: Optional[subprocess.Popen] = None
        self.tools: list[dict] = []
        self._req_id = 0
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Start MCP server process and initialize."""
        try:
            # Merge env with current env
            proc_env = dict(os.environ)
            if self.env:
                proc_env.update(self.env)

            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=proc_env,
            )

            # Send initialize
            resp = self._request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "aura", "version": "1.0.0"},
            })
            if not resp:
                logger.error(f"[MCPClient] No response from {self.name} during initialize")
                self.disconnect()
                return False

            # Send initialized notification
            self._notify("notifications/initialized", {})

            # List tools
            tools_resp = self._request("tools/list", {})
            if tools_resp and "tools" in tools_resp:
                self.tools = tools_resp["tools"]
                logger.info(f"[MCPClient] Connected to {self.name}: {len(self.tools)} tools")
            else:
                logger.info(f"[MCPClient] Connected to {self.name}: 0 tools")

            return True
        except Exception as e:
            logger.error(f"[MCPClient] Failed to connect to {self.name}: {e}")
            self.disconnect()
            return False

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on the MCP server. Returns result text."""
        resp = self._request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        if resp and "content" in resp:
            parts = []
            for item in resp["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item["text"])
            return "\n".join(parts) if parts else json.dumps(resp)
        return json.dumps(resp or {"error": "No response"})

    def disconnect(self):
        """Terminate the MCP server process."""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception as e:
                    logger.debug(f"[MCPClient] non-critical: {e}")
            self.process = None

    def _request(self, method: str, params: dict) -> Optional[dict]:
        """Send a JSON-RPC request and read the response."""
        if not self.process or self.process.poll() is not None:
            return None
        with self._lock:
            self._req_id += 1
            msg = json.dumps({
                "jsonrpc": "2.0",
                "id": self._req_id,
                "method": method,
                "params": params,
            }) + "\n"
            try:
                self.process.stdin.write(msg.encode())
                self.process.stdin.flush()
                line = self.process.stdout.readline()
                if line:
                    resp = json.loads(line)
                    if "error" in resp:
                        logger.warning(f"[MCPClient] {self.name} error: {resp['error']}")
                        return None
                    return resp.get("result")
            except Exception as e:
                logger.error(f"[MCPClient] Request to {self.name} failed: {e}")
            return None

    def _notify(self, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)."""
        if not self.process or self.process.poll() is not None:
            return
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }) + "\n"
        try:
            self.process.stdin.write(msg.encode())
            self.process.stdin.flush()
        except Exception as e:
            logger.debug(f"[MCPClient] non-critical: {e}")
class MCPClientManager:
    """Manage multiple MCP server connections."""

    def __init__(self):
        self.connections: dict[str, MCPClientConnection] = {}

    def connect(self, name: str, command: list[str], env: dict | None = None) -> bool:
        """Connect to an MCP server by spawning its process."""
        conn = MCPClientConnection(name, command, env)
        if conn.connect():
            self.connections[name] = conn
            return True
        return False

    def list_all_tools(self) -> list[dict]:
        """Return all tools from all connected servers, prefixed with server name."""
        all_tools = []
        for name, conn in self.connections.items():
            for tool in conn.tools:
                prefixed = dict(tool)
                prefixed["_mcp_server"] = name
                prefixed["name"] = f"mcp_{name}__{tool['name']}"
                all_tools.append(prefixed)
        return all_tools

    def call_tool(self, prefixed_name: str, arguments: dict) -> str:
        """Route a tool call to the correct MCP server."""
        if not prefixed_name.startswith("mcp_"):
            return json.dumps({"error": f"Not an MCP tool: {prefixed_name}"})
        rest = prefixed_name[4:]
        parts = rest.split("__", 1)
        if len(parts) != 2:
            return json.dumps({"error": f"Invalid MCP tool name: {prefixed_name}"})
        server_name, tool_name = parts
        conn = self.connections.get(server_name)
        if not conn:
            return json.dumps({"error": f"MCP server not connected: {server_name}"})
        return conn.call_tool(tool_name, arguments)

    def disconnect_all(self):
        """Disconnect all MCP server connections."""
        for conn in self.connections.values():
            conn.disconnect()
        self.connections.clear()

    def load_from_config(self, config: dict):
        """Load MCP server configs from AURA.md frontmatter.

        Format in AURA.md:
        mcp_servers:
          filesystem:
            command: ["npx", "-y", "@anthropic/mcp-filesystem"]
            args: ["/path/to/dir"]
          github:
            command: ["npx", "-y", "@anthropic/mcp-github"]
            env:
              GITHUB_TOKEN: "..."
        """
        servers = config.get("mcp_servers", {})
        if not servers or not isinstance(servers, dict):
            return
        for name, cfg in servers.items():
            if not isinstance(cfg, dict):
                continue
            cmd = cfg.get("command", [])
            if isinstance(cmd, str):
                cmd = [cmd]
            args = cfg.get("args", [])
            if isinstance(args, str):
                args = [args]
            full_cmd = list(cmd) + list(args)
            env_vars = cfg.get("env")
            if full_cmd:
                logger.info(f"[MCPClient] Connecting to {name}: {' '.join(full_cmd)}")
                self.connect(name, full_cmd, env_vars)

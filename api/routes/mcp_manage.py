"""MCP server management — list/add/remove outbound MCP connections.

Used by the Chrome extension's Connections panel to configure which external
MCP servers Aura's agent loop connects to. Config persists to `~/.aura/mcp.json`
mirroring the Claude Code / Cursor config shape.

Scope: v1 supports stdio and streamable-http outbound transports. The stdio
path uses the existing `aura/core/mcp_client.py::MCPClientConnection`.

Deployment:
  1. git pull on Hetzner
  2. systemctl restart aura-api
  3. Config file auto-created at ~/.aura/mcp.json on first write

Added as part of SOTA Round 2 (Cluster 2).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp-manage"], dependencies=[Depends(require_api_key)])


CONFIG_PATH = Path.home() / ".aura" / "mcp.json"


class McpServerConfig(BaseModel):
    name: str = Field(..., pattern=r"^[a-zA-Z0-9_\-]{1,40}$")
    transport: str = Field(..., pattern=r"^(stdio|http)$")
    url: Optional[str] = None         # for http transport
    command: Optional[List[str]] = None  # for stdio transport
    env: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    enabled: bool = True


class McpServerStatus(BaseModel):
    name: str
    transport: str
    enabled: bool
    connected: bool
    tool_count: int
    tools: List[str] = Field(default_factory=list)
    error: Optional[str] = None


# ── Config file load/save ──────────────────────────────────────────────────

def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {"servers": []}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("[MCP/manage] Failed to read config: %s", e)
        return {"servers": []}


def _save_config(data: Dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.get("/servers")
async def list_servers() -> Dict[str, Any]:
    """Return all configured outbound MCP servers plus their current status."""
    config = _load_config()
    servers = config.get("servers", [])
    statuses: List[McpServerStatus] = []
    for srv in servers:
        statuses.append(McpServerStatus(
            name=srv.get("name", ""),
            transport=srv.get("transport", ""),
            enabled=bool(srv.get("enabled", True)),
            connected=False,  # probed on /test endpoint
            tool_count=int(srv.get("cached_tool_count", 0)),
            tools=list(srv.get("cached_tools", []))[:50],
        ))
    return {"servers": [s.model_dump() for s in statuses], "count": len(statuses)}


@router.post("/servers")
async def add_server(server: McpServerConfig) -> Dict[str, Any]:
    """Add a new outbound MCP server to the registry."""
    if server.transport == "stdio" and not server.command:
        raise HTTPException(400, "stdio transport requires `command`")
    if server.transport == "http" and not server.url:
        raise HTTPException(400, "http transport requires `url`")
    config = _load_config()
    servers = config.get("servers", [])
    if any(s.get("name") == server.name for s in servers):
        raise HTTPException(409, f"Server named '{server.name}' already exists")
    servers.append(server.model_dump())
    config["servers"] = servers
    _save_config(config)
    return {"ok": True, "name": server.name}


@router.delete("/servers/{name}")
async def delete_server(name: str) -> Dict[str, Any]:
    config = _load_config()
    before = len(config.get("servers", []))
    config["servers"] = [s for s in config.get("servers", []) if s.get("name") != name]
    after = len(config["servers"])
    if before == after:
        raise HTTPException(404, f"Server '{name}' not found")
    _save_config(config)
    return {"ok": True, "name": name}


@router.post("/servers/{name}/enable")
async def enable_server(name: str) -> Dict[str, Any]:
    return _set_enabled(name, True)


@router.post("/servers/{name}/disable")
async def disable_server(name: str) -> Dict[str, Any]:
    return _set_enabled(name, False)


def _set_enabled(name: str, enabled: bool) -> Dict[str, Any]:
    config = _load_config()
    found = False
    for s in config.get("servers", []):
        if s.get("name") == name:
            s["enabled"] = enabled
            found = True
            break
    if not found:
        raise HTTPException(404, f"Server '{name}' not found")
    _save_config(config)
    return {"ok": True, "name": name, "enabled": enabled}


@router.post("/servers/{name}/test")
async def test_server(name: str) -> Dict[str, Any]:
    """Attempt to connect to the server and list its tools."""
    config = _load_config()
    srv = next((s for s in config.get("servers", []) if s.get("name") == name), None)
    if not srv:
        raise HTTPException(404, f"Server '{name}' not found")

    transport = srv.get("transport", "")
    try:
        if transport == "stdio":
            from aura.core.mcp_client import MCPClientConnection
            conn = MCPClientConnection(
                name=name,
                command=list(srv.get("command") or []),
                env=srv.get("env") or None,
            )
            if not conn.connect():
                return {"ok": False, "error": "Failed to start MCP subprocess"}
            tools = list(conn.tools or [])
            conn.disconnect()
        elif transport == "http":
            # HTTP path: use a fresh client that speaks Streamable HTTP.
            # If the extension-side HTTPMCPClientConnection isn't available yet,
            # do an inline fetch.
            import httpx
            url = srv.get("url") or ""
            headers = dict(srv.get("headers") or {})
            headers.setdefault("Content-Type", "application/json")
            async with httpx.AsyncClient(timeout=10) as c:
                # initialize
                init = await c.post(url, json={
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "aura", "version": "1.0.0"}},
                }, headers=headers)
                if init.status_code >= 400:
                    return {"ok": False, "error": f"HTTP {init.status_code}"}
                list_resp = await c.post(url, json={
                    "jsonrpc": "2.0", "id": 2, "method": "tools/list",
                }, headers=headers)
                if list_resp.status_code >= 400:
                    return {"ok": False, "error": f"tools/list HTTP {list_resp.status_code}"}
                data = list_resp.json()
                tools = (data.get("result") or {}).get("tools", [])
        else:
            return {"ok": False, "error": f"Unknown transport: {transport}"}
    except Exception as e:
        logger.warning("[MCP/manage] test failed for %s: %s", name, e)
        return {"ok": False, "error": safe_error_detail(e)}

    # Cache tool list in config for quick display
    srv["cached_tool_count"] = len(tools)
    srv["cached_tools"] = [t.get("name", "") if isinstance(t, dict) else str(t) for t in tools][:50]
    _save_config(config)

    return {
        "ok": True,
        "name": name,
        "tool_count": len(tools),
        "tools": [
            {"name": t.get("name", ""), "description": t.get("description", "")}
            if isinstance(t, dict) else {"name": str(t), "description": ""}
            for t in tools
        ],
    }

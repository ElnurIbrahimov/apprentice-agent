"""Aura-as-MCP-server over Streamable HTTP.

Exposes Aura's AGENTIC_TOOLS (click, type, scroll, navigate, snapshot, memory
queries, etc.) as a JSON-RPC 2.0 MCP endpoint over HTTP so external MCP clients
(the Aura Chrome extension, Claude Code on the laptop, Cursor, ChatGPT desktop)
can drive the real browser without needing a stdio subprocess.

This sits alongside the existing stdio server at `aura/core/mcp_server.py`
(invoked via `aura mcp-serve`). The stdio version is still the preferred local
IDE integration; this HTTP version is for remote clients that can't spawn
subprocesses.

Transport: Streamable HTTP as of MCP spec 2025-06-18
  POST /mcp with body {jsonrpc, id, method, params}
  Response: single JSON body (we don't use SSE streaming in v1 — tool calls
  return inline).
  `Mcp-Session-Id` header tracks session continuity (opaque uuid).
  Auth: reuses the standard `X-API-Key` header via `require_api_key`.

Deployment:
  1. git pull on Hetzner
  2. systemctl restart aura-api
  3. Test with curl:
       curl -X POST https://aura-elnur.duckdns.org/mcp \\
         -H 'X-API-Key: ...' \\
         -H 'Content-Type: application/json' \\
         -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

Added as part of SOTA Round 2 (Cluster 2).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

# Note: no `/api` prefix — MCP clients expect the server at a top-level path.
router = APIRouter(prefix="/mcp", tags=["mcp"], dependencies=[Depends(require_api_key)])


PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "aura"
SERVER_VERSION = "1.0.0"

# Allowed origins for DNS-rebinding defense. Extended via env.
ALLOWED_ORIGINS = {
    "https://aura-elnur.duckdns.org",
    "http://localhost",
    "http://127.0.0.1",
    "chrome-extension://*",  # matched via prefix below
}
ENV_ORIGINS = os.getenv("AURA_MCP_ALLOWED_ORIGINS", "")
if ENV_ORIGINS:
    for o in ENV_ORIGINS.split(","):
        o = o.strip()
        if o:
            ALLOWED_ORIGINS.add(o)


def _origin_allowed(origin: Optional[str]) -> bool:
    if not origin:
        return True  # same-origin / no Origin header
    if origin in ALLOWED_ORIGINS:
        return True
    for allowed in ALLOWED_ORIGINS:
        if allowed.endswith("*") and origin.startswith(allowed.rstrip("*")):
            return True
    return False


# Lazy-loaded server instance (reuses existing AuraMCPServer machinery).
_server_singleton: Any = None


def _get_server() -> Any:
    global _server_singleton
    if _server_singleton is None:
        try:
            from aura.core.mcp_server import AuraMCPServer
        except Exception as e:
            logger.error("[MCP/HTTP] Failed to import AuraMCPServer: %s", e)
            raise HTTPException(503, detail="Aura MCP server unavailable")
        _server_singleton = AuraMCPServer(os.getcwd())
    return _server_singleton


def _jsonrpc_result(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _dispatch(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Dispatch a JSON-RPC request and return the response dict, or None for notifications."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {}) or {}

    # Notifications (no id) — acknowledge silently.
    if req_id is None and method.startswith("notifications/"):
        return None

    server = _get_server()

    if method == "initialize":
        return _jsonrpc_result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })

    if method == "ping":
        return _jsonrpc_result(req_id, {})

    if method == "tools/list":
        return _jsonrpc_result(req_id, {"tools": server.tools})

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {}) or {}
        if not tool_name:
            return _jsonrpc_error(req_id, -32602, "Missing tool name")
        known = {t["name"] for t in server.tools}
        if tool_name not in known:
            return _jsonrpc_error(req_id, -32602, f"Unknown tool: {tool_name}")
        try:
            result_text = server.executor.execute(tool_name, arguments)
            return _jsonrpc_result(req_id, {
                "content": [{"type": "text", "text": result_text}],
            })
        except Exception as e:
            logger.warning("[MCP/HTTP] tool %s failed: %s", tool_name, e)
            return _jsonrpc_result(req_id, {
                "content": [{"type": "text", "text": json.dumps({"error": safe_error_detail(e)})}],
                "isError": True,
            })

    return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")


@router.post("")
async def mcp_endpoint(
    request: Request,
    origin: Optional[str] = Header(default=None),
    mcp_session_id: Optional[str] = Header(default=None, alias="Mcp-Session-Id"),
):
    """Streamable HTTP MCP endpoint. Single POST → single JSON response."""
    if not _origin_allowed(origin):
        raise HTTPException(403, detail=f"Origin not allowed: {origin}")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, detail="Invalid JSON body")

    session_id = mcp_session_id or str(uuid.uuid4())

    # Accept both single request and batch [requests] per JSON-RPC spec.
    if isinstance(body, list):
        responses = []
        for req in body:
            try:
                resp = _dispatch(req)
                if resp is not None:
                    responses.append(resp)
            except Exception as e:
                responses.append(_jsonrpc_error(req.get("id") if isinstance(req, dict) else None, -32603, safe_error_detail(e)))
        headers = {"Mcp-Session-Id": session_id}
        return Response(
            content=json.dumps(responses),
            media_type="application/json",
            headers=headers,
        )

    if not isinstance(body, dict):
        raise HTTPException(400, detail="Body must be a JSON object or array")

    try:
        resp = _dispatch(body)
    except Exception as e:
        logger.error("[MCP/HTTP] dispatch error: %s", e)
        resp = _jsonrpc_error(body.get("id"), -32603, safe_error_detail(e))

    # Notifications: return 204 No Content.
    if resp is None:
        return Response(status_code=204, headers={"Mcp-Session-Id": session_id})

    return Response(
        content=json.dumps(resp),
        media_type="application/json",
        headers={"Mcp-Session-Id": session_id},
    )


@router.get("")
async def mcp_get_info() -> Dict[str, Any]:
    """Simple discovery endpoint — some MCP clients GET first to check the server is alive."""
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "transport": "streamable-http",
    }

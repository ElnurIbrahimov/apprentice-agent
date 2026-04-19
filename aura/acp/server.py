"""Aura ACP server — stdio JSON-RPC 2.0.

Minimal ACP (Agent Communication Protocol) implementation. Supports:
  - `initialize` handshake
  - `session/new` / `session/load` / `session/prompt` / `session/cancel`
  - Streaming `session/update` notifications during prompt execution
  - Basic permission routing (allow-all in unrestricted tier; blocked in read-only sandbox)

This is a FUNCTIONAL scaffolding, not a full port of Hermes 2179-LOC adapter.
Advanced features (tool-kind events, resource loading, fork, model switching)
are stubs that return reasonable defaults. Extend as needed.

Entry point:
    python -m aura.acp
    # or: aura acp-serve
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "0.1.0"
SERVER_NAME = "aura-acp"
SERVER_VERSION = "1.0.0"


@dataclass
class ACPSession:
    id: str
    history: List[dict] = field(default_factory=list)
    model_override: Optional[str] = None
    cancelled: bool = False


class AuraACPServer:
    """Stdio JSON-RPC server implementing a minimal ACP subset."""

    def __init__(self, project_root: str = "."):
        self.project_root = os.path.abspath(project_root)
        self._sessions: Dict[str, ACPSession] = {}
        self._sessions_lock = threading.Lock()
        self._initialized = False
        self._brain = None  # Lazy-loaded

    # ── Transport ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Read JSON-RPC from stdin, dispatch, write to stdout."""
        if hasattr(sys.stdin, "buffer"):
            stream = sys.stdin.buffer
        else:
            stream = sys.stdin

        for line in stream:
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
            try:
                self._handle(request)
            except Exception as exc:
                logger.exception("[ACP] handler crash")
                self._write_error(request.get("id"), -32603, f"Internal error: {exc}")

    def _handle(self, request: dict) -> None:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {}) or {}

        # Notifications
        if req_id is None and method.startswith("notifications/"):
            self._handle_notification(method, params)
            return

        if method == "initialize":
            self._handle_initialize(req_id, params)
        elif method == "session/new":
            self._handle_session_new(req_id, params)
        elif method == "session/load":
            self._handle_session_load(req_id, params)
        elif method == "session/prompt":
            self._handle_session_prompt(req_id, params)
        elif method == "session/cancel":
            self._handle_session_cancel(req_id, params)
        elif method == "session/list":
            self._handle_session_list(req_id)
        elif method == "ping":
            self._write_result(req_id, {})
        else:
            self._write_error(req_id, -32601, f"Method not found: {method}")

    def _handle_notification(self, method: str, params: dict) -> None:
        """Silent ack for notifications; log at debug."""
        logger.debug("[ACP] notification: %s", method)

    # ── Handlers ────────────────────────────────────────────────────────

    def _handle_initialize(self, req_id, params: dict) -> None:
        self._initialized = True
        self._write_result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "sessions": True,
                "streaming": True,
                "cancellation": True,
                "tool_call_events": False,  # not yet
                "resources": False,  # not yet
                "fork": False,
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        })

    def _handle_session_new(self, req_id, params: dict) -> None:
        session_id = params.get("id") or str(uuid.uuid4())
        model_override = params.get("model")
        session = ACPSession(id=session_id, model_override=model_override)
        with self._sessions_lock:
            self._sessions[session_id] = session
        self._write_result(req_id, {"sessionId": session_id})

    def _handle_session_load(self, req_id, params: dict) -> None:
        # Minimal: treat load-of-unknown as a fresh session to avoid dead-end
        session_id = params.get("id") or str(uuid.uuid4())
        with self._sessions_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = ACPSession(id=session_id)
        self._write_result(req_id, {"sessionId": session_id})

    def _handle_session_list(self, req_id) -> None:
        with self._sessions_lock:
            ids = list(self._sessions.keys())
        self._write_result(req_id, {"sessions": [{"id": sid} for sid in ids]})

    def _handle_session_cancel(self, req_id, params: dict) -> None:
        session_id = params.get("sessionId") or params.get("id") or ""
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session:
                session.cancelled = True
        self._write_result(req_id, {"cancelled": True})

    def _handle_session_prompt(self, req_id, params: dict) -> None:
        session_id = params.get("sessionId") or params.get("id") or ""
        prompt_blocks = params.get("prompt") or params.get("content") or []

        prompt_text = self._extract_prompt_text(prompt_blocks)
        if not prompt_text.strip():
            self._write_error(req_id, -32602, "Empty prompt")
            return

        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = ACPSession(id=session_id or str(uuid.uuid4()))
                self._sessions[session.id] = session

        session.cancelled = False
        session.history.append({"role": "user", "content": prompt_text})

        # Sandbox check — READ_ONLY blocks any agentic work
        try:
            from aura.core.permissions import SandboxTier, get_sandbox_tier
            if get_sandbox_tier() == SandboxTier.READ_ONLY:
                self._write_error(req_id, -32000,
                    "BLOCKED by READ_ONLY sandbox tier")
                return
        except Exception:
            pass

        # Run prompt in a worker thread so we can stream notifications while
        # the caller waits on the RPC result.
        def _worker():
            try:
                response = self._call_brain(session, prompt_text)
                if session.cancelled:
                    return
                session.history.append({"role": "assistant", "content": response})
                self._emit_session_update(session.id, "message", {
                    "role": "assistant",
                    "content": response,
                })
                self._write_result(req_id, {
                    "sessionId": session.id,
                    "response": response,
                })
            except Exception as exc:
                logger.exception("[ACP] prompt worker crashed")
                self._write_error(req_id, -32603, f"Prompt failed: {exc}")

        threading.Thread(target=_worker, daemon=True, name=f"acp-prompt-{session.id}").start()

    # ── Brain integration ───────────────────────────────────────────────

    def _call_brain(self, session: ACPSession, prompt: str) -> str:
        """Delegate to Aura's brain for a response."""
        if self._brain is None:
            from aura.brain import OllamaBrain
            self._brain = OllamaBrain(warmup=False)

        model = session.model_override
        try:
            if model:
                result = self._brain.think(prompt, model=model)
            else:
                result = self._brain.think(prompt)
        except Exception as exc:
            logger.exception("[ACP] brain.think failed")
            return f"[error] {exc}"

        if isinstance(result, dict):
            return result.get("response", "") or str(result)
        return str(result or "")

    # ── Helpers ─────────────────────────────────────────────────────────

    def _extract_prompt_text(self, blocks: Any) -> str:
        if isinstance(blocks, str):
            return blocks
        if not isinstance(blocks, list):
            return ""
        parts = []
        for b in blocks:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif "text" in b:
                    parts.append(b["text"])
        return "\n".join(parts)

    def _emit_session_update(self, session_id: str, update_type: str, data: dict) -> None:
        """Push a session/update notification to the client."""
        notification = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "type": update_type,
                **data,
            },
        }
        self._write(notification)

    def _write_result(self, req_id, result: dict) -> None:
        self._write({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": result,
        })

    def _write_error(self, req_id, code: int, message: str) -> None:
        self._write({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        })

    def _write(self, data: dict) -> None:
        line = json.dumps(data, ensure_ascii=False) + "\n"
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except BrokenPipeError:
            pass


def run_acp_server(project_root: Optional[str] = None) -> None:
    """Entry point for `aura acp-serve`."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    server = AuraACPServer(project_root or os.getcwd())
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    run_acp_server()

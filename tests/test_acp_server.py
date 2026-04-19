"""Tests for aura.acp.server minimal JSON-RPC handling."""
from __future__ import annotations

import io
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from aura.acp.server import AuraACPServer, PROTOCOL_VERSION


def _drive(server: AuraACPServer, requests: list[dict]) -> list[dict]:
    """Feed JSON-RPC requests through server, capture stdout."""
    captured: list[str] = []

    def _capture(data: dict):
        captured.append(json.dumps(data))

    server._write = _capture  # monkey-patch the writer
    for req in requests:
        server._handle(req)
    return [json.loads(line) for line in captured]


def test_initialize_handshake():
    server = AuraACPServer(project_root=".")
    responses = _drive(server, [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}])
    assert len(responses) == 1
    r = responses[0]
    assert r["id"] == 1
    assert r["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert r["result"]["capabilities"]["sessions"] is True
    assert r["result"]["capabilities"]["streaming"] is True


def test_session_new_returns_id():
    server = AuraACPServer(project_root=".")
    responses = _drive(server, [
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}},
    ])
    assert "sessionId" in responses[0]["result"]
    assert isinstance(responses[0]["result"]["sessionId"], str)


def test_session_new_respects_provided_id():
    server = AuraACPServer(project_root=".")
    responses = _drive(server, [
        {"jsonrpc": "2.0", "id": 1, "method": "session/new",
         "params": {"id": "custom-session-123"}},
    ])
    assert responses[0]["result"]["sessionId"] == "custom-session-123"


def test_session_list_returns_known_sessions():
    server = AuraACPServer(project_root=".")
    _drive(server, [
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"id": "a"}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"id": "b"}},
    ])
    responses = _drive(server, [
        {"jsonrpc": "2.0", "id": 3, "method": "session/list", "params": {}},
    ])
    ids = {s["id"] for s in responses[0]["result"]["sessions"]}
    assert ids == {"a", "b"}


def test_unknown_method_returns_method_not_found():
    server = AuraACPServer(project_root=".")
    responses = _drive(server, [
        {"jsonrpc": "2.0", "id": 1, "method": "nonexistent/method", "params": {}},
    ])
    assert responses[0]["error"]["code"] == -32601


def test_prompt_empty_returns_error():
    server = AuraACPServer(project_root=".")
    _drive(server, [{"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"id": "s"}}])
    responses = _drive(server, [
        {"jsonrpc": "2.0", "id": 2, "method": "session/prompt",
         "params": {"sessionId": "s", "prompt": []}},
    ])
    assert responses[0]["error"]["code"] == -32602
    assert "Empty" in responses[0]["error"]["message"]


def test_prompt_read_only_sandbox_blocks():
    from aura.core.permissions import SandboxTier, set_sandbox_tier, get_sandbox_tier

    original = get_sandbox_tier()
    set_sandbox_tier(SandboxTier.READ_ONLY)
    try:
        server = AuraACPServer(project_root=".")
        _drive(server, [{"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"id": "s"}}])
        # Need to drive synchronously — prompt spawns a thread, but the read-only
        # check fires before the thread starts.
        responses = _drive(server, [
            {"jsonrpc": "2.0", "id": 2, "method": "session/prompt",
             "params": {"sessionId": "s", "prompt": [{"type": "text", "text": "hi"}]}},
        ])
        assert responses[0]["error"]["code"] == -32000
        assert "READ_ONLY" in responses[0]["error"]["message"]
    finally:
        set_sandbox_tier(original)


def test_session_cancel():
    server = AuraACPServer(project_root=".")
    _drive(server, [{"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"id": "s"}}])
    responses = _drive(server, [
        {"jsonrpc": "2.0", "id": 2, "method": "session/cancel", "params": {"sessionId": "s"}},
    ])
    assert responses[0]["result"]["cancelled"] is True


def test_extract_prompt_text_variants():
    server = AuraACPServer(project_root=".")
    assert server._extract_prompt_text("plain") == "plain"
    assert server._extract_prompt_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert server._extract_prompt_text([{"text": "x"}]) == "x"
    assert server._extract_prompt_text([]) == ""
    assert server._extract_prompt_text(None) == ""


def test_ping():
    server = AuraACPServer(project_root=".")
    responses = _drive(server, [{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}])
    assert responses[0]["result"] == {}

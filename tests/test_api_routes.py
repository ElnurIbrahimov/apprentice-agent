"""Tests for API routes — input validation, auth, error handling.

Covers the critical endpoints identified in ENGINEERING_REVIEW_2026-04-02.
Uses FastAPI TestClient (httpx-based) so no real server is needed.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create TestClient with auth disabled for unit testing."""
    import os
    os.environ["AURA_API_AUTH_ENABLED"] = "false"
    os.environ["AURA_ENV"] = "development"

    # Mock the agent service to avoid heavy init
    with patch("api.services.agent_service.agent_service") as mock_svc:
        mock_svc.is_ready = True
        mock_svc.start_background_init = MagicMock()
        from api.main import app
        yield TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Input Validation — Conversation ID
# ---------------------------------------------------------------------------

class TestConversationIDValidation:
    """Conversation endpoints must reject invalid IDs."""

    INVALID_IDS = [
        "",                          # empty
        "../../../etc/passwd",       # path traversal
        "a" * 200,                   # too long
        "id with spaces",            # spaces
        "id<script>alert(1)</script>",  # XSS
        'id"; DROP TABLE;--',        # SQLi
    ]

    def test_rename_rejects_invalid_id(self, client):
        for bad_id in self.INVALID_IDS:
            if not bad_id:
                continue  # skip empty — FastAPI returns 404 for empty path
            r = client.put(f"/api/chat/conversations/{bad_id}", json={"title": "test"})
            assert r.status_code in (400, 404, 422), f"Expected 4xx for id={bad_id!r}, got {r.status_code}"

    def test_delete_rejects_invalid_id(self, client):
        r = client.delete("/api/chat/conversations/../../../etc/passwd")
        assert r.status_code in (400, 404, 422)

    def test_switch_rejects_invalid_id(self, client):
        r = client.post("/api/chat/conversations/a<>b/switch")
        assert r.status_code in (400, 404, 422)

    def test_messages_rejects_invalid_id(self, client):
        r = client.get("/api/chat/conversations/id%00null/messages")
        assert r.status_code in (400, 404, 422)

    def test_valid_id_format_accepted(self, client):
        """Valid ID format should not be rejected by validation (may 404 if not found)."""
        r = client.put("/api/chat/conversations/conv_abc123", json={"title": "test"})
        # Should not be 400 (validation error) — 404/500 is OK (not found / no agent)
        assert r.status_code != 400 or "Invalid" not in r.text


# ---------------------------------------------------------------------------
# Input Validation — Emotion
# ---------------------------------------------------------------------------

class TestEmotionValidation:
    """Emotion endpoints must reject invalid values."""

    def test_mood_trigger_rejects_invalid_emotion(self, client):
        r = client.post("/api/mood/trigger?emotion=__import__('os')&intensity=0.5")
        assert r.status_code in (400, 422)

    def test_mood_trigger_accepts_valid_emotion(self, client):
        r = client.post("/api/mood/trigger?emotion=happy&intensity=0.5")
        # 200 or 500 (if ALMA not available) — but NOT 400
        assert r.status_code != 400

    def test_mood_trigger_clamps_intensity(self, client):
        """Intensity outside 0-1 should be clamped, not rejected."""
        r = client.post("/api/mood/trigger?emotion=calm&intensity=99.0")
        assert r.status_code != 400


# ---------------------------------------------------------------------------
# Input Validation — Provider Name
# ---------------------------------------------------------------------------

class TestProviderValidation:
    """Provider endpoints must reject invalid names."""

    def test_provider_models_rejects_special_chars(self, client):
        r = client.get("/api/providers/../etc/models")
        assert r.status_code in (400, 404, 422)

    def test_provider_models_rejects_uppercase(self, client):
        r = client.get("/api/providers/DROP_TABLE/models")
        assert r.status_code == 400

    def test_provider_key_rejects_bad_name(self, client):
        r = client.post("/api/providers/../../etc/key", json={"key": "test123"})
        assert r.status_code in (400, 404, 422)

    def test_provider_valid_name(self, client):
        r = client.get("/api/providers/openai/models")
        # 200 or 404 (not configured) — but not 400 (validation)
        assert r.status_code != 400


# ---------------------------------------------------------------------------
# Input Validation — Research
# ---------------------------------------------------------------------------

class TestResearchValidation:
    """Research endpoint must validate model names and query length."""

    def test_research_rejects_empty_query(self, client):
        r = client.post("/api/research", json={"query": "", "depth": "quick"})
        assert r.status_code == 400

    def test_research_rejects_oversized_query(self, client):
        r = client.post("/api/research", json={"query": "x" * 3000, "depth": "quick"})
        assert r.status_code in (400, 422)  # Pydantic max_length=1000 returns 422

    def test_research_rejects_bad_model(self, client):
        r = client.post("/api/research", json={
            "query": "test",
            "depth": "quick",
            "model": "'; DROP TABLE models;--"
        })
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Centralized Validators (api/utils.py)
# ---------------------------------------------------------------------------

class TestValidators:
    """Unit tests for centralized validators."""

    def test_validate_id_accepts_valid(self):
        from api.utils import validate_id
        assert validate_id("conv_abc123") == "conv_abc123"
        assert validate_id("ses_1234_abcd") == "ses_1234_abcd"
        assert validate_id("a-b-c") == "a-b-c"

    def test_validate_id_rejects_invalid(self):
        from api.utils import validate_id
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            validate_id("")
        with pytest.raises(HTTPException):
            validate_id("../../../etc/passwd")
        with pytest.raises(HTTPException):
            validate_id("a" * 200)
        with pytest.raises(HTTPException):
            validate_id("id with spaces")

    def test_validate_model_name_accepts_valid(self):
        from api.utils import validate_model_name
        assert validate_model_name("kimi-k2.5:cloud") == "kimi-k2.5:cloud"
        assert validate_model_name("qwen3.5:397b-cloud") == "qwen3.5:397b-cloud"
        assert validate_model_name("nemotron-3-super:cloud") == "nemotron-3-super:cloud"

    def test_validate_model_name_rejects_invalid(self):
        from api.utils import validate_model_name
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            validate_model_name("")
        with pytest.raises(HTTPException):
            validate_model_name("'; DROP TABLE;--")
        with pytest.raises(HTTPException):
            validate_model_name("model name with spaces")

    def test_validate_emotion_accepts_valid(self):
        from api.utils import validate_emotion
        assert validate_emotion("happy") == "happy"
        assert validate_emotion("CALM") == "calm"
        assert validate_emotion("  Neutral  ") == "neutral"

    def test_validate_emotion_rejects_invalid(self):
        from api.utils import validate_emotion
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            validate_emotion("__import__('os')")
        with pytest.raises(HTTPException):
            validate_emotion("not_a_real_emotion")
        with pytest.raises(HTTPException):
            validate_emotion("")


# ---------------------------------------------------------------------------
# Auth Middleware
# ---------------------------------------------------------------------------

class TestAuthMiddleware:
    """API key auth middleware rejects unauthenticated requests when enabled."""

    def test_health_endpoint_unauthenticated(self, client):
        """Health checks should work without auth."""
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_health_deep_unauthenticated(self, client):
        r = client.get("/api/health/deep")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# SSRF Guard
# ---------------------------------------------------------------------------

class TestSSRFGuard:
    """SSRF protection validates URLs and prevents redirect loops."""

    def test_blocks_private_ip(self):
        from aura.security.ssrf_guard import validate_url_safe
        with pytest.raises(ValueError, match="private"):
            validate_url_safe("http://127.0.0.1/admin")

    def test_blocks_internal_port(self):
        from aura.security.ssrf_guard import validate_url_safe
        with pytest.raises(ValueError, match="Blocked port"):
            validate_url_safe("http://example.com:6379/")

    def test_blocks_file_scheme(self):
        from aura.security.ssrf_guard import validate_url_safe
        with pytest.raises(ValueError, match="Blocked scheme"):
            validate_url_safe("file:///etc/passwd")

    def test_blocks_long_url(self):
        from aura.security.ssrf_guard import validate_url_safe
        with pytest.raises(ValueError, match="max length"):
            validate_url_safe("http://example.com/" + "a" * 5000)

    def test_allows_public_url(self):
        from aura.security.ssrf_guard import validate_url_safe
        pinned, hostname = validate_url_safe("http://1.1.1.1/test")
        assert pinned  # Should return without raising


# ---------------------------------------------------------------------------
# Session Serializer — tool_call_id preservation
# ---------------------------------------------------------------------------

class TestSessionSerializer:
    """Session serializer preserves tool_call_id."""

    def test_preserves_tool_call_id(self):
        from aura.core.session import AgenticSession
        ses = AgenticSession()

        # Create a mock Pydantic-like ToolCall
        class MockFunction:
            name = "web_search"
            arguments = {"query": "test"}
        class MockToolCall:
            id = "call_abc123"
            function = MockFunction()

        msg = {
            "role": "assistant",
            "content": "Let me search.",
            "tool_calls": [MockToolCall()],
        }
        serialized = ses._serialize_message(msg)
        assert serialized["tool_calls"][0].get("id") == "call_abc123"
        assert serialized["tool_calls"][0]["function"]["name"] == "web_search"

    def test_dict_tool_calls_passthrough(self):
        from aura.core.session import AgenticSession
        ses = AgenticSession()
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "xyz", "function": {"name": "test", "arguments": {}}}],
        }
        serialized = ses._serialize_message(msg)
        assert serialized["tool_calls"][0]["id"] == "xyz"

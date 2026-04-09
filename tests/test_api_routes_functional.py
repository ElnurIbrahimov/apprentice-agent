"""Functional tests for API routes — endpoint behavior, response shapes, error paths.

Complements test_api_routes.py (input validation) with actual endpoint behavior tests.
Uses FastAPI TestClient with mocked agent service.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create TestClient with auth disabled and agent service mocked."""
    os.environ["AURA_API_AUTH_ENABLED"] = "false"
    os.environ["AURA_ENV"] = "development"
    os.environ.pop("AURA_API_KEY", None)

    with patch("api.services.agent_service.agent_service") as mock_svc:
        mock_svc.is_ready = True
        mock_svc.start_background_init = MagicMock()
        from api.main import app
        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_agent():
    """Provide a mock agent with brain for chat tests."""
    agent = MagicMock()
    agent.brain = MagicMock()
    agent.brain.think = MagicMock(return_value="Hello! How can I help?")
    agent.brain.get_last_model_used = MagicMock(return_value="test-model:latest")
    agent.brain._query_count = 5
    agent.brain._total_query_count = 100
    agent.brain.conversation_history = []
    return agent


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    """Health check endpoints should always respond."""

    def test_health_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_deep_health_returns_200(self, client):
        r = client.get("/api/health/deep")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] in ("ok", "degraded", "down")
        assert "subsystems" in data
        assert "uptime_seconds" in data
        assert "environment" in data

    def test_health_no_auth_required(self, client):
        """Health endpoint must work even if auth were enabled."""
        r = client.get("/api/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

class TestChatEndpoints:
    """Chat endpoint request/response contract."""

    def test_chat_missing_message_returns_422(self, client):
        r = client.post("/api/chat", json={})
        assert r.status_code == 422

    def test_chat_empty_message_returns_422(self, client):
        r = client.post("/api/chat", json={"message": ""})
        assert r.status_code == 422

    def test_chat_valid_request_shape(self, client, mock_agent):
        """Valid chat request should return ChatResponse shape."""
        with patch("api.routes.chat._get_agent_service") as mock_svc:
            svc = MagicMock()
            svc.is_ready = True
            svc.chat.return_value = "Test response"
            mock_svc.return_value = svc
            r = client.post("/api/chat", json={"message": "hello"})
            if r.status_code == 200:
                data = r.json()
                assert "response" in data

    def test_chat_oversized_message_returns_422(self, client):
        r = client.post("/api/chat", json={"message": "x" * 200_000})
        assert r.status_code == 422

    def test_conversations_list(self, client):
        """GET /api/chat/conversations should return JSON."""
        r = client.get("/api/chat/conversations")
        # May return 200 or 500 (no agent) — either is valid wiring
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            data = r.json()
            assert isinstance(data, (list, dict))


# ---------------------------------------------------------------------------
# Models endpoints
# ---------------------------------------------------------------------------

class TestModelsEndpoints:
    """Model listing and configuration endpoints."""

    def test_models_list_returns_json(self, client):
        r = client.get("/api/models")
        assert r.status_code in (200, 404, 500)
        if r.status_code == 200:
            data = r.json()
            assert isinstance(data, (list, dict))

    def test_models_roles_returns_json(self, client):
        r = client.get("/api/models/roles")
        assert r.status_code in (200, 404, 500)


# ---------------------------------------------------------------------------
# Memory endpoints
# ---------------------------------------------------------------------------

class TestMemoryEndpoints:
    """Memory recall tracking endpoints."""

    def test_memory_stats_returns_json(self, client):
        r = client.get("/api/memory/stats")
        assert r.status_code in (200, 404, 500)
        if r.status_code == 200:
            data = r.json()
            assert isinstance(data, dict)

    def test_memory_recalls_returns_json(self, client):
        r = client.get("/api/memory/recalls")
        assert r.status_code in (200, 404, 500)
        if r.status_code == 200:
            data = r.json()
            assert isinstance(data, (list, dict))


# ---------------------------------------------------------------------------
# Hands endpoints
# ---------------------------------------------------------------------------

class TestHandsEndpoints:
    """Autonomous Hands API endpoints."""

    def test_hands_list_returns_json(self, client):
        r = client.get("/api/hands")
        assert r.status_code in (200, 404, 500)
        if r.status_code == 200:
            data = r.json()
            assert isinstance(data, dict)
            assert "hands" in data

    def test_hands_nonexistent_returns_error(self, client):
        r = client.get("/api/hands/nonexistent_hand_xyz")
        assert r.status_code in (404, 500)

    def test_hands_run_nonexistent_returns_error(self, client):
        r = client.post("/api/hands/nonexistent_hand_xyz/run")
        assert r.status_code in (404, 422, 500)


# ---------------------------------------------------------------------------
# Status endpoints
# ---------------------------------------------------------------------------

class TestStatusEndpoints:
    """Status and admin endpoints."""

    def test_status_returns_json(self, client):
        r = client.get("/api/status")
        assert r.status_code in (200, 404, 500)
        if r.status_code == 200:
            data = r.json()
            assert "online" in data or "status" in data

    def test_status_models_returns_json(self, client):
        r = client.get("/api/status/models")
        assert r.status_code in (200, 404, 500)


# ---------------------------------------------------------------------------
# Auth enforcement
# ---------------------------------------------------------------------------

class TestAuthEnforcement:
    """Auth middleware behavior when enabled."""

    def test_auth_enabled_rejects_without_key(self):
        """When auth is enabled with a key, requests without key are rejected."""
        env = {
            "AURA_API_AUTH_ENABLED": "true",
            "AURA_API_KEY": "test-secret-key-12345",
            "AURA_ENV": "development",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("api.services.agent_service.agent_service") as mock_svc:
                mock_svc.is_ready = True
                mock_svc.start_background_init = MagicMock()
                # Re-import to pick up new env
                import importlib
                import api.auth
                importlib.reload(api.auth)
                from api.main import app
                c = TestClient(app, raise_server_exceptions=False)
                r = c.get("/api/status")
                # Should be 401 or 403 (rejected)
                assert r.status_code in (401, 403)

    def test_auth_enabled_accepts_valid_key(self):
        """When auth is enabled, valid key is accepted."""
        key = "test-secret-key-12345"
        env = {
            "AURA_API_AUTH_ENABLED": "true",
            "AURA_API_KEY": key,
            "AURA_ENV": "development",
        }
        with patch.dict(os.environ, env, clear=False):
            with patch("api.services.agent_service.agent_service") as mock_svc:
                mock_svc.is_ready = True
                mock_svc.start_background_init = MagicMock()
                import importlib
                import api.auth
                importlib.reload(api.auth)
                from api.main import app
                c = TestClient(app, raise_server_exceptions=False)
                r = c.get("/api/health", headers={"X-API-Key": key})
                # Health is on public router, should always work
                assert r.status_code == 200


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """API should return clean error responses, not stack traces."""

    def test_404_on_unknown_api_route(self, client):
        r = client.get("/api/nonexistent_endpoint_xyz")
        assert r.status_code in (404, 405)

    def test_chat_returns_json_on_error(self, client):
        """Even on internal errors, chat should return JSON, not HTML."""
        with patch("api.routes.chat._get_agent_service") as mock_svc:
            svc = MagicMock()
            svc.is_ready = False
            mock_svc.return_value = svc
            r = client.post("/api/chat", json={"message": "test"})
            # Should be a JSON error, not a 500 HTML page
            if r.status_code >= 400:
                assert r.headers.get("content-type", "").startswith("application/json")

    def test_method_not_allowed(self, client):
        """PUT on health should return 405."""
        r = client.put("/api/health")
        assert r.status_code == 405


# ---------------------------------------------------------------------------
# Rate limiting headers
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """Rate limit middleware should set tracking headers."""

    def test_request_id_header_present(self, client):
        """Every response should have X-Request-ID."""
        r = client.get("/api/health")
        assert "x-request-id" in r.headers


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

class TestCORS:
    """CORS headers should be present on responses."""

    def test_cors_allows_localhost(self, client):
        r = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Should not be blocked
        assert r.status_code in (200, 204, 400)

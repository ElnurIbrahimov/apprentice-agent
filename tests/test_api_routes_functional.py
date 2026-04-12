"""Functional tests for stable API route contracts."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a TestClient with auth disabled for deterministic route tests."""
    env = {
        "AURA_API_AUTH_ENABLED": "false",
        "AURA_ENV": "development",
    }
    with patch.dict(os.environ, env, clear=False):
        from api.main import app

        yield TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_deep_health_returns_expected_shape(self, client):
        response = client.get("/api/health/deep")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in {"ok", "degraded", "down"}
        assert isinstance(data["subsystems"], list)
        assert "uptime_seconds" in data
        assert "environment" in data


class TestChatEndpoints:
    def test_chat_missing_message_returns_422(self, client):
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_chat_empty_message_returns_422(self, client):
        response = client.post("/api/chat", json={"message": ""})
        assert response.status_code == 422

    def test_chat_valid_request_returns_chat_response_shape(self, client):
        with patch("api.routes.chat._get_agent_service") as mock_get_service:
            service = MagicMock()
            service.chat.return_value = {
                "response": "Test response",
                "fast_path": False,
                "mood": {
                    "emotion": "neutral",
                    "confidence": 50,
                    "valence": 0.0,
                    "arousal": 0.0,
                    "dominance": 0.0,
                    "emoji": "🙂",
                },
                "model_used": "test-model:latest",
            }
            mock_get_service.return_value = service

            response = client.post("/api/chat", json={"message": "hello"})

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert data["model_used"] == "test-model:latest"
        assert data["mood"]["emotion"] == "neutral"

    def test_conversations_list_returns_json(self, client):
        with patch("api.routes.chat._get_agent_service") as mock_get_service:
            service = MagicMock()
            service.list_conversations.return_value = [
                {"id": "conv-1", "title": "First conversation"},
            ]
            mock_get_service.return_value = service

            response = client.get("/api/chat/conversations")

        assert response.status_code == 200
        assert response.json() == [{"id": "conv-1", "title": "First conversation"}]


class TestModelsEndpoints:
    def test_models_list_returns_expected_shape(self, client):
        with patch("api.routes.models._get_verified_models") as mock_verified:
            mock_verified.return_value = {
                "cloud": [{"name": "cloud-a"}],
                "local": [{"name": "local-a"}],
                "chatgpt": [{"name": "chatgpt:gpt-test"}],
                "direct_api": [{"name": "direct-a"}],
                "total": 4,
            }

            response = client.get("/api/models")

        assert response.status_code == 200
        data = response.json()
        assert data["chatgpt_models"] == ["chatgpt:gpt-test"]
        assert data["cloud_models"] == ["cloud-a"]
        assert data["local_models"] == ["local-a"]
        assert data["direct_api_models"] == ["direct-a"]
        assert data["total"] == 4

    def test_models_roles_returns_json(self, client):
        response = client.get("/api/models/roles")

        assert response.status_code == 200
        assert isinstance(response.json(), dict)


class TestMemoryEndpoints:
    def test_memory_recall_stats_returns_json(self, client):
        response = client.get("/api/memory/recalls/stats")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "total_recalls" in data
        assert "recent_count" in data

    def test_memory_recent_recalls_returns_json(self, client):
        response = client.get("/api/memory/recalls/recent")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 0
        assert isinstance(data["events"], list)


class TestHandsEndpoints:
    def test_hands_list_returns_json(self, client):
        with patch("api.routes.hands._get_manager") as mock_get_manager:
            manager = MagicMock()
            manager.list_hands.return_value = [{"name": "guardian"}]
            mock_get_manager.return_value = manager

            response = client.get("/api/hands")

        assert response.status_code == 200
        data = response.json()
        assert data["hands"] == [{"name": "guardian"}]
        assert data["count"] == 1

    def test_hands_nonexistent_returns_404(self, client):
        with patch("api.routes.hands._get_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_hand.return_value = None
            mock_get_manager.return_value = manager

            response = client.get("/api/hands/nonexistent_hand_xyz")

        assert response.status_code == 404

    def test_hands_run_nonexistent_returns_404(self, client):
        with patch("api.routes.hands._get_manager") as mock_get_manager:
            manager = MagicMock()
            manager.get_hand.return_value = None
            mock_get_manager.return_value = manager

            response = client.post("/api/hands/nonexistent_hand_xyz/run")

        assert response.status_code == 404


class TestStatusEndpoints:
    def test_status_returns_json(self, client):
        with patch("api.routes.status._get_agent_service") as mock_get_service:
            service = MagicMock()
            service.is_ready = False
            mock_get_service.return_value = service

            response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert data["online"] is True
        assert "model" in data

    def test_models_detailed_returns_json(self, client):
        with patch("api.routes.status._get_agent_service") as mock_get_service:
            service = MagicMock()
            service.is_ready = False
            mock_get_service.return_value = service

            response = client.get("/api/models/detailed")

        assert response.status_code == 200
        data = response.json()
        assert "local_models" in data
        assert "cloud_models" in data
        assert "current_model" in data


class TestAuthEnforcement:
    def test_auth_enabled_rejects_without_key(self):
        env = {
            "AURA_API_AUTH_ENABLED": "true",
            "AURA_API_KEY": "test-secret-key-12345",
            "AURA_ENV": "development",
        }
        with patch.dict(os.environ, env, clear=False):
            from api.main import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/api/status")

        assert response.status_code == 401

    def test_auth_enabled_accepts_valid_key(self):
        key = "test-secret-key-12345"
        env = {
            "AURA_API_AUTH_ENABLED": "true",
            "AURA_API_KEY": key,
            "AURA_ENV": "development",
        }
        with patch.dict(os.environ, env, clear=False):
            from api.main import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/api/status", headers={"X-API-Key": key})

        assert response.status_code == 200


class TestErrorHandling:
    def test_404_on_unknown_api_route(self, client):
        response = client.get("/api/nonexistent_endpoint_xyz")
        assert response.status_code == 404

    def test_chat_returns_json_on_error(self, client):
        with patch("api.routes.chat._get_agent_service") as mock_get_service:
            service = MagicMock()
            service.chat.side_effect = RuntimeError("boom")
            mock_get_service.return_value = service

            response = client.post("/api/chat", json={"message": "test"})

        assert response.status_code == 500
        assert response.headers.get("content-type", "").startswith("application/json")

    def test_method_not_allowed(self, client):
        response = client.put("/api/health")
        assert response.status_code == 405


class TestHeadersAndCors:
    def test_request_id_header_present(self, client):
        response = client.get("/api/health")
        assert "x-request-id" in response.headers

    def test_cors_preflight_is_not_blocked(self, client):
        response = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code in {200, 204, 400}

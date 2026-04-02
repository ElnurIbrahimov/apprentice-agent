import os
import pytest

pytest.importorskip("fastapi")

from unittest.mock import patch


@pytest.fixture
def auth_client():
    """Create a TestClient with auth enabled."""
    with patch.dict(os.environ, {
        "AURA_API_KEY": "test-key-123",
        "AURA_API_AUTH_ENABLED": "true",
    }):
        from fastapi.testclient import TestClient
        from api.main import app
        yield TestClient(app)


def test_proactive_start_requires_auth(auth_client):
    response = auth_client.post("/api/proactive/start")
    assert response.status_code == 401


def test_proactive_start_with_valid_key(auth_client):
    response = auth_client.post("/api/proactive/start", headers={"X-API-Key": "test-key-123"})
    assert response.status_code != 401


def test_memory_recalls_requires_auth(auth_client):
    response = auth_client.get("/api/memory/recalls/recent")
    assert response.status_code == 401

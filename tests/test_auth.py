import os
import pytest
from fastapi.testclient import TestClient

# Set API key before importing app
os.environ["AURA_API_KEY"] = "test-key-123"

from api.main import app

client = TestClient(app)

def test_proactive_start_requires_auth():
    response = client.post("/api/proactive/start")
    assert response.status_code == 401

def test_proactive_start_with_valid_key():
    response = client.post("/api/proactive/start", headers={"X-API-Key": "test-key-123"})
    assert response.status_code != 401

def test_memory_recalls_requires_auth():
    response = client.get("/api/memory/recalls/recent")
    assert response.status_code == 401

"""Pytest fixtures for integration tests.

Provides:
- mock_ollama: starts/stops a MockOllamaServer and patches env vars
- mock_agent: creates an ApprenticeAgent wired to mock Ollama
- clean_env: saves and restores environment variables
"""

import os
import sys
import tempfile

import pytest

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tests.helpers.mock_ollama import MockOllamaServer


@pytest.fixture()
def mock_ollama(monkeypatch, tmp_path):
    """Start a mock Ollama server and patch environment to point at it.

    Yields the running MockOllamaServer instance.
    After the test, the server is stopped and env vars are restored.
    """
    server = MockOllamaServer()
    server.start()

    # Patch environment so all Aura code uses the mock
    monkeypatch.setenv("OLLAMA_HOST", server.base_url)
    # Use temp dirs to avoid polluting real data
    monkeypatch.setenv("AURA_DATA_DIR", str(tmp_path / "aura_data"))
    # Disable cloud models (no API key in tests)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    # Set model names to local-looking names so warmup hits the mock
    monkeypatch.setenv("MODEL_FAST", "test-model:latest")
    monkeypatch.setenv("MODEL_REASON", "test-model:latest")
    monkeypatch.setenv("MODEL_CODE", "test-model:latest")
    monkeypatch.setenv("MODEL_VISION", "test-model:latest")
    monkeypatch.setenv("MODEL_THINK", "test-model:latest")
    monkeypatch.setenv("MODEL_LONGCTX", "test-model:latest")

    yield server

    server.stop()


@pytest.fixture()
def patched_config(mock_ollama):
    """Reload Config class attributes to pick up the monkeypatched env vars.

    Must be used after mock_ollama fixture. Returns the Config class.
    """
    from aura.config import Config

    # Force Config class attributes to re-read from (now patched) env
    Config.OLLAMA_HOST = os.environ["OLLAMA_HOST"]
    Config.MODEL_FAST = os.environ.get("MODEL_FAST", Config.MODEL_FAST)
    Config.MODEL_REASON = os.environ.get("MODEL_REASON", Config.MODEL_REASON)
    Config.MODEL_CODE = os.environ.get("MODEL_CODE", Config.MODEL_CODE)
    Config.MODEL_VISION = os.environ.get("MODEL_VISION", Config.MODEL_VISION)
    Config.MODEL_THINK = os.environ.get("MODEL_THINK", Config.MODEL_THINK)
    Config.MODEL_LONGCTX = os.environ.get("MODEL_LONGCTX", Config.MODEL_LONGCTX)
    Config.MODEL_NAME = Config.MODEL_REASON

    yield Config

    # Note: monkeypatch fixture restores env vars automatically after test


@pytest.fixture()
def mock_brain(patched_config):
    """Create an OllamaBrain connected to the mock Ollama server.

    Skips warmup for speed.
    """
    from aura.brain import OllamaBrain
    brain = OllamaBrain(warmup=False)
    return brain


@pytest.fixture()
def mock_agent(patched_config):
    """Create an ApprenticeAgent in fast_init mode, wired to mock Ollama.

    Uses fast_init=True to skip heavy tool loading and warmup.
    """
    from aura.agent import ApprenticeAgent
    agent = ApprenticeAgent(fast_init=True)
    return agent

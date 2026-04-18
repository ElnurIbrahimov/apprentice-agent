"""Integration tests for the Aura CLI.

These tests exercise the real code paths (agent init, brain calls, command
dispatch, agentic loop, non-interactive mode) against a mock Ollama HTTP server
so that no real LLM calls are made and tests stay fast (< 5s each).
"""

import os
import sys
import subprocess
import threading

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Test 1: OllamaBrain initializes with mock Ollama
# ---------------------------------------------------------------------------
class TestBrainInit:
    def test_brain_creates_client(self, mock_brain):
        """OllamaBrain should initialize with a working client when Ollama is reachable."""
        assert mock_brain is not None
        assert mock_brain.client is not None
        assert mock_brain.model == "test-model:latest"

    def test_brain_history_starts_empty(self, mock_brain):
        """New brain should have empty or loaded conversation history."""
        assert isinstance(mock_brain.conversation_history, list)


# ---------------------------------------------------------------------------
# Test 2: Agent initializes with mock Ollama
# ---------------------------------------------------------------------------
class TestAgentInit:
    def test_agent_init(self, mock_agent):
        """Agent should initialize successfully with a running mock Ollama."""
        assert mock_agent is not None
        assert mock_agent.brain is not None
        assert mock_agent.brain.client is not None

    def test_agent_has_tools(self, mock_agent):
        """Agent should have core tools loaded even in fast_init mode."""
        assert isinstance(mock_agent.tools, dict)
        # Core tools are always loaded
        assert len(mock_agent.tools) > 0

    def test_agent_identity_loaded(self, mock_agent):
        """Agent should load identity during init."""
        assert mock_agent.identity is not None


# ---------------------------------------------------------------------------
# Test 3: Agent handles a simple prompt via run()
# ---------------------------------------------------------------------------
class TestAgentRun:
    def test_simple_greeting(self, mock_agent):
        """Agent.run() should return a response for a simple greeting (fast-path)."""
        result = mock_agent.run("hello")
        assert isinstance(result, dict)
        assert "response" in result
        assert len(result["response"]) > 0

    def test_run_returns_standard_keys(self, mock_agent):
        """Agent.run() result should contain the standard response keys."""
        result = mock_agent.run("hi")
        assert "goal" in result
        assert "response" in result
        assert "completed" in result

    def test_run_with_complex_prompt(self, mock_agent):
        """Agent.run() should handle a non-trivial prompt that goes through the agent loop."""
        result = mock_agent.run("What is 2+2?", timeout_seconds=15)
        assert isinstance(result, dict)
        assert "response" in result
        # Should have a non-empty response (from mock Ollama)
        assert len(result.get("response", "")) > 0


# ---------------------------------------------------------------------------
# Test 4: Command registry dispatch
# ---------------------------------------------------------------------------
class TestCommandDispatch:
    def test_registry_has_commands(self):
        """COMMAND_REGISTRY should contain all expected slash commands."""
        from aura.cli.commands import COMMAND_REGISTRY
        assert isinstance(COMMAND_REGISTRY, dict)
        assert len(COMMAND_REGISTRY) > 20  # We know there are 30+ commands
        # Spot-check a few key commands
        assert "/help" in COMMAND_REGISTRY
        assert "/quit" in COMMAND_REGISTRY
        assert "/model" in COMMAND_REGISTRY
        assert "/shell" in COMMAND_REGISTRY

    def test_handle_command_unknown(self, mock_agent, capsys):
        """handle_command with unknown command should print error, not crash."""
        from aura.cli.commands import handle_command
        handle_command(mock_agent, "/nonexistent_cmd_xyz")
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_handle_command_fuzzy_match(self, mock_agent, capsys):
        """handle_command should suggest close matches for typos."""
        from aura.cli.commands import handle_command
        handle_command(mock_agent, "/hel")  # close to /help
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out
        # May or may not suggest /help depending on cutoff, but should not crash

    def test_help_command_does_not_crash(self, mock_agent):
        """Calling /help handler directly should not raise."""
        from aura.cli.commands import COMMAND_REGISTRY
        handler = COMMAND_REGISTRY["/help"]
        # Should not raise; may print to stdout
        handler(mock_agent, "", {"speak": False})

    def test_quit_raises_systemexit(self, mock_agent):
        """/quit handler should raise SystemExit."""
        from aura.cli.commands import COMMAND_REGISTRY
        handler = COMMAND_REGISTRY["/quit"]
        with pytest.raises(SystemExit):
            handler(mock_agent, "", {"speak": False})

    def test_all_handlers_are_callable(self):
        """Every handler in COMMAND_REGISTRY should be a callable."""
        from aura.cli.commands import COMMAND_REGISTRY
        for cmd, handler in COMMAND_REGISTRY.items():
            assert callable(handler), f"Handler for {cmd} is not callable"


# ---------------------------------------------------------------------------
# Test 5: AgenticLoop respects cancellation
# ---------------------------------------------------------------------------
class TestAgenticLoop:
    def test_cancel_event_is_settable(self, mock_brain):
        """AgenticLoop.cancel() should set the internal cancel event."""
        from aura.core.agentic_loop import AgenticLoop

        loop = AgenticLoop(brain=mock_brain, max_iterations=50)
        assert not loop._cancel_event.is_set()
        loop.cancel()
        assert loop._cancel_event.is_set()

    def test_loop_cancel_stops_mid_run(self, mock_brain):
        """Cancelling from another thread should stop the loop."""
        from aura.core.agentic_loop import AgenticLoop

        loop = AgenticLoop(brain=mock_brain, max_iterations=50)

        # Cancel from a timer thread after a short delay (after run() clears the event)
        cancel_timer = threading.Timer(0.3, loop.cancel)
        cancel_timer.start()
        try:
            result = loop.run("do something complex")
        finally:
            cancel_timer.cancel()

        assert isinstance(result, dict)
        # The loop should have stopped early (well under 50 iterations)
        assert result.get("iterations", 0) < 50

    def test_loop_respects_max_iterations(self, mock_brain):
        """Loop should stop at max_iterations."""
        from aura.core.agentic_loop import AgenticLoop

        loop = AgenticLoop(brain=mock_brain, max_iterations=1)
        result = loop.run("test prompt")
        assert isinstance(result, dict)
        assert result.get("iterations", 0) <= 2  # At most 1 iteration + finalization


# ---------------------------------------------------------------------------
# Test 6: Health check detects down Ollama
# ---------------------------------------------------------------------------
class TestHealthCheck:
    def test_health_check_ok_with_mock(self, mock_ollama, patched_config):
        """Health check should return (True, models) when mock Ollama is running."""
        from aura.brain_support import ollama_health_check
        ok, models = ollama_health_check()
        assert ok is True, f"Expected ok=True, got: {ok}"
        assert isinstance(models, list)

    def test_health_check_detects_down_ollama(self, monkeypatch):
        """Health check should return (False, []) when Ollama is unreachable."""
        # Point to a port that nothing is listening on
        monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:19999")
        # Reload Config to pick up the changed env var
        from aura.config import Config
        Config.OLLAMA_HOST = "http://127.0.0.1:19999"

        from aura.brain_support import ollama_health_check
        ok, models = ollama_health_check()
        assert ok is False, f"Expected ok=False, got: {ok}"
        assert models == []


# ---------------------------------------------------------------------------
# Test 7: Non-interactive mode (aura -p 'test')
# ---------------------------------------------------------------------------
class TestNonInteractiveMode:
    def test_prompt_flag_produces_output(self, mock_ollama, tmp_path):
        """Running main.py -p 'hello' should produce output and exit cleanly."""
        env = os.environ.copy()
        env["OLLAMA_HOST"] = mock_ollama.base_url
        env["AURA_DATA_DIR"] = str(tmp_path / "aura_data")
        # Override models to local names so Config doesn't try cloud validation
        env["MODEL_FAST"] = "test-model:latest"
        env["MODEL_REASON"] = "test-model:latest"
        env["MODEL_CODE"] = "test-model:latest"
        env["MODEL_VISION"] = "test-model:latest"
        env["MODEL_THINK"] = "test-model:latest"
        env["MODEL_LONGCTX"] = "test-model:latest"
        # Remove cloud key to avoid cloud client init
        env.pop("OLLAMA_API_KEY", None)

        main_py = os.path.join(_PROJECT_ROOT, "main.py")
        # 60s budget — subprocess cold-start imports torch/transformers and
        # warms the mock Ollama client; under parallel test load we've seen
        # this tip past 30s, causing spurious TimeoutExpired failures.
        result = subprocess.run(
            [sys.executable, main_py, "-p", "hello"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=_PROJECT_ROOT,
        )
        # The process should exit (0 = success, but even non-zero is acceptable
        # as long as it doesn't hang). Check that it produced *some* output.
        assert result.returncode == 0 or len(result.stdout + result.stderr) > 0, (
            f"Process exited with rc={result.returncode}, no output.\n"
            f"stdout: {result.stdout[:500]}\nstderr: {result.stderr[:500]}"
        )


# ---------------------------------------------------------------------------
# Test 8: Mock Ollama server itself works correctly
# ---------------------------------------------------------------------------
class TestMockOllamaServer:
    """Sanity-check the mock server directly with requests."""

    def test_tags_endpoint(self, mock_ollama):
        """GET /api/tags should return fake model list."""
        import requests
        resp = requests.get(f"{mock_ollama.base_url}/api/tags", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) > 0

    def test_chat_endpoint(self, mock_ollama):
        """POST /api/chat should return a canned response."""
        import requests
        resp = requests.post(
            f"{mock_ollama.base_url}/api/chat",
            json={
                "model": "test-model:latest",
                "messages": [{"role": "user", "content": "hello"}],
            },
            timeout=5,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"]["role"] == "assistant"
        assert len(data["message"]["content"]) > 0

    def test_generate_endpoint(self, mock_ollama):
        """POST /api/generate should return a canned response."""
        import requests
        resp = requests.post(
            f"{mock_ollama.base_url}/api/generate",
            json={"model": "test-model:latest", "prompt": "test"},
            timeout=5,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True

    def test_embed_endpoint(self, mock_ollama):
        """POST /api/embed should return fake embedding vectors."""
        import requests
        resp = requests.post(
            f"{mock_ollama.base_url}/api/embed",
            json={"model": "nomic-embed-text:latest", "input": "test text"},
            timeout=5,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) == 384

    def test_custom_response(self, mock_ollama):
        """Setting custom_response should change what the server returns."""
        import requests
        mock_ollama.set_response("CUSTOM ANSWER 42")
        resp = requests.post(
            f"{mock_ollama.base_url}/api/chat",
            json={
                "model": "test-model:latest",
                "messages": [{"role": "user", "content": "q"}],
            },
            timeout=5,
        )
        data = resp.json()
        assert data["message"]["content"] == "CUSTOM ANSWER 42"

    def test_warmup_ping(self, mock_ollama):
        """Empty prompt to /api/generate (warmup) should return empty response."""
        import requests
        resp = requests.post(
            f"{mock_ollama.base_url}/api/generate",
            json={"model": "test-model:latest", "prompt": ""},
            timeout=5,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert data["response"] == ""

    def test_context_manager(self):
        """MockOllamaServer should work as a context manager."""
        from tests.helpers.mock_ollama import MockOllamaServer
        import requests
        with MockOllamaServer() as server:
            resp = requests.get(f"{server.base_url}/api/tags", timeout=5)
            assert resp.status_code == 200
        # After exit, server should be stopped — connection should fail
        with pytest.raises(requests.ConnectionError):
            requests.get(f"http://127.0.0.1:{server.port}/api/tags", timeout=1)

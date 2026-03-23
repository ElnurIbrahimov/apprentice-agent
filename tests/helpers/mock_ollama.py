"""Mock Ollama HTTP server for integration tests.

Provides a lightweight HTTP server that mimics Ollama API endpoints:
- POST /api/chat       - returns canned chat completion
- POST /api/generate   - returns canned generation
- GET  /api/tags       - returns list of fake models
- POST /api/embed      - returns fake embedding vector
- POST /api/embeddings - returns fake embedding vector (alt endpoint)
- GET  /                - health check (Ollama root)

Usage as context manager:
    with MockOllamaServer() as server:
        os.environ["OLLAMA_HOST"] = f"http://localhost:{server.port}"
        # ... run tests against the mock ...

Usage as pytest fixture: see tests/integration/conftest.py
"""

import json
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional


# Default canned response text
DEFAULT_RESPONSE = "This is a mock response from the test Ollama server."

# Fake model list matching what Aura Config expects
FAKE_MODELS = [
    {"name": "nemotron-3-super:cloud", "size": 0, "digest": "abc123", "modified_at": "2026-01-01T00:00:00Z"},
    {"name": "kimi-k2.5:cloud", "size": 0, "digest": "def456", "modified_at": "2026-01-01T00:00:00Z"},
    {"name": "qwen3.5:397b-cloud", "size": 0, "digest": "ghi789", "modified_at": "2026-01-01T00:00:00Z"},
    {"name": "minimax-m2.7:cloud", "size": 0, "digest": "jkl012", "modified_at": "2026-01-01T00:00:00Z"},
    {"name": "test-model:latest", "size": 1_000_000, "digest": "test000", "modified_at": "2026-01-01T00:00:00Z"},
]

# Fake 384-dim embedding (nomic-embed-text dimension)
FAKE_EMBEDDING = [0.01] * 384


def _find_free_port() -> int:
    """Find a random free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class _MockOllamaHandler(BaseHTTPRequestHandler):
    """HTTP request handler that mimics Ollama API responses."""

    # Suppress default stderr logging from BaseHTTPRequestHandler
    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def do_GET(self):
        if self.path == "/api/tags":
            self._send_json({"models": FAKE_MODELS})
        elif self.path == "/":
            # Ollama root health check — returns "Ollama is running"
            self._send_json({"status": "Ollama is running"})
        else:
            self.send_error(404)

    def do_POST(self):
        body = self._read_body()
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            payload = {}

        if self.path == "/api/chat":
            self._handle_chat(payload)
        elif self.path == "/api/generate":
            self._handle_generate(payload)
        elif self.path in ("/api/embed", "/api/embeddings"):
            self._handle_embed(payload)
        else:
            self.send_error(404)

    def _handle_chat(self, payload: dict):
        """Respond to POST /api/chat with a canned message."""
        model = payload.get("model", "test-model:latest")
        # Echo the last user message if present, otherwise default
        messages = payload.get("messages", [])
        user_msgs = [m for m in messages if m.get("role") == "user"]
        prompt_text = user_msgs[-1]["content"] if user_msgs else ""

        response_text = self.server.custom_response or DEFAULT_RESPONSE

        self._send_json({
            "model": model,
            "created_at": "2026-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "done": True,
            "total_duration": 100_000_000,
            "load_duration": 10_000_000,
            "prompt_eval_count": len(prompt_text.split()),
            "prompt_eval_duration": 50_000_000,
            "eval_count": len(response_text.split()),
            "eval_duration": 40_000_000,
        })

    def _handle_generate(self, payload: dict):
        """Respond to POST /api/generate with a canned completion."""
        model = payload.get("model", "test-model:latest")
        prompt = payload.get("prompt", "")

        # If prompt is empty (warmup/keep-alive ping), return minimal response
        if not prompt:
            self._send_json({
                "model": model,
                "response": "",
                "done": True,
            })
            return

        response_text = self.server.custom_response or DEFAULT_RESPONSE

        self._send_json({
            "model": model,
            "created_at": "2026-01-01T00:00:00Z",
            "response": response_text,
            "done": True,
            "total_duration": 100_000_000,
            "prompt_eval_count": len(prompt.split()),
            "eval_count": len(response_text.split()),
        })

    def _handle_embed(self, payload: dict):
        """Respond to POST /api/embed or /api/embeddings with fake vectors."""
        model = payload.get("model", "nomic-embed-text:latest")
        # Support both single string and list of strings
        input_data = payload.get("input") or payload.get("prompt", "")
        if isinstance(input_data, str):
            embeddings = [FAKE_EMBEDDING]
        else:
            embeddings = [FAKE_EMBEDDING for _ in input_data]

        self._send_json({
            "model": model,
            "embeddings": embeddings,
        })


class MockOllamaServer:
    """A mock Ollama server that runs in a background thread.

    Args:
        custom_response: Override the default canned response text.
        port: Specific port to use (0 or None = auto-find free port).
    """

    def __init__(self, custom_response: Optional[str] = None, port: Optional[int] = None):
        self.custom_response = custom_response
        self._port = port or _find_free_port()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def start(self):
        """Start the mock server in a background thread."""
        self._server = HTTPServer(("127.0.0.1", self._port), _MockOllamaHandler)
        self._server.custom_response = self.custom_response
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="MockOllama",
        )
        self._thread.start()

    def stop(self):
        """Shut down the mock server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def set_response(self, text: str):
        """Change the canned response for subsequent requests."""
        self.custom_response = text
        if self._server:
            self._server.custom_response = text

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

"""WebSocket test infrastructure for Aura.

Provides a reusable WebSocket test client that hooks into the FastAPI/Starlette
test framework. Uses `httpx-ws` or `websockets` (auto-detected) to connect to
the running test server and exercise WebSocket endpoints.

Usage in tests:

    from tests.ws_helpers import ws_connect

    async def test_my_ws():
        async with ws_connect("/api/ws/chat") as ws:
            resp = await ws.send_json_and_recv({"type": "ping"})
            assert resp["type"] == "pong"
"""

import json
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Try to import the WebSocket testing backend.
# Preference order: httpx-ws (Starlette TestClient compatible) -> websockets (generic)

_ws_backend = None
_ws_error = None

try:
    import httpx
    from httpx_ws import AsyncWebSocketSession  # type: ignore
    _ws_backend = "httpx_ws"
except ImportError:
    try:
        import websockets  # type: ignore
        _ws_backend = "websockets"
    except ImportError:
        _ws_error = "Neither httpx-ws nor websockets installed. Install: pip install httpx-ws"
        logger.warning(_ws_error)


class WSTestClient:
    """Simple WebSocket test client for Aura endpoints."""

    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url
        self._ws = None

    async def connect(self, path: str) -> None:
        """Connect to a WebSocket endpoint."""
        if _ws_backend == "websockets":
            self._ws = await websockets.connect(f"{self.base_url}{path}")
        elif _ws_backend == "httpx_ws":
            client = httpx.AsyncClient(base_url=self.base_url)
            url = f"{self.base_url}{path}"
            self._ws = AsyncWebSocketSession(client, url)
            await self._ws.start()
        else:
            raise RuntimeError(f"WebSocket backend not available: {_ws_error}")

    async def send_json(self, data: dict) -> None:
        """Send a JSON message."""
        if _ws_backend == "websockets":
            await self._ws.send(json.dumps(data))
        elif _ws_backend == "httpx_ws":
            await self._ws.send_json(data)

    async def recv_json(self) -> dict:
        """Receive a JSON message."""
        if _ws_backend == "websockets":
            msg = await self._ws.recv()
            return json.loads(msg)
        elif _ws_backend == "httpx_ws":
            return await self._ws.receive_json()

    async def send_json_and_recv(self, data: dict) -> dict:
        """Send JSON and receive the next JSON response."""
        await self.send_json(data)
        return await self.recv_json()

    async def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws:
            if _ws_backend == "websockets":
                await self._ws.close()
            elif _ws_backend == "httpx_ws":
                await self._ws.close()


@asynccontextmanager
async def ws_connect(path: str, base_url: str = "ws://localhost:8000"):
    """Context manager for a WebSocket test connection.

    Usage:
        async with ws_connect("/api/ws/chat") as ws:
            resp = await ws.send_json_and_recv({"message": "hello"})
            assert resp["type"] == "response"
    """
    client = WSTestClient(base_url)
    try:
        await client.connect(path)
        yield client
    finally:
        await client.close()

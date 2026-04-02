"""
ExtensionChannel -- WebSocket adapter for CLI Bridge.

Runs a local WebSocket server on port 9828 that the browser extension
connects to. When running ``aura --channels extension``, the extension
talks to the local CLI agent instead of the remote API server, giving
it full local tool access (filesystem, git, code execution, etc.).

The server runs in a dedicated background thread with its own event loop
when started from a sync context (CLI), or as a task on the existing loop
when started from an async context (API server).

Protocol (Client = browser extension, Server = this adapter):

    Client -> Server:
        {"type": "chat", "message": "hello", "id": "msg_123"}
        {"type": "ping"}
        {"type": "stop"}
        {"type": "status"}

    Server -> Client:
        {"type": "chunk", "content": "partial response...", "source": "cli"}
        {"type": "done", "response": "full response", "reply_to": "msg_123", "source": "cli"}
        {"type": "channel_message", "channel": "telegram", "username": "Elnur", "text": "..."}
        {"type": "pong"}
        {"type": "stopped"}
        {"type": "status", "channels": [...], "bridge": true}
        {"type": "error", "error": "..."}
        {"type": "typing"}

Health endpoint:
    GET http://localhost:9828/health  ->  {"status": "ok", "bridge": true}
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from http import HTTPStatus
from typing import Any, Callable, Dict, List, Optional

from .bridge import ChannelAdapter, ChannelMessage, ChannelResponse, ChannelSource

logger = logging.getLogger("aura.channels.extension")

# Default port for the local WebSocket server
DEFAULT_PORT = 9828
DEFAULT_HOST = "127.0.0.1"


class _Connection:
    """Tracks a single WebSocket connection from the extension."""

    __slots__ = ("ws", "conn_id", "connected_at", "last_ping")

    def __init__(self, ws: Any, conn_id: str) -> None:
        self.ws = ws
        self.conn_id = conn_id
        self.connected_at = time.time()
        self.last_ping = time.time()

    def __repr__(self) -> str:
        return f"<Connection {self.conn_id[:8]}>"


class ExtensionChannel(ChannelAdapter):
    """
    Browser extension channel adapter for CLI Bridge.

    Runs a local WebSocket server on ``localhost:9828`` that the browser
    extension connects to. Messages from the extension are queued for
    the CLI agent. Responses are streamed back via WebSocket.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._host = host
        self._port = port
        self._connections: Dict[str, _Connection] = {}
        self._on_message: Optional[Callable[[ChannelMessage], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Any = None  # websockets serve instance
        self._running: bool = False
        self._bridge: Any = None  # optional ChannelBridge reference

    # -- ChannelAdapter interface ------------------------------------------

    @property
    def source(self) -> ChannelSource:
        return ChannelSource.EXTENSION

    @property
    def name(self) -> str:
        return "extension"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, on_message: Callable[[ChannelMessage], None]) -> None:
        """Start the WebSocket server in a background thread.

        Called by ChannelBridge.start(). Creates a dedicated event loop
        in a daemon thread so it works from both sync CLI and async contexts.
        """
        if self._running:
            logger.warning("[ExtensionChannel] Already running — ignoring start()")
            return

        self._on_message = on_message
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="extension-channel",
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut down the WebSocket server and clean up."""
        if not self._running:
            return

        logger.info("[ExtensionChannel] Stopping...")
        self._running = False

        # Schedule async cleanup on the server's own loop
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._stop_server(),
                self._loop,
            )
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("[ExtensionChannel] Thread did not stop in time")

        self._thread = None
        self._loop = None
        self._server = None
        self._connections.clear()
        logger.info("[ExtensionChannel] Stopped")

    def send(self, response: ChannelResponse) -> None:
        """Send a response back through the extension WebSocket.

        Called by ChannelBridge.send_response(). Routes the response to
        the server's background event loop for async delivery.
        """
        if not self._loop or not self._connections:
            return

        reply_to = None
        if response.reply_to and response.reply_to.metadata:
            reply_to = response.reply_to.metadata.get("msg_id")

        chat_id = response.chat_id or ""

        asyncio.run_coroutine_threadsafe(
            self.send_response(chat_id, response.text, reply_to),
            self._loop,
        )

    # -- Optional async entry point ----------------------------------------

    async def start_async(self, bridge: Any = None) -> None:
        """Start the WebSocket server from an async context.

        Use this instead of ``start()`` when you already have an event loop
        (e.g., from the API server). For CLI Bridge mode, use ``start()``.

        Args:
            bridge: Optional ChannelBridge reference for cross-channel info.
        """
        self._bridge = bridge
        self._loop = asyncio.get_running_loop()
        await self._start_ws_server()

    async def stop_async(self) -> None:
        """Stop from an async context."""
        await self._stop_server()

    # -- Internal: background thread + event loop --------------------------

    def _run_loop(self) -> None:
        """Entry point for the background thread — creates its own event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_server_and_block())
        except Exception:
            logger.exception("[ExtensionChannel] Event loop crashed")
        finally:
            self._running = False
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()

    async def _start_ws_server(self) -> None:
        """Start the websockets server (non-blocking)."""
        if self._running:
            logger.warning("[ExtensionChannel] Already running")
            return

        try:
            from websockets.asyncio.server import serve
        except ImportError:
            logger.error(
                "[ExtensionChannel] 'websockets' package not installed. "
                "Extension bridge disabled. Run: pip install websockets"
            )
            return

        self._running = True

        try:
            self._server = await serve(
                self._handle_connection,
                self._host,
                self._port,
                process_request=self._process_http_request,
            )
        except OSError as e:
            logger.error(
                "[ExtensionChannel] Could not bind to %s:%d — %s",
                self._host, self._port, e,
            )
            self._running = False
            return

        logger.info(
            "[ExtensionChannel] WebSocket server listening on "
            "ws://%s:%d", self._host, self._port,
        )
        logger.info(
            "[ExtensionChannel] Health: http://%s:%d/health",
            self._host, self._port,
        )

    async def _start_server_and_block(self) -> None:
        """Start the server and block until _running is cleared."""
        await self._start_ws_server()
        if not self._server:
            return
        # Block until stopped
        while self._running:
            await asyncio.sleep(0.5)
        await self._stop_server()

    async def _stop_server(self) -> None:
        """Close all connections and shut down the server."""
        self._running = False

        for conn in list(self._connections.values()):
            try:
                await conn.ws.close(1001, "Server shutting down")
            except Exception:
                pass
        self._connections.clear()

        if self._server:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None

    # -- HTTP request handler (health endpoint) ----------------------------

    async def _process_http_request(self, connection, request):
        """Handle plain HTTP requests before WebSocket upgrade.

        Returns a Response for non-upgrade requests (health, status),
        or None to proceed with WebSocket upgrade.
        """
        import websockets.datastructures
        from websockets.http11 import Response

        cors = {"Access-Control-Allow-Origin": "*"}

        if request.path == "/health":
            body = json.dumps({
                "status": "ok",
                "bridge": self._bridge is not None,
                "connections": len(self._connections),
            }).encode()
            return Response(
                HTTPStatus.OK,
                "OK",
                websockets.datastructures.Headers({
                    "Content-Type": "application/json",
                    **cors,
                }),
                body,
            )

        if request.path == "/status":
            body = json.dumps({
                "channels": self._get_active_channels(),
                "bridge": self._bridge is not None,
                "connections": len(self._connections),
            }).encode()
            return Response(
                HTTPStatus.OK,
                "OK",
                websockets.datastructures.Headers({
                    "Content-Type": "application/json",
                    **cors,
                }),
                body,
            )

        # All other paths proceed to WebSocket upgrade.
        # Note: websockets v16 Request has no .method — all requests
        # reaching process_request are GET (the HTTP parser rejects others).
        return None

    # -- WebSocket connection handler --------------------------------------

    async def _handle_connection(self, websocket) -> None:
        """Handle a single WebSocket connection lifecycle."""
        conn_id = uuid.uuid4().hex
        conn = _Connection(websocket, conn_id)
        self._connections[conn_id] = conn

        logger.info(
            "[ExtensionChannel] Connection opened: %s (total: %d)",
            conn, len(self._connections),
        )

        # Send initial status on connect
        await self._send_json(conn, {
            "type": "status",
            "channels": self._get_active_channels(),
            "bridge": self._bridge is not None,
        })

        try:
            async for raw_message in websocket:
                await self._handle_message(conn, raw_message)
        except Exception as e:
            logger.debug("[ExtensionChannel] Connection %s ended: %s", conn, e)
        finally:
            self._connections.pop(conn_id, None)
            logger.info(
                "[ExtensionChannel] Connection closed: %s (remaining: %d)",
                conn, len(self._connections),
            )

    async def _handle_message(self, conn: _Connection, raw: str) -> None:
        """Parse and route a single incoming message."""
        # Reject oversized messages
        if len(raw) > 1_000_000:
            await self._send_json(conn, {"type": "error", "error": "Message too large (1MB max)"})
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_json(conn, {"type": "error", "error": "Invalid JSON"})
            return
        if not isinstance(msg, dict):
            await self._send_json(conn, {"type": "error", "error": "Expected JSON object"})
            return

        msg_type = msg.get("type", "")

        # -- Ping / keepalive --
        if msg_type == "ping":
            conn.last_ping = time.time()
            await self._send_json(conn, {"type": "pong"})
            return

        # -- Status request --
        if msg_type == "status":
            await self._send_json(conn, {
                "type": "status",
                "channels": self._get_active_channels(),
                "bridge": self._bridge is not None,
            })
            return

        # -- Stop current generation --
        if msg_type == "stop":
            # Forward a stop signal to the bridge so the agent can halt
            if self._on_message:
                stop_msg = ChannelMessage(
                    source=ChannelSource.EXTENSION,
                    text="",
                    user_id=f"ext_{conn.conn_id[:8]}",
                    user_name="Extension User",
                    chat_id=conn.conn_id,
                    metadata={"type": "stop", "conn_id": conn.conn_id},
                )
                try:
                    self._on_message(stop_msg)
                except Exception:
                    logger.exception(
                        "[ExtensionChannel] on_message callback failed (stop)"
                    )
            await self._send_json(conn, {"type": "stopped"})
            return

        # -- Chat message --
        if msg_type == "chat":
            text = msg.get("message", "").strip()
            msg_id = msg.get("id", uuid.uuid4().hex)

            if not text:
                await self._send_json(conn, {
                    "type": "error",
                    "error": "Empty message",
                    "reply_to": msg_id,
                })
                return

            # Build a ChannelMessage and forward to the bridge callback
            channel_msg = ChannelMessage(
                source=ChannelSource.EXTENSION,
                text=text,
                user_id=f"ext_{conn.conn_id[:8]}",
                user_name=msg.get("user_name", "Extension User"),
                chat_id=conn.conn_id,
                metadata={
                    "msg_id": msg_id,
                    "conn_id": conn.conn_id,
                    "url": msg.get("url", ""),
                    "page_content": msg.get("page_content", ""),
                },
            )

            if self._on_message:
                try:
                    result = self._on_message(channel_msg)
                    # Support async callbacks too
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error("[ExtensionChannel] on_message error: %s", e)
                    await self._send_json(conn, {
                        "type": "error",
                        "error": "Internal error processing message",
                        "reply_to": msg_id,
                    })
            else:
                logger.warning("[ExtensionChannel] No on_message callback registered")
                await self._send_json(conn, {
                    "type": "error",
                    "error": "Bridge not connected",
                    "reply_to": msg_id,
                })
            return

        # -- Unknown --
        await self._send_json(conn, {
            "type": "error",
            "error": f"Unknown message type: {msg_type}",
        })

    # -- Outbound messaging ------------------------------------------------

    async def send_response(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
    ) -> None:
        """Send a complete response to the extension.

        Args:
            chat_id: Connection ID to route to (broadcasts if not found).
            text: Full response text.
            reply_to: Original message ID this replies to.
        """
        payload: Dict[str, Any] = {
            "type": "done",
            "response": text,
            "source": "cli",
        }
        if reply_to:
            payload["reply_to"] = reply_to

        if chat_id and chat_id in self._connections:
            await self._send_json(self._connections[chat_id], payload)
        else:
            await self._broadcast(payload)

    async def send_chunk(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> None:
        """Send a streaming chunk to the extension.

        Args:
            chat_id: Connection ID to route to.
            content: Partial content chunk.
            reply_to: Original message ID.
        """
        payload: Dict[str, Any] = {
            "type": "chunk",
            "content": content,
            "source": "cli",
        }
        if reply_to:
            payload["reply_to"] = reply_to

        if chat_id and chat_id in self._connections:
            await self._send_json(self._connections[chat_id], payload)
        else:
            await self._broadcast(payload)

    async def send_typing(self, chat_id: str) -> None:
        """Send typing indicator to the extension."""
        payload = {"type": "typing"}
        if chat_id and chat_id in self._connections:
            await self._send_json(self._connections[chat_id], payload)
        else:
            await self._broadcast(payload)

    async def send_channel_event(
        self,
        channel: str,
        username: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Forward a message from another channel to all extension connections.

        This lets the extension display messages from Telegram, Discord, etc.

        Args:
            channel: Source channel name (e.g. "telegram").
            username: Sender display name.
            text: Message text.
            metadata: Optional extra data.
        """
        payload: Dict[str, Any] = {
            "type": "channel_message",
            "channel": channel,
            "username": username,
            "text": text,
        }
        if metadata:
            payload["metadata"] = metadata

        await self._broadcast(payload)

    # -- Internal helpers --------------------------------------------------

    async def _send_json(self, conn: _Connection, data: dict) -> None:
        """Send JSON to a single connection, removing it on failure."""
        try:
            await conn.ws.send(json.dumps(data))
        except Exception as e:
            logger.debug("[ExtensionChannel] Send failed to %s: %s", conn, e)
            self._connections.pop(conn.conn_id, None)

    async def _broadcast(self, data: dict) -> None:
        """Send JSON to all active connections."""
        dead: List[str] = []
        payload = json.dumps(data)

        for conn_id, conn in list(self._connections.items()):
            try:
                await conn.ws.send(payload)
            except Exception:
                dead.append(conn_id)

        for conn_id in dead:
            self._connections.pop(conn_id, None)

    def _get_active_channels(self) -> List[str]:
        """Return list of active channel names."""
        channels = [self.name]
        if self._bridge and hasattr(self._bridge, "active_channels"):
            channels = self._bridge.active_channels
        return channels

    @property
    def connection_count(self) -> int:
        """Number of active WebSocket connections."""
        return len(self._connections)

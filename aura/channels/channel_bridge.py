"""
ChannelBridge -- connects messaging channels to the CLI agent.

When running ``aura --channels telegram``, this bridge:

1. Runs the Telegram bot (or any adapter) in a background thread
2. Queues incoming messages for the CLI agent to process
3. Routes agent responses back to the originating channel
4. Shows channel messages in the CLI terminal via event callbacks

Architecture
~~~~~~~~~~~~
The CLI main thread stays synchronous (blocking on user input).  Each
channel adapter may be async internally, so the bridge spins up a single
background daemon thread with its own ``asyncio`` event loop and starts
every adapter inside that loop.  Messages flow through a thread-safe
``queue.Queue``.

    ChannelAdapter  (background thread)
          |
          v
    queue.Queue     <-- thread-safe handoff
          |
          v
    CLI main thread (polls via get_pending_message)
          |
          v
    agent.run(msg.text)
          |
          v
    bridge.send_response(msg, text)  --> routes back to adapter
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import queue
import threading
import time
from typing import Callable, Dict, List, Optional

from .bridge import ChannelAdapter, ChannelMessage, ChannelResponse, ChannelSource

logger = logging.getLogger("aura.channel_bridge")


class ChannelBridge:
    """
    Manages background channel adapters and routes messages to/from the CLI.

    Usage::

        bridge = ChannelBridge()
        bridge.add_channel(telegram_adapter)
        bridge.start()                       # daemon thread

        # Inside the CLI REPL:
        while True:
            msg = bridge.get_pending_message()   # non-blocking
            if msg:
                response = agent.run(msg.text)
                bridge.send_response(msg, response)
            else:
                user_input = get_input()
                ...

        bridge.stop()
    """

    def __init__(self) -> None:
        # Registered adapters keyed by ChannelSource
        self._channels: Dict[ChannelSource, ChannelAdapter] = {}

        # Thread-safe queue: adapters push, CLI main thread pops
        self._message_queue: queue.Queue[ChannelMessage] = queue.Queue()

        # Callbacks fired when a message is enqueued (on the adapter thread).
        # Used by the CLI display to print an inline notification.
        self._event_callbacks: List[Callable[[ChannelMessage], None]] = []

        # Callbacks fired when a response is sent back.
        self._response_callbacks: List[Callable[[ChannelResponse], None]] = []

        # Background asyncio event loop + thread
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Per-channel error tracking (bounded)
        self._errors: Dict[str, List[str]] = {}

    # ── Channel registration ──────────────────────────────────────────────

    def add_channel(self, adapter: ChannelAdapter) -> None:
        """Register a channel adapter.  Must be called before ``start()``."""
        key = adapter.source
        if key in self._channels:
            logger.warning("Channel %s already registered — replacing", key.value)
        self._channels[key] = adapter
        self._errors[key.value] = []
        logger.info("Channel registered: %s", key.value)

    def remove_channel(self, source: ChannelSource) -> None:
        """Remove and stop a channel adapter."""
        adapter = self._channels.pop(source, None)
        if adapter is None:
            return
        if adapter.is_running:
            try:
                adapter.stop()
            except Exception:
                logger.debug("Error stopping removed adapter %s", source.value, exc_info=True)
        self._errors.pop(source.value, None)
        logger.info("Channel removed: %s", source.value)

    def get_channels(self) -> List[ChannelAdapter]:
        """Return all registered adapters."""
        return list(self._channels.values())

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all registered channel adapters in a background daemon thread."""
        if self._running:
            logger.warning("ChannelBridge.start() called but already running")
            return
        if not self._channels:
            logger.warning("ChannelBridge.start() called with no channels registered")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_background_loop,
            name="channel-bridge",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "ChannelBridge started — channels: %s",
            ", ".join(ch.value for ch in self._channels),
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Gracefully stop all channels and the background event loop."""
        if not self._running:
            return

        self._running = False

        # Stop adapters
        for source, adapter in self._channels.items():
            try:
                result = adapter.stop()
                if inspect.isawaitable(result):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            from aura.pools import fire_and_forget
                            fire_and_forget(result)
                        else:
                            asyncio.run(result)
                    except RuntimeError:
                        asyncio.run(result)
                logger.info("Stopped channel: %s", source.value if hasattr(source, 'value') else source)
            except Exception as e:
                logger.warning("Error stopping %s: %s", source, e)

        # Shut down the background event loop
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        # Drain leftover messages
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("ChannelBridge stopped")

    # ── Incoming message queue ────────────────────────────────────────────

    def get_pending_message(self, timeout: float = 0.1) -> Optional[ChannelMessage]:
        """
        Non-blocking fetch from the message queue.

        Returns the next ``ChannelMessage`` or ``None`` if the queue is
        empty after *timeout* seconds.  Use ``timeout=0`` for a pure
        non-blocking check.
        """
        try:
            return self._message_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def has_pending(self) -> bool:
        """Check whether there are queued messages without consuming them."""
        return not self._message_queue.empty()

    def queue_message(self, message: ChannelMessage) -> None:
        """
        Called by channel adapters (via the ``on_message`` callback) to
        enqueue an incoming message.

        Also fires all registered event callbacks so the CLI can react
        immediately (e.g. print a notification line).
        """
        self._message_queue.put(message)
        logger.debug(
            "Queued message from %s/%s: %.80s",
            message.source.value,
            message.user_name,
            message.text,
        )
        for cb in self._event_callbacks:
            try:
                cb(message)
            except Exception:
                logger.debug("Event callback error", exc_info=True)

    # ── Response routing ──────────────────────────────────────────────────

    def send_response(self, original: ChannelMessage, text: str) -> None:
        """
        Send a response back to the channel that produced *original*.

        Builds a ``ChannelResponse`` and dispatches it to the correct
        adapter's ``send()`` method.  Safe to call from the CLI main thread.
        """
        adapter = self._channels.get(original.source)
        if adapter is None:
            logger.error("No adapter for channel %s — dropping response", original.source.value)
            return

        response = ChannelResponse(
            text=text,
            target_source=original.source,
            chat_id=original.chat_id,
            reply_to=original,
        )

        if adapter.is_running:
            try:
                adapter.send(response)
            except Exception:
                logger.error(
                    "Failed to send response via %s", original.source.value,
                    exc_info=True,
                )
                self._record_error(original.source.value, "send_response failed")

        # Notify response listeners
        for cb in self._response_callbacks:
            try:
                cb(response)
            except Exception:
                logger.debug("Response callback error", exc_info=True)

    def broadcast_response(
        self, text: str, exclude: Optional[ChannelSource] = None
    ) -> None:
        """Send a response to ALL active channels (except *exclude*)."""
        for source, adapter in self._channels.items():
            if source == exclude or not adapter.is_running:
                continue
            try:
                adapter.send(ChannelResponse(text=text, target_source=source))
            except Exception:
                logger.error("Broadcast error to %s", source.value, exc_info=True)

    def send_typing(self, original: ChannelMessage) -> None:
        """
        Send a typing indicator to the channel that produced *original*.

        No-op if the adapter doesn't support ``send_typing``.
        """
        adapter = self._channels.get(original.source)
        if adapter is None:
            return
        fn = getattr(adapter, "send_typing", None)
        if callable(fn):
            try:
                fn(original.chat_id)
            except Exception:
                pass  # typing indicators are best-effort

    # ── Event & response callbacks ────────────────────────────────────────

    def on_message(self, callback: Callable[[ChannelMessage], None]) -> None:
        """
        Register a callback that fires whenever a message is queued.

        The callback receives the ``ChannelMessage`` and runs on the
        adapter's background thread — keep it fast and thread-safe.
        """
        self._event_callbacks.append(callback)

    def off_message(self, callback: Callable[[ChannelMessage], None]) -> None:
        """Remove a previously registered message callback."""
        self._event_callbacks = [cb for cb in self._event_callbacks if cb is not callback]

    def on_response(self, callback: Callable[[ChannelResponse], None]) -> None:
        """Register a callback that fires when a response is sent."""
        self._response_callbacks.append(callback)

    def off_response(self, callback: Callable[[ChannelResponse], None]) -> None:
        """Remove a previously registered response callback."""
        self._response_callbacks = [cb for cb in self._response_callbacks if cb is not callback]

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def active_channels(self) -> List[str]:
        """Names of all registered channels."""
        return [s.value for s in self._channels]

    @property
    def running_channels(self) -> List[str]:
        """Names of channels whose adapters report ``is_running == True``."""
        return [s.value for s, a in self._channels.items() if a.is_running]

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        """Return a status dict suitable for ``/status`` display."""
        return {
            "running": self._running,
            "channels": {
                source.value: {
                    "running": adapter.is_running,
                    "errors": self._errors.get(source.value, [])[-5:],
                }
                for source, adapter in self._channels.items()
            },
            "queue_size": self._message_queue.qsize(),
            "callback_count": len(self._event_callbacks),
        }

    def status(self) -> List[dict]:
        """Return a list of per-channel status dicts for CLI display.

        Each dict has keys: ``channel``, ``running``, ``pending``.
        This is the format consumed by ``chat_session.py``'s /channels command.
        """
        q_size = self._message_queue.qsize()
        return [
            {
                "channel": source.value,
                "running": adapter.is_running,
                "pending": q_size,
            }
            for source, adapter in self._channels.items()
        ]

    def set_on_message_callback(self, callback: Callable[[ChannelMessage], None]) -> None:
        """Register a single on-message callback (alias for ``on_message()``).

        Replaces any previously registered callbacks set via this method.
        Used by ``chat_session.py`` to wire the CLI notification display.
        """
        # Remove the previous callback stored under this method if any
        if hasattr(self, '_session_callback') and self._session_callback in self._event_callbacks:
            self.off_message(self._session_callback)
        self._session_callback = callback
        self.on_message(callback)

    # ── Internal: background event loop ───────────────────────────────────

    def _run_background_loop(self) -> None:
        """
        Entry point for the background daemon thread.

        Creates a new asyncio event loop and runs forever until
        ``stop()`` is called.  Adapters that need async (e.g. Telegram's
        polling loop) can schedule coroutines on ``self._loop``.
        """
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            # Start all adapters (synchronous start, they may spin up
            # their own tasks on this loop if needed)
            for source, adapter in self._channels.items():
                try:
                    adapter.start(on_message=self.queue_message)
                    logger.info("Adapter started: %s", source.value)
                except Exception as exc:
                    logger.error("Failed to start %s: %s", source.value, exc)
                    self._record_error(source.value, f"start failed: {exc}")

            # Keep the loop alive for async adapters
            self._loop.run_forever()
        except Exception:
            logger.error("Background loop crashed", exc_info=True)
        finally:
            # Clean up pending async tasks
            pending = asyncio.all_tasks(self._loop) if hasattr(asyncio, "all_tasks") else []
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()
            self._loop = None
            self._running = False
            logger.debug("Background loop thread exiting")

    # ── Error tracking ────────────────────────────────────────────────────

    def _record_error(self, channel: str, message: str) -> None:
        """Keep a bounded list of recent errors per channel."""
        errors = self._errors.setdefault(channel, [])
        errors.append(f"{time.strftime('%H:%M:%S')} {message}")
        if len(errors) > 20:
            self._errors[channel] = errors[-20:]

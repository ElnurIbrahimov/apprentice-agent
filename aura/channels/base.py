"""Base channel adapter interface for AURA multi-channel support.

Inspired by ClawdBot's typed channel plugin system. Each channel
(Telegram, Discord, Signal, LINE, etc.) implements ChannelAdapter.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Capability flags ──────────────────────────────────────────────────────────

class ChannelCapability(Enum):
    TEXT = auto()           # Plain text messages
    MARKDOWN = auto()       # Markdown formatting
    IMAGES = auto()         # Sending/receiving images
    VOICE = auto()          # Voice messages / calls
    REACTIONS = auto()      # Emoji reactions (ack while processing)
    THREADS = auto()        # Threaded replies
    BUTTONS = auto()        # Interactive buttons / inline keyboard
    FILE_UPLOAD = auto()    # File attachments


# ── Message models ────────────────────────────────────────────────────────────

@dataclass
class InboundMessage:
    """Normalised incoming message from any channel."""
    channel_id: str           # Unique channel identifier (e.g. "discord", "telegram")
    chat_id: str              # Conversation / channel ID within the platform
    user_id: str              # Sender identifier
    user_name: str            # Display name of the sender
    text: str                 # Message text content
    message_id: Optional[str] = None
    reply_to_id: Optional[str] = None
    attachments: List[str] = field(default_factory=list)   # URLs / file paths
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutboundMessage:
    """Normalised outgoing message to send on a channel."""
    text: str
    chat_id: str
    reply_to_id: Optional[str] = None
    parse_mode: str = "markdown"   # "markdown" | "html" | "plain"
    attachments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Abstract base ─────────────────────────────────────────────────────────────

class ChannelAdapter(ABC):
    """Abstract base for all channel adapters.

    Subclass this to add a new messaging channel to AURA.

    Lifecycle
    ---------
    1. ``start()``  — connect / authenticate / start polling
    2. ``on_message`` callback fires for each inbound message
    3. ``send()``   — called by AURA agent to deliver responses
    4. ``stop()``   — graceful shutdown
    """

    channel_id: str = "unknown"
    display_name: str = "Unknown Channel"
    capabilities: List[ChannelCapability] = []

    def __init__(self):
        self._on_message_cb: Optional[Callable[[InboundMessage], Coroutine]] = None
        self._running = False

    # ── Must implement ────────────────────────────────────────────────────────

    @abstractmethod
    async def start(self) -> None:
        """Connect to the channel and begin receiving messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Disconnect and clean up resources."""

    @abstractmethod
    async def send(self, message: OutboundMessage) -> bool:
        """Send a message. Returns True on success."""

    # ── Optional ──────────────────────────────────────────────────────────────

    async def ack(self, chat_id: str, message_id: str, emoji: str = "⏳") -> None:
        """Send a reaction/ack while processing (if REACTIONS capability exists)."""

    async def remove_ack(self, chat_id: str, message_id: str, emoji: str = "⏳") -> None:
        """Remove processing reaction after response is sent."""

    async def health_check(self) -> Dict[str, Any]:
        """Return health/status dict for diagnostics."""
        return {"channel": self.channel_id, "running": self._running}

    # ── Callback registration ─────────────────────────────────────────────────

    def on_message(self, callback: Callable[[InboundMessage], Coroutine]) -> None:
        """Register the callback that fires when a message arrives."""
        self._on_message_cb = callback

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Internal: dispatch inbound message to registered callback."""
        if self._on_message_cb:
            try:
                await self._on_message_cb(msg)
            except Exception as e:
                logger.error(f"[{self.channel_id}] Message handler error: {e}")

    def has_capability(self, cap: ChannelCapability) -> bool:
        return cap in self.capabilities

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} channel={self.channel_id} running={self._running}>"

"""
Channel Bridge — Base types for multi-channel message routing.

Defines the data classes and adapter interface that all channel
implementations (Telegram, Discord, Extension, etc.) must use.
The orchestrator lives in ``channel_bridge.py``.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ChannelSource(Enum):
    """Identifies which channel a message came from."""
    CLI = "cli"
    TELEGRAM = "telegram"
    EXTENSION = "extension"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    SLACK = "slack"
    MATRIX = "matrix"
    API = "api"


@dataclass
class ChannelMessage:
    """A message arriving from any channel."""
    source: ChannelSource
    text: str
    user_id: str = ""
    user_name: str = ""
    chat_id: str = ""
    timestamp: float = field(default_factory=time.time)
    attachments: List[Any] = field(default_factory=list)
    message_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelResponse:
    """A response to be sent back through a channel."""
    text: str
    target_source: ChannelSource
    chat_id: str = ""
    reply_to: Optional[ChannelMessage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

class ChannelAdapter(ABC):
    """
    Abstract base for channel adapters.

    Each adapter runs a listener for its platform and forwards incoming
    messages to the bridge via the ``on_message`` callback.  The bridge
    calls ``send()`` to push responses back through the adapter.
    """

    @property
    @abstractmethod
    def source(self) -> ChannelSource:
        """The channel source this adapter handles."""
        ...

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Whether the adapter is currently listening."""
        ...

    @abstractmethod
    def start(self, on_message: Callable[[ChannelMessage], None]) -> None:
        """
        Start listening for messages.

        Args:
            on_message: Callback the adapter calls with each incoming
                        ``ChannelMessage``.  The bridge supplies this.
        """
        ...

    @abstractmethod
    def send(self, response: ChannelResponse) -> None:
        """Send a response back through this channel."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the listener and clean up resources."""
        ...

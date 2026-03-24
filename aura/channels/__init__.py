"""
aura.channels — Multi-channel message routing for the CLI agent.

Provides the ChannelBridge orchestrator that connects remote messaging
channels (Telegram, Discord, browser extension, etc.) to the local CLI
REPL, plus the base adapter interface and data classes.
"""

from .bridge import (
    ChannelAdapter,
    ChannelMessage,
    ChannelResponse,
    ChannelSource,
)
from .channel_bridge import ChannelBridge
from .extension_channel import ExtensionChannel

__all__ = [
    "ChannelAdapter",
    "ChannelBridge",
    "ChannelMessage",
    "ChannelResponse",
    "ChannelSource",
    "ExtensionChannel",
]

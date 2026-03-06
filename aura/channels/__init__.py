"""AURA multi-channel abstraction layer.

Adapters
--------
- DiscordAdapter  — discord.py bot (TEXT, MARKDOWN, IMAGES, REACTIONS, THREADS)
- SignalAdapter   — signal-cli REST API (TEXT, IMAGES, VOICE, FILES)
- LINEAdapter     — LINE Messaging API webhook (TEXT, IMAGES, BUTTONS)
"""
from .channel_manager import get_channel_manager, ChannelManager
from .base import ChannelAdapter, InboundMessage, OutboundMessage, ChannelCapability

__all__ = [
    "get_channel_manager",
    "ChannelManager",
    "ChannelAdapter",
    "InboundMessage",
    "OutboundMessage",
    "ChannelCapability",
]

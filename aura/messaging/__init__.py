"""
AURA Messaging Integration

Connect AURA to WhatsApp, Telegram, and other platforms.
"""

from .base_platform import BasePlatform, IncomingMessage, OutgoingMessage, MessageType
from .router import MessageRouter

__all__ = [
    "BasePlatform",
    "IncomingMessage",
    "OutgoingMessage",
    "MessageType",
    "MessageRouter",
]

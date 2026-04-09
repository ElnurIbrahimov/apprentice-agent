"""
AURA Messaging Integration

Connect AURA to WhatsApp, Telegram, and other platforms.
"""

from .base_platform import BasePlatform, IncomingMessage, MessageType, OutgoingMessage
from .router import MessageRouter

__all__ = [
    "BasePlatform",
    "IncomingMessage",
    "MessageRouter",
    "MessageType",
    "OutgoingMessage",
]

"""
Base class for all messaging platforms.
Defines the interface that WhatsApp, Telegram, etc. must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

from .sanitizer import sanitize_outgoing

logger = logging.getLogger(__name__)


class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    DOCUMENT = "document"
    STICKER = "sticker"
    LOCATION = "location"


@dataclass
class IncomingMessage:
    """Standardized incoming message format"""
    platform: str                    # "telegram", "whatsapp"
    user_id: str                     # Platform-specific user ID
    chat_id: str                     # Platform-specific chat ID
    username: Optional[str]          # Username if available
    display_name: Optional[str]      # Display name if available
    message_type: MessageType        # Type of message
    text: Optional[str]              # Text content (if text message)
    media_url: Optional[str]         # Media URL (if media message)
    timestamp: datetime              # When message was sent
    raw_message: Any                 # Original platform message object


@dataclass
class OutgoingMessage:
    """Standardized outgoing message format"""
    chat_id: str
    text: str
    reply_to_message_id: Optional[str] = None
    parse_mode: Optional[str] = None  # "markdown", "html", etc.


class BasePlatform(ABC):
    """Abstract base class for messaging platforms"""

    def __init__(self, aura_engine, config: dict):
        self.aura = aura_engine
        self.config = config
        self.is_running = False
        self._on_message_callback: Optional[Callable] = None

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return platform name (e.g., 'telegram', 'whatsapp')"""
        pass

    @abstractmethod
    async def start(self):
        """Start the platform bot/connection"""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the platform bot/connection"""
        pass

    @abstractmethod
    async def send_message(self, message: OutgoingMessage) -> bool:
        """Send a message to a user"""
        pass

    @abstractmethod
    async def send_typing_indicator(self, chat_id: str):
        """Show typing indicator"""
        pass

    async def handle_incoming(self, message: IncomingMessage) -> Optional[str]:
        """
        Handle an incoming message through AURA.
        Returns the response text.
        """

        if message.message_type not in (MessageType.TEXT, MessageType.VOICE):
            return "I can only process text messages right now."

        if not message.text:
            return None

        # Process through AURA
        try:
            # Update user context in AURA
            self._update_user_context(message)

            # Get response from AURA
            response = await self._process_with_aura(message.text, message.user_id)

            return response

        except Exception as e:
            logger.error(f"[{self.platform_name}] Error processing message: {e}")
            return "Oops, something went wrong. Give me a sec..."

    def _update_user_context(self, message: IncomingMessage):
        """Update AURA's context about this user"""
        # Store platform-specific user info — currently informational only.
        # The agent's memory system handles profile extraction
        # automatically within its chat() pipeline.
        pass

    async def _process_with_aura(self, text: str, user_id: str) -> str:
        """Process message through AURA agent.

        Uses ApprenticeAgent.chat() which handles fast-path, memory,
        emotion, LLM generation, and humanization internally.
        """
        try:
            if hasattr(self.aura, 'chat'):
                response = self.aura.chat(text)
                if response:
                    return response
        except Exception as e:
            logger.error(f"Agent chat error: {e}")

        return "Hey! I got your message. What's up?"

    async def send_proactive(self, chat_id: str, message: str):
        """Send a proactive message (for follow-ups, greetings, etc.)

        All proactive messages are sanitized before sending to prevent
        prompt injection exfiltration from LLM-generated content.
        """
        clean_message, flagged = sanitize_outgoing(message, source="proactive_awareness")
        if flagged:
            logger.warning(
                f"[{self.platform_name}] Proactive message to {chat_id} was flagged by sanitizer"
            )
        outgoing = OutgoingMessage(
            chat_id=chat_id,
            text=clean_message
        )
        return await self.send_message(outgoing)

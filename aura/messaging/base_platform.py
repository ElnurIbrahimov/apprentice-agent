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

        if message.message_type != MessageType.TEXT:
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
        # Store platform-specific user info
        user_info = {
            "platform": self.platform_name,
            "user_id": message.user_id,
            "username": message.username,
            "display_name": message.display_name,
            "last_seen": message.timestamp.isoformat()
        }

        # Could update AURA's user profile here
        if hasattr(self.aura, 'memory') and hasattr(self.aura.memory, 'extract_and_store_profile'):
            # Extract profile from message
            self.aura.memory.extract_and_store_profile(message.text or "")

    async def _process_with_aura(self, text: str, user_id: str) -> str:
        """Process message through AURA engine"""

        # Try fast path first (for instant responses)
        if hasattr(self.aura, 'fast_path'):
            from aura.fast_path import FastPathHandler
            fast_handler = FastPathHandler(self.aura)
            fast_response = fast_handler.try_fast_path(text)
            if fast_response:
                return fast_response

        # Fall back to full AURA processing
        if hasattr(self.aura, 'process_input'):
            # Use the engine's process_input/process_response flow
            context = self.aura.process_input(text)

            # For messaging, we need to generate a response
            # This would normally come from the LLM, but for fast responses:
            mock_response = "I hear you! Let me think about that..."

            response = self.aura.process_response(mock_response, context)
            return response.content

        # Direct fallback
        return "Hey! I got your message. What's up?"

    async def send_proactive(self, chat_id: str, message: str):
        """Send a proactive message (for follow-ups, greetings, etc.)"""
        outgoing = OutgoingMessage(
            chat_id=chat_id,
            text=message
        )
        return await self.send_message(outgoing)

"""
DiscordChannel — Lightweight Discord adapter for CLI Bridge.

Runs discord.py in a background thread, forwards messages to the CLI agent
via ChannelBridge, and sends responses back to Discord.

Like TelegramChannel, this is a thin relay — the CLI's agent handles
all processing. No commands, no slash commands, just message forwarding.

Requires: pip install discord.py>=2.3.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from typing import Callable, Dict, List, Optional

from .bridge import (
    ChannelAdapter,
    ChannelMessage,
    ChannelResponse,
    ChannelSource,
)

logger = logging.getLogger(__name__)

_DISCORD_MSG_LIMIT = 2000


def _split_message(text: str, limit: int = _DISCORD_MSG_LIMIT) -> List[str]:
    """Split a long message into chunks that fit Discord's limit."""
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at newline
        split_at = text.rfind('\n', 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip('\n')
    return chunks


class DiscordChannel(ChannelAdapter):
    """Discord adapter — relays messages between Discord and the CLI agent."""

    def __init__(
        self,
        token: Optional[str] = None,
        allowed_users: Optional[set] = None,
        allowed_channels: Optional[set] = None,
    ):
        self._token = token or os.environ.get("DISCORD_BOT_TOKEN", "")
        self._allowed_users = allowed_users or set()
        self._allowed_channels = allowed_channels or set()
        # Load allowed users from env if not provided
        env_users = os.environ.get("DISCORD_ALLOWED_USERS", "")
        if env_users and not self._allowed_users:
            self._allowed_users = {u.strip() for u in env_users.split(",") if u.strip()}

        self._running = False
        self._client = None
        self._on_message: Optional[Callable[[ChannelMessage], None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        # Map channel_id -> last message for response routing
        self._pending_responses: Dict[str, asyncio.Future] = {}

    @property
    def source(self) -> ChannelSource:
        return ChannelSource.DISCORD

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, on_message: Callable[[ChannelMessage], None]) -> None:
        """Start the Discord bot in a background thread."""
        if not self._token:
            logger.error("[Discord] No DISCORD_BOT_TOKEN set — cannot start")
            return
        if self._running:
            return

        self._on_message = on_message
        self._running = True

        self._thread = threading.Thread(
            target=self._run_bot,
            daemon=True,
            name="discord-channel",
        )
        self._thread.start()
        logger.info("[Discord] Channel adapter starting...")

    def _run_bot(self):
        """Run the Discord client in its own event loop."""
        try:
            import discord

            intents = discord.Intents.default()
            intents.message_content = True
            client = discord.Client(intents=intents)
            self._client = client

            @client.event
            async def on_ready():
                logger.info(f"[Discord] Connected as {client.user} ({client.user.id})")
                self._loop = asyncio.get_running_loop()

            @client.event
            async def on_message(message):
                # Ignore own messages
                if message.author == client.user:
                    return

                # Filter by allowed users (if set)
                if self._allowed_users:
                    user_id = str(message.author.id)
                    user_name = str(message.author)
                    if user_id not in self._allowed_users and user_name not in self._allowed_users:
                        return

                # Filter by allowed channels (if set)
                if self._allowed_channels and str(message.channel.id) not in self._allowed_channels:
                    return

                # Check for mention or DM
                is_dm = isinstance(message.channel, discord.DMChannel)
                is_mentioned = client.user in message.mentions if client.user else False
                has_prefix = message.content.strip().lower().startswith(("aura ", "aura,"))

                # In guilds, only respond to mentions or prefix
                if not is_dm and not is_mentioned and not has_prefix:
                    return

                # Clean the message text
                text = message.content
                if is_mentioned and client.user:
                    text = re.sub(rf'<@!?{client.user.id}>\s*', '', text).strip()
                if has_prefix:
                    text = text[5:].strip()

                if not text:
                    return

                # Create ChannelMessage
                ch_msg = ChannelMessage(
                    source=ChannelSource.DISCORD,
                    text=text,
                    user_id=str(message.author.id),
                    user_name=str(message.author),
                    chat_id=str(message.channel.id),
                    message_id=str(message.id),
                    metadata={
                        "guild": str(message.guild.id) if message.guild else "DM",
                        "guild_name": message.guild.name if message.guild else "DM",
                        "channel_name": getattr(message.channel, 'name', 'DM'),
                        "is_dm": is_dm,
                    },
                )

                # Show typing indicator
                async with message.channel.typing():
                    if self._on_message:
                        self._on_message(ch_msg)

            client.run(self._token, log_handler=None)

        except ImportError:
            logger.error("[Discord] discord.py not installed. Run: pip install discord.py>=2.3.0")
            self._running = False
        except Exception as e:
            logger.error(f"[Discord] Bot crashed: {e}")
            self._running = False

    def send(self, response: ChannelResponse) -> None:
        """Send a response back through Discord."""
        if not self._client or not self._loop or self._loop.is_closed():
            logger.warning("[Discord] Cannot send — bot not running")
            return

        chat_id = response.chat_id
        text = response.text or ""

        async def _send():
            try:
                channel = self._client.get_channel(int(chat_id))
                if not channel:
                    channel = await self._client.fetch_channel(int(chat_id))
                if channel:
                    for chunk in _split_message(text):
                        await channel.send(chunk)
            except Exception as e:
                logger.error(f"[Discord] Send failed to {chat_id}: {e}")

        asyncio.run_coroutine_threadsafe(_send(), self._loop)

    def stop(self) -> None:
        """Stop the Discord bot."""
        self._running = False
        if self._client and self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._client.close(), self._loop)
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("[Discord] Channel adapter stopped")

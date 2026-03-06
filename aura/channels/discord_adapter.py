"""Discord channel adapter for AURA.

Requires: pip install discord.py
Set DISCORD_BOT_TOKEN in environment or aura/.env

Features
--------
- Slash commands: /ask, /status, /dream
- Ack reaction (⏳) while processing, removed on reply
- Markdown passthrough (Discord renders it natively)
- Multi-server / multi-channel support

Usage
-----
    from aura.channels.discord_adapter import DiscordAdapter
    from aura.channels.channel_manager import get_channel_manager

    adapter = DiscordAdapter(token=os.getenv("DISCORD_BOT_TOKEN"))
    get_channel_manager().register(adapter)
    await get_channel_manager().start_all()
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .base import (
    ChannelAdapter,
    ChannelCapability,
    InboundMessage,
    OutboundMessage,
)

logger = logging.getLogger(__name__)

# Discord.py is optional — adapter is importable without it; errors at start()
try:
    import discord
    from discord.ext import commands as discord_commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    discord = None


class DiscordAdapter(ChannelAdapter):
    """AURA channel adapter for Discord.

    Parameters
    ----------
    token:
        Discord bot token. Falls back to DISCORD_BOT_TOKEN env var.
    command_prefix:
        Classic prefix for text commands (default: "!").
    allowed_guilds:
        If non-empty, only respond in these guild IDs.
    """

    channel_id = "discord"
    display_name = "Discord"
    capabilities = [
        ChannelCapability.TEXT,
        ChannelCapability.MARKDOWN,
        ChannelCapability.IMAGES,
        ChannelCapability.REACTIONS,
        ChannelCapability.THREADS,
        ChannelCapability.BUTTONS,
        ChannelCapability.FILE_UPLOAD,
    ]

    def __init__(
        self,
        token: Optional[str] = None,
        command_prefix: str = "!",
        allowed_guilds: Optional[List[int]] = None,
    ):
        super().__init__()
        self._token = token or os.getenv("DISCORD_BOT_TOKEN", "")
        self._prefix = command_prefix
        self._allowed_guilds = set(allowed_guilds or [])
        self._bot: Optional[Any] = None  # discord.ext.commands.Bot
        self._ack_reactions: Dict[str, str] = {}  # message_id → channel_id

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not DISCORD_AVAILABLE:
            raise ImportError(
                "discord.py not installed. Run: pip install discord.py"
            )
        if not self._token:
            raise ValueError(
                "DISCORD_BOT_TOKEN not set. Export it or pass token= to DiscordAdapter()."
            )

        intents = discord.Intents.default()
        intents.message_content = True

        self._bot = discord_commands.Bot(
            command_prefix=self._prefix,
            intents=intents,
            description="AURA AI assistant",
        )
        self._register_events()
        self._register_slash_commands()

        self._running = True
        logger.info("[Discord] Starting bot...")
        await self._bot.start(self._token)

    async def stop(self) -> None:
        self._running = False
        if self._bot:
            await self._bot.close()
            logger.info("[Discord] Bot stopped.")

    # ── Send ──────────────────────────────────────────────────────────────────

    async def send(self, message: OutboundMessage) -> bool:
        if not self._bot:
            return False
        try:
            channel = self._bot.get_channel(int(message.chat_id))
            if channel is None:
                logger.warning(f"[Discord] Channel {message.chat_id} not found")
                return False

            # Chunk long messages (Discord 2000-char limit)
            text = message.text
            chunks = [text[i:i+1900] for i in range(0, len(text), 1900)]

            for chunk in chunks:
                if message.reply_to_id:
                    try:
                        ref_msg = await channel.fetch_message(int(message.reply_to_id))
                        await ref_msg.reply(chunk)
                        message.reply_to_id = None  # only first chunk as reply
                    except Exception:
                        await channel.send(chunk)
                else:
                    await channel.send(chunk)

            return True
        except Exception as e:
            logger.error(f"[Discord] Send error: {e}")
            return False

    # ── Reactions (ack while processing) ─────────────────────────────────────

    async def ack(self, chat_id: str, message_id: str, emoji: str = "⏳") -> None:
        if not self._bot:
            return
        try:
            channel = self._bot.get_channel(int(chat_id))
            if channel:
                msg = await channel.fetch_message(int(message_id))
                await msg.add_reaction(emoji)
                self._ack_reactions[message_id] = emoji
        except Exception:
            pass

    async def remove_ack(self, chat_id: str, message_id: str, emoji: str = "⏳") -> None:
        if not self._bot:
            return
        try:
            channel = self._bot.get_channel(int(chat_id))
            if channel:
                msg = await channel.fetch_message(int(message_id))
                me = self._bot.user
                await msg.remove_reaction(emoji, me)
                self._ack_reactions.pop(message_id, None)
        except Exception:
            pass

    # ── Health ────────────────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        ok = self._bot is not None and not self._bot.is_closed()
        return {
            "channel": self.channel_id,
            "running": self._running,
            "bot_ready": ok,
            "guilds": len(self._bot.guilds) if ok else 0,
            "latency_ms": round(self._bot.latency * 1000, 1) if ok else None,
        }

    # ── Internal event wiring ─────────────────────────────────────────────────

    def _is_guild_allowed(self, guild_id: Optional[int]) -> bool:
        if not self._allowed_guilds:
            return True
        return guild_id in self._allowed_guilds

    def _register_events(self) -> None:
        bot = self._bot

        @bot.event
        async def on_ready():
            logger.info(
                f"[Discord] Logged in as {bot.user} ({bot.user.id}) — "
                f"{len(bot.guilds)} guild(s)"
            )
            self._running = True

        @bot.event
        async def on_message(message: discord.Message):
            if message.author.bot:
                return
            guild_id = message.guild.id if message.guild else None
            if not self._is_guild_allowed(guild_id):
                return

            # Only respond to direct mentions or DMs
            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mention = bot.user in message.mentions
            if not (is_dm or is_mention):
                # Still process text commands (! prefix)
                await bot.process_commands(message)
                return

            text = message.clean_content.strip()
            # Strip mention prefix
            for mention in [f"<@{bot.user.id}>", f"<@!{bot.user.id}>"]:
                text = text.replace(mention, "").strip()

            if not text:
                return

            inbound = InboundMessage(
                channel_id=self.channel_id,
                chat_id=str(message.channel.id),
                user_id=str(message.author.id),
                user_name=message.author.display_name,
                text=text,
                message_id=str(message.id),
                reply_to_id=str(message.reference.message_id) if message.reference else None,
                attachments=[a.url for a in message.attachments],
                metadata={"guild_id": guild_id, "channel_name": getattr(message.channel, "name", "dm")},
            )

            # Ack reaction while processing
            await self.ack(str(message.channel.id), str(message.id))

            await self._dispatch(inbound)

            # Ack removal is handled by channel_manager after response is sent

            await bot.process_commands(message)

    def _register_slash_commands(self) -> None:
        bot = self._bot

        @bot.command(name="aura_status")
        async def cmd_status(ctx: discord_commands.Context):
            """Show AURA system status."""
            health = await self.health_check()
            await ctx.send(
                f"**AURA Discord Adapter**\n"
                f"Bot: {'✅' if health['bot_ready'] else '❌'}\n"
                f"Guilds: {health['guilds']}\n"
                f"Latency: {health['latency_ms']}ms"
            )

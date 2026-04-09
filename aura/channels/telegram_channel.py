"""
TelegramChannel — Lightweight Telegram adapter for CLI Bridge.

Runs python-telegram-bot in a background thread, forwards messages to the
CLI agent via ChannelBridge, and sends responses back to Telegram.

Unlike the full TelegramBot class (which has its own agent, command handlers,
proactive messaging, etc.), this adapter is intentionally minimal: it only
listens for messages from allowed users and shuttles them through the bridge.
The CLI's agent handles all processing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import threading
import time
from typing import Callable, List, Optional

from .bridge import (
    ChannelAdapter,
    ChannelMessage,
    ChannelResponse,
    ChannelSource,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Telegram MarkdownV2 formatting utilities (shared module)
# ---------------------------------------------------------------------------

from aura.messaging.telegram_formatting import (
    _TELEGRAM_MSG_LIMIT,
)
from aura.messaging.telegram_formatting import (
    format_telegram_response as _format_telegram_response,
)
from aura.messaging.telegram_formatting import (
    split_message as _split_message,
)

# ---------------------------------------------------------------------------
#  TelegramChannel adapter
# ---------------------------------------------------------------------------

class TelegramChannel(ChannelAdapter):
    """
    Lightweight Telegram adapter for CLI bridge mode.

    Unlike the full TelegramBot (which runs its own agent), this is a thin
    adapter that just relays messages to/from the CLI agent via ChannelBridge.

    Usage::

        channel = TelegramChannel(token="BOT_TOKEN", allowed_users=[12345])
        bridge.add_channel(channel)
        bridge.start()
    """

    def __init__(
        self,
        token: Optional[str] = None,
        allowed_users: Optional[List[int]] = None,
    ):
        self._token: str = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._allowed_users: List[int] = allowed_users or self._load_allowed_users()
        self._on_message: Optional[Callable[[ChannelMessage], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._app = None  # telegram.ext.Application (lazy import)
        self._running: bool = False
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # ChannelAdapter interface
    # ------------------------------------------------------------------

    @property
    def source(self) -> ChannelSource:
        return ChannelSource.TELEGRAM

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, on_message: Callable[[ChannelMessage], None]) -> None:
        """Start the Telegram listener in a background thread."""
        if self._running:
            logger.warning("[TelegramChannel] Already running -- ignoring start()")
            return

        if not self._token or self._token == "YOUR_BOT_TOKEN_HERE":
            raise ValueError(
                "TELEGRAM_BOT_TOKEN not set. "
                "Get one from @BotFather on Telegram and set the env var."
            )

        if not self._allowed_users:
            logger.warning(
                "[TelegramChannel] No allowed users configured. "
                "Set TELEGRAM_ALLOWED_USERS env var (comma-separated IDs). "
                "Bot will reject ALL messages until at least one user is allowed."
            )

        self._on_message = on_message
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="telegram-channel",
        )
        self._thread.start()

    def send(self, response: ChannelResponse) -> None:
        """Send a response back to Telegram (thread-safe, called from CLI thread)."""
        if not self._loop or not self._app:
            logger.warning("[TelegramChannel] Cannot send -- bot not started")
            return

        chat_id = response.chat_id
        if not chat_id and response.reply_to:
            chat_id = response.reply_to.chat_id
        if not chat_id:
            logger.warning("[TelegramChannel] No chat_id for response -- dropping")
            return

        reply_to_id = response.metadata.get("reply_to_message_id") if response.metadata else None

        asyncio.run_coroutine_threadsafe(
            self._send_response(chat_id, response.text, reply_to=reply_to_id),
            self._loop,
        )

    def stop(self) -> None:
        """Stop the Telegram listener and clean up."""
        if not self._running:
            return

        logger.info("[TelegramChannel] Stopping...")
        self._running = False

        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("[TelegramChannel] Thread did not stop in time")

        self._thread = None
        self._loop = None
        self._app = None
        logger.info("[TelegramChannel] Stopped")

    # ------------------------------------------------------------------
    # Internal: event loop + bot lifecycle
    # ------------------------------------------------------------------

    def _load_allowed_users(self) -> List[int]:
        """Load allowed user IDs from TELEGRAM_ALLOWED_USERS env var."""
        raw = os.getenv("TELEGRAM_ALLOWED_USERS", "")
        if not raw:
            return []
        ids: List[int] = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
        return ids

    def _run_loop(self) -> None:
        """Entry point for the background thread -- creates its own event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_bot())
        except Exception:
            logger.exception("[TelegramChannel] Event loop crashed")
        finally:
            self._running = False
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass
            self._loop.close()

    async def _start_bot(self) -> None:
        """Build and run the Telegram Application."""
        try:
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            logger.error(
                "[TelegramChannel] python-telegram-bot not installed. "
                "Run: pip install python-telegram-bot>=20.0"
            )
            return

        self._app = Application.builder().token(self._token).build()

        # --- Command handlers ---
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(CommandHandler("help", self._handle_help))
        self._app.add_handler(CommandHandler("status", self._handle_status))
        self._app.add_handler(CommandHandler("bridge", self._handle_bridge))

        # --- Media handlers ---
        self._app.add_handler(MessageHandler(
            filters.VOICE | filters.AUDIO,
            self._handle_voice,
        ))
        self._app.add_handler(MessageHandler(
            filters.PHOTO,
            self._handle_photo,
        ))
        self._app.add_handler(MessageHandler(
            filters.Document.ALL,
            self._handle_document,
        ))

        # --- Text handler (must be last) ---
        self._app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message,
        ))

        # --- Error handler ---
        self._app.add_error_handler(self._handle_error)

        self._running = True
        logger.info(
            "[TelegramChannel] Starting listener (allowed users: %s)",
            self._allowed_users or "NONE -- all messages will be rejected",
        )

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        # Log bot identity
        try:
            me = await self._app.bot.get_me()
            logger.info("[TelegramChannel] Bot online: @%s", me.username)
        except Exception:
            logger.info("[TelegramChannel] Bot online (could not fetch username)")

        # Block until stopped
        try:
            while self._running:
                await asyncio.sleep(0.5)
        finally:
            logger.info("[TelegramChannel] Shutting down bot...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def _get_allowed_users(self) -> set:
        """Build the set of allowed user IDs.

        Reads TELEGRAM_ALLOWED_USERS from the environment on every call so
        that hot-reloads work without restarting the bot (same pattern as
        the full TelegramBot._is_user_allowed).
        """
        ids: set = set(self._allowed_users)

        raw = os.environ.get("TELEGRAM_ALLOWED_USERS", "")
        if raw:
            for part in raw.split(","):
                part = part.strip()
                if part.isdigit():
                    ids.add(int(part))

        return ids

    _rejection_log_count: int = 0
    _rejection_log_last: float = 0.0
    _rejection_log_reset_at: float = 0.0

    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to interact with the bot."""
        import time as _t
        allowed = self._get_allowed_users()
        now = _t.time()
        # Daily reset so the counter never silently suppresses logs forever
        if now - self._rejection_log_reset_at >= 86400:
            self._rejection_log_count = 0
            self._rejection_log_reset_at = now
        if not allowed:
            # Rate-limit rejection logs: first 3, then once per 60s
            self._rejection_log_count += 1
            if self._rejection_log_count <= 3 or (now - self._rejection_log_last) > 60:
                logger.warning(
                    "[TelegramChannel] Rejected user %s -- no allowed users configured (count=%d)",
                    user_id, self._rejection_log_count,
                )
                self._rejection_log_last = now
            return False
        if user_id not in allowed:
            self._rejection_log_count += 1
            if self._rejection_log_count <= 3 or (now - self._rejection_log_last) > 60:
                logger.warning(
                    "[TelegramChannel] Rejected user %s -- not in allowed list (count=%d)",
                    user_id, self._rejection_log_count,
                )
                self._rejection_log_last = now
            return False
        return True

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _handle_start(self, update, context) -> None:
        """Handle /start command."""
        user = update.effective_user
        if user is None or not self._is_user_allowed(user.id):
            await update.message.reply_text("Sorry, I'm currently in private mode.")
            return

        welcome = (
            f"Hey {user.first_name or 'there'}!\n\n"
            "I'm connected to AURA via CLI bridge mode.\n\n"
            "Just send me a message and it'll be handled by the CLI agent.\n\n"
            "Commands:\n"
            "/help - Show available commands\n"
            "/status - Bot & bridge status\n"
            "/bridge - Bridge connection info\n"
        )
        await update.message.reply_text(welcome)

    async def _handle_help(self, update, context) -> None:
        """Handle /help command."""
        user = update.effective_user
        if user is None or not self._is_user_allowed(user.id):
            return

        help_text = (
            "AURA Bridge Mode\n\n"
            "Send any text message and the CLI agent will handle it.\n\n"
            "Commands:\n"
            "/start - Welcome message\n"
            "/help - This help text\n"
            "/status - Bot & bridge status\n"
            "/bridge - Bridge connection info\n\n"
            "Supported inputs:\n"
            "- Text messages\n"
            "- Voice messages (noted, transcription not available)\n"
            "- Photos (noted as attachment)\n"
            "- Documents (noted with filename)\n"
        )
        await update.message.reply_text(help_text)

    async def _handle_status(self, update, context) -> None:
        """Handle /status command."""
        user = update.effective_user
        if user is None or not self._is_user_allowed(user.id):
            return

        uptime_sec = int(time.time() - self._start_time)
        hours, remainder = divmod(uptime_sec, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        allowed_count = len(self._get_allowed_users())

        status_text = (
            "Bridge Status\n\n"
            f"Mode: CLI Bridge\n"
            f"Status: Running\n"
            f"Uptime: {uptime_str}\n"
            f"Allowed users: {allowed_count}\n"
        )
        await update.message.reply_text(status_text)

    async def _handle_bridge(self, update, context) -> None:
        """Handle /bridge command -- show bridge connection info."""
        user = update.effective_user
        if user is None or not self._is_user_allowed(user.id):
            return

        hostname = platform.node() or "unknown"
        bridge_text = f"Connected to CLI bridge at [{hostname}]"
        await update.message.reply_text(bridge_text)

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_message(self, update, context) -> None:
        """Handle incoming Telegram text message -> queue for CLI agent."""
        user = update.effective_user
        if user is None or not self._is_user_allowed(user.id):
            return

        msg = update.message
        if not msg or not msg.text:
            return

        channel_msg = ChannelMessage(
            source=ChannelSource.TELEGRAM,
            text=msg.text,
            user_id=str(user.id),
            user_name=user.first_name or user.username or str(user.id),
            chat_id=str(msg.chat_id),
            metadata={
                "message_id": msg.message_id,
                "username": user.username or "",
            },
            timestamp=time.time(),
        )

        logger.info(
            "[TelegramChannel] Message from %s: %s",
            channel_msg.user_name,
            msg.text[:80] + ("..." if len(msg.text) > 80 else ""),
        )

        # Show typing indicator
        await self._send_typing(str(msg.chat_id))

        self._dispatch(channel_msg)

    async def _handle_voice(self, update, context) -> None:
        """Handle voice messages -- queue with a note."""
        user = update.effective_user
        if user is None or not self._is_user_allowed(user.id):
            return

        msg = update.message
        if not msg:
            return

        text = "(Voice message -- transcription not available in bridge mode)"

        # Include caption if present
        if msg.caption:
            text = f"{msg.caption}\n\n{text}"

        channel_msg = ChannelMessage(
            source=ChannelSource.TELEGRAM,
            text=text,
            user_id=str(user.id),
            user_name=user.first_name or user.username or str(user.id),
            chat_id=str(msg.chat_id),
            metadata={
                "message_id": msg.message_id,
                "type": "voice",
                "username": user.username or "",
            },
            timestamp=time.time(),
        )

        logger.info("[TelegramChannel] Voice message from %s", channel_msg.user_name)
        await self._send_typing(str(msg.chat_id))
        self._dispatch(channel_msg)

    async def _handle_photo(self, update, context) -> None:
        """Handle photo messages -- queue with a note."""
        user = update.effective_user
        if user is None or not self._is_user_allowed(user.id):
            return

        msg = update.message
        if not msg:
            return

        text = "(Photo attached)"

        # Include caption if present
        if msg.caption:
            text = f"{msg.caption}\n\n{text}"

        channel_msg = ChannelMessage(
            source=ChannelSource.TELEGRAM,
            text=text,
            user_id=str(user.id),
            user_name=user.first_name or user.username or str(user.id),
            chat_id=str(msg.chat_id),
            metadata={
                "message_id": msg.message_id,
                "type": "photo",
                "username": user.username or "",
            },
            timestamp=time.time(),
        )

        logger.info("[TelegramChannel] Photo from %s", channel_msg.user_name)
        await self._send_typing(str(msg.chat_id))
        self._dispatch(channel_msg)

    async def _handle_document(self, update, context) -> None:
        """Handle document messages -- queue with filename note."""
        user = update.effective_user
        if user is None or not self._is_user_allowed(user.id):
            return

        msg = update.message
        if not msg or not msg.document:
            return

        filename = msg.document.file_name or "unknown"
        text = f"(Document: {filename})"

        # Include caption if present
        if msg.caption:
            text = f"{msg.caption}\n\n{text}"

        channel_msg = ChannelMessage(
            source=ChannelSource.TELEGRAM,
            text=text,
            user_id=str(user.id),
            user_name=user.first_name or user.username or str(user.id),
            chat_id=str(msg.chat_id),
            metadata={
                "message_id": msg.message_id,
                "type": "document",
                "filename": filename,
                "username": user.username or "",
            },
            timestamp=time.time(),
        )

        logger.info(
            "[TelegramChannel] Document from %s: %s",
            channel_msg.user_name,
            filename,
        )
        await self._send_typing(str(msg.chat_id))
        self._dispatch(channel_msg)

    # ------------------------------------------------------------------
    # Error handler
    # ------------------------------------------------------------------

    async def _handle_error(self, update, context) -> None:
        """Log errors from the Telegram dispatcher."""
        logger.error(
            "[TelegramChannel] Update %s caused error: %s",
            update,
            context.error,
        )

    # ------------------------------------------------------------------
    # Dispatch helper
    # ------------------------------------------------------------------

    def _dispatch(self, channel_msg: ChannelMessage) -> None:
        """Forward a ChannelMessage to the bridge callback."""
        if self._on_message:
            try:
                self._on_message(channel_msg)
            except Exception:
                logger.exception("[TelegramChannel] on_message callback failed")

    # ------------------------------------------------------------------
    # Sending messages
    # ------------------------------------------------------------------

    async def send_response(self, chat_id: str, text: str, reply_to: Optional[str] = None) -> None:
        """Send response back to Telegram with MarkdownV2 formatting.

        This is a public coroutine for direct async use. The ``send()``
        method (ChannelAdapter interface) wraps this for cross-thread calls.
        """
        await self._send_response(chat_id, text, reply_to=reply_to)

    async def _send_response(self, chat_id: str, text: str, reply_to: Optional[str] = None) -> None:
        """Send a response back to Telegram with MarkdownV2, falling back to plain text."""
        if not self._app or not self._app.bot:
            return

        try:
            from telegram.constants import ParseMode
        except ImportError:
            return

        # Try MarkdownV2 first
        try:
            chunks = _format_telegram_response(text)
            for chunk in chunks:
                kwargs: dict = {
                    "chat_id": int(chat_id),
                    "text": chunk,
                    "parse_mode": ParseMode.MARKDOWN_V2,
                }
                if reply_to:
                    kwargs["reply_to_message_id"] = int(reply_to)
                    reply_to = None  # Only reply-to on the first chunk

                await self._app.bot.send_message(**kwargs)
                if len(chunks) > 1:
                    await asyncio.sleep(0.3)
            return
        except Exception as e:
            logger.debug(
                "[TelegramChannel] MarkdownV2 send failed, falling back to plain text: %s", e
            )

        # Fallback: send as plain text, split at limit
        try:
            for chunk in _split_message(text, _TELEGRAM_MSG_LIMIT):
                kwargs = {"chat_id": int(chat_id), "text": chunk}
                if reply_to:
                    kwargs["reply_to_message_id"] = int(reply_to)
                    reply_to = None
                await self._app.bot.send_message(**kwargs)
                await asyncio.sleep(0.1)
        except Exception:
            logger.exception(
                "[TelegramChannel] Failed to send message to chat %s", chat_id
            )

    async def send_typing(self, chat_id: str) -> None:
        """Show typing indicator -- public API."""
        await self._send_typing(chat_id)

    async def _send_typing(self, chat_id: str) -> None:
        """Show typing indicator in chat."""
        if not self._app or not self._app.bot:
            return
        try:
            from telegram.constants import ChatAction
            await self._app.bot.send_chat_action(
                chat_id=int(chat_id),
                action=ChatAction.TYPING,
            )
        except Exception as e:
            logger.debug("[TelegramChannel] Could not send typing indicator: %s", e)

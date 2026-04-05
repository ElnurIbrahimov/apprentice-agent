"""
TelegramBot — core class and module-level scheduler callbacks.

This module contains:
  - Module-level rate limiter (_RateLimiter, _check_rate_limit)
  - Module-level scheduler callbacks (APScheduler-compatible, pickleable)
  - class TelegramBot(BasePlatform + all mixins)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time as _time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from aura.messaging.telegram_formatting import (
    split_message as _split_message,
)

try:
    from telegram import (
        Update, Bot, InlineQueryResultArticle, InputTextMessageContent,
        InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice,
        ReplyKeyboardMarkup, ReplyKeyboardRemove, BotCommand,
    )
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        InlineQueryHandler,
        CallbackQueryHandler,
        PreCheckoutQueryHandler,
        ChatMemberHandler,
        MessageReactionHandler,
        ContextTypes,
        filters,
    )
    from telegram.constants import ParseMode, ChatAction

    try:
        from telegram import ReactionTypeEmoji
        REACTIONS_AVAILABLE = True
    except ImportError:
        REACTIONS_AVAILABLE = False

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None
    Bot = None

from aura.messaging.base_platform import (
    BasePlatform,
    IncomingMessage,
    OutgoingMessage,
    MessageType,
)
from aura.core.conversation_manager import get_conversation_manager

try:
    from aura_skill_library import (
        SkillStore, SkillLearner, Skill, SkillCategory, SkillExample, SkillMetadata
    )
    SKILL_LIBRARY_AVAILABLE = True
except ImportError:
    SKILL_LIBRARY_AVAILABLE = False

try:
    from aura.evolution import GEPAEngine, GEPAConfig, AuraSkillAdapter, Candidate, GEPAResult
    from aura.evolution.types import EvalExample
    GEPA_AVAILABLE = True
except (ImportError, Exception):
    GEPA_AVAILABLE = False

from aura.messaging.telegram.constants import _LRUCache
from aura.messaging.telegram.mixins import (
    CommandsMixin,
    ResearchMixin,
    SessionsMixin,
    MediaMixin,
    AgentCoreMixin,
    SkillsMixin,
    SchedulingMixin,
    LocationMixin,
    SocialMixin,
    PaymentsMixin,
    MiscMixin,
)

logger = logging.getLogger(__name__)

# ============================================================================
#  Per-user rate limiter
# ============================================================================

class _RateLimiter:
    """Bounded per-user rate limiter with periodic cleanup of stale entries."""

    def __init__(self) -> None:
        self._timestamps: Dict[str, list] = {}
        self._last_cleanup: float = _time.time()

    def check(self, user_id: str, max_per_min: int = 20) -> bool:
        """Return True if the user is within rate limits, False if throttled."""
        now = _time.time()
        # Periodic cleanup: remove users with no activity in the last hour
        if now - self._last_cleanup > 3600:
            cutoff = now - 3600
            stale = [uid for uid, ts in self._timestamps.items() if not ts or ts[-1] < cutoff]
            for uid in stale:
                del self._timestamps[uid]
            self._last_cleanup = now
        stamps = self._timestamps.get(user_id)
        if stamps is None:
            stamps = []
            self._timestamps[user_id] = stamps
        stamps[:] = [t for t in stamps if now - t < 60]
        if len(stamps) >= max_per_min:
            return False
        stamps.append(now)
        return True


_rate_limiter = _RateLimiter()


def _check_rate_limit(user_id: str, max_per_min: int = 20) -> bool:
    """Return True if the user is within rate limits, False if throttled."""
    return _rate_limiter.check(user_id, max_per_min)


# ============================================================================
#  Scheduled task / reminder callbacks (module-level for APScheduler pickling)
# ============================================================================

# Global ref to the running bot instance so scheduler callbacks can send messages.
# Set in TelegramBot.start(), cleared in TelegramBot.stop().
_active_bot_instance: Optional["TelegramBot"] = None
_active_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _send_telegram_reminder(chat_id: str, message: str):
    """APScheduler callback: send a one-shot reminder message to Telegram.

    Runs in APScheduler's thread pool — uses run_coroutine_threadsafe to bridge
    into the bot's async event loop.
    """
    bot_inst = _active_bot_instance
    loop = _active_event_loop
    if not bot_inst or not bot_inst.bot or not loop or loop.is_closed():
        logging.getLogger(__name__).warning(
            f"[Scheduler] Cannot deliver reminder — bot not running. chat={chat_id}"
        )
        return

    text = f"\u23f0 Reminder: {message}"

    async def _send():
        try:
            await bot_inst.bot.send_message(chat_id=chat_id, text=text)
        except Exception as e:
            logging.getLogger(__name__).error(f"[Scheduler] Reminder send failed: {e}")

    asyncio.run_coroutine_threadsafe(_send(), loop)


def _run_telegram_scheduled_task(chat_id: str, task_prompt: str, is_agent_task: bool = False):
    """APScheduler callback: run a scheduled task and send results to Telegram.

    If is_agent_task is True, runs the prompt through the AURA agent first.
    Otherwise just sends the task_prompt as a notification.
    """
    bot_inst = _active_bot_instance
    loop = _active_event_loop
    if not bot_inst or not bot_inst.bot or not loop or loop.is_closed():
        logging.getLogger(__name__).warning(
            f"[Scheduler] Cannot deliver scheduled task — bot not running. chat={chat_id}"
        )
        return

    async def _execute_and_send():
        try:
            if is_agent_task:
                # Run through the agent, then send results
                try:
                    response_text, _ = await asyncio.to_thread(
                        bot_inst._run_agent_sync, task_prompt
                    )
                except Exception as e:
                    response_text = f"Scheduled task failed: {e}"

                header = f"\U0001f4c5 Scheduled Task: {task_prompt}\n\n"
                full_text = header + (response_text or "No output.")

                # Split if needed
                for chunk in _split_message(full_text, 4096):
                    await bot_inst.bot.send_message(chat_id=chat_id, text=chunk)
            else:
                await bot_inst.bot.send_message(
                    chat_id=chat_id,
                    text=f"\U0001f4c5 Scheduled: {task_prompt}"
                )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"[Scheduler] Scheduled task send failed: {e}"
            )

    asyncio.run_coroutine_threadsafe(_execute_and_send(), loop)


def notify_hand_result(result) -> None:
    """Push a Hand execution result to all authorized Telegram chats.

    Called from HandManager's notify callback. Skips routine "all clear" results.
    Only notifies on: failures, Guardian issues, or completed research with findings.
    """
    bot_inst = _active_bot_instance
    loop = _active_event_loop
    if not bot_inst or not bot_inst.bot or not loop or loop.is_closed():
        return

    # Skip routine "all clear" Guardian reports and empty Memory maintenance
    hand_name = result.hand_name if hasattr(result, 'hand_name') else str(result.get("hand", ""))
    success = result.success if hasattr(result, 'success') else result.get("success", True)
    summary = result.summary if hasattr(result, 'summary') else result.get("summary", "")

    if success and hand_name == "guardian" and "ALL CLEAR" in summary.upper():
        return
    if success and hand_name == "memory" and "no action needed" in summary.lower():
        return

    icon = "\u2705" if success else "\u274c"
    text = f"{icon} <b>Hand '{hand_name}' completed</b>\n\n{summary[:800]}"

    async def _send():
        try:
            # Send to all known authorized chats
            chat_ids = list(getattr(bot_inst, '_authorized_chats', set()))
            if not chat_ids and hasattr(bot_inst, '_admin_chat_id'):
                chat_ids = [bot_inst._admin_chat_id]
            for cid in chat_ids:
                try:
                    await bot_inst.bot.send_message(chat_id=cid, text=text, parse_mode="HTML")
                except Exception:
                    pass
        except Exception as e:
            logging.getLogger(__name__).debug(f"[Hands] Telegram notify failed: {e}")

    asyncio.run_coroutine_threadsafe(_send(), loop)


def notify_hand_approval_request(request: dict) -> None:
    """Send a Hand approval request to Telegram with InlineKeyboard."""
    bot_inst = _active_bot_instance
    loop = _active_event_loop
    if not bot_inst or not bot_inst.bot or not loop or loop.is_closed():
        return

    request_id = request.get("request_id", "unknown")
    hand_name = request.get("hand_name", "unknown")
    tool_name = request.get("tool_name", "unknown")

    text = (
        f"\U0001f510 <b>Approval Required</b>\n\n"
        f"Hand '<b>{hand_name}</b>' wants to use <b>{tool_name}</b>\n"
        f"Request: {request_id}"
    )

    async def _send():
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("\u2705 Approve", callback_data=f"hand_approve_{request_id}"),
                InlineKeyboardButton("\u274c Deny", callback_data=f"hand_deny_{request_id}"),
            ]])
            chat_ids = list(getattr(bot_inst, '_authorized_chats', set()))
            if not chat_ids and hasattr(bot_inst, '_admin_chat_id'):
                chat_ids = [bot_inst._admin_chat_id]
            for cid in chat_ids:
                try:
                    await bot_inst.bot.send_message(
                        chat_id=cid, text=text, parse_mode="HTML", reply_markup=keyboard
                    )
                except Exception:
                    pass
        except Exception as e:
            logging.getLogger(__name__).debug(f"[Hands] Telegram approval request failed: {e}")

    asyncio.run_coroutine_threadsafe(_send(), loop)


# ============================================================================
#  TelegramBot — core class
# ============================================================================

class TelegramBot(
    BasePlatform,
    CommandsMixin,
    ResearchMixin,
    SessionsMixin,
    MediaMixin,
    AgentCoreMixin,
    SkillsMixin,
    SchedulingMixin,
    LocationMixin,
    SocialMixin,
    PaymentsMixin,
    MiscMixin,
):
    """Telegram Bot integration for AURA"""

    def __init__(self, aura_engine, config: dict):
        super().__init__(aura_engine, config)

        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot not installed. "
                "Run: pip install python-telegram-bot>=20.0"
            )

        self.token = config.get("telegram_token")
        if not self.token or self.token == "YOUR_BOT_TOKEN_HERE":
            raise ValueError(
                "telegram_token is required in config. "
                "Get one from @BotFather on Telegram."
            )

        self.allowed_users: List[str] = config.get("allowed_users", [])
        self.admin_users: List[str] = config.get("admin_users", [])
        self._allowed_users_cache: set = set()
        self._allowed_cache_time: float = 0.0

        self.app: Optional[Application] = None
        self.bot: Optional[Bot] = None

        # ===== SQLite-backed persistent store =====
        from aura.messaging.telegram_store import TelegramStore
        db_path = config.get("telegram_db_path", "data/telegram_state.db")
        self.store = TelegramStore(db_path=db_path)
        # One-time migration from legacy JSON files
        self.store.migrate_from_json()

        # Track active chats for proactive messaging (backed by store)
        self.active_chats: Dict[str, dict] = self.store.get_active_chats()
        self._pending_forget: Dict[str, datetime] = {}

        # Document context TTL (store handles persistence)
        self._DOC_CONTEXT_TTL = 30 * 60  # 30 minutes

        # Lazy-loaded code executor
        self._code_executor = None

        # Inline query state (cache in-memory for speed, store for persistence)
        self._inline_cache = _LRUCache(maxsize=50)
        self._inline_last_query: Dict[int, Tuple[str, float]] = {}  # user_id -> (query, timestamp)
        self._INLINE_DEBOUNCE_SEC = 1.0
        self._INLINE_TIMEOUT_SEC = 15.0

        # Skill system state (in-memory cache, write-through to store)
        self._skill_pending: Dict[int, dict] = {}
        self._last_exchange: Dict[int, dict] = {}
        self._skill_store = None   # Lazy-loaded SkillStore
        self._skill_learner = None  # Lazy-loaded SkillLearner
        self._skill_create_state: Dict[int, dict] = {}

        # Premium/payment state (backed by store)
        self._premium_users: Dict[str, dict] = self.store.get_premium_users()

        # Group message cache (backed by store, read-through cache)
        self._group_message_cache: Dict[str, List[dict]] = {}

        # Location sharing state (backed by store)
        self._user_locations: Dict[str, dict] = {}

        # === World-class UX improvements ===
        # Per-user processing locks (backpressure — one message at a time)
        self._user_locks: Dict[str, asyncio.Lock] = {}
        self._processing_users: set = set()

        # Conversation context window (sliding window of recent messages)
        self._chat_history: Dict[str, list] = defaultdict(list)
        self._MAX_CONTEXT_MESSAGES = 10

        # Failed messages for retry
        self._failed_messages: Dict[str, str] = {}

        # Voice reply synthesizer (TTS for voice message replies)
        from aura.messaging.telegram.voice_reply import VoiceReply
        self._voice_reply = VoiceReply()

        # Digest job tracking (backed by store)
        self._digest_job_ids: Dict[str, str] = self.store.get_digest_jobs()

    @property
    def platform_name(self) -> str:
        return "telegram"

    def _get_code_executor(self):
        """Lazy-load CodeExecutorTool to avoid import cost at startup."""
        if self._code_executor is None:
            from aura.tools.code_executor import CodeExecutorTool
            self._code_executor = CodeExecutorTool(timeout=30)
        return self._code_executor

    def _get_brain(self):
        """Get the OllamaBrain instance from the agent wrapper."""
        agent = getattr(self.aura, 'agent', None)
        if agent:
            return getattr(agent, 'brain', None)
        return getattr(self.aura, 'brain', None)

    def _save_state(self):
        """Persist current active_chats to SQLite store."""
        try:
            for chat_id, info in self.active_chats.items():
                self.store.upsert_active_chat(
                    chat_id=chat_id,
                    user_id=info.get("user_id", ""),
                    first_name=info.get("first_name", ""),
                    username=info.get("username", ""),
                )
        except Exception as e:
            logger.error(f"Could not save Telegram state: {e}")

    def _save_premium_state(self):
        """Premium state is auto-persisted via store — this is a no-op for compatibility."""
        pass

    def is_premium(self, user_id: str) -> bool:
        """Check if a user has premium status."""
        return user_id in self._premium_users or self.store.is_premium(user_id)

    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed — cached for 300s, refreshed from env on expiry."""
        now = _time.time()
        if now - self._allowed_cache_time > 300:
            env_val = os.environ.get("TELEGRAM_ALLOWED_USERS", "")
            allowed: set = {u.strip() for u in env_val.split(",") if u.strip()} if env_val else set()
            for u in self.allowed_users:
                if u:
                    allowed.add(u)
            self._allowed_users_cache = allowed
            self._allowed_cache_time = now
        if not self._allowed_users_cache:
            logger.warning(f"[TelegramBot] Rejected user {user_id} — no allowed_users configured.")
            return False
        is_allowed = str(user_id) in self._allowed_users_cache
        if not is_allowed:
            logger.warning(f"[TelegramBot] Rejected user {user_id} — not in allowed list")
        return is_allowed

    def _is_admin(self, user_id: int) -> bool:
        """Check if user is an admin"""
        return str(user_id) in self.admin_users

    async def _send_formatted(self, update_or_msg, text: str, **kwargs):
        """Send text with MarkdownV2 formatting, falling back to plain text.

        Works with both Update objects and Message objects (for reply_text).
        Handles splitting long messages into chunks.
        """
        from telegram.constants import ParseMode
        from aura.messaging.telegram_formatting import format_telegram_response

        msg = update_or_msg
        if hasattr(msg, 'message'):
            msg = msg.message

        try:
            chunks = format_telegram_response(text)
        except Exception:
            chunks = [text] if len(text) <= 4096 else self._split_message(text, 4096)

        for chunk in chunks:
            for parse_mode in (ParseMode.MARKDOWN_V2, None):
                try:
                    await msg.reply_text(chunk, parse_mode=parse_mode, **kwargs)
                    break
                except Exception:
                    if parse_mode is None:
                        logger.warning(f"Failed to send formatted chunk ({len(chunk)} chars)")

    async def start(self):
        """Start the Telegram bot"""

        logger.info("Starting Telegram bot...")

        # Build application
        self.app = Application.builder().token(self.token).build()
        self.bot = self.app.bot

        # Add handlers
        self.app.add_handler(CommandHandler("start", self._handle_start))
        self.app.add_handler(CommandHandler("help", self._handle_help))
        self.app.add_handler(CommandHandler("status", self._handle_status))
        self.app.add_handler(CommandHandler("mood", self._handle_mood))
        self.app.add_handler(CommandHandler("memory", self._handle_memory))
        self.app.add_handler(CommandHandler("forget", self._handle_forget))
        self.app.add_handler(CommandHandler("research", self._handle_research))
        self.app.add_handler(CommandHandler("search", self._handle_search))
        self.app.add_handler(CommandHandler("summarize", self._handle_summarize))
        self.app.add_handler(CommandHandler("image", self._handle_image))
        self.app.add_handler(CommandHandler("code", self._handle_code))
        self.app.add_handler(CommandHandler("model", self._handle_model))
        self.app.add_handler(CommandHandler("compare", self._handle_compare))
        self.app.add_handler(CommandHandler("session", self._handle_session))
        self.app.add_handler(CommandHandler("webhook", self._handle_webhook))
        self.app.add_handler(CommandHandler("remind", self._handle_remind))
        self.app.add_handler(CommandHandler("schedule", self._handle_schedule))
        self.app.add_handler(CommandHandler("tasks", self._handle_tasks))
        self.app.add_handler(CommandHandler("cancel", self._handle_cancel))
        self.app.add_handler(CommandHandler("agent", self._handle_agent))
        self.app.add_handler(CommandHandler("fleet", self._handle_fleet))
        self.app.add_handler(CommandHandler("learn", self._handle_learn))
        self.app.add_handler(CommandHandler("skill", self._handle_skill))
        self.app.add_handler(CommandHandler("premium", self._handle_premium))
        self.app.add_handler(CommandHandler("donate", self._handle_donate))

        # --- Improvement commands ---
        self.app.add_handler(CommandHandler("keyboard", self._handle_keyboard))
        self.app.add_handler(CommandHandler("stars", self._handle_stars))
        self.app.add_handler(CommandHandler("file", self._handle_file_gen))
        self.app.add_handler(CommandHandler("export", self._handle_export))
        self.app.add_handler(CommandHandler("digest", self._handle_digest))
        self.app.add_handler(CommandHandler("lang", self._handle_lang))
        self.app.add_handler(CommandHandler("stickers", self._handle_stickers_cmd))
        self.app.add_handler(CommandHandler("pin", self._handle_pin))
        self.app.add_handler(CommandHandler("hand", self._handle_hand))
        self.app.add_handler(CommandHandler("tts", self._handle_tts))

        # Hand approval callbacks
        self.app.add_handler(CallbackQueryHandler(self._handle_hand_approval_callback, pattern="^hand_"))

        # Payment handlers
        self.app.add_handler(CallbackQueryHandler(self._handle_callback, pattern="^buy_"))
        self.app.add_handler(CallbackQueryHandler(self._handle_stars_callback, pattern="^stars_"))
        self.app.add_handler(CallbackQueryHandler(self._handle_action_callback, pattern="^act_"))
        self.app.add_handler(CallbackQueryHandler(self._handle_pin_callback, pattern="^pin_"))
        self.app.add_handler(CallbackQueryHandler(self._handle_retry_callback, pattern="^retry_"))
        self.app.add_handler(CallbackQueryHandler(self._handle_onboarding_callback, pattern="^onboard_"))
        self.app.add_handler(PreCheckoutQueryHandler(self._handle_pre_checkout))
        self.app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, self._handle_successful_payment))

        # Inline query handler (enable via @BotFather -> /setinline)
        self.app.add_handler(InlineQueryHandler(self._handle_inline))

        # Voice/audio message handler
        self.app.add_handler(MessageHandler(
            filters.VOICE | filters.AUDIO,
            self._handle_voice
        ))

        # Document upload handler
        self.app.add_handler(MessageHandler(
            filters.Document.ALL,
            self._handle_document
        ))

        # Photo handler (images sent as photos, not documents)
        self.app.add_handler(MessageHandler(
            filters.PHOTO,
            self._handle_photo_upload
        ))

        # Location handler (Phase 5 — contextual location info)
        self.app.add_handler(MessageHandler(
            filters.LOCATION,
            self._handle_location
        ))

        # Nearby search command (uses last shared location)
        self.app.add_handler(CommandHandler("nearby", self._handle_nearby))

        # Sticker handler (Phase 5 — contextual reactions)
        self.app.add_handler(MessageHandler(
            filters.Sticker.ALL,
            self._handle_sticker
        ))

        # Group-specific command handlers
        self.app.add_handler(CommandHandler("summarize_group", self._handle_summarize_group))
        self.app.add_handler(CommandHandler("summarize_thread", self._handle_summarize_thread))

        # ChatMember handler — bot added/removed from groups
        self.app.add_handler(ChatMemberHandler(self._handle_chat_member, ChatMemberHandler.MY_CHAT_MEMBER))

        # Mini App web_app_data handler (commands from Mini App tools tab)
        self.app.add_handler(MessageHandler(
            filters.StatusUpdate.WEB_APP_DATA,
            self._handle_webapp_data
        ))

        # Edited message handler (re-process edits)
        self.app.add_handler(MessageHandler(
            filters.UpdateType.EDITED_MESSAGE & filters.TEXT,
            self._handle_edited_message
        ))

        # Reaction feedback capture
        try:
            self.app.add_handler(MessageReactionHandler(self._handle_reaction_update))
        except Exception as e:
            logger.debug(f"[Telegram] MessageReactionHandler not available: {e}")

        # Message handler (must be last)
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message
        ))

        # Error handler
        self.app.add_error_handler(self._handle_error)

        # Start bot (webhook or polling)
        self.is_running = True
        _allowed_updates = [
            "message", "edited_message", "inline_query", "chosen_inline_result",
            "my_chat_member", "chat_member", "callback_query",
            "pre_checkout_query", "message_reaction",
        ]

        await self.app.initialize()
        await self.app.start()

        # Webhook mode: set TELEGRAM_WEBHOOK_URL to enable
        # e.g. https://yourdomain.com/api/telegram/webhook
        webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL", "").strip()
        webhook_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
        webhook_port = int(os.environ.get("TELEGRAM_WEBHOOK_PORT", "8443"))

        if webhook_url:
            # Generate a secret token if none provided
            if not webhook_secret:
                webhook_secret = hashlib.sha256(self.token.encode()).hexdigest()[:32]

            await self.app.bot.set_webhook(
                url=webhook_url,
                secret_token=webhook_secret,
                allowed_updates=_allowed_updates,
                drop_pending_updates=True,
            )
            # Start built-in webhook server
            # Listens on 0.0.0.0:webhook_port — Nginx proxies to this
            await self.app.updater.start_webhook(
                listen="0.0.0.0",
                port=webhook_port,
                url_path="/api/telegram/webhook",
                webhook_url=webhook_url,
                secret_token=webhook_secret,
            )
            logger.info(f"Telegram bot started in WEBHOOK mode on port {webhook_port}")
            logger.info(f"Webhook URL: {webhook_url}")
        else:
            # Polling mode (default — no webhook URL configured)
            await self.app.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=_allowed_updates,
            )
            logger.info("Telegram bot started in POLLING mode")

        # Expose bot instance + event loop for scheduler callbacks
        global _active_bot_instance, _active_event_loop
        _active_bot_instance = self
        _active_event_loop = asyncio.get_running_loop()

        # Get bot info
        me = await self.bot.get_me()
        logger.info(f"Bot: @{me.username} ({me.first_name})")

        # Set bot commands menu (autocomplete in Telegram)
        try:
            await self.bot.set_my_commands([
                BotCommand("start", "Start the bot"),
                BotCommand("help", "Show help"),
                BotCommand("status", "Current status"),
                BotCommand("mood", "Check my mood"),
                BotCommand("memory", "What I remember"),
                BotCommand("research", "Deep research"),
                BotCommand("search", "Web search"),
                BotCommand("summarize", "Summarize text/URL"),
                BotCommand("image", "Generate an image"),
                BotCommand("code", "Run Python code"),
                BotCommand("model", "View/switch AI models"),
                BotCommand("compare", "Compare models"),
                BotCommand("session", "Manage sessions"),
                BotCommand("remind", "Set a reminder"),
                BotCommand("schedule", "Recurring tasks"),
                BotCommand("tasks", "List scheduled tasks"),
                BotCommand("agent", "Run specialist agent"),
                BotCommand("fleet", "Multi-agent parallel"),
                BotCommand("keyboard", "Toggle reply keyboard"),
                BotCommand("stars", "Support with Telegram Stars"),
                BotCommand("file", "Generate document"),
                BotCommand("export", "Export conversation"),
                BotCommand("digest", "Daily digest settings"),
                BotCommand("lang", "Set language"),
                BotCommand("pin", "Pin a message"),
                BotCommand("nearby", "Nearby places"),
                BotCommand("premium", "Premium tiers"),
                BotCommand("donate", "Support AURA"),
            ])
            logger.info("[Telegram] Bot commands menu set successfully")
        except Exception as e:
            logger.warning(f"[Telegram] Could not set bot commands: {e}")

        # Set Mini App as persistent menu button (replaces hamburger menu)
        webapp_url = os.environ.get("TELEGRAM_WEBAPP_URL", "").strip()
        if webapp_url:
            try:
                from telegram import MenuButtonWebApp, WebAppInfo
                await self.bot.set_chat_menu_button(
                    menu_button=MenuButtonWebApp(
                        text="Open AURA",
                        web_app=WebAppInfo(url=webapp_url),
                    )
                )
                logger.info(f"[Telegram] Mini App menu button set: {webapp_url}")
            except Exception as e:
                logger.warning(f"[Telegram] Could not set Mini App menu button: {e}")
        else:
            logger.info("[Telegram] No TELEGRAM_WEBAPP_URL set — Mini App button skipped")

        # Connect proactive system to Telegram
        await self._connect_proactive_system()

    async def stop(self):
        """Stop the Telegram bot"""

        logger.info("Stopping Telegram bot...")
        self.is_running = False

        # Clear scheduler callback references
        global _active_bot_instance, _active_event_loop
        _active_bot_instance = None
        _active_event_loop = None

        # Cancel proactive polling task
        if hasattr(self, '_proactive_task') and self._proactive_task:
            self._proactive_task.cancel()
            try:
                await self._proactive_task
            except asyncio.CancelledError:
                pass
            logger.info("Proactive polling stopped")

        # Persist final state and close SQLite
        self._save_state()
        if hasattr(self, 'store'):
            self.store.close()
            logger.info("TelegramStore closed")

        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

        self._save_state()
        logger.info("Telegram bot stopped.")

    async def send_message(self, message: OutgoingMessage) -> bool:
        """Send a message"""

        try:
            parse_mode = None
            if message.parse_mode == "markdown":
                parse_mode = ParseMode.MARKDOWN

            await self.bot.send_message(
                chat_id=message.chat_id,
                text=message.text,
                parse_mode=parse_mode,
                reply_to_message_id=message.reply_to_message_id
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    async def send_typing_indicator(self, chat_id: str):
        """Show typing indicator"""
        try:
            await self.bot.send_chat_action(
                chat_id=chat_id,
                action=ChatAction.TYPING
            )
        except Exception as e:
            logger.warning(f"Could not send typing indicator: {e}")

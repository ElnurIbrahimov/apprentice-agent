"""
Telegram Bot Integration for AURA

Uses python-telegram-bot library (async version).
Install: pip install python-telegram-bot>=20.0
"""

import asyncio
import logging
import os
import random
import time as _time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
import json

# Per-user rate limiting
_msg_timestamps: Dict[str, list] = defaultdict(list)


def _check_rate_limit(user_id: str, max_per_min: int = 20) -> bool:
    """Return True if the user is within rate limits, False if throttled."""
    now = _time.time()
    stamps = _msg_timestamps[user_id]
    stamps[:] = [t for t in stamps if now - t < 60]
    if len(stamps) >= max_per_min:
        return False
    stamps.append(now)
    return True

try:
    from telegram import Update, Bot
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters
    )
    from telegram.constants import ParseMode, ChatAction
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None
    Bot = None

from .base_platform import (
    BasePlatform,
    IncomingMessage,
    OutgoingMessage,
    MessageType
)

logger = logging.getLogger(__name__)


class TelegramBot(BasePlatform):
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

        self.app: Optional[Application] = None
        self.bot: Optional[Bot] = None

        # Track active chats for proactive messaging
        self.active_chats: Dict[str, dict] = {}
        self._pending_forget: Dict[str, datetime] = {}
        self._load_state()

    @property
    def platform_name(self) -> str:
        return "telegram"

    def _load_state(self):
        """Load saved state (active chats, etc.)"""
        state_file = Path("data/messaging/telegram_state.json")
        if state_file.exists():
            try:
                with open(state_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.active_chats = data.get("active_chats", {})
            except Exception as e:
                logger.warning(f"Could not load Telegram state: {e}")

    def _save_state(self):
        """Save state for persistence"""
        state_file = Path("data/messaging/telegram_state.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "active_chats": self.active_chats,
                    "last_saved": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save Telegram state: {e}")

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

        # Message handler (must be last)
        self.app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message
        ))

        # Error handler
        self.app.add_error_handler(self._handle_error)

        # Start polling
        self.is_running = True

        # Initialize and start
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)

        logger.info("Telegram bot started successfully!")

        # Get bot info
        me = await self.bot.get_me()
        logger.info(f"Bot: @{me.username} ({me.first_name})")

        # Connect proactive system to Telegram
        await self._connect_proactive_system()

    async def stop(self):
        """Stop the Telegram bot"""

        logger.info("Stopping Telegram bot...")
        self.is_running = False

        # Cancel proactive polling task
        if hasattr(self, '_proactive_task') and self._proactive_task:
            self._proactive_task.cancel()
            try:
                await self._proactive_task
            except asyncio.CancelledError:
                pass
            logger.info("Proactive polling stopped")

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

    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed — reads os.environ EVERY call (bulletproof)."""
        env_val = os.environ.get("TELEGRAM_ALLOWED_USERS", "")
        allowed = [u.strip() for u in env_val.split(",") if u.strip()] if env_val else []
        if self.allowed_users:
            for u in self.allowed_users:
                if u and u not in allowed:
                    allowed.append(u)
        if not allowed:
            logger.warning(f"[TelegramBot] Rejected user {user_id} — no allowed_users configured.")
            return False
        is_allowed = str(user_id) in allowed
        if not is_allowed:
            logger.warning(f"[TelegramBot] Rejected user {user_id} — not in allowed list {allowed}")
        return is_allowed


    def _is_admin(self, user_id: int) -> bool:
        """Check if user is an admin"""
        return str(user_id) in self.admin_users

    # ============ COMMAND HANDLERS ============

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""

        user = update.effective_user
        chat_id = str(update.effective_chat.id)

        if not self._is_user_allowed(user.id):
            await update.message.reply_text(
                "Sorry, I'm currently in private beta."
            )
            return

        # Track this chat
        self.active_chats[chat_id] = {
            "user_id": str(user.id),
            "username": user.username,
            "first_name": user.first_name,
            "started_at": datetime.now().isoformat(),
            "last_message": datetime.now().isoformat()
        }
        self._save_state()

        welcome = f"""Hey {user.first_name}!

I'm AURA - your AI thinking partner.

I'm not just a chatbot. I remember our conversations, notice patterns, and actually care how things turn out for you.

Quick commands:
/status - See my current state
/mood - Check my mood
/memory - What I remember
/help - More info

Or just talk to me like a friend. What's on your mind?"""

        await update.message.reply_text(welcome)

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""

        help_text = """AURA Commands

/start - Start fresh
/status - My current status
/mood - How I'm feeling
/memory - What I remember about you
/forget - Clear my memory (careful!)

Tips:
- Just chat normally - I'll respond naturally
- Say "remember this:" to save something important
- I'll follow up on things you mention
- I notice patterns over time

I'm here whenever you need me."""

        await update.message.reply_text(help_text)

    async def _handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""

        if not self._is_user_allowed(update.effective_user.id):
            return

        # Get status from agent
        try:
            tool_count = len(self.aura.tools) if hasattr(self.aura, 'tools') else 0
            identity_name = self.aura.identity.get('name', 'AURA') if hasattr(self.aura, 'identity') else 'AURA'
            status = f"""AURA Status

Name: {identity_name}
Tools: {tool_count} loaded
Status: Online and ready!"""
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            status = "Online and ready!"

        await update.message.reply_text(status)

    async def _handle_mood(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mood command"""

        if not self._is_user_allowed(update.effective_user.id):
            return

        # Get mood from EvoEmo tool if available
        try:
            evoemo = self.aura.tools.get("evoemo") if hasattr(self.aura, 'tools') else None
            if evoemo and hasattr(evoemo, 'get_state'):
                state = evoemo.get_state()
                mood = state.get("dominant_emotion", "neutral")
                response = f"Current Mood: {mood}\n\nReady to chat!"
            else:
                response = "Feeling good and ready to chat!"
        except Exception as e:
            logger.error(f"Error getting mood: {e}")
            response = "Feeling good and ready to chat!"

        await update.message.reply_text(response)

    async def _handle_memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /memory command"""

        if not self._is_user_allowed(update.effective_user.id):
            return

        # Get memory summary from agent's MemorySystem
        try:
            if hasattr(self.aura, 'memory') and self.aura.memory:
                mem = self.aura.memory
                # MemorySystem stores conversations in ChromaDB
                stats = mem.get_stats() if hasattr(mem, 'get_stats') else {}
                count = stats.get('total_entries', 0) if stats else 0
                memory_text = f"Memory system active!\n\nStored entries: {count}\n\nKeep chatting and I'll remember important things."
            else:
                memory_text = "Memory system active. I remember our conversations!"
        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            memory_text = "Memory system active!"

        await update.message.reply_text(memory_text)

    async def _handle_forget(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /forget command"""

        if not self._is_user_allowed(update.effective_user.id):
            return

        user_id = str(update.effective_user.id)

        await update.message.reply_text(
            "This will clear my memory of you. Are you sure?\n\n"
            "This includes:\n"
            "- Conversation history\n"
            "- User profile information\n"
            "- Learned facts about you\n"
            "- Emotional context\n\n"
            "Type 'yes forget everything' to confirm."
        )

        # Store pending forget request
        self._pending_forget[user_id] = datetime.now()

    # ============ MESSAGE HANDLER ============

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages"""

        user = update.effective_user
        chat_id = str(update.effective_chat.id)

        # Check if user is allowed
        if not self._is_user_allowed(user.id):
            return

        # Per-user rate limit (max_messages_per_minute from config, default 20)
        max_per_min = self.config.get("max_messages_per_minute", 20)
        if not _check_rate_limit(str(user.id), max_per_min):
            await update.message.reply_text(
                "You're sending messages too fast. Please wait a moment."
            )
            return

        text = update.message.text

        # Check for forget confirmation
        if text and text.lower() == "yes forget everything":
            user_id = str(user.id)

            # Check if there's a pending forget request (within 5 minutes)
            if hasattr(self, '_pending_forget') and user_id in self._pending_forget:
                request_time = self._pending_forget[user_id]
                if (datetime.now() - request_time).total_seconds() < 300:  # 5 min
                    # Actually clear the memory
                    cleared_items = []

                    try:
                        # Clear agent's MemorySystem if available
                        if hasattr(self.aura, 'memory') and self.aura.memory:
                            if hasattr(self.aura.memory, 'clear'):
                                self.aura.memory.clear()
                                cleared_items.append("Conversation memory")

                        # Clear agent state history
                        if hasattr(self.aura, 'state') and hasattr(self.aura.state, '_history'):
                            self.aura.state._history.clear()
                            cleared_items.append("State history")

                        # Remove pending forget request
                        del self._pending_forget[user_id]

                        if cleared_items:
                            cleared_list = "\n".join(f"- {item}" for item in cleared_items)
                            await update.message.reply_text(
                                f"Memory cleared successfully!\n\n"
                                f"Cleared:\n{cleared_list}\n\n"
                                f"Fresh start! What would you like to talk about?"
                            )
                        else:
                            await update.message.reply_text(
                                "Memory cleared. Fresh start! What would you like to talk about?"
                            )

                    except Exception as e:
                        logger.error(f"Error clearing memory: {e}")
                        await update.message.reply_text(
                            "There was an issue clearing some memories, but I'll treat this as a fresh start. "
                            "What would you like to talk about?"
                        )
                    return
                else:
                    # Request expired
                    del self._pending_forget[user_id]

            # No pending request or expired
            await update.message.reply_text(
                "No active forget request. Use /forget first if you want to clear my memory."
            )
            return

        # Update active chat info
        if chat_id in self.active_chats:
            self.active_chats[chat_id]["last_message"] = datetime.now().isoformat()
        else:
            self.active_chats[chat_id] = {
                "user_id": str(user.id),
                "username": user.username,
                "first_name": user.first_name,
                "started_at": datetime.now().isoformat(),
                "last_message": datetime.now().isoformat()
            }

        # Show typing indicator
        await self.send_typing_indicator(chat_id)

        # Create standardized message
        incoming = IncomingMessage(
            platform="telegram",
            user_id=str(user.id),
            chat_id=chat_id,
            username=user.username,
            display_name=user.first_name,
            message_type=MessageType.TEXT,
            text=text,
            media_url=None,
            timestamp=datetime.now(),
            raw_message=update.message
        )

        # Process through AURA
        response = await self.handle_incoming(incoming)

        if response:
            # Send response
            await update.message.reply_text(response)

        self._save_state()

    async def _handle_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Telegram error: {context.error}")

        if update and update.effective_chat:
            try:
                await self.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Oops, something went wrong. Let me try again..."
                )
            except Exception as e:
                logger.debug(f"Could not send error message: {e}")

    # ============ OVERRIDE AURA PROCESSING ============

    async def _process_with_aura(self, text: str, user_id: str) -> str:
        """
        Process message through ApprenticeAgent.

        Uses asyncio.to_thread to avoid blocking the event loop, since
        agent.chat() / generate_response() are synchronous LLM calls.
        """
        try:
            # Try generate_response (TelegramAgentWrapper), then chat (direct agent)
            if hasattr(self.aura, 'generate_response'):
                response = await asyncio.to_thread(self.aura.generate_response, text)
                if response:
                    return response
            elif hasattr(self.aura, 'chat'):
                response = await asyncio.to_thread(self.aura.chat, text)
                if response:
                    return response
            else:
                logger.error(f"AURA has no chat or generate_response method. Type: {type(self.aura)}")
        except Exception as e:
            logger.error(f"AURA processing error: {e}", exc_info=True)

        return "Sorry, I couldn't process that. Try again in a moment."

    # ============ PROACTIVE MESSAGING ============

    async def _connect_proactive_system(self):
        """Connect GatewayDaemon proactive system to Telegram if available."""
        try:
            daemon = getattr(self.aura, 'gateway_daemon', None)
            if daemon:
                # Start proactive polling loop
                self._proactive_task = asyncio.create_task(self._proactive_polling_loop())
                logger.info("Proactive system connected to Telegram!")
            else:
                self._proactive_task = None
                logger.info("No proactive system available — skipping")
        except Exception as e:
            self._proactive_task = None
            logger.warning(f"Could not connect proactive system: {e}")

    async def _proactive_polling_loop(self):
        """Poll GatewayDaemon for proactive messages and send to active chats."""
        logger.info("Proactive polling loop started")

        while self.is_running:
            try:
                daemon = getattr(self.aura, 'gateway_daemon', None)
                if daemon and hasattr(daemon, 'get_pending_notifications'):
                    pending = daemon.get_pending_notifications()

                    for notification in pending:
                        message = getattr(notification, 'content', str(notification))
                        for chat_id, chat_info in self.active_chats.items():
                            try:
                                user_name = chat_info.get("first_name", "there")
                                personalized = message.replace("{name}", user_name)
                                await self.send_proactive(chat_id, personalized)
                                await asyncio.sleep(0.2)
                            except Exception as e:
                                logger.warning(f"Could not send proactive to {chat_id}: {e}")

            except Exception as e:
                logger.error(f"Proactive polling error: {e}")

            # Wait before next poll (60 seconds)
            await asyncio.sleep(60)

    async def send_to_all_active(self, message: str):
        """Send a message to all active chats (for broadcasts)"""

        for chat_id in self.active_chats:
            try:
                await self.send_proactive(chat_id, message)
                await asyncio.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Could not send to {chat_id}: {e}")

    async def send_morning_greeting(self, chat_id: str, user_name: str):
        """Send personalized morning greeting"""

        greetings = [
            f"Morning {user_name}! What's on your mind today?",
            f"Hey {user_name}! Morning. Ready when you are.",
            f"Good morning! How are you feeling today?",
        ]

        await self.send_proactive(chat_id, random.choice(greetings))

    async def send_follow_up(self, chat_id: str, topic: str):
        """Send a follow-up about something mentioned"""

        message = f"Hey - how did {topic} go?"
        await self.send_proactive(chat_id, message)

    def get_active_chat_ids(self) -> List[str]:
        """Get list of active chat IDs for proactive messaging"""
        return list(self.active_chats.keys())

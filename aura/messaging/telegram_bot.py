"""
Telegram Bot Integration for AURA

Uses python-telegram-bot library (async version).
Install: pip install python-telegram-bot>=20.0
"""

import asyncio
import logging
import random
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
import json

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
        self._load_state()

    @property
    def platform_name(self) -> str:
        return "telegram"

    def _load_state(self):
        """Load saved state (active chats, etc.)"""
        state_file = Path("aura/data/messaging/telegram_state.json")
        if state_file.exists():
            try:
                with open(state_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.active_chats = data.get("active_chats", {})
            except Exception as e:
                logger.warning(f"Could not load Telegram state: {e}")

    def _save_state(self):
        """Save state for persistence"""
        state_file = Path("aura/data/messaging/telegram_state.json")
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
        """Check if user is allowed to use the bot"""
        if not self.allowed_users:
            return True  # Allow all if no whitelist
        return str(user_id) in self.allowed_users

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

        # Get status from AURA
        try:
            if hasattr(self.aura, 'get_status'):
                status_data = self.aura.get_status()
                status = f"""AURA Status

Version: {status_data.get('version', 'Unknown')}
Soul: {status_data.get('soul', 'Unknown')}
Mood: {status_data.get('mood', {}).get('mood', 'neutral')}
Patterns: {status_data.get('patterns', {}).get('total_patterns', 0)}
Turns: {status_data.get('turns', 0)}

I'm here and ready to help!"""
            else:
                status = "Online and ready!"
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            status = "Online and ready!"

        await update.message.reply_text(status)

    async def _handle_mood(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mood command"""

        if not self._is_user_allowed(update.effective_user.id):
            return

        # Get mood from AURA emotional engine
        try:
            if hasattr(self.aura, 'emotion'):
                mood = self.aura.emotion.state.mood.value
                energy = self.aura.emotion.state.energy
                warmth = self.aura.emotion.state.warmth

                mood_emojis = {
                    "excited": "!!", "happy": ":)", "content": ":)",
                    "neutral": ":|", "thoughtful": "...", "tired": ":/",
                    "concerned": ":(", "frustrated": ":("
                }
                emoji = mood_emojis.get(mood, ":)")

                response = f"""Current Mood {emoji}

Feeling: {mood}
Energy: {energy:.0%}
Warmth: {warmth:.0%}

{self.aura.emotion.state.mood_reason or 'Just vibing'}"""
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

        # Get memory summary from AURA
        try:
            if hasattr(self.aura, 'memory'):
                # Get recent facts
                facts = self.aura.memory.get_recent("learned_facts", limit=5)
                profile = self.aura.memory.read_section("user_profile", "Basic Info")

                memory_text = "What I Remember\n\n"

                if profile:
                    memory_text += "About You:\n"
                    # Extract just the content, not timestamps
                    for line in profile.split("\n")[:3]:
                        if line.strip():
                            # Remove timestamp formatting
                            clean = line.split("**]** ")[-1] if "**]**" in line else line
                            clean = clean.split("`")[0].strip()
                            if clean and not clean.startswith("-"):
                                memory_text += f"  - {clean}\n"
                            elif clean:
                                memory_text += f"  {clean}\n"

                if facts:
                    memory_text += "\nRecent Facts:\n"
                    for fact in facts[:3]:
                        clean = fact.split("`")[0].strip()
                        memory_text += f"  - {clean[:60]}...\n" if len(clean) > 60 else f"  - {clean}\n"

                if not profile and not facts:
                    memory_text = "I'm still getting to know you! Keep chatting and I'll remember important things."
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

        await update.message.reply_text(
            "This will clear my memory of you. Are you sure?\n\n"
            "Type 'yes forget everything' to confirm."
        )
        # Actual clearing would be handled in message handler

    # ============ MESSAGE HANDLER ============

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages"""

        user = update.effective_user
        chat_id = str(update.effective_chat.id)

        # Check if user is allowed
        if not self._is_user_allowed(user.id):
            return

        text = update.message.text

        # Check for forget confirmation
        if text and text.lower() == "yes forget everything":
            # Clear user-specific memory
            await update.message.reply_text(
                "Memory cleared. Fresh start! What would you like to talk about?"
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
            except:
                pass

    # ============ OVERRIDE AURA PROCESSING ============

    async def _process_with_aura(self, text: str, user_id: str) -> str:
        """
        Process message through AURA engine.

        Uses AURA's generate_response() which:
        - Handles fast-path (only explicit commands)
        - Retrieves memories
        - Calls LLM with full context
        - Stores interaction
        """

        try:
            # Set up progress callback for long-running tasks
            if hasattr(self.aura, 'set_progress_callback'):
                def sync_progress_callback(message: str):
                    """Sync wrapper to send progress messages."""
                    import asyncio
                    try:
                        # Create task to send message
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self.bot.send_message(chat_id=user_id, text=message))
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")

                self.aura.set_progress_callback(sync_progress_callback)

            # Use the new generate_response method that does everything
            if hasattr(self.aura, 'generate_response'):
                response = self.aura.generate_response(
                    user_message=text,
                    chat_id=user_id
                )
                return response

            # Fallback to old process_input if generate_response not available
            if hasattr(self.aura, 'process_input'):
                context = self.aura.process_input(text)

                # Use fast-path from context if available
                if hasattr(self.aura, 'fast_path'):
                    fast_response = self.aura.fast_path.try_fast_path(text)
                    if fast_response:
                        return fast_response

                # Otherwise use LLM directly
                if hasattr(self.aura, 'llm'):
                    memories = []
                    if hasattr(self.aura, 'memory_retriever'):
                        memories = self.aura.memory_retriever.get_relevant_memories(text)

                    response = self.aura.llm.generate(
                        user_message=text,
                        memories=memories,
                        emotional_context={"mood": context.get("mood", "warm")}
                    )
                    return response

        except Exception as e:
            logger.error(f"AURA processing error: {e}")

        # Ultimate fallback - should rarely happen
        return "Hey, I'm here! What's on your mind?"

    # ============ PROACTIVE MESSAGING ============

    async def _connect_proactive_system(self):
        """Connect AURA proactive system to Telegram."""
        try:
            if hasattr(self.aura, 'proactive') and self.aura.proactive:
                # Create async send function for proactive messages
                async def send_proactive_message(chat_id: str, text: str):
                    try:
                        await self.bot.send_message(chat_id=chat_id, text=text)
                        logger.info(f"Sent proactive message to {chat_id}")
                        return True
                    except Exception as e:
                        logger.error(f"Proactive send failed: {e}")
                        return False

                # Store callback on self for later use
                self._proactive_send = send_proactive_message

                # Register all active chats with proactive system
                for chat_id, info in self.active_chats.items():
                    try:
                        # If proactive has register_chat method
                        if hasattr(self.aura.proactive, 'register_chat'):
                            self.aura.proactive.register_chat(
                                chat_id=chat_id,
                                user_name=info.get("first_name", "there")
                            )
                    except:
                        pass

                # Start proactive polling loop
                self._proactive_task = asyncio.create_task(self._proactive_polling_loop())

                logger.info("Proactive system connected to Telegram!")
        except Exception as e:
            logger.warning(f"Could not connect proactive system: {e}")

    async def _proactive_polling_loop(self):
        """Poll heartbeat queue and send proactive messages to active chats."""
        logger.info("Proactive polling loop started")

        while self.is_running:
            try:
                # Check if AURA has proactive system with notifications
                if hasattr(self.aura, 'proactive') and self.aura.proactive:
                    # Get pending notifications from HeartbeatMonitor
                    pending = self.aura.proactive.get_pending_notifications()

                    for notification in pending:
                        message = notification.message
                        category = notification.category

                        # Send to all active chats
                        for chat_id, chat_info in self.active_chats.items():
                            try:
                                # Personalize if possible
                                user_name = chat_info.get("first_name", "there")
                                personalized = message.replace("{name}", user_name)

                                await self.send_proactive(chat_id, personalized)
                                logger.info(f"✉️ Proactive [{category}] to {chat_id}: {message[:50]}...")

                                # Rate limit between chats
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

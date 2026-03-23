"""
Telegram Bot Integration for AURA

Uses python-telegram-bot library (async version).
Install: pip install python-telegram-bot>=20.0
"""

import asyncio
import hashlib
import logging
import os
import random
import re
import io
import tempfile
import time as _time
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json
import base64

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
    from telegram import Update, Bot, InlineQueryResultArticle, InputTextMessageContent
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        InlineQueryHandler,
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

# Max chars for code output before truncation in Telegram messages
_MAX_OUTPUT_CHARS = 3500


def _extract_code_from_message(text: str) -> str:
    """Extract Python code from a /code message.

    Supports:
      /code print("hello")
      /code ```python\\nprint("hello")\\n```
      /code ```\\nprint("hello")\\n```
    """
    # Remove the /code prefix (may include @botname)
    stripped = re.sub(r"^/code(@\w+)?\s*", "", text, count=1)

    # Check for fenced code block: ```python ... ``` or ``` ... ```
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", stripped, re.DOTALL)
    if match:
        return match.group(1).strip()

    return stripped.strip()


class _LRUCache:
    """Simple LRU cache using OrderedDict."""

    def __init__(self, maxsize: int = 50):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> "Optional[str]":
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: str):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


# ============================================================================
#  Telegram MarkdownV2 formatting utilities
# ============================================================================

# Characters that must be escaped in Telegram MarkdownV2 (outside code blocks)
_MARKDOWNV2_ESCAPE = re.compile(r'([_*\[\]()~`>#+\-=|{}.!\\])')

# Maximum Telegram message length
_TELEGRAM_MSG_LIMIT = 4096


def _escape_mdv2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    return _MARKDOWNV2_ESCAPE.sub(r'\\\1', text)


def format_telegram_response(text: str) -> List[str]:
    """Convert markdown text to Telegram MarkdownV2 and split into chunks.

    Handles code blocks, inline code, bold, italic, links.
    Escapes all MarkdownV2 special chars outside of formatting spans.
    Returns a list of strings, each <= 4096 chars, split at paragraph boundaries.
    """
    if not text:
        return [""]

    # --- Phase 1: Extract code blocks and inline code to protect them ---
    placeholders: Dict[str, str] = {}
    counter = [0]

    def _save_code_block(m: re.Match) -> str:
        key = f"\x00CB{counter[0]}\x00"
        counter[0] += 1
        lang = m.group(1) or ""
        code = m.group(2)
        placeholders[key] = f"```{lang}\n{code}\n```"
        return key

    def _save_inline_code(m: re.Match) -> str:
        key = f"\x00IC{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = f"`{m.group(1)}`"
        return key

    result = re.sub(r'```(\w*)\n?(.*?)```', _save_code_block, text, flags=re.DOTALL)
    result = re.sub(r'`([^`\n]+)`', _save_inline_code, result)

    # --- Phase 2: Extract links, bold, italic before escaping ---
    link_phs: Dict[str, str] = {}

    def _save_link(m: re.Match) -> str:
        key = f"\x00LK{counter[0]}\x00"
        counter[0] += 1
        link_text = _escape_mdv2(m.group(1))
        url = m.group(2)
        link_phs[key] = f"[{link_text}]({url})"
        return key

    result = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', _save_link, result)

    fmt_phs: Dict[str, str] = {}

    def _save_bold(m: re.Match) -> str:
        key = f"\x00BD{counter[0]}\x00"
        counter[0] += 1
        inner = _escape_mdv2(m.group(1))
        fmt_phs[key] = f"*{inner}*"
        return key

    def _save_italic(m: re.Match) -> str:
        key = f"\x00IT{counter[0]}\x00"
        counter[0] += 1
        inner = _escape_mdv2(m.group(1))
        fmt_phs[key] = f"_{inner}_"
        return key

    # Bold: **text** or __text__
    result = re.sub(r'\*\*(.+?)\*\*', _save_bold, result)
    result = re.sub(r'__(.+?)__', _save_bold, result)
    # Italic: *text* or _text_ (single, non-greedy)
    result = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', _save_italic, result)
    result = re.sub(r'(?<!_)_([^_]+?)_(?!_)', _save_italic, result)

    # --- Phase 3: Escape remaining text ---
    result = _escape_mdv2(result)

    # --- Phase 4: Restore placeholders (reverse order of extraction) ---
    for key, val in fmt_phs.items():
        result = result.replace(_escape_mdv2(key), val)
    for key, val in link_phs.items():
        result = result.replace(_escape_mdv2(key), val)
    for key, val in placeholders.items():
        result = result.replace(_escape_mdv2(key), val)

    # --- Phase 5: Split into <= 4096-char chunks ---
    return _split_message(result, _TELEGRAM_MSG_LIMIT)


def _split_message(text: str, limit: int) -> List[str]:
    """Split text into chunks of at most `limit` chars at paragraph breaks."""
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        # Try paragraph break, then newline, then space, then hard cut
        split_at = remaining.rfind('\n\n', 0, limit)
        if split_at == -1:
            split_at = remaining.rfind('\n', 0, limit)
        if split_at == -1:
            split_at = remaining.rfind(' ', 0, limit)
        if split_at == -1:
            split_at = limit

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip('\n')

    return chunks if chunks else [""]


def format_research_citations(text: str, sources: List[Dict]) -> str:
    """Format research results with numbered citations and clickable source list.

    Args:
        text: The research report body (may already contain [N] refs).
        sources: List of dicts with 'url' and 'title' keys.

    Returns:
        Formatted text with a numbered source list appended.
    """
    if not sources:
        return text

    # De-duplicate by URL
    seen_urls: set = set()
    unique: List[Dict] = []
    for s in sources:
        url = s.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(s)

    lines = ["\n\n---\n**Sources:**"]
    for i, src in enumerate(unique, 1):
        title = src.get("title", "Untitled")
        url = src.get("url", "")
        if url:
            lines.append(f"[{i}] [{title}]({url})")
        else:
            lines.append(f"[{i}] {title}")

    return text + "\n".join(lines)


# ============================================================================
#  TelegramProgressReporter — editable progress message with rate limiting
# ============================================================================

class TelegramProgressReporter:
    """Sends and edits a single progress message in a Telegram chat.

    Rate-limited to one edit every 2 seconds (Telegram API constraint).

    Usage:
        reporter = TelegramProgressReporter(bot, chat_id)
        await reporter.start("Working on it...")
        await reporter.update("Step 1/3...")
        await reporter.finish("Here's the answer.")
    """

    MIN_EDIT_INTERVAL = 2.0

    def __init__(self, bot, chat_id: str):
        self._bot = bot
        self._chat_id = chat_id
        self._message_id: Optional[int] = None
        self._last_edit_time: float = 0.0
        self._last_text: str = ""
        self._pending_update: Optional[str] = None
        self._flush_task: Optional[asyncio.Task] = None

    async def start(self, text: str = "\u23f3 Working...") -> int:
        """Send the initial progress message. Returns message_id."""
        try:
            msg = await self._bot.send_message(chat_id=self._chat_id, text=text)
            self._message_id = msg.message_id
            self._last_text = text
            self._last_edit_time = _time.time()
            return msg.message_id
        except Exception as e:
            logger.warning(f"[ProgressReporter] Could not send initial message: {e}")
            return 0

    async def update(self, text: str):
        """Edit the progress message. Rate-limited: queues if too fast."""
        if not self._message_id or text == self._last_text:
            return

        now = _time.time()
        elapsed = now - self._last_edit_time

        if elapsed >= self.MIN_EDIT_INTERVAL:
            await self._do_edit(text)
        else:
            self._pending_update = text
            if not self._flush_task or self._flush_task.done():
                delay = self.MIN_EDIT_INTERVAL - elapsed
                self._flush_task = asyncio.create_task(self._flush_after(delay))

    async def _flush_after(self, delay: float):
        """Wait then send the most recent pending update."""
        await asyncio.sleep(delay)
        if self._pending_update:
            text = self._pending_update
            self._pending_update = None
            await self._do_edit(text)

    async def _do_edit(self, text: str):
        """Actually edit the Telegram message."""
        try:
            await self._bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=self._message_id,
                text=text,
            )
            self._last_text = text
            self._last_edit_time = _time.time()
        except Exception as e:
            if "not modified" not in str(e).lower():
                logger.debug(f"[ProgressReporter] Edit failed: {e}")

    async def finish(self, text: str, parse_mode: Optional[str] = None):
        """Replace the progress message with the final response.

        If text > 4096 chars, deletes the progress message and sends
        the response as multiple new messages.
        """
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()

        if not self._message_id:
            return

        if len(text) <= _TELEGRAM_MSG_LIMIT:
            try:
                kwargs = {
                    "chat_id": self._chat_id,
                    "message_id": self._message_id,
                    "text": text,
                }
                if parse_mode:
                    kwargs["parse_mode"] = parse_mode
                await self._bot.edit_message_text(**kwargs)
            except Exception as e:
                logger.debug(f"[ProgressReporter] Final edit failed: {e}")
                try:
                    await self._bot.send_message(
                        chat_id=self._chat_id, text=text, parse_mode=parse_mode
                    )
                except Exception:
                    pass
        else:
            # Delete progress message, send multi-part
            try:
                await self._bot.delete_message(
                    chat_id=self._chat_id, message_id=self._message_id
                )
            except Exception:
                pass

            for chunk in _split_message(text, _TELEGRAM_MSG_LIMIT):
                try:
                    await self._bot.send_message(
                        chat_id=self._chat_id, text=chunk, parse_mode=parse_mode
                    )
                    await asyncio.sleep(0.3)
                except Exception as e:
                    logger.warning(f"[ProgressReporter] Chunk send failed: {e}")

    async def finish_parts(self, parts: List[str], parse_mode: Optional[str] = None):
        """Replace the progress message with pre-split parts."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        if not parts:
            return

        # Edit progress message with first part
        if self._message_id:
            try:
                kwargs = {
                    "chat_id": self._chat_id,
                    "message_id": self._message_id,
                    "text": parts[0],
                }
                if parse_mode:
                    kwargs["parse_mode"] = parse_mode
                await self._bot.edit_message_text(**kwargs)
            except Exception:
                try:
                    await self._bot.send_message(
                        chat_id=self._chat_id, text=parts[0], parse_mode=parse_mode
                    )
                except Exception:
                    pass

        for part in parts[1:]:
            try:
                await self._bot.send_message(
                    chat_id=self._chat_id, text=part, parse_mode=parse_mode
                )
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.warning(f"[ProgressReporter] Part send failed: {e}")


# ============================================================================
#  ToolStatusCallback — wires agent tool execution into progress updates
# ============================================================================

class ToolStatusCallback:
    """Provides on_tool_start / on_tool_end / on_research_progress callbacks.

    Wire into the agent's tool execution flow:
        reporter = TelegramProgressReporter(bot, chat_id)
        cb = ToolStatusCallback(reporter)
        # then pass cb.on_tool_start, cb.on_tool_end as callbacks
    """

    _TOOL_LABELS = {
        "web_search":     "\U0001f50d Searching the web",
        "deep_research":  "\U0001f4da Researching",
        "code_execute":   "\u26a1 Executing code",
        "code_run":       "\u26a1 Running code",
        "plot":           "\U0001f4ca Generating plot",
        "browse":         "\U0001f310 Browsing",
        "file_read":      "\U0001f4c4 Reading file",
        "file_write":     "\u270d\ufe0f Writing file",
        "memory_search":  "\U0001f9e0 Searching memory",
        "summarize":      "\U0001f4dd Summarizing",
    }

    def __init__(self, reporter: TelegramProgressReporter):
        self._reporter = reporter
        self._active_tools: List[str] = []

    def _label_for(self, tool_name: str) -> str:
        return self._TOOL_LABELS.get(tool_name, f"\U0001f527 Using {tool_name}")

    async def on_tool_start(self, tool_name: str, args: Optional[Dict] = None):
        """Called when a tool starts executing."""
        self._active_tools.append(tool_name)
        label = self._label_for(tool_name)

        detail = ""
        if args:
            if "query" in args:
                detail = f': "{args["query"][:60]}"'
            elif "url" in args:
                detail = f": {args['url'][:60]}"
            elif "topic" in args:
                detail = f': "{args["topic"][:60]}"'

        await self._reporter.update(f"{label}{detail}...")

    async def on_tool_end(self, tool_name: str, success: bool = True,
                          summary: Optional[str] = None):
        """Called when a tool finishes executing."""
        if tool_name in self._active_tools:
            self._active_tools.remove(tool_name)

        if summary:
            status = "\u2705" if success else "\u274c"
            await self._reporter.update(f"{status} {summary[:100]}")
        elif self._active_tools:
            current = self._label_for(self._active_tools[-1])
            await self._reporter.update(f"{current}...")
        else:
            await self._reporter.update("\U0001f4ad Thinking...")

    async def on_research_progress(self, event: Dict):
        """Handle ResearchProgressEmitter events.

        Pass this as callback to ResearchProgressEmitter:
            emitter = ResearchProgressEmitter(callback=
                lambda evt: asyncio.create_task(cb.on_research_progress(evt))
            )
        """
        data = event.get("data", {})
        stage = event.get("stage", "")

        if stage == "search":
            step = data.get("step", 0)
            total = data.get("total", 0)
            query = data.get("query", "")
            text = f"\U0001f50d Searching ({step}/{total}): {query[:60]}"
        elif stage == "plan":
            n = len(data.get("subtopics", []))
            text = f"\U0001f4cb Planning: {n} sub-queries"
        elif stage == "source":
            title = data.get("title", "")[:60]
            text = f"\U0001f4d6 Found: {title}"
        elif stage == "finding":
            text = f"\U0001f4a1 {data.get('message', 'New finding')[:80]}"
        elif stage == "synthesis":
            text = "\u270d\ufe0f Synthesizing report..."
        else:
            text = data.get("message", "\U0001f50d Researching...")[:100]

        await self._reporter.update(text)


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

        # Per-user document context for Q&A after upload
        # Maps user_id -> {"text": str, "filename": str, "timestamp": float}
        self._user_doc_context: Dict[int, dict] = {}
        self._DOC_CONTEXT_TTL = 30 * 60  # 30 minutes

        # Lazy-loaded code executor
        self._code_executor = None

        # Inline query state
        self._inline_cache = _LRUCache(maxsize=50)
        self._inline_last_query: Dict[int, Tuple[str, float]] = {}  # user_id -> (query, timestamp)
        self._INLINE_DEBOUNCE_SEC = 1.0
        self._INLINE_TIMEOUT_SEC = 15.0

        self._load_state()

    @property
    def platform_name(self) -> str:
        return "telegram"

    def _get_code_executor(self):
        """Lazy-load CodeExecutorTool to avoid import cost at startup."""
        if self._code_executor is None:
            from aura.tools.code_executor import CodeExecutorTool
            self._code_executor = CodeExecutorTool(timeout=30)
        return self._code_executor

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
        self.app.add_handler(CommandHandler("research", self._handle_research))
        self.app.add_handler(CommandHandler("search", self._handle_search))
        self.app.add_handler(CommandHandler("summarize", self._handle_summarize))
        self.app.add_handler(CommandHandler("image", self._handle_image))
        self.app.add_handler(CommandHandler("code", self._handle_code))
        self.app.add_handler(CommandHandler("model", self._handle_model))
        self.app.add_handler(CommandHandler("compare", self._handle_compare))

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
        await self.app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message", "inline_query", "chosen_inline_result"]
        )

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
/research <topic> - Deep research
/search <query> - Web search
/summarize - Summarize URL or text
/image <prompt> - Generate an image
/code <python> - Run Python code
/model - View/switch AI models
/compare <prompt> - Compare models side-by-side
/help - More info

Inline mode: Type @Aura828Bot in any chat to ask me anything!

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
/research <topic> - Deep research on any topic
/search <query> - Quick web search
/summarize - Summarize a URL or text (reply to a message or provide a URL)
/image <prompt> - Generate an image from a text description
/code <python> - Execute Python code (supports code blocks)
/model - View current models and available options
/model <name> - Switch the default model
/compare <prompt> - Run same prompt through 3 models side-by-side

Inline mode (use @Aura828Bot in any chat):
- Type any question to get a quick answer
- "translate <text>" - Translation
- "explain <topic>" - Explanation
- "summarize <text>" - Summary

Tips:
- Just chat normally - I'll respond naturally
- Say "remember this:" to save something important
- Reply to a /code message to run more code
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

    async def _handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /image <prompt> command — generate an image via ImageGenerationTool."""

        if not self._is_user_allowed(update.effective_user.id):
            return

        chat_id = str(update.effective_chat.id)

        # Extract prompt: everything after /image
        raw_text = update.message.text or ""
        prompt = raw_text.partition(" ")[2].strip()

        if not prompt:
            await update.message.reply_text(
                "Usage: /image <prompt>\n\n"
                "Example: /image a sunset over mountains in watercolor style"
            )
            return

        # Show upload_photo action while generating
        try:
            await self.bot.send_chat_action(
                chat_id=chat_id,
                action=ChatAction.UPLOAD_PHOTO
            )
        except Exception:
            pass

        await update.message.reply_text(
            f"Generating image for: {prompt}\nThis may take a moment..."
        )

        # Try to get the image_gen tool from the agent's loaded tools
        image_tool = None
        if hasattr(self.aura, 'tools') and isinstance(self.aura.tools, dict):
            image_tool = (
                self.aura.tools.get("image_gen")
                or self.aura.tools.get("image_generation")
            )

        # If no pre-loaded tool, try to instantiate ImageGenerationTool directly
        if image_tool is None:
            try:
                from aura.tools.image_gen import ImageGenerationTool
                image_tool = ImageGenerationTool()
                logger.info("ImageGenerationTool instantiated on-demand for /image command")
            except Exception as e:
                logger.warning(f"Could not load ImageGenerationTool: {e}")

        if image_tool is None:
            await update.message.reply_text(
                "Image generation is not available on this server. "
                "The image_gen tool could not be loaded (torch/diffusers may not be installed)."
            )
            return

        # Run generation in a thread to avoid blocking the event loop
        try:
            generate_fn = getattr(image_tool, 'generate', None)
            if generate_fn is None:
                await update.message.reply_text(
                    "Image generation tool found but has no generate() method."
                )
                return

            result = await asyncio.to_thread(generate_fn, prompt)
        except Exception as e:
            logger.error(f"Image generation failed: {e}", exc_info=True)
            await update.message.reply_text(f"Image generation failed: {e}")
            return

        if not result or not result.get("success"):
            error_msg = (
                result.get("error", "Unknown error") if result else "No result returned"
            )
            await update.message.reply_text(f"Image generation failed: {error_msg}")
            return

        # Send the image as a Telegram photo
        try:
            await self.bot.send_chat_action(
                chat_id=chat_id,
                action=ChatAction.UPLOAD_PHOTO
            )

            pil_image = result.get("image")
            image_path = result.get("image_path")

            if pil_image is not None:
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                buf.seek(0)
                await self.bot.send_photo(
                    chat_id=chat_id,
                    photo=buf,
                    caption=prompt[:1024],
                    reply_to_message_id=update.message.message_id
                )
            elif image_path and Path(image_path).exists():
                with open(image_path, "rb") as f:
                    await self.bot.send_photo(
                        chat_id=chat_id,
                        photo=f,
                        caption=prompt[:1024],
                        reply_to_message_id=update.message.message_id
                    )
            else:
                await update.message.reply_text(
                    "Image was generated but could not be retrieved. "
                    "Check server logs for details."
                )
        except Exception as e:
            logger.error(f"Failed to send generated image: {e}", exc_info=True)
            image_path = result.get("image_path")
            if image_path:
                await update.message.reply_text(
                    f"Image generated but failed to send: {e}\n"
                    f"Saved at: {image_path}"
                )
            else:
                await update.message.reply_text(
                    f"Image generated but failed to send: {e}"
                )

    # ============ CODE EXECUTION HANDLER ============

    async def _handle_code(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /code command — execute Python code and return results."""

        if not self._is_user_allowed(update.effective_user.id):
            return

        chat_id = str(update.effective_chat.id)
        text = update.message.text or ""
        code = _extract_code_from_message(text)

        if not code:
            await update.message.reply_text(
                "Usage: /code <python_code>\n\n"
                "Examples:\n"
                "  /code print('hello world')\n"
                "  /code 2 ** 100\n\n"
                "You can also use code blocks:\n"
                "  /code ```python\n"
                "  for i in range(5):\n"
                "      print(i)\n"
                "  ```"
            )
            return

        # Show typing while executing
        await self.send_typing_indicator(chat_id)

        # Store message id so replies can continue context
        context.chat_data["_last_code_msg_id"] = update.message.message_id
        context.chat_data["_last_code"] = code

        # Run in thread to avoid blocking the event loop
        try:
            executor = self._get_code_executor()
            result = await asyncio.to_thread(executor.execute, code)
        except Exception as e:
            logger.error(f"Code execution failed: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Execution failed: {e}")
            return

        await self._send_code_result(update, result)

    async def _send_code_result(self, update: Update, result: dict):
        """Format and send code execution results back to the user."""
        chat_id = update.effective_chat.id

        success = result.get("success", False)
        output = result.get("output", "")
        error = result.get("error", "") or result.get("errors", "")

        # Check for base64 plot data (from E2B tier or future extensions)
        plot_data = result.get("plot_base64") or result.get("image")
        if plot_data:
            try:
                img_bytes = base64.b64decode(plot_data)
                await self.bot.send_photo(
                    chat_id=chat_id,
                    photo=io.BytesIO(img_bytes),
                    reply_to_message_id=update.message.message_id,
                )
            except Exception as e:
                logger.warning(f"Could not send plot image: {e}")

        if success:
            if output:
                truncated = False
                if len(output) > _MAX_OUTPUT_CHARS:
                    output = output[:_MAX_OUTPUT_CHARS]
                    truncated = True

                msg = f"<pre>{self._escape_html(output)}</pre>"
                if truncated:
                    msg += "\n\n(output truncated)"

                try:
                    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
                except Exception:
                    # Fallback to plain text if HTML parsing fails
                    plain = output
                    if truncated:
                        plain += "\n\n(output truncated)"
                    await update.message.reply_text(plain)
            elif not plot_data:
                await update.message.reply_text("Code executed successfully (no output).")
        else:
            # Error case
            err_msg = str(error) if error else "Unknown error"
            truncated = False
            if len(err_msg) > _MAX_OUTPUT_CHARS:
                err_msg = err_msg[:_MAX_OUTPUT_CHARS]
                truncated = True

            text = f"Error:\n<pre>{self._escape_html(err_msg)}</pre>"
            if truncated:
                text += "\n\n(error output truncated)"

            # Include partial stdout if any
            if output:
                partial = output[:1000]
                text = f"<pre>{self._escape_html(partial)}</pre>\n\n" + text

            try:
                await update.message.reply_text(text, parse_mode=ParseMode.HTML)
            except Exception:
                plain = f"Error:\n{err_msg}"
                if truncated:
                    plain += "\n\n(error output truncated)"
                await update.message.reply_text(plain)

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters for Telegram HTML parse mode."""
        return (
            text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    # ============ RESEARCH / SEARCH / SUMMARIZE ============

    @staticmethod
    def _split_message(text: str, limit: int = 4096) -> list:
        """Split a long message into chunks respecting Telegram's char limit."""
        if len(text) <= limit:
            return [text]
        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break
            split_at = text.rfind("\n", 0, limit)
            if split_at < limit // 2:
                split_at = text.rfind(" ", 0, limit)
            if split_at < limit // 4:
                split_at = limit
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        return chunks

    async def _handle_research(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /research <topic> -- deep multi-phase research pipeline."""
        user = update.effective_user
        if not self._is_user_allowed(user.id):
            return

        chat_id = str(update.effective_chat.id)
        raw_text = update.message.text or ""
        topic = raw_text.partition(" ")[2].strip()
        if not topic:
            await update.message.reply_text(
                "Usage: /research <topic>\n\n"
                "Example: /research quantum computing breakthroughs 2025"
            )
            return

        # Editable progress message
        status_msg = await update.message.reply_text(
            f"Researching: {topic}\n\nSearching the web..."
        )
        await self.send_typing_indicator(chat_id)

        # Progress callback -- edits the status message in-place
        progress_lines = [f"Researching: {topic}"]
        last_edit_time = [_time.time()]

        async def _update_status(line: str):
            now = _time.time()
            if now - last_edit_time[0] < 2.0:
                return
            last_edit_time[0] = now
            progress_lines.append(line)
            display = progress_lines[-6:]
            try:
                await status_msg.edit_text("\n".join(display))
                await self.send_typing_indicator(chat_id)
            except Exception:
                pass

        def _sync_progress(msg: str):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(_update_status(msg))
            except Exception:
                pass

        try:
            from aura.tools.deep_research import DeepResearchTool

            tool = DeepResearchTool()
            tool.set_progress_callback(_sync_progress)

            result = await asyncio.to_thread(tool.research, topic, "standard")

            # Needs clarification?
            if result.get("needs_clarification"):
                await status_msg.edit_text(
                    f"Could you be more specific?\n\n{result.get('question', '')}"
                )
                return

            if not result.get("success"):
                await status_msg.edit_text(
                    f"Research failed: {result.get('error', 'Unknown error')}"
                )
                return

            # Build report
            parts = [
                f"Research: {topic}",
                f"({result.get('pages_read', 0)} sources, "
                f"{result.get('time_seconds', 0):.0f}s)",
                "",
            ]

            synthesis = (
                result.get("synthesis")
                or result.get("summary")
                or result.get("content", "")
            )
            if synthesis:
                parts.append(synthesis)
            else:
                for ps in result.get("page_summaries", [])[:8]:
                    title = ps.get("title", "Source")
                    body = ps.get("summary", ps.get("content", ""))[:300]
                    parts.append(f"- {title}: {body}")

            contradictions = result.get("contradictions", [])
            if contradictions:
                parts.append("")
                parts.append("Contradictions found:")
                for c in contradictions[:3]:
                    parts.append(f"  - {c}")

            # Clickable source citations
            sources = result.get("sources", [])
            if sources:
                parts.append("")
                parts.append("Sources:")
                seen = set()
                for s in sources[:10]:
                    url = s if isinstance(s, str) else s.get("url", "")
                    title = s.get("title", url) if isinstance(s, dict) else url
                    if url and url not in seen:
                        seen.add(url)
                        parts.append(f"  {title}\n  {url}")

            report = "\n".join(parts)

            try:
                await status_msg.delete()
            except Exception:
                pass

            for chunk in self._split_message(report):
                await update.message.reply_text(chunk)

        except ImportError:
            await status_msg.edit_text(
                "Deep research tool not available. "
                "Check that aura.tools.deep_research is installed."
            )
        except Exception as e:
            logger.error(f"Research command error: {e}", exc_info=True)
            try:
                await status_msg.edit_text(f"Research error: {str(e)[:200]}")
            except Exception:
                await update.message.reply_text(f"Research error: {str(e)[:200]}")

    async def _handle_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /search <query> -- quick web search via SearXNG."""
        user = update.effective_user
        if not self._is_user_allowed(user.id):
            return

        chat_id = str(update.effective_chat.id)
        raw_text = update.message.text or ""
        query = raw_text.partition(" ")[2].strip()
        if not query:
            await update.message.reply_text(
                "Usage: /search <query>\n\n"
                "Example: /search best Python async frameworks"
            )
            return

        await self.send_typing_indicator(chat_id)

        try:
            from aura.tools.web_search import WebSearchTool

            tool = WebSearchTool()
            result = await asyncio.to_thread(tool.search, query, 8)

            if not result.get("success"):
                await update.message.reply_text(
                    f"Search failed: {result.get('error', 'Unknown error')}"
                )
                return

            results = result.get("results", [])
            if not results:
                await update.message.reply_text(f"No results found for: {query}")
                return

            lines = [f"Search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "Untitled")
                snippet = r.get("snippet", "")[:150]
                url = r.get("url", "")
                lines.append(f"{i}. {title}")
                if snippet:
                    lines.append(f"   {snippet}")
                if url:
                    lines.append(f"   {url}")
                lines.append("")

            search_msg = "\n".join(lines)
            for chunk in self._split_message(search_msg):
                await update.message.reply_text(chunk)

        except ImportError:
            await update.message.reply_text("Web search tool is not available.")
        except Exception as e:
            logger.error(f"Search command error: {e}", exc_info=True)
            await update.message.reply_text(f"Search error: {str(e)[:200]}")

    async def _handle_summarize(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /summarize -- summarize a URL or replied-to text."""
        user = update.effective_user
        if not self._is_user_allowed(user.id):
            return

        chat_id = str(update.effective_chat.id)
        raw_text = (update.message.text or "").partition(" ")[2].strip()
        reply = update.message.reply_to_message

        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        target_url = None
        target_text = None

        # Priority 1: URL in command text
        url_match = url_pattern.search(raw_text)
        if url_match:
            target_url = url_match.group(0)
        # Priority 2: Reply to a message with URL
        elif reply and reply.text:
            reply_match = url_pattern.search(reply.text)
            if reply_match:
                target_url = reply_match.group(0)
            elif len(reply.text) > 50:
                # Priority 3: Reply to long text
                target_text = reply.text
            else:
                await update.message.reply_text(
                    "The replied message is too short to summarize. "
                    "Reply to a longer message or provide a URL."
                )
                return
        else:
            await update.message.reply_text(
                "Usage:\n"
                "  /summarize <url> - Summarize a webpage\n"
                "  Reply to a message with /summarize - Summarize that text\n\n"
                "Example: /summarize https://example.com/article"
            )
            return

        await self.send_typing_indicator(chat_id)

        try:
            if target_url:
                status_msg = await update.message.reply_text(
                    f"Fetching: {target_url[:60]}..."
                )
                page_text = await self._fetch_url_content(target_url)
                if not page_text:
                    await status_msg.edit_text(
                        "Could not fetch the page content. "
                        "The site may be blocking requests."
                    )
                    return
                target_text = page_text[:8000]
                try:
                    await status_msg.edit_text("Summarizing...")
                except Exception:
                    pass
            else:
                status_msg = await update.message.reply_text("Summarizing...")

            await self.send_typing_indicator(chat_id)

            prompt = (
                "Summarize the following content concisely using bullet points. "
                "Focus on key facts, main arguments, and important details. "
                "Keep it under 500 words.\n\n"
                f"Content:\n{target_text}"
            )

            summary = await self._process_with_aura(prompt, str(user.id))

            if not summary or summary.startswith("Sorry, I couldn't"):
                # Extractive fallback
                sentences = [
                    s.strip() for s in target_text.split(".")
                    if len(s.strip()) > 30
                ]
                summary = "Key excerpts:\n\n" + "\n".join(
                    f"- {s.strip()}." for s in sentences[:8]
                )

            final_parts = []
            if target_url:
                final_parts.append(f"Summary of: {target_url}\n")
            final_parts.append(summary)
            final = "\n".join(final_parts)

            try:
                await status_msg.delete()
            except Exception:
                pass

            for chunk in self._split_message(final):
                await update.message.reply_text(chunk)

        except Exception as e:
            logger.error(f"Summarize command error: {e}", exc_info=True)
            await update.message.reply_text(
                f"Summarization error: {str(e)[:200]}"
            )

    async def _fetch_url_content(self, url: str):
        """Fetch and extract text content from a URL."""
        try:
            import requests

            resp = await asyncio.to_thread(
                lambda: requests.get(
                    url,
                    timeout=15,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36"
                        ),
                    },
                )
            )
            if resp.status_code != 200:
                return None

            html = resp.text

            # Prefer trafilatura for clean text extraction
            try:
                import trafilatura
                extracted = trafilatura.extract(html)
                if extracted and len(extracted) > 100:
                    return extracted
            except ImportError:
                pass

            # Fallback: strip HTML tags
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text if len(text) > 50 else None

        except Exception as e:
            logger.warning(f"Failed to fetch URL {url}: {e}")
            return None

    # ============ MODEL COMMANDS ============

    def _get_brain(self):
        """Get the OllamaBrain instance from the agent wrapper."""
        agent = getattr(self.aura, 'agent', None)
        if agent:
            return getattr(agent, 'brain', None)
        return getattr(self.aura, 'brain', None)

    async def _handle_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /model command -- list or switch models."""

        if not self._is_user_allowed(update.effective_user.id):
            return

        raw_text = update.message.text or ""
        arg = raw_text.partition(" ")[2].strip()

        try:
            from aura.config import Config, VERIFIED_CLOUD_MODELS
        except ImportError:
            await update.message.reply_text("Config module not available.")
            return

        if not arg:
            current = Config.get_all_models()
            roles_display = {
                'fast': 'Fast',
                'reason': 'Reason',
                'code': 'Code',
                'vision': 'Vision',
                'think': 'Think',
                'longctx': 'Long Context',
            }

            lines = ["Current Models:\n"]
            for role, display_name in roles_display.items():
                model = current.get(role, 'unknown')
                lines.append(f"  {display_name}: {model}")

            lines.append("\nAvailable Cloud Models:\n")
            current_values = set(current.values())
            for m in sorted(VERIFIED_CLOUD_MODELS):
                marker_str = " [active]" if m in current_values else ""
                lines.append(f"  {m}{marker_str}")

            lines.append("\nUsage:\n  /model <name> - switch default model\n  /model auto - return to auto-routing")

            await update.message.reply_text("\n".join(lines))
            return

        brain = self._get_brain()

        if arg.lower() == "auto":
            if brain:
                brain.set_model_override(None)
            await update.message.reply_text("Switched to auto-routing.")
            return

        all_known = set(VERIFIED_CLOUD_MODELS)
        for chain_attr in ['MODEL_FAST_CHAIN', 'MODEL_REASON_CHAIN', 'MODEL_CODE_CHAIN',
                           'MODEL_VISION_CHAIN', 'MODEL_THINK_CHAIN', 'MODEL_LONGCTX_CHAIN']:
            all_known.update(getattr(Config, chain_attr, []))

        if arg not in all_known:
            matches = [m for m in sorted(all_known) if arg.lower() in m.lower()]
            if len(matches) == 1:
                arg = matches[0]
            elif matches:
                match_list = "\n".join(f"  {m}" for m in matches)
                await update.message.reply_text(
                    f"Multiple matches for '{arg}':\n{match_list}\n\nBe more specific."
                )
                return
            else:
                await update.message.reply_text(
                    f"Model '{arg}' not found.\n\nUse /model to see available models."
                )
                return

        if brain:
            brain.set_model_override(arg)

        await update.message.reply_text(f"Switched to {arg}")

    async def _handle_compare(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare <prompt> -- run same prompt through 3 models in parallel."""

        if not self._is_user_allowed(update.effective_user.id):
            return

        chat_id = str(update.effective_chat.id)
        raw_text = update.message.text or ""
        prompt = raw_text.partition(" ")[2].strip()

        if not prompt:
            await update.message.reply_text(
                "Usage: /compare <prompt>\n\n"
                "Example: /compare What causes auroras?"
            )
            return

        try:
            from aura.config import Config
        except ImportError:
            await update.message.reply_text("Config module not available.")
            return

        brain = self._get_brain()
        if not brain:
            await update.message.reply_text("Brain not available.")
            return

        # Pick 3 distinct models: fast, reason, code (fall back to think, longctx)
        candidates = [
            ("Fast", Config.get_model('fast')),
            ("Reason", Config.get_model('reason')),
            ("Code", Config.get_model('code')),
            ("Think", Config.get_model('think')),
            ("Long Context", Config.get_model('longctx')),
        ]
        models = []
        seen = set()
        for label, model in candidates:
            if model not in seen and len(models) < 3:
                seen.add(model)
                models.append((label, model))

        await self.send_typing_indicator(chat_id)
        model_names = ", ".join(m for _, m in models)
        status_msg = await update.message.reply_text(
            f"Comparing {len(models)} models: {model_names}\nRunning..."
        )

        def _call_model(label, model_name):
            start = _time.time()
            try:
                client, actual_model = brain._get_client_for_model(model_name)
                response = client.chat(
                    model=actual_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                if response is None:
                    text = "(no response)"
                elif isinstance(response, dict):
                    msg = response.get("message", {})
                    text = msg.get("content", "") if isinstance(msg, dict) else ""
                else:
                    msg = getattr(response, "message", None)
                    text = getattr(msg, "content", "") if msg else ""
                if not text:
                    text = "(empty response)"
            except Exception as e:
                text = f"(error: {e})"
            elapsed = _time.time() - start
            return label, text, elapsed

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                loop.run_in_executor(pool, _call_model, label, model_name)
                for label, model_name in models
            ]
            results = await asyncio.gather(*futures, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                await self.bot.send_message(chat_id=chat_id, text=f"Error: {result}")
                continue

            label, response_text, elapsed = result
            model_name = next((m for l, m in models if l == label), "unknown")

            max_len = 3800
            if len(response_text) > max_len:
                response_text = response_text[:max_len] + "\n\n[...truncated]"

            header = f"--- {label} ({model_name}) ---\nTime: {elapsed:.1f}s\n\n"
            await self.bot.send_message(chat_id=chat_id, text=header + response_text)

        try:
            await status_msg.edit_text(f"Comparison complete. {len(models)} models responded.")
        except Exception:
            pass

    # ============ MESSAGE HANDLER ============

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages through the full ReAct agent loop."""

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

        # Check if this is a reply to a /code message — treat as continuation code
        reply = update.message.reply_to_message
        if reply and context.chat_data.get("_last_code_msg_id") == reply.message_id:
            code = text.strip()
            if code:
                await self.send_typing_indicator(chat_id)
                context.chat_data["_last_code_msg_id"] = update.message.message_id
                context.chat_data["_last_code"] = code
                try:
                    executor = self._get_code_executor()
                    result = await asyncio.to_thread(executor.execute, code)
                except Exception as e:
                    logger.error(f"Code execution (reply) failed: {e}", exc_info=True)
                    await update.message.reply_text(f"❌ Execution failed: {e}")
                    return
                await self._send_code_result(update, result)
                return

        # Check for forget confirmation
        if text and text.lower() == "yes forget everything":
            user_id = str(user.id)

            # Check if there's a pending forget request (within 5 minutes)
            if hasattr(self, '_pending_forget') and user_id in self._pending_forget:
                request_time = self._pending_forget[user_id]
                if (datetime.now() - request_time).total_seconds() < 300:  # 5 min
                    cleared_items = []
                    try:
                        if hasattr(self.aura, 'memory') and self.aura.memory:
                            if hasattr(self.aura.memory, 'clear'):
                                self.aura.memory.clear()
                                cleared_items.append("Conversation memory")
                        if hasattr(self.aura, 'state') and hasattr(self.aura.state, '_history'):
                            self.aura.state._history.clear()
                            cleared_items.append("State history")
                        if hasattr(self, '_user_doc_context') and int(user_id) in self._user_doc_context:
                            del self._user_doc_context[int(user_id)]
                            cleared_items.append("Document context")
                        del self._pending_forget[user_id]
                        if cleared_items:
                            cleared_list = "\n".join(f"- {item}" for item in cleared_items)
                            await update.message.reply_text(
                                f"Memory cleared successfully!\n\nCleared:\n{cleared_list}\n\n"
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
                    del self._pending_forget[user_id]

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

        # Prepend document context if the user has an active uploaded document
        if hasattr(self, '_build_doc_augmented_text'):
            text = self._build_doc_augmented_text(user.id, text)

        # Route through the full agent loop with typing indicator + file artifacts
        await self._run_agent_and_reply(update, text)

        self._save_state()

    # ============ PHOTO / VISION HANDLERS ============

    # Image extensions recognized when sent as document attachments
    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

    # Text/code file extensions for document upload Q&A
    _TEXT_EXTENSIONS = {
        ".txt", ".csv", ".json", ".py", ".md", ".js", ".ts", ".jsx", ".tsx",
        ".html", ".css", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".sh", ".bash", ".sql", ".log", ".env", ".rs", ".go", ".java",
        ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
    }

    _MAX_DOC_SIZE = 20 * 1024 * 1024  # 20 MB

    def _analyze_image_sync(self, img_b64: str, prompt: str) -> str:
        """Analyze a base64-encoded image with the vision model (synchronous).

        Uses VisionTool with its full fallback chain. Meant to be called
        via asyncio.to_thread from async handlers.
        """
        try:
            from aura.tools.vision import VisionTool

            # Reuse brain's Ollama client if available
            brain = getattr(self.aura, 'brain', None)
            if brain is None:
                brain = getattr(getattr(self.aura, 'agent', None), 'brain', None)

            vt = VisionTool(brain=brain)
            content, model_used = vt._analyze_with_fallback(img_b64, prompt)
            logger.info(f"[TelegramBot] Vision analysis done with {model_used}")
            return content
        except Exception as e:
            logger.error(f"[TelegramBot] Vision analysis failed: {e}", exc_info=True)
            return "Sorry, I couldn't analyze that image right now. The vision model may be unavailable."

    async def _handle_photo_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle photos sent as compressed Telegram images -- analyze with vision model."""

        user = update.effective_user
        chat_id = str(update.effective_chat.id)

        if not self._is_user_allowed(user.id):
            return

        max_per_min = self.config.get("max_messages_per_minute", 20)
        if not _check_rate_limit(str(user.id), max_per_min):
            await update.message.reply_text("You're sending messages too fast. Please wait a moment.")
            return

        # Show typing while we download and process
        await self.send_typing_indicator(chat_id)

        # Get highest resolution photo (last in the list)
        photo = update.message.photo[-1]
        prompt = update.message.caption or "Describe this image in detail."

        # Download photo bytes from Telegram
        try:
            tg_file = await context.bot.get_file(photo.file_id)
            buf = io.BytesIO()
            await tg_file.download_to_memory(buf)
            image_bytes = buf.getvalue()
        except Exception as e:
            logger.error(f"[TelegramBot] Failed to download photo: {e}")
            await update.message.reply_text("Couldn't download the photo. Please try again.")
            return

        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Refresh typing -- vision models can take a while
        await self.send_typing_indicator(chat_id)

        # Run synchronous vision analysis in a thread
        response = await asyncio.to_thread(self._analyze_image_sync, img_b64, prompt)

        await update.message.reply_text(response)
        self._save_state()

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document uploads -- extract text for PDFs/text files, route images to vision."""
        user = update.effective_user
        chat_id = str(update.effective_chat.id)

        if not self._is_user_allowed(user.id):
            return

        max_per_min = self.config.get("max_messages_per_minute", 20)
        if not _check_rate_limit(str(user.id), max_per_min):
            await update.message.reply_text(
                "You're sending messages too fast. Please wait a moment."
            )
            return

        doc = update.message.document
        if not doc:
            return

        filename = doc.file_name or "unknown"
        file_size = doc.file_size or 0
        mime = doc.mime_type or ""
        ext = os.path.splitext(filename)[1].lower()

        # Size guard
        if file_size > self._MAX_DOC_SIZE:
            await update.message.reply_text(
                f"That file is too large ({file_size / (1024*1024):.1f} MB). "
                f"I can handle files up to 20 MB."
            )
            return

        await self.send_typing_indicator(chat_id)

        try:
            # --- PDF ---
            if ext == ".pdf" or mime == "application/pdf":
                text, char_count = await self._extract_pdf(doc)
                self._store_doc_context(user.id, text, filename)
                await update.message.reply_text(
                    f"\U0001f4c4 Extracted {char_count:,} characters from {filename}.\n"
                    f"Ask me anything about this document!"
                )

            # --- Text / code files ---
            elif ext in self._TEXT_EXTENSIONS or mime.startswith("text/"):
                text, char_count = await self._extract_text_file(doc)
                self._store_doc_context(user.id, text, filename)
                await update.message.reply_text(
                    f"\U0001f4c4 Read {char_count:,} characters from {filename}.\n"
                    f"Ask me anything about this file!"
                )

            # --- Images sent as documents ---
            elif ext in self._IMAGE_EXTENSIONS or mime.startswith("image/"):
                prompt = update.message.caption or "Describe this image in detail."
                try:
                    tg_file = await context.bot.get_file(doc.file_id)
                    buf = io.BytesIO()
                    await tg_file.download_to_memory(buf)
                    image_bytes = buf.getvalue()
                except Exception as e:
                    logger.error(f"[TelegramBot] Failed to download image document: {e}")
                    await update.message.reply_text(
                        "Couldn't download the image. Please try again."
                    )
                    return

                img_b64 = base64.b64encode(image_bytes).decode("utf-8")
                await self.send_typing_indicator(chat_id)
                response = await asyncio.to_thread(self._analyze_image_sync, img_b64, prompt)
                await update.message.reply_text(response)
                self._save_state()
                return

            # --- Unsupported ---
            else:
                await update.message.reply_text(
                    f"I can't read .{ext or '?'} files yet. "
                    f"I support PDFs, text/code files, and images."
                )

        except Exception as e:
            logger.error(f"Document handler error for {filename}: {e}", exc_info=True)
            await update.message.reply_text(
                f"Sorry, I couldn't process {filename}. "
                f"The file might be corrupted or in an unsupported format."
            )

    async def _extract_pdf(self, doc) -> "Tuple[str, int]":
        """Download a Telegram document and extract PDF text.

        Tries PyMuPDF (fitz) first, falls back to pdfplumber.
        Returns (extracted_text, char_count).
        """
        tg_file = await doc.get_file()
        file_bytes = await tg_file.download_as_bytearray()

        def _extract_with_fitz(raw: bytes) -> str:
            import fitz  # PyMuPDF
            text_parts = []
            with fitz.open(stream=raw, filetype="pdf") as pdf_doc:
                for page in pdf_doc:
                    text_parts.append(page.get_text())
            return "\n\n".join(text_parts)

        def _extract_with_pdfplumber(raw: bytes) -> str:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                for page in pdf.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n\n".join(text_parts)

        raw = bytes(file_bytes)
        text = ""

        for extractor, name in [
            (_extract_with_fitz, "PyMuPDF"),
            (_extract_with_pdfplumber, "pdfplumber"),
        ]:
            try:
                text = await asyncio.to_thread(extractor, raw)
                if text.strip():
                    logger.info(f"PDF extracted with {name}: {len(text)} chars")
                    break
            except ImportError:
                continue
            except Exception as e:
                logger.warning(f"PDF extraction with {name} failed: {e}")
                continue

        if not text.strip():
            raise ValueError(
                "Could not extract text -- the PDF may be image-based or empty."
            )

        if len(text) > 80_000:
            text = text[:80_000] + "\n\n[... truncated at 80,000 characters ...]"

        return text, len(text)

    async def _extract_text_file(self, doc) -> "Tuple[str, int]":
        """Download a Telegram document and read it as UTF-8 text."""
        tg_file = await doc.get_file()
        file_bytes = await tg_file.download_as_bytearray()

        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")

        if len(text) > 80_000:
            text = text[:80_000] + "\n\n[... truncated at 80,000 characters ...]"

        return text, len(text)

    # ============ DOCUMENT CONTEXT HELPERS ============

    def _store_doc_context(self, user_id: int, text: str, filename: str):
        """Store extracted document text for subsequent Q&A."""
        self._user_doc_context[user_id] = {
            "text": text,
            "filename": filename,
            "timestamp": _time.time(),
        }
        logger.info(
            f"Stored doc context for user {user_id}: {filename} ({len(text)} chars)"
        )

    def _get_doc_context(self, user_id: int):
        """Get active document context, or None if expired/missing."""
        ctx = self._user_doc_context.get(user_id)
        if not ctx:
            return None
        if _time.time() - ctx["timestamp"] > self._DOC_CONTEXT_TTL:
            del self._user_doc_context[user_id]
            logger.info(f"Doc context expired for user {user_id}")
            return None
        return ctx

    def _build_doc_augmented_text(self, user_id: int, text: str) -> str:
        """If the user has active document context, prepend it to their message."""
        ctx = self._get_doc_context(user_id)
        if not ctx:
            return text

        doc_text = ctx["text"]
        if len(doc_text) > 30_000:
            doc_text = doc_text[:30_000] + "\n[... document truncated ...]"

        return (
            f"[DOCUMENT CONTEXT -- file: {ctx['filename']}]\n"
            f"{doc_text}\n"
            f"[END DOCUMENT CONTEXT]\n\n"
            f"User question: {text}"
        )


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

    # ============ VOICE MESSAGE HANDLER ============

    async def _handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages and audio files — transcribe and respond."""

        user = update.effective_user
        chat_id = str(update.effective_chat.id)

        if not self._is_user_allowed(user.id):
            return

        # Rate limit same as text
        max_per_min = self.config.get("max_messages_per_minute", 20)
        if not _check_rate_limit(str(user.id), max_per_min):
            await update.message.reply_text(
                "You're sending messages too fast. Please wait a moment."
            )
            return

        # Get the voice or audio file object
        voice = update.message.voice
        audio = update.message.audio

        if voice:
            file_id = voice.file_id
            duration = voice.duration or 0
            suffix = ".ogg"
        elif audio:
            file_id = audio.file_id
            duration = audio.duration or 0
            mime = audio.mime_type or ""
            ext_map = {
                "audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
                "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/flac": ".flac",
                "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/webm": ".webm",
            }
            suffix = ext_map.get(mime, ".ogg")
        else:
            await update.message.reply_text(
                "Could not read the audio. Try sending as a voice message."
            )
            return

        # Reject very long audio (>5 minutes)
        if duration > 300:
            await update.message.reply_text(
                "That audio is too long (max 5 minutes for voice messages). "
                "Try sending a shorter clip."
            )
            return

        # Show typing while we process
        await self.send_typing_indicator(chat_id)

        # Download the file from Telegram
        tmp_path = None
        try:
            tg_file = await context.bot.get_file(file_id)
            buf = io.BytesIO()
            await tg_file.download_to_memory(buf)
            buf.seek(0)

            # Write to temp file for Whisper
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(buf.read())

            logger.info(f"[Voice] Downloaded {suffix} from user {user.id} ({duration}s)")

        except Exception as e:
            logger.error(f"[Voice] Failed to download voice file: {e}")
            await update.message.reply_text(
                "I couldn't download that voice message. Please try again."
            )
            return

        # Transcribe
        transcription = await self._transcribe_audio_file(tmp_path)

        # Clean up temp file
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass

        if not transcription:
            await update.message.reply_text(
                "I couldn't transcribe that voice message. "
                "Transcription isn't available right now — try typing your message instead."
            )
            return

        # Send transcription as italic quote
        try:
            await update.message.reply_text(
                f"_You said: {transcription}_",
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception:
            # Fallback without markdown if special chars break it
            await update.message.reply_text(f"You said: {transcription}")

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

        # Show typing while AURA thinks
        await self.send_typing_indicator(chat_id)

        # Process transcribed text through AURA (same as if user typed it)
        incoming = IncomingMessage(
            platform="telegram",
            user_id=str(user.id),
            chat_id=chat_id,
            username=user.username,
            display_name=user.first_name,
            message_type=MessageType.VOICE,
            text=transcription,
            media_url=None,
            timestamp=datetime.now(),
            raw_message=update.message
        )

        response = await self.handle_incoming(incoming)

        if response:
            await update.message.reply_text(response)

        self._save_state()

    async def _transcribe_audio_file(self, file_path: str) -> Optional[str]:
        """Transcribe an audio file. Tries in priority order:
        1. Local Whisper via AudioTranscriberTool
        2. /api/transcribe endpoint on the backend
        3. None (caller shows error)
        """
        loop = asyncio.get_running_loop()

        # --- Method 1: Local Whisper via AudioTranscriberTool ---
        try:
            from aura.tools.audio_transcriber import AudioTranscriberTool
            transcriber = AudioTranscriberTool()

            result = await loop.run_in_executor(
                None, lambda: transcriber.transcribe(file_path)
            )

            if result.get("success") and result.get("text", "").strip():
                text = result["text"].strip()
                logger.info(f"[Voice] Whisper transcribed: {text[:80]}...")
                return text
            else:
                logger.warning(
                    f"[Voice] Whisper returned no text: {result.get('error', 'empty')}"
                )
        except ImportError:
            logger.info("[Voice] Whisper not installed, trying API fallback")
        except Exception as e:
            logger.warning(f"[Voice] Whisper transcription failed: {e}")

        # --- Method 2: Backend /api/transcribe endpoint ---
        try:
            import aiohttp

            api_url = os.environ.get("AURA_API_URL", "http://127.0.0.1:8000")
            api_key = os.environ.get("AURA_API_KEY", "")

            headers = {}
            if api_key:
                headers["x-api-key"] = api_key

            async with aiohttp.ClientSession() as session:
                with open(file_path, "rb") as f:
                    form = aiohttp.FormData()
                    form.add_field(
                        "file", f,
                        filename=os.path.basename(file_path),
                        content_type="audio/ogg"
                    )
                    async with session.post(
                        f"{api_url}/api/transcribe",
                        data=form,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            text = data.get("text", "").strip()
                            if text:
                                logger.info(f"[Voice] API transcribed: {text[:80]}...")
                                return text
                        else:
                            body = await resp.text()
                            logger.warning(
                                f"[Voice] API transcribe returned {resp.status}: {body[:200]}"
                            )
        except ImportError:
            logger.info("[Voice] aiohttp not installed, skipping API fallback")
        except Exception as e:
            logger.warning(f"[Voice] API transcription failed: {e}")

        # All methods exhausted
        logger.warning("[Voice] All transcription methods failed")
        return None


    # ============ INLINE QUERY HANDLER ============

    # Quick-action prefixes: keyword -> (system_prompt_prefix, result_title_prefix)
    _INLINE_ACTIONS = {
        "translate": (
            "Translate the following text. Auto-detect the source language. "
            "If the text is in English, translate to Russian. Otherwise translate to English. "
            "Only output the translation, nothing else:\n\n",
            "Translate"
        ),
        "explain": (
            "Give a clear, concise explanation of the following topic. "
            "Keep it under 200 words:\n\n",
            "Explain"
        ),
        "summarize": (
            "Summarize the following text in 2-3 sentences. Be concise:\n\n",
            "Summarize"
        ),
    }

    async def _handle_inline(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline queries from any chat via @Aura828Bot <query>."""

        inline_query = update.inline_query
        if not inline_query:
            return

        user = inline_query.from_user
        query = (inline_query.query or "").strip()
        user_id = user.id

        # Auth check
        if not self._is_user_allowed(user_id):
            await inline_query.answer([], cache_time=60)
            return

        # Ignore empty or too-short queries
        if len(query) < 3:
            await inline_query.answer([], cache_time=5)
            return

        # Debounce: skip if the user sent another query within the debounce window
        now = _time.time()
        prev = self._inline_last_query.get(user_id)
        self._inline_last_query[user_id] = (query, now)

        if prev and prev[0] != query:
            elapsed = now - prev[1]
            if elapsed < self._INLINE_DEBOUNCE_SEC:
                await asyncio.sleep(self._INLINE_DEBOUNCE_SEC - elapsed)
                current = self._inline_last_query.get(user_id)
                if current and current[0] != query:
                    return

        # Build result list
        results = []

        # Detect quick-action prefix
        query_lower = query.lower()
        detected_action = None
        action_body = query

        for prefix, (prompt_prefix, title_prefix) in self._INLINE_ACTIONS.items():
            if query_lower.startswith(prefix + " ") and len(query) > len(prefix) + 1:
                detected_action = prefix
                action_body = query[len(prefix):].strip()
                break

        if detected_action:
            prompt_prefix, title_prefix = self._INLINE_ACTIONS[detected_action]
            full_prompt = prompt_prefix + action_body

            response = await self._inline_generate(full_prompt, user_id)
            if response:
                result_id = hashlib.md5(
                    f"{detected_action}:{action_body}".encode()
                ).hexdigest()[:16]
                results.append(InlineQueryResultArticle(
                    id=result_id,
                    title=f"{title_prefix}: {action_body[:40]}",
                    description=response[:100],
                    input_message_content=InputTextMessageContent(
                        message_text=response
                    )
                ))

            # Also offer a general AURA answer as second option
            general_response = await self._inline_generate(query, user_id)
            if general_response and general_response != response:
                general_id = hashlib.md5(
                    f"general:{query}".encode()
                ).hexdigest()[:16]
                results.append(InlineQueryResultArticle(
                    id=general_id,
                    title=f"Ask AURA: {query[:40]}",
                    description=general_response[:100],
                    input_message_content=InputTextMessageContent(
                        message_text=general_response
                    )
                ))
        else:
            # General query
            response = await self._inline_generate(query, user_id)
            if response:
                result_id = hashlib.md5(query.encode()).hexdigest()[:16]
                results.append(InlineQueryResultArticle(
                    id=result_id,
                    title=response[:50],
                    description=query,
                    input_message_content=InputTextMessageContent(
                        message_text=response
                    )
                ))

        # Fallback if nothing generated
        if not results:
            results.append(InlineQueryResultArticle(
                id="error",
                title="Could not generate a response",
                description="Try a different query or ask in the direct chat.",
                input_message_content=InputTextMessageContent(
                    message_text="Sorry, I couldn't process that query. Try asking me directly!"
                )
            ))

        try:
            await inline_query.answer(results, cache_time=30)
        except Exception as e:
            logger.error(f"Failed to answer inline query: {e}")

    async def _inline_generate(self, prompt: str, user_id: int) -> Optional[str]:
        """Generate a response for an inline query with caching and timeout."""

        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        cached = self._inline_cache.get(cache_key)
        if cached:
            return cached

        try:
            response = await asyncio.wait_for(
                self._process_with_aura(prompt, str(user_id)),
                timeout=self._INLINE_TIMEOUT_SEC
            )

            if response and response != "Sorry, I couldn't process that. Try again in a moment.":
                if len(response) > 4000:
                    response = response[:3997] + "..."
                self._inline_cache.put(cache_key, response)
                return response
        except asyncio.TimeoutError:
            logger.warning(f"Inline query timed out for user {user_id}")
        except Exception as e:
            logger.error(f"Inline generation error: {e}")

        return None

    # ============ REACT AGENT LOOP ============

    _AGENT_TIMEOUT = 120  # seconds

    async def _run_agent_and_reply(self, update: Update, goal: str):
        """Run the full ReAct agent loop and send the result back to the user.

        Flow:
        1. Send a "Thinking..." placeholder message
        2. Start a background typing indicator loop
        3. Run agent.run() in a thread with timeout (full ReAct loop with tools)
        4. Edit the placeholder with the final response
        5. Send any file artifacts (screenshots, plots) as photos/documents
        6. Fall back to agent.chat() / brain.think() on failure
        """
        chat_id = str(update.effective_chat.id)
        start_time = _time.time()

        # Send placeholder — we'll edit it with the real response
        placeholder = await update.message.reply_text("Thinking...")

        # Start typing indicator loop (cancelled when done)
        typing_task = asyncio.create_task(self._typing_loop(chat_id))

        try:
            response_text, artifacts = await asyncio.wait_for(
                asyncio.to_thread(self._run_agent_sync, goal),
                timeout=self._AGENT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            elapsed = _time.time() - start_time
            logger.warning(f"[Telegram] Agent timed out after {elapsed:.1f}s for: {goal[:80]}")
            response_text = "That took too long. Try a simpler question or break it into parts."
            artifacts = []
        except Exception as e:
            logger.error(f"[Telegram] Agent error: {e}", exc_info=True)
            response_text = "Something went wrong processing your request. Please try again."
            artifacts = []
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        elapsed = _time.time() - start_time
        logger.info(
            f"[Telegram] Response ready in {elapsed:.1f}s "
            f"({len(response_text)} chars, {len(artifacts)} artifacts)"
        )

        # Edit the placeholder with the real response
        await self._edit_or_send_response(placeholder, chat_id, response_text, update)

        # Send any file artifacts (screenshots, plots, generated files)
        for artifact_path in artifacts:
            await self._send_file_artifact(chat_id, artifact_path, update)

    def _run_agent_sync(self, goal: str):
        """Synchronous agent execution — called via asyncio.to_thread.

        Returns (response_text, list_of_artifact_paths).

        Priority chain:
        1. agent.run()  — full ReAct loop with tool calling
        2. agent.chat() — single LLM call with memory/emotion
        3. brain.think() — raw LLM call (last resort)
        """
        wrapper = self.aura  # TelegramAgentWrapper
        agent = getattr(wrapper, 'agent', wrapper)  # Unwrap to ApprenticeAgent

        response_text = ""
        artifacts = []

        # === Primary: Full ReAct agent loop ===
        try:
            logger.info(f"[Telegram] agent.run() starting: {goal[:80]}")
            result = agent.run(goal, timeout_seconds=self._AGENT_TIMEOUT - 5)

            if isinstance(result, dict):
                response_text = result.get("response", "")

                if result.get("timeout"):
                    response_text = response_text or "That request took too long. Please try a simpler query."

                # Fallback: check final_evaluation
                if not response_text:
                    fe = result.get("final_evaluation", {})
                    response_text = fe.get("progress", "")

                # Fallback: check history for last tool output
                if not response_text:
                    history = result.get("history", [])
                    if history:
                        last = history[-1]
                        if isinstance(last, dict):
                            response_text = last.get("result", {}).get("output", str(last))

                artifacts = self._collect_artifacts(agent)

            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result) if result else ""

            if response_text:
                logger.info(f"[Telegram] agent.run() succeeded ({len(response_text)} chars)")
                return (response_text, artifacts)

        except Exception as e:
            logger.warning(f"[Telegram] agent.run() failed: {e}, falling back to chat()")

        # === Fallback 1: agent.chat() ===
        try:
            logger.info(f"[Telegram] Fallback to agent.chat(): {goal[:80]}")
            response_text = agent.chat(goal)
            if response_text:
                return (response_text, [])
        except Exception as e:
            logger.warning(f"[Telegram] agent.chat() failed: {e}, falling back to brain.think()")

        # === Fallback 2: Direct brain.think() ===
        try:
            brain = getattr(agent, 'brain', None)
            if brain and hasattr(brain, 'think'):
                logger.info(f"[Telegram] Fallback to brain.think(): {goal[:80]}")
                response_text = brain.think(goal)
                if response_text:
                    return (response_text, [])
        except Exception as e:
            logger.error(f"[Telegram] brain.think() also failed: {e}")

        return (response_text or "Sorry, I couldn't process that right now.", [])

    def _collect_artifacts(self, agent) -> list:
        """Collect file artifacts produced by the agent's tool calls."""
        artifacts = []

        brain = getattr(agent, 'brain', None)
        if brain:
            screenshot = getattr(brain, '_last_screenshot_path', None)
            if screenshot and Path(screenshot).exists():
                artifacts.append(screenshot)
                brain._last_screenshot_path = None

        output_dirs = [
            Path("data/output"),
            Path("data/screenshots"),
            Path("data/generated"),
        ]
        cutoff = _time.time() - 60

        for out_dir in output_dirs:
            if out_dir.exists():
                try:
                    for f in out_dir.iterdir():
                        if f.is_file() and f.stat().st_mtime > cutoff:
                            fpath = str(f)
                            if fpath not in artifacts:
                                artifacts.append(fpath)
                except OSError:
                    pass

        return artifacts

    async def _typing_loop(self, chat_id: str):
        """Send typing indicator every 4 seconds until cancelled."""
        try:
            while True:
                await self.send_typing_indicator(chat_id)
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass

    async def _edit_or_send_response(self, placeholder, chat_id: str, text: str, update: Update):
        """Edit the placeholder message with the response, splitting if > 4096 chars."""
        if not text:
            text = "I processed your request but have nothing to report."

        MAX_LEN = 4096
        text = text.strip()

        if len(text) <= MAX_LEN:
            try:
                await placeholder.edit_text(text)
            except Exception as e:
                logger.warning(f"Could not edit placeholder: {e}")
                try:
                    await self.bot.send_message(chat_id=chat_id, text=text)
                except Exception:
                    pass
        else:
            chunks = self._split_message(text, MAX_LEN)
            for i, chunk in enumerate(chunks):
                try:
                    if i == 0:
                        await placeholder.edit_text(chunk)
                    else:
                        await self.bot.send_message(chat_id=chat_id, text=chunk)
                except Exception as e:
                    logger.warning(f"Error sending chunk {i}: {e}")

    @staticmethod
    def _split_message(text: str, max_len: int = 4096) -> list:
        """Split a long message into chunks, preferring paragraph boundaries."""
        if len(text) <= max_len:
            return [text]

        chunks = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            split_at = text.rfind("\n\n", 0, max_len)
            if split_at == -1 or split_at < max_len // 2:
                split_at = text.rfind("\n", 0, max_len)
            if split_at == -1 or split_at < max_len // 2:
                split_at = text.rfind(" ", 0, max_len)
            if split_at == -1:
                split_at = max_len
            chunks.append(text[:split_at].rstrip())
            text = text[split_at:].lstrip()

        return chunks

    async def _send_file_artifact(self, chat_id: str, filepath: str, update: Update):
        """Send a file artifact as a Telegram photo or document."""
        p = Path(filepath)
        if not p.exists():
            return

        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
        try:
            if p.suffix.lower() in image_exts and p.stat().st_size < 10 * 1024 * 1024:
                with open(p, "rb") as f:
                    await self.bot.send_photo(
                        chat_id=chat_id,
                        photo=f,
                        caption=p.name[:200],
                        reply_to_message_id=update.message.message_id,
                    )
            else:
                with open(p, "rb") as f:
                    await self.bot.send_document(
                        chat_id=chat_id,
                        document=f,
                        filename=p.name,
                        caption=p.name[:200],
                        reply_to_message_id=update.message.message_id,
                    )
            logger.info(f"[Telegram] Sent artifact: {filepath}")
        except Exception as e:
            logger.warning(f"[Telegram] Could not send artifact {filepath}: {e}")

    # ============ OVERRIDE AURA PROCESSING (kept for BasePlatform compat) ============

    async def _process_with_aura(self, text: str, user_id: str) -> str:
        """Process message through the full ReAct agent loop.

        This override is kept for BasePlatform.handle_incoming() compatibility.
        The primary path is _run_agent_and_reply() for regular messages.
        Voice/inline handlers still use this via handle_incoming() or directly.
        """
        try:
            response_text, _artifacts = await asyncio.to_thread(self._run_agent_sync, text)
            if response_text:
                return response_text
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

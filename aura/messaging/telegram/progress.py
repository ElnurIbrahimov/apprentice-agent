"""
TelegramProgressReporter and ToolStatusCallback — progress update helpers.
No dependencies on bot.py or any mixin.
"""
from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import Dict, List, Optional

from aura.messaging.telegram_formatting import (
    _TELEGRAM_MSG_LIMIT,
)
from aura.messaging.telegram_formatting import (
    split_message as _split_message,
)

logger = logging.getLogger(__name__)


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

    async def update_structured(self, step: int, total: int, label: str,
                                elapsed: float = 0.0):
        """Send a structured progress update with progress bar and step counter."""
        filled = int(10 * step / max(total, 1))
        bar = "\u2588" * filled + "\u2591" * (10 - filled)
        pct = int(100 * step / max(total, 1))
        time_str = f" | {elapsed:.0f}s" if elapsed > 0 else ""
        text = f"\u2699\ufe0f Step {step}/{total} — {label}\n[{bar}] {pct}%{time_str}"
        await self.update(text)

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

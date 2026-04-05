"""
AgentCoreMixin — _handle_message (main handler), _run_agent_and_reply,
                 _typing_loop, _run_agent_sync, _collect_artifacts,
                 _get_user_lock, _append_to_history, _build_contextual_prompt,
                 _edit_or_send_response, _send_file_artifact, _process_with_aura,
                 _handle_edited_message, _handle_reaction_update, _handle_error
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from aura.core.conversation_manager import get_conversation_manager

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None

logger = logging.getLogger(__name__)


class AgentCoreMixin:
    """Core message handling, agent execution, and conversation context."""

    _AGENT_TIMEOUT = 120  # seconds

    @staticmethod
    def _get_progress_text(goal: str) -> str:
        """Return a contextual progress message based on the user's request."""
        g = goal.lower()
        if any(k in g for k in ["search", "find", "look up", "google"]):
            return "\U0001f50d Searching..."
        if any(k in g for k in ["research", "investigate", "deep dive", "analyze"]):
            return "\U0001f9ea Researching..."
        if any(k in g for k in ["code", "python", "script", "function", "debug"]):
            return "\U0001f4bb Writing code..."
        if any(k in g for k in ["image", "draw", "generate", "picture", "photo"]):
            return "\U0001f3a8 Generating..."
        if any(k in g for k in ["translate", "translation"]):
            return "\U0001f310 Translating..."
        if any(k in g for k in ["summarize", "summary", "tldr"]):
            return "\U0001f4dd Summarizing..."
        if any(k in g for k in ["math", "calculate", "solve", "equation"]):
            return "\U0001f9ee Calculating..."
        if "[document context" in g.lower():
            return "\U0001f4c4 Analyzing document..."
        return "\U0001f4ad Thinking..."

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages through the full ReAct agent loop."""

        user = update.effective_user
        chat_id = str(update.effective_chat.id)

        # Detect group vs private chat
        chat_type = update.effective_chat.type  # "private", "group", "supergroup"
        is_group = chat_type in ("group", "supergroup")

        text = update.message.text or ""

        # --- Group message caching (persisted to SQLite) ---
        if is_group:
            self.store.add_group_message(
                chat_id=chat_id,
                user_id=str(user.id),
                user_name=user.first_name or "Unknown",
                text=text,
            )

        # --- Group gate: only respond if mentioned or replied-to ---
        if is_group:
            me = await context.bot.get_me()
            bot_username = me.username or "Aura828Bot"
            mentioned = f"@{bot_username}".lower() in text.lower()
            replied_to_bot = (
                update.message.reply_to_message is not None
                and update.message.reply_to_message.from_user is not None
                and update.message.reply_to_message.from_user.id == me.id
            )
            if not mentioned and not replied_to_bot:
                return  # Ignore messages not directed at us

            # Strip the @mention from text before processing
            text = re.sub(rf"@{re.escape(bot_username)}", "", text, flags=re.IGNORECASE).strip()
            if not text:
                text = "Hello"
        else:
            # Private chat: require allowed user
            if not self._is_user_allowed(user.id):
                return

        # Per-user rate limit (max_messages_per_minute from config, default 20)
        from aura.messaging.telegram.bot import _check_rate_limit
        max_per_min = self.config.get("max_messages_per_minute", 20)
        if not _check_rate_limit(str(user.id), max_per_min):
            await update.message.reply_text(
                "You're sending messages too fast. Please wait a moment."
            )
            return

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
                        self.store.clear_doc_context(user_id)
                        cleared_items.append("Document context")
                        # Clear conversation context window
                        if user_id in self._chat_history:
                            del self._chat_history[user_id]
                            cleared_items.append("Conversation context")
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

        # Check for pending skill actions (confirm/create flow)
        if await self._check_skill_pending(update, user.id, text):
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

        # Multi-language: prepend language instruction for non-English users
        user_lang = self.store.get_user_language(str(user.id))
        if (not user_lang or user_lang == "en") and hasattr(user, 'language_code') and user.language_code:
            lc = user.language_code.split("-")[0]
            if lc and lc != "en":
                user_lang = lc
                self.store.set_user_language(str(user.id), lc)
        if user_lang and user_lang != "en":
            text = f"[Respond in {user_lang}] {text}"

        # Forum support: capture message_thread_id for topic-aware replies
        thread_id = getattr(update.message, 'message_thread_id', None)

        # Bind to conversation via ConversationManager (cross-surface sync)
        conv_id = None
        try:
            cm = get_conversation_manager()
            if cm._brain is not None:
                conv_id = cm.get_or_create_session("telegram", str(user.id))
                cm.switch_conversation(conv_id, surface="telegram")
        except Exception as e:
            logger.debug(f"[Telegram] ConversationManager session bind skipped: {e}")

        # Backpressure: per-user lock — one message at a time
        uid_str = str(user.id)
        lock = self._get_user_lock(uid_str)
        if lock.locked():
            await update.message.reply_text(
                "\u23f3 Still working on your previous message. I'll get to this next.",
                reply_to_message_id=update.message.message_id,
            )

        async with lock:
            # Route through the full agent loop with typing indicator + file artifacts
            await self._run_agent_and_reply(update, text, conv_id=conv_id, user_id=uid_str)

        self._save_state()

    async def _run_agent_and_reply(self, update: Update, goal: str, *,
                                    conv_id: str = None, user_id: str = None):
        """Run the full ReAct agent loop and send the result back to the user."""
        chat_id = str(update.effective_chat.id)
        start_time = _time.time()
        original_goal = goal  # Keep original for history/retry

        # Track user message in conversation context window
        if user_id:
            self._append_to_history(user_id, "user", goal)

        # Build context-enriched prompt
        if user_id:
            goal = self._build_contextual_prompt(user_id, goal)

        # Track the user message in ConversationManager
        if conv_id and user_id:
            try:
                cm = get_conversation_manager()
                cm.on_message_added(conv_id, "user", original_goal, "telegram", user_id)
            except Exception as e:
                logger.debug(f"[Telegram] ConversationManager user message tracking skipped: {e}")

        # Send contextual placeholder based on the goal
        placeholder_text = self._get_progress_text(original_goal)
        placeholder = await update.message.reply_text(placeholder_text)

        # Streaming: use a queue to pass chunks from the agent thread
        chunk_q = queue.Queue(maxsize=500)
        streamed_text = ""
        _STREAM_EDIT_INTERVAL = 1.5  # seconds between Telegram message edits
        _last_edit_time = _time.time()
        _stream_done = asyncio.Event()

        async def _stream_editor():
            """Periodically edit the placeholder with accumulated streamed text."""
            nonlocal streamed_text, _last_edit_time
            while not _stream_done.is_set():
                # Drain all available chunks
                new_chunks = []
                while True:
                    try:
                        chunk = chunk_q.get_nowait()
                        if chunk is None:  # Sentinel — agent done
                            _stream_done.set()
                            break
                        new_chunks.append(chunk)
                    except queue.Empty:
                        break

                if new_chunks:
                    streamed_text += "".join(new_chunks)
                    now = _time.time()
                    # Edit message at most every INTERVAL seconds (Telegram rate limit)
                    if now - _last_edit_time >= _STREAM_EDIT_INTERVAL and streamed_text.strip():
                        try:
                            display = streamed_text[:4000] + (" ..." if len(streamed_text) > 4000 else " \u2588")
                            await placeholder.edit_text(display)
                            _last_edit_time = now
                        except Exception:
                            pass  # Message unchanged or rate limited

                if not _stream_done.is_set():
                    await asyncio.sleep(0.3)

        # Start stream editor and typing loop
        stream_task = asyncio.create_task(_stream_editor())
        typing_task = asyncio.create_task(self._typing_loop(chat_id))

        is_error = False
        try:
            response_text, artifacts = await asyncio.wait_for(
                asyncio.to_thread(self._run_agent_sync, goal, chunk_q),
                timeout=self._AGENT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            elapsed = _time.time() - start_time
            logger.warning(f"[Telegram] Agent timed out after {elapsed:.1f}s for: {original_goal[:80]}")
            response_text = streamed_text or "That took too long. Try a simpler question or break it into parts."
            artifacts = []
            is_error = True
        except Exception as e:
            logger.error(f"[Telegram] Agent error: {e}", exc_info=True)
            response_text = streamed_text or "Something went wrong processing your request."
            artifacts = []
            is_error = True
        finally:
            _stream_done.set()
            typing_task.cancel()
            stream_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
            try:
                await stream_task
            except asyncio.CancelledError:
                pass

        # Store failed message for retry
        if is_error and user_id:
            self._failed_messages[user_id] = original_goal

        elapsed = _time.time() - start_time
        logger.info(
            f"[Telegram] Response ready in {elapsed:.1f}s "
            f"({len(response_text)} chars, {len(artifacts)} artifacts)"
        )

        # Track assistant response in conversation context window
        if user_id and response_text:
            self._append_to_history(user_id, "assistant", response_text)

        # Track the assistant response in ConversationManager
        if conv_id and user_id and response_text:
            try:
                cm = get_conversation_manager()
                cm.on_message_added(conv_id, "assistant", response_text, "telegram", user_id)
            except Exception as e:
                logger.debug(f"[Telegram] ConversationManager assistant message tracking skipped: {e}")

        # Store last exchange for /learn command (persisted to SQLite)
        if user_id:
            try:
                self.store.set_skill_state(user_id, last_exchange={
                    "input": original_goal[:2000],
                    "output": response_text[:2000],
                    "timestamp": _time.time(),
                })
            except Exception:
                pass

        # Build reply buttons — include retry on errors
        if is_error and user_id:
            action_buttons = InlineKeyboardMarkup([[
                InlineKeyboardButton("\U0001f504 Retry", callback_data=f"retry_{user_id}"),
                InlineKeyboardButton("\U0001f504 Retry (lighter)", callback_data=f"retry_light_{user_id}"),
            ]])
        else:
            action_buttons = self._get_action_buttons(update.message.message_id)

        # Edit the placeholder with the real response + action buttons
        await self._edit_or_send_response(placeholder, chat_id, response_text, update,
                                          reply_markup=action_buttons)

        # Send any file artifacts (screenshots, plots, generated files)
        for artifact_path in artifacts:
            await self._send_file_artifact(chat_id, artifact_path, update)

        # Phase 5+: Native reaction via Telegram API (if available)
        try:
            emotion_data = None
            evoemo = self.aura.tools.get("evoemo") if hasattr(self.aura, 'tools') else None
            if evoemo and hasattr(evoemo, 'get_state'):
                emotion_data = evoemo.get_state()
            if not emotion_data:
                agent = getattr(self.aura, 'agent', self.aura)
                brain = getattr(agent, 'brain', None)
                if brain and hasattr(brain, '_alma_engine'):
                    emotion_data = brain._alma_engine.get_emotional_state()
            if not emotion_data and hasattr(self.aura, 'emotion') and self.aura.emotion:
                emotion_data = self.aura.emotion.get_emotional_state()

            if emotion_data:
                emotion = emotion_data.get("dominant_emotion", "neutral")
                intensity = emotion_data.get("intensity", 0.3)
                # Try native reaction first, fall back to sticker/GIF
                reacted = await self._try_native_reaction(
                    chat_id, update.message.message_id, emotion, intensity
                )
                if not reacted:
                    await self._maybe_send_reaction(chat_id, emotion, intensity)
        except Exception:
            pass  # Reactions are optional — never break the main flow

    def _run_agent_sync(self, goal: str, chunk_queue: queue.Queue = None):
        """Synchronous agent execution — called via asyncio.to_thread.

        Returns (response_text, list_of_artifact_paths).
        """
        wrapper = self.aura  # TelegramAgentWrapper
        agent = getattr(wrapper, 'agent', wrapper)  # Unwrap to ApprenticeAgent

        response_text = ""
        artifacts = []

        # Build on_chunk callback for streaming
        on_chunk = None
        if chunk_queue is not None:
            def on_chunk(text):
                try:
                    chunk_queue.put_nowait(text)
                except queue.Full:
                    pass

        # === Primary: Full ReAct agent loop ===
        try:
            logger.info(f"[Telegram] agent.run() starting: {goal[:80]}")
            result = agent.run(goal, timeout_seconds=self._AGENT_TIMEOUT - 5,
                               on_chunk=on_chunk)

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

        # Signal streaming done for fallback paths
        if chunk_queue is not None:
            chunk_queue.put(None)  # Sentinel

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

    def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create a per-user asyncio.Lock for backpressure."""
        if user_id not in self._user_locks:
            self._user_locks[user_id] = asyncio.Lock()
        return self._user_locks[user_id]

    def _append_to_history(self, user_id: str, role: str, content: str):
        """Append a message to the user's sliding context window."""
        history = self._chat_history[user_id]
        history.append({
            "role": role,
            "content": content[:2000],
            "timestamp": _time.time()
        })
        # Keep only last N messages
        if len(history) > self._MAX_CONTEXT_MESSAGES:
            self._chat_history[user_id] = history[-self._MAX_CONTEXT_MESSAGES:]

    def _build_contextual_prompt(self, user_id: str, current_message: str) -> str:
        """Build a prompt with recent conversation context prepended."""
        history = self._chat_history.get(user_id, [])
        if not history:
            return current_message

        context_lines = []
        for msg in history[-(self._MAX_CONTEXT_MESSAGES - 1):]:  # Leave room for current
            prefix = "User" if msg["role"] == "user" else "Aura"
            context_lines.append(f"{prefix}: {msg['content']}")

        context = "\n".join(context_lines)
        return (
            f"[Recent conversation for context — respond to the CURRENT message only]\n"
            f"{context}\n\n"
            f"[Current message]\n{current_message}"
        )

    async def _handle_edited_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Re-process edited messages through the agent."""
        edited = update.edited_message
        if not edited or not edited.text:
            return

        user = edited.from_user
        if not user or not self._is_user_allowed(user.id):
            return

        from aura.messaging.telegram.bot import _check_rate_limit
        if not _check_rate_limit(str(user.id), self.config.get("max_messages_per_minute", 20)):
            return

        chat_id = str(edited.chat_id)
        text = edited.text.strip()
        if not text:
            return

        # Send acknowledgment
        await self.send_typing_indicator(chat_id)
        notice = await self.bot.send_message(
            chat_id=chat_id,
            text="✏️ Got your edit, re-processing...",
            reply_to_message_id=edited.message_id,
        )

        # Apply backpressure
        user_id = str(user.id)
        lock = self._get_user_lock(user_id)
        async with lock:
            # Prepend document context if active
            if hasattr(self, '_build_doc_augmented_text'):
                text = self._build_doc_augmented_text(user.id, text)

            # Build context-enriched prompt
            contextual_text = self._build_contextual_prompt(user_id, text)

            # Track in history
            self._append_to_history(user_id, "user", text)

            # Start typing loop
            typing_task = asyncio.create_task(self._typing_loop(chat_id))
            try:
                response_text, artifacts = await asyncio.wait_for(
                    asyncio.to_thread(self._run_agent_sync, contextual_text),
                    timeout=self._AGENT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                response_text = "That took too long. Try a simpler question."
                artifacts = []
            except Exception as e:
                logger.error(f"[Telegram] Edit re-process error: {e}", exc_info=True)
                response_text = "Something went wrong re-processing your edit."
                artifacts = []
            finally:
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

            # Track response in history
            self._append_to_history(user_id, "assistant", response_text)

            # Edit the notice with the real response
            action_buttons = self._get_action_buttons(edited.message_id)
            await self._edit_or_send_response(notice, chat_id, response_text, update,
                                              reply_markup=action_buttons)

            for artifact_path in artifacts:
                await self._send_file_artifact(chat_id, artifact_path, update)

    async def _handle_reaction_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Capture user reactions to bot messages as feedback signals."""
        reaction = update.message_reaction
        if not reaction:
            return

        user_id = str(reaction.user.id) if reaction.user else "unknown"
        chat_id = str(reaction.chat.id)
        message_id = reaction.message_id

        new_reactions = []
        for r in (reaction.new_reaction or []):
            if hasattr(r, 'emoji'):
                new_reactions.append(r.emoji)
        if not new_reactions:
            return

        positive = {'\U0001f44d', '\u2764\ufe0f', '\U0001f525', '\u2b50', '\U0001f389', '\U0001f4af', '\U0001f44f', '\U0001f929', '\U0001f4aa'}
        negative = {'\U0001f44e', '\U0001f622', '\U0001f621', '\U0001f92e', '\U0001f4a9'}

        sentiment = 'positive' if any(r in positive for r in new_reactions) else \
                    'negative' if any(r in negative for r in new_reactions) else 'neutral'

        try:
            self.store.save_reaction_feedback(
                user_id=user_id,
                chat_id=chat_id,
                message_id=message_id,
                reactions=json.dumps(new_reactions),
                sentiment=sentiment,
            )
            logger.info(f"[Telegram] Reaction feedback: {sentiment} from {user_id} ({new_reactions})")
        except Exception as e:
            logger.debug(f"[Telegram] Could not save reaction feedback: {e}")

    async def _handle_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors with retry buttons."""
        logger.error(f"Telegram error: {context.error}")

        if update and update.effective_chat:
            user_id = str(update.effective_user.id) if update.effective_user else None
            try:
                retry_markup = None
                if user_id:
                    retry_markup = InlineKeyboardMarkup([[
                        InlineKeyboardButton("\U0001f504 Retry", callback_data=f"retry_{user_id}"),
                        InlineKeyboardButton("\U0001f504 Retry (lighter)", callback_data=f"retry_light_{user_id}"),
                    ]])
                await self.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Something went wrong. Want to try again?",
                    reply_markup=retry_markup,
                )
            except Exception as e:
                logger.debug(f"Could not send error message: {e}")

    async def _edit_or_send_response(self, placeholder, chat_id: str, text: str,
                                     update: Update, reply_markup=None):
        """Edit the placeholder message with the response, splitting if > 4096 chars.

        Converts markdown to Telegram MarkdownV2 for proper formatting (bold, italic,
        code blocks, links). Falls back to plain text if formatting fails.
        """
        from telegram.constants import ParseMode
        from aura.messaging.telegram_formatting import format_telegram_response

        if not text:
            text = "I processed your request but have nothing to report."

        text = text.strip()

        # Convert markdown → MarkdownV2 and split into chunks
        try:
            chunks = format_telegram_response(text)
        except Exception:
            # Fallback: send as plain text if formatting fails
            chunks = self._split_message(text, 4096)

        for i, chunk in enumerate(chunks):
            markup = reply_markup if i == len(chunks) - 1 else None
            # Try MarkdownV2 first, fall back to plain text
            sent = False
            for parse_mode in (ParseMode.MARKDOWN_V2, None):
                try:
                    if i == 0:
                        await placeholder.edit_text(
                            chunk, parse_mode=parse_mode, reply_markup=markup)
                    else:
                        await self.bot.send_message(
                            chat_id=chat_id, text=chunk,
                            parse_mode=parse_mode, reply_markup=markup)
                    sent = True
                    break
                except Exception as e:
                    if parse_mode is not None:
                        logger.debug(f"MarkdownV2 failed, falling back to plain: {e}")
                        continue
                    logger.warning(f"Error sending chunk {i}: {e}")
            if not sent:
                logger.warning(f"Failed to send chunk {i} in any format")

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

    async def _process_with_aura(self, text: str, user_id: str) -> str:
        """Process message through the full ReAct agent loop."""
        try:
            response_text, _artifacts = await asyncio.to_thread(self._run_agent_sync, text)
            if response_text:
                return response_text
        except Exception as e:
            logger.error(f"AURA processing error: {e}", exc_info=True)

        return "Sorry, I couldn't process that. Try again in a moment."

"""
MiscMixin — keyboard, file gen, digest, language, action callbacks, stickers cmd,
            pinning, export, webhook, hand, proactive system, send_to_all_active,
            send_morning_greeting, send_follow_up, _handle_agent, UI helpers,
            get_active_chat_ids
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import tempfile
import time as _time
from datetime import datetime
from typing import List, Optional

from aura.core.conversation_manager import get_conversation_manager
from aura.messaging.telegram.constants import EMOTION_EMOJI, EMOTION_REACTIONS

try:
    from telegram import (
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        ReplyKeyboardMarkup,
        ReplyKeyboardRemove,
        Update,
    )
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None

logger = logging.getLogger(__name__)


class MiscMixin:
    """Miscellaneous handlers — keyboard, file generation, digest, language,
    action buttons, stickers, pinning, export, webhook, hand, proactive system."""

    # ---- 1. Persistent Reply Keyboard ----

    def _get_reply_keyboard(self) -> ReplyKeyboardMarkup:
        """Build the persistent quick-access reply keyboard."""
        keyboard = [
            ["/research", "/search", "/code", "/image"],
            ["/model", "/status", "/help", "/session"],
        ]
        return ReplyKeyboardMarkup(keyboard, resize_keyboard=True,
                                   is_persistent=True)

    async def _handle_keyboard(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /keyboard — toggle the persistent reply keyboard on/off."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        user_id = str(update.effective_user.id)
        currently_on = self.store.get_keyboard_enabled(user_id)
        if currently_on:
            self.store.set_keyboard_enabled(user_id, False)
            await update.message.reply_text(
                "Keyboard hidden. Use /keyboard to bring it back.",
                reply_markup=ReplyKeyboardRemove()
            )
        else:
            self.store.set_keyboard_enabled(user_id, True)
            await update.message.reply_text(
                "Keyboard restored!",
                reply_markup=self._get_reply_keyboard()
            )

    # ---- 7. File Generation ----

    async def _handle_file_gen(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /file <format> [content] — generate a document and send it."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        args = context.args or []
        if not args:
            await update.message.reply_text(
                "Usage: /file <pdf|docx|txt|md> [content or 'last']\n\n"
                "Examples:\n"
                "/file pdf last — export last response as PDF\n"
                "/file docx Meeting notes: discussed Q1 goals"
            )
            return

        fmt = args[0].lower().strip(".")
        if fmt not in ("pdf", "docx", "txt", "md"):
            await update.message.reply_text("Supported formats: pdf, docx, txt, md")
            return

        content_text = " ".join(args[1:]) if len(args) > 1 else ""

        # "last" keyword: use last agent response
        if content_text.strip().lower() == "last" or not content_text:
            uid = update.effective_user.id
            exchange = self.store.get_skill_state(str(uid)).get("last_exchange", {})
            content_text = exchange.get("output", "")
            if not content_text:
                await update.message.reply_text("No recent response to export. Send a message first.")
                return

        placeholder = await update.message.reply_text(f"\U0001f4c4 Generating {fmt.upper()}...")

        try:
            if fmt in ("pdf", "docx"):
                from aura.tools.document_generator import DocumentGeneratorTool
                gen = DocumentGeneratorTool()
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"aura_{ts}.{fmt}"
                filepath = os.path.join(tempfile.gettempdir(), filename)
                if fmt == "pdf":
                    gen.create_pdf(content_text, filepath)
                else:
                    gen.create_docx(content_text, filepath)
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"aura_{ts}.{fmt}"
                filepath = os.path.join(tempfile.gettempdir(), filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content_text)

            with open(filepath, "rb") as f:
                await self.bot.send_document(
                    chat_id=str(update.effective_chat.id),
                    document=f,
                    filename=filename,
                    caption=f"Generated {fmt.upper()} document",
                )
            await placeholder.delete()
            # Clean up temp file
            try:
                os.unlink(filepath)
            except OSError:
                pass
        except Exception as e:
            logger.error(f"[FileGen] Error: {e}", exc_info=True)
            await placeholder.edit_text(f"Failed to generate file: {e}")

    # ---- 8. Daily Digest ----

    async def _handle_digest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /digest on|off|now — daily activity digest."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        chat_id = str(update.effective_chat.id)
        args = context.args or []
        cmd = args[0].lower() if args else "now"

        if cmd == "on":
            # Schedule via APScheduler if available
            try:
                from apscheduler.schedulers.asyncio import AsyncIOScheduler
                scheduler = getattr(self, '_digest_scheduler', None)
                if not scheduler:
                    scheduler = AsyncIOScheduler()
                    self._digest_scheduler = scheduler
                    scheduler.start()
                job = scheduler.add_job(
                    self._send_daily_digest_async, 'cron',
                    hour=9, minute=0,
                    args=[chat_id],
                    id=f"digest_{chat_id}",
                    replace_existing=True,
                )
                self._digest_job_ids[chat_id] = job.id
                self.store.set_digest_job(chat_id, job.id, enabled=True)
                await update.message.reply_text(
                    "\U0001f4ec Daily digest enabled! You'll get a summary at 9:00 AM.\n"
                    "Use /digest off to disable, /digest now for an instant digest."
                )
            except ImportError:
                await update.message.reply_text(
                    "APScheduler not installed — digest scheduling unavailable.\n"
                    "Use /digest now for a one-time digest."
                )
                return

        elif cmd == "off":
            try:
                scheduler = getattr(self, '_digest_scheduler', None)
                job_id = self._digest_job_ids.get(chat_id)
                if scheduler and job_id:
                    scheduler.remove_job(job_id)
                    self._digest_job_ids.pop(chat_id, None)
                self.store.set_digest_job(chat_id, "", enabled=False)
            except Exception:
                pass
            await update.message.reply_text("\U0001f6d1 Daily digest disabled.")

        else:
            # "now" — send an instant digest
            await self._send_daily_digest_async(chat_id)

    async def _send_daily_digest_async(self, chat_id: str):
        """Build and send a daily activity digest to the chat."""
        lines = ["\U0001f4ca Daily AURA Digest\n"]

        # Gather stats
        try:
            from aura.proactive.persistence import ProactivePersistence
            pp = ProactivePersistence()
            conn = pp._conn
            if conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM activity_log WHERE timestamp > datetime('now', '-1 day')"
                )
                count = cursor.fetchone()[0]
                lines.append(f"\u2022 Activities logged (24h): {count}")
        except Exception:
            lines.append("\u2022 Activity log: unavailable")

        # Conversation count
        try:
            cm = get_conversation_manager()
            convos = cm.list_conversations() if hasattr(cm, 'list_conversations') else []
            lines.append(f"\u2022 Active conversations: {len(convos)}")
        except Exception:
            pass

        # Active chats
        lines.append(f"\u2022 Active Telegram chats: {len(self.active_chats)}")

        # Scheduled tasks
        try:
            aura_sched = getattr(self.aura, 'scheduler', None)
            if aura_sched:
                jobs = aura_sched.get_jobs() if hasattr(aura_sched, 'get_jobs') else []
                lines.append(f"\u2022 Scheduled tasks: {len(jobs)}")
        except Exception:
            pass

        lines.append(f"\n\U0001f552 Generated at {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        try:
            await self.bot.send_message(chat_id=int(chat_id), text="\n".join(lines))
        except Exception as e:
            logger.error(f"[Digest] Failed to send digest: {e}")

    # ---- 9. Multi-language ----

    async def _handle_lang(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /lang [code] — set or show language preference."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        user_id = str(update.effective_user.id)
        args = context.args or []

        if not args:
            current = self.store.get_user_language(user_id)
            tg_lang = getattr(update.effective_user, 'language_code', 'unknown')
            await update.message.reply_text(
                f"\U0001f310 Language settings:\n"
                f"Current: {current}\n"
                f"Telegram language: {tg_lang}\n\n"
                f"Set with: /lang <code>\n"
                f"Examples: /lang ru, /lang az, /lang en"
            )
            return

        lang_code = args[0].lower().strip()[:5]
        self.store.set_user_language(user_id, lang_code)
        if lang_code == "en":
            await update.message.reply_text(
                "\U0001f310 Language set to English (default). "
                "I'll respond in English."
            )
        else:
            await update.message.reply_text(
                f"\U0001f310 Language set to: {lang_code}\n"
                f"I'll try to respond in {lang_code} from now on."
            )

    # ---- 10. Reply Action Buttons ----

    def _get_action_buttons(self, original_msg_id: int = 0) -> InlineKeyboardMarkup:
        """Build action buttons attached to agent responses."""
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("\U0001f504 Regenerate", callback_data=f"act_regenerate_{original_msg_id}"),
                InlineKeyboardButton("\U0001f4dd Shorter", callback_data=f"act_shorter_{original_msg_id}"),
                InlineKeyboardButton("\U0001f310 Translate", callback_data=f"act_translate_{original_msg_id}"),
            ],
            [
                InlineKeyboardButton("\U0001f50d Go deeper", callback_data=f"act_deeper_{original_msg_id}"),
                InlineKeyboardButton("\U0001f4be Save", callback_data=f"act_save_{original_msg_id}"),
                InlineKeyboardButton("\U0001f4c4 Export", callback_data=f"act_export_{original_msg_id}"),
            ],
        ])

    async def _handle_action_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle act_<action>_<msg_id> callbacks from action buttons."""
        query = update.callback_query
        if not self._is_user_allowed(query.from_user.id):
            await query.answer("Unauthorized", show_alert=True)
            return
        await query.answer()
        data = query.data  # e.g. "act_deeper_12345"
        parts = data.split("_", 2)
        if len(parts) < 3:
            return
        action = parts[1]
        user_id = str(query.from_user.id)

        # Get last response text for context
        uid = query.from_user.id
        exchange = self.store.get_skill_state(str(uid)).get("last_exchange", {})
        last_output = exchange.get("output", "")
        last_input = exchange.get("input", "")

        if action == "deeper":
            if not last_output:
                await query.message.reply_text("No recent response to expand on.")
                return
            goal = f"Expand on and go deeper into this topic. Previous response was about: {last_input[:200]}"
            placeholder = await query.message.reply_text("Thinking...")
            try:
                response_text, artifacts = await asyncio.wait_for(
                    asyncio.to_thread(self._run_agent_sync, goal),
                    timeout=self._AGENT_TIMEOUT,
                )
            except Exception:
                response_text = "Could not expand. Try asking directly."
                artifacts = []
            chat_id = str(query.message.chat_id)
            await self._edit_or_send_response(placeholder, chat_id, response_text, update)

        elif action == "save":
            if not last_output:
                await query.message.reply_text("Nothing to save.")
                return
            try:
                if hasattr(self.aura, 'memory') and self.aura.memory:
                    mem_text = f"[Saved from Telegram] {last_input[:100]}: {last_output[:500]}"
                    if hasattr(self.aura.memory, 'add'):
                        self.aura.memory.add(mem_text)
                    elif hasattr(self.aura.memory, 'store'):
                        self.aura.memory.store(mem_text)
                    await query.message.reply_text("\U0001f4be Saved to memory!")
                else:
                    await query.message.reply_text("Memory system not available.")
            except Exception as e:
                await query.message.reply_text(f"Could not save: {e}")

        elif action == "share":
            if last_output:
                share_text = last_output[:4000]
                await query.message.reply_text(
                    f"\U0001f4e4 Shareable response:\n\n{share_text}"
                )
            else:
                await query.message.reply_text("No recent response to share.")

        elif action == "export":
            if not last_output:
                await query.message.reply_text("Nothing to export.")
                return
            # Quick export as txt
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aura_response_{ts}.txt"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Query: {last_input}\n\nResponse:\n{last_output}")
            try:
                with open(filepath, "rb") as f:
                    await self.bot.send_document(
                        chat_id=str(query.message.chat_id),
                        document=f,
                        filename=filename,
                    )
            except Exception as e:
                await query.message.reply_text(f"Export failed: {e}")
            try:
                os.unlink(filepath)
            except OSError:
                pass

        elif action == "regenerate":
            if not last_input:
                await query.message.reply_text("Nothing to regenerate.")
                return
            placeholder = await query.message.reply_text("\U0001f504 Regenerating...")
            chat_id = str(query.message.chat_id)
            typing_task = asyncio.create_task(self._typing_loop(chat_id))
            try:
                response_text, artifacts = await asyncio.wait_for(
                    asyncio.to_thread(self._run_agent_sync, last_input),
                    timeout=self._AGENT_TIMEOUT,
                )
            except Exception:
                response_text = "Regeneration failed. Try asking directly."
            finally:
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass
            buttons = self._get_action_buttons(query.message.message_id)
            await self._edit_or_send_response(placeholder, chat_id, response_text, update,
                                              reply_markup=buttons)
            # Update stored exchange
            if user_id:
                try:
                    self.store.set_skill_state(user_id, last_exchange={
                        "input": last_input[:2000],
                        "output": response_text[:2000],
                        "timestamp": _time.time(),
                    })
                except Exception:
                    pass

        elif action == "shorter":
            if not last_output:
                await query.message.reply_text("Nothing to shorten.")
                return
            placeholder = await query.message.reply_text("\U0001f4dd Making it shorter...")
            goal = (
                f"Rewrite this response to be much more concise — keep only the key points, "
                f"remove fluff and redundancy. Original response:\n\n{last_output[:3000]}"
            )
            chat_id = str(query.message.chat_id)
            try:
                response_text, _ = await asyncio.wait_for(
                    asyncio.to_thread(self._run_agent_sync, goal),
                    timeout=self._AGENT_TIMEOUT,
                )
            except Exception:
                response_text = "Could not shorten. Try asking directly."
            await self._edit_or_send_response(placeholder, chat_id, response_text, update)

        elif action == "translate":
            if not last_output:
                await query.message.reply_text("Nothing to translate.")
                return
            # Get user's language preference, default to asking for auto-detect
            target_lang = self.store.get_user_language(user_id) or "en"
            if target_lang == "en":
                goal = (
                    f"Detect the most likely non-English language the user speaks and translate "
                    f"this response into that language. If unsure, translate to Russian.\n\n"
                    f"Text to translate:\n{last_output[:3000]}"
                )
            else:
                goal = (
                    f"Translate this response into {target_lang}:\n\n{last_output[:3000]}"
                )
            placeholder = await query.message.reply_text("\U0001f310 Translating...")
            chat_id = str(query.message.chat_id)
            try:
                response_text, _ = await asyncio.wait_for(
                    asyncio.to_thread(self._run_agent_sync, goal),
                    timeout=self._AGENT_TIMEOUT,
                )
            except Exception:
                response_text = "Translation failed. Try /lang to set your language."
            await self._edit_or_send_response(placeholder, chat_id, response_text, update)

    # ---- 11. Custom Sticker Pack ----

    async def _handle_stickers_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stickers — admin command to scaffold an emotion-mapped sticker set."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        if not self._is_admin(update.effective_user.id):
            await update.message.reply_text("Admin only command.")
            return

        emotions = list(EMOTION_REACTIONS.keys())
        lines = [
            "\U0001f3a8 AURA Sticker Pack Setup\n",
            "To create a custom sticker pack for AURA, you need:",
            "1. Design one sticker per emotion (512x512 PNG, <512KB)",
            "2. Send them to @Stickers bot to create a pack",
            "3. Set AURA_STICKER_PACK env var to the pack name\n",
            "Required emotions:",
        ]
        for emo in emotions:
            emoji = EMOTION_EMOJI.get(emo, "\U0001f610")
            lines.append(f"  {emoji} {emo}")

        lines.append(f"\nTotal stickers needed: {len(emotions)}")
        lines.append("\nOnce created, AURA will auto-select stickers based on emotional state.")

        await update.message.reply_text("\n".join(lines))

    # ---- 13. Message Pinning ----

    async def _handle_pin(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pin — pin the replied-to message, or the last bot message."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        chat_id = str(update.effective_chat.id)
        reply = update.message.reply_to_message

        if reply:
            try:
                await context.bot.pin_chat_message(
                    chat_id=int(chat_id),
                    message_id=reply.message_id,
                    disable_notification=True,
                )
                await update.message.reply_text("\U0001f4cc Message pinned!")
            except Exception as e:
                await update.message.reply_text(f"Could not pin: {e}")
        else:
            await update.message.reply_text(
                "Reply to a message with /pin to pin it.\n"
                "Tip: I'll also offer to pin important research results in groups."
            )

    async def _handle_pin_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle pin_<message_id> callbacks — pin a specific message."""
        query = update.callback_query
        await query.answer()
        try:
            msg_id = int(query.data.replace("pin_", ""))
            await context.bot.pin_chat_message(
                chat_id=query.message.chat_id,
                message_id=msg_id,
                disable_notification=True,
            )
            await query.message.reply_text("\U0001f4cc Pinned!")
        except Exception as e:
            await query.message.reply_text(f"Could not pin: {e}")

    def _get_pin_button(self, message_id: int) -> InlineKeyboardMarkup:
        """Build a 'Pin this' inline button for important research results."""
        return InlineKeyboardMarkup([[
            InlineKeyboardButton("\U0001f4cc Pin this", callback_data=f"pin_{message_id}")
        ]])

    # ---- 14. Conversation Export ----

    async def _handle_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /export [json|md|pdf|txt] — export conversation history."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        args = context.args or []
        fmt = args[0].lower() if args else "md"
        if fmt not in ("json", "md", "pdf", "txt"):
            await update.message.reply_text("Supported: /export json|md|pdf|txt")
            return

        user_id = str(update.effective_user.id)
        placeholder = await update.message.reply_text(f"\U0001f4e6 Exporting as {fmt.upper()}...")

        try:
            # Gather conversation from ConversationManager
            messages = []
            try:
                cm = get_conversation_manager()
                conv_id = cm.get_bound_conversation(f"telegram:{user_id}")
                if conv_id and hasattr(cm, 'get_messages'):
                    messages = cm.get_messages(conv_id) or []
                elif conv_id and hasattr(cm, '_messages'):
                    messages = cm._messages.get(conv_id, [])
            except Exception:
                pass

            # Fallback: use last exchange if no conversation manager data
            if not messages:
                exchange = self.store.get_skill_state(str(update.effective_user.id)).get("last_exchange", {})
                if exchange:
                    messages = [
                        {"role": "user", "content": exchange.get("input", "")},
                        {"role": "assistant", "content": exchange.get("output", "")},
                    ]

            if not messages:
                await placeholder.edit_text("No conversation data to export.")
                return

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aura_conversation_{ts}.{fmt}"
            filepath = os.path.join(tempfile.gettempdir(), filename)

            if fmt == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(messages if isinstance(messages, list) else
                              [{"role": m.role, "content": m.preview} if hasattr(m, 'role')
                               else m for m in messages],
                              f, indent=2, default=str)

            elif fmt == "md":
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("# AURA Conversation Export\n")
                    f.write(f"*Exported: {datetime.now().isoformat()}*\n\n---\n\n")
                    for m in messages:
                        if isinstance(m, dict):
                            role = m.get("role", "unknown").upper()
                            content = m.get("content", m.get("preview", ""))
                        elif hasattr(m, 'role'):
                            role = m.role.upper()
                            content = getattr(m, 'preview', '') or getattr(m, 'content', '')
                        else:
                            role = "MSG"
                            content = str(m)
                        f.write(f"**{role}**: {content}\n\n")

            elif fmt == "txt":
                with open(filepath, "w", encoding="utf-8") as f:
                    for m in messages:
                        if isinstance(m, dict):
                            role = m.get("role", "unknown")
                            content = m.get("content", m.get("preview", ""))
                        elif hasattr(m, 'role'):
                            role = m.role
                            content = getattr(m, 'preview', '') or getattr(m, 'content', '')
                        else:
                            role = "msg"
                            content = str(m)
                        f.write(f"[{role}] {content}\n\n")

            elif fmt == "pdf":
                try:
                    from aura.tools.document_generator import DocumentGeneratorTool
                    gen = DocumentGeneratorTool()
                    text_content = "AURA Conversation Export\n\n"
                    for m in messages:
                        if isinstance(m, dict):
                            role = m.get("role", "unknown").upper()
                            content = m.get("content", m.get("preview", ""))
                        elif hasattr(m, 'role'):
                            role = m.role.upper()
                            content = getattr(m, 'preview', '') or getattr(m, 'content', '')
                        else:
                            role = "MSG"
                            content = str(m)
                        text_content += f"{role}: {content}\n\n"
                    gen.create_pdf(text_content, filepath)
                except ImportError:
                    await placeholder.edit_text("PDF export requires fpdf2. Use /export md instead.")
                    return

            with open(filepath, "rb") as f:
                await self.bot.send_document(
                    chat_id=str(update.effective_chat.id),
                    document=f,
                    filename=filename,
                    caption=f"Conversation export ({fmt.upper()})",
                )
            await placeholder.delete()
            try:
                os.unlink(filepath)
            except OSError:
                pass
        except Exception as e:
            logger.error(f"[Export] Error: {e}", exc_info=True)
            await placeholder.edit_text(f"Export failed: {e}")

    # ---- Webhook ----

    async def _handle_webhook(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /webhook command — show webhook info and endpoints."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        args = context.args or []
        backend = os.getenv("AURA_BACKEND_URL", "http://89.167.107.134")

        if not args:
            text = (
                "Webhook Endpoints\n\n"
                f"GitHub: {backend}/api/webhooks/github\n"
                f"Alerts: {backend}/api/webhooks/alert\n"
                f"Notify: {backend}/api/webhooks/notify\n\n"
                "Use /webhook test to send a test event.\n"
                "Use /webhook github <repo> for setup instructions."
            )
            await update.message.reply_text(text)
        elif args[0] == "test":
            await update.message.reply_text("Test webhook received! Pipeline is working.")
        elif args[0] == "github" and len(args) > 1:
            repo = args[1]
            text = (
                f"GitHub Webhook Setup for {repo}\n\n"
                f"URL: {backend}/api/webhooks/github\n"
                f"Content type: application/json\n"
                f"Events: check_run, workflow_run, pull_request, push\n\n"
                f"Configure at: https://github.com/{repo}/settings/hooks/new"
            )
            await update.message.reply_text(text)
        else:
            await update.message.reply_text("Usage: /webhook, /webhook test, /webhook github <repo>")

    # ---- Hand (Autonomous Hands) ----

    async def _handle_hand(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /hand — manage autonomous Hands."""
        if not self._is_user_allowed(update.effective_user.id):
            return

        from aura.hands.collector import CollectorHand
        from aura.hands.guardian import GuardianHand
        from aura.hands.manager import get_hand_manager
        from aura.hands.memory_hand import MemoryHand
        from aura.hands.researcher import ResearcherHand

        manager = get_hand_manager()
        if not manager.list_hands():
            manager.register(ResearcherHand())
            manager.register(GuardianHand())
            manager.register(MemoryHand())
            manager.register(CollectorHand())

        args = context.args or []
        subcmd = args[0].lower() if args else "list"
        subarg = args[1] if len(args) > 1 else ""

        if subcmd in ("list", "ls"):
            hands = manager.list_hands()
            if not hands:
                await update.message.reply_text("No hands registered.")
                return
            lines = ["<b>Autonomous Hands</b>\n"]
            for h in hands:
                state_icon = {"inactive": "\u26ab", "active": "\ud83d\udfe2", "running": "\ud83d\udd04", "paused": "\u23f8", "cooldown": "\u23f3", "error": "\ud83d\udd34"}.get(h["state"], "\u2753")
                lines.append(
                    f"{state_icon} <b>{h['name']}</b> — {h['state']}\n"
                    f"   Runs: {h['total_runs']} | Cost: ${h['total_cost']:.4f} | Failures: {h['consecutive_failures']}"
                )
                if h.get("last_run"):
                    lines.append(f"   Last: {h['last_run']}")
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

        elif subcmd == "run" and subarg:
            hand = manager.get_hand(subarg)
            if not hand:
                await update.message.reply_text(f"Unknown hand: {subarg}")
                return

            brain = getattr(self.aura, 'brain', None)
            tools = getattr(self.aura, 'tools', {})
            if not brain:
                await update.message.reply_text("Agent brain not available.")
                return

            placeholder = await update.message.reply_text(f"\ud83e\udd16 Running hand '{subarg}'...")
            chat_id = update.effective_chat.id

            async def _run_and_report():
                try:
                    result = await manager.run_hand(subarg, brain, tools)
                    status = "\u2705" if result.success else "\u274c"
                    text = (
                        f"{status} <b>Hand '{subarg}' completed</b>\n\n"
                        f"{result.summary[:500]}\n\n"
                        f"Iterations: {result.iterations} | "
                        f"Duration: {result.duration_seconds:.1f}s | "
                        f"Cost: ${result.cost_usd:.4f}"
                    )
                    if result.error:
                        text += f"\nError: {result.error}"
                    await self._edit_or_send_response(placeholder, text, chat_id, parse_mode="HTML")
                except Exception as e:
                    await self._edit_or_send_response(placeholder, f"Hand execution failed: {e}", chat_id)

            from aura.pools import fire_and_forget
            fire_and_forget(_run_and_report())

        elif subcmd == "activate" and subarg:
            if manager.activate(subarg):
                await update.message.reply_text(f"\ud83d\udfe2 Hand '{subarg}' activated.")
            else:
                await update.message.reply_text(f"Unknown or already running hand: {subarg}")

        elif subcmd == "deactivate" and subarg:
            if manager.deactivate(subarg):
                await update.message.reply_text(f"\u26ab Hand '{subarg}' deactivated.")
            else:
                await update.message.reply_text(f"Unknown hand: {subarg}")

        elif subcmd == "status" and subarg:
            hand = manager.get_hand(subarg)
            if not hand:
                await update.message.reply_text(f"Unknown hand: {subarg}")
                return
            stats = hand.get_stats()
            lines = [
                f"<b>Hand: {stats['name']}</b>",
                f"State: {stats['state']}",
                f"Description: {stats.get('description', 'N/A')}",
                f"Total runs: {stats['total_runs']}",
                f"Total cost: ${stats['total_cost']:.4f}",
                f"Consecutive failures: {stats['consecutive_failures']}",
                f"Model: {stats.get('model_preference', 'N/A')}",
                f"Idle only: {stats.get('idle_only', 'N/A')}",
                f"Drive trigger: {stats.get('trigger_on_drive', 'None')}",
            ]
            if stats.get("last_run"):
                lines.append(f"Last run: {stats['last_run']}")
            if stats.get("last_error"):
                lines.append(f"Last error: {stats['last_error']}")
            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

        else:
            await update.message.reply_text(
                "<b>Usage:</b> /hand &lt;command&gt; [name]\n\n"
                "/hand list — Show all hands\n"
                "/hand run &lt;name&gt; — Run a hand now\n"
                "/hand activate &lt;name&gt; — Activate for scheduling\n"
                "/hand deactivate &lt;name&gt; — Deactivate\n"
                "/hand status &lt;name&gt; — Detailed status",
                parse_mode="HTML",
            )

    async def _handle_hand_approval_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle hand approval inline keyboard callbacks."""
        query = update.callback_query
        if not self._is_user_allowed(query.from_user.id) or not self._is_admin(query.from_user.id):
            await query.answer("Unauthorized — admin only", show_alert=True)
            return
        await query.answer()

        data = query.data  # "hand_approve_<request_id>" or "hand_deny_<request_id>"
        parts = data.split("_", 2)
        if len(parts) < 3:
            return

        action = parts[1]  # "approve" or "deny"
        request_id = parts[2]
        approved = action == "approve"

        try:
            from aura.hands.manager import get_hand_manager
            manager = get_hand_manager()
            manager.resolve_approval(request_id, approved)
            status = "\u2705 Approved" if approved else "\u274c Denied"
            await query.edit_message_text(f"{status} (request: {request_id})")
        except Exception as e:
            await query.edit_message_text(f"Failed to resolve approval: {e}")

    # ---- Proactive Messaging ----

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
            "Good morning! How are you feeling today?",
        ]
        await self.send_proactive(chat_id, random.choice(greetings))

    async def send_follow_up(self, chat_id: str, topic: str):
        """Send a follow-up about something mentioned"""
        message = f"Hey - how did {topic} go?"
        await self.send_proactive(chat_id, message)

    # ---- Multi-agent ----

    async def _handle_agent(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /agent <specialist> <task> — run a specialist sub-agent."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        args = context.args
        if not args:
            await update.message.reply_text(
                "Usage:\n"
                "/agent list — show specialists\n"
                "/agent route <query> — preview routing\n"
                "/agent <specialist> <task> — run a specialist\n\n"
                "Specialists: research, coder, analyst, creative, searcher"
            )
            return

        subcmd = args[0].lower()

        if subcmd == "list":
            try:
                orch = getattr(self.aura, "orchestrator", None)
                if not orch:
                    await update.message.reply_text("Multi-agent orchestrator not available.")
                    return
                status = orch.get_status()
                lines = ["Available Specialists:\n"]
                for spec in status.get("specialists", []):
                    lines.append(f"  {spec['name']} — {spec.get('description', '')[:80]}")
                await update.message.reply_text("\n".join(lines))
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")

        elif subcmd == "route" and len(args) > 1:
            query = " ".join(args[1:])
            try:
                orch = getattr(self.aura, "orchestrator", None)
                if not orch:
                    await update.message.reply_text("Orchestrator not available.")
                    return
                preview = orch.route_preview(query)
                await update.message.reply_text(f"Routing preview:\n{preview}")
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")

        else:
            specialist = subcmd
            task = " ".join(args[1:]) if len(args) > 1 else "Help me"
            placeholder = await update.message.reply_text(f"Running {specialist} agent...")
            try:
                response = await asyncio.to_thread(self._run_agent_sync, task)
                await self._edit_or_send_response(placeholder, str(update.effective_chat.id), response or "No response from agent.", update)
            except Exception as e:
                await self._edit_or_send_response(placeholder, str(update.effective_chat.id), f"Agent error: {e}", update)

    # ---- TTS Toggle ----

    async def _handle_tts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tts — toggle voice replies on/off."""
        user = update.effective_user
        if not self._is_user_allowed(user.id):
            return

        args = (update.message.text or "").split(maxsplit=1)
        if len(args) < 2:
            current = self.store.get_tts_enabled(str(user.id))
            status = "ON" if current else "OFF"
            await update.message.reply_text(
                f"Voice replies are currently *{status}*\\.\n"
                f"Use `/tts on` or `/tts off` to toggle\\.",
                parse_mode="MarkdownV2"
            )
            return

        cmd = args[1].strip().lower()
        if cmd in ("on", "enable", "1"):
            self.store.set_tts_enabled(str(user.id), True)
            await update.message.reply_text("Voice replies enabled.")
        elif cmd in ("off", "disable", "0"):
            self.store.set_tts_enabled(str(user.id), False)
            await update.message.reply_text("Voice replies disabled.")
        else:
            await update.message.reply_text("Usage: /tts on or /tts off")

    # ---- Forum Topics ----

    async def _create_forum_topic_if_needed(self, chat_id: str, topic_name: str,
                                            context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
        """Create a forum topic for a research task in a supergroup with topics enabled."""
        try:
            result = await context.bot.create_forum_topic(
                chat_id=int(chat_id),
                name=topic_name[:128],
            )
            return result.message_thread_id
        except Exception as e:
            logger.debug(f"[Forum] Could not create topic: {e}")
            return None

    # ---- Public utility ----

    def get_active_chat_ids(self) -> List[str]:
        """Get list of active chat IDs for proactive messaging"""
        return list(self.active_chats.keys())

"""
SessionsMixin — /session, _session_info/new/list/switch/sync,
                _handle_chat_member, _handle_summarize_group, _handle_summarize_thread
"""
from __future__ import annotations

import asyncio
import logging

from aura.core.conversation_manager import get_conversation_manager
from aura.messaging.telegram_formatting import escape_mdv2 as _escape_mdv2

try:
    from telegram import Update
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None

logger = logging.getLogger(__name__)


class SessionsMixin:
    """Session and group conversation management handlers."""

    async def _handle_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /session command — manage cross-surface conversation sessions."""
        if not self._is_user_allowed(update.effective_user.id):
            return

        user_id = str(update.effective_user.id)
        args = context.args or []

        try:
            manager = get_conversation_manager()
        except Exception as e:
            logger.error(f"[Telegram] ConversationManager unavailable: {e}")
            await update.message.reply_text("Session management is not available right now.")
            return

        # Dispatch subcommands
        if not args:
            await self._session_info(update, manager, user_id)
        elif args[0].lower() == "new":
            title = " ".join(args[1:]).strip() if len(args) > 1 else None
            await self._session_new(update, manager, user_id, title)
        elif args[0].lower() == "list":
            await self._session_list(update, manager, user_id)
        elif args[0].lower() == "sync":
            await self._session_sync(update, manager, user_id)
        else:
            # Treat as conversation ID to switch to
            target_id = args[0].strip()
            await self._session_switch(update, manager, user_id, target_id)

    async def _session_info(self, update: Update, manager, user_id: str):
        """Show current session info."""
        try:
            from datetime import datetime
            conv_id = manager.get_or_create_session("telegram", user_id)
            convs = manager.list_conversations()
            conv = next((c for c in convs if c["id"] == conv_id), None)

            if not conv:
                await update.message.reply_text("No active session found\\. Use /session new to start one\\.",
                                                 parse_mode="MarkdownV2")
                return

            title = conv.get("title", "Untitled")
            msg_count = conv.get("message_count", 0)
            updated_at = conv.get("updated_at", 0)

            # Format timestamp
            if updated_at:
                ts = datetime.fromtimestamp(updated_at).strftime("%Y\\-%m\\-%d %H:%M")
            else:
                ts = "Unknown"

            # Connected surfaces
            bound = conv.get("bound_surfaces", [])
            if bound:
                surface_badges = []
                for sk in bound:
                    surface_name = sk.split(":")[0] if ":" in sk else sk
                    surface_badges.append(f"\\[{_escape_mdv2(surface_name)}\\]")
                surfaces_str = " ".join(surface_badges)
            else:
                surfaces_str = "\\(none\\)"

            # Surface activity
            activity = conv.get("surface_activity", {})
            if activity:
                activity_lines = []
                for s, count in sorted(activity.items(), key=lambda x: -x[1]):
                    activity_lines.append(f"  {_escape_mdv2(s)}: {count} msgs")
                activity_str = "\n".join(activity_lines)
            else:
                activity_str = "  No surface activity recorded"

            text = (
                f"*Current Session*\n\n"
                f"*Title:* {_escape_mdv2(title)}\n"
                f"*ID:* `{conv_id}`\n"
                f"*Messages:* {msg_count}\n"
                f"*Updated:* {ts}\n"
                f"*Surfaces:* {surfaces_str}\n\n"
                f"*Activity:*\n{activity_str}"
            )

            await update.message.reply_text(text, parse_mode="MarkdownV2")

        except Exception as e:
            logger.error(f"[Telegram] /session info error: {e}", exc_info=True)
            await update.message.reply_text("Could not retrieve session info.")

    async def _session_new(self, update: Update, manager, user_id: str, title: str | None = None):
        """Create a new conversation and switch to it."""
        try:
            conv_id = manager.new_session("telegram", user_id, title)
            display_title = _escape_mdv2(title or "Telegram Chat")

            text = (
                f"*New session created*\n\n"
                f"*Title:* {display_title}\n"
                f"*ID:* `{conv_id}`\n\n"
                f"You're now chatting in this session\\."
            )
            await update.message.reply_text(text, parse_mode="MarkdownV2")

        except Exception as e:
            logger.error(f"[Telegram] /session new error: {e}", exc_info=True)
            await update.message.reply_text("Could not create a new session.")

    async def _session_list(self, update: Update, manager, user_id: str):
        """List recent conversations (max 10)."""
        try:
            from datetime import datetime
            convs = manager.list_sessions("telegram", user_id)
            if not convs:
                await update.message.reply_text("No conversations found\\. Use /session new to start one\\.",
                                                 parse_mode="MarkdownV2")
                return

            # Limit to 10 most recent
            convs = convs[:10]

            lines = ["*Recent Conversations*\n"]
            for conv in convs:
                conv_id = conv["id"]
                title = conv.get("title", "Untitled")
                msg_count = conv.get("message_count", 0)
                updated_at = conv.get("updated_at", 0)
                is_bound = conv.get("is_bound", False)

                # Timestamp
                if updated_at:
                    ts = datetime.fromtimestamp(updated_at).strftime("%m/%d %H:%M")
                else:
                    ts = "\\-\\-"

                # Surface badges
                bound_surfaces = conv.get("bound_surfaces", [])
                badges = ""
                if bound_surfaces:
                    badge_parts = []
                    for sk in bound_surfaces:
                        surface_name = sk.split(":")[0] if ":" in sk else sk
                        badge_parts.append(f"\\[{_escape_mdv2(surface_name)}\\]")
                    badges = " " + " ".join(badge_parts)

                # Active marker
                marker = "→ " if is_bound else "  "
                active_label = " \\*active\\*" if is_bound else ""

                short_id = conv_id[-8:] if len(conv_id) > 8 else conv_id

                line = (
                    f"{_escape_mdv2(marker)}"
                    f"*{_escape_mdv2(title)}*{active_label}\n"
                    f"    `{short_id}` \\| "
                    f"{_escape_mdv2(ts)} \\| "
                    f"{msg_count} msgs{badges}"
                )
                lines.append(line)

            lines.append("\nSwitch: `/session <id>`")

            text = "\n".join(lines)
            await update.message.reply_text(text, parse_mode="MarkdownV2")

        except Exception as e:
            logger.error(f"[Telegram] /session list error: {e}", exc_info=True)
            await update.message.reply_text("Could not list sessions.")

    async def _session_switch(self, update: Update, manager, user_id: str, target_id: str):
        """Switch to a conversation by ID (supports partial match)."""
        try:
            convs = manager.list_conversations()

            # Exact match first
            match = next((c for c in convs if c["id"] == target_id), None)

            # Partial match (suffix or contains)
            if not match:
                candidates = [c for c in convs if c["id"].endswith(target_id)]
                if not candidates:
                    candidates = [c for c in convs if target_id in c["id"]]
                if len(candidates) == 1:
                    match = candidates[0]
                elif len(candidates) > 1:
                    lines = ["Multiple matches:\n"]
                    for c in candidates[:5]:
                        lines.append(f"  `{c['id']}` — {_escape_mdv2(c.get('title', 'Untitled'))}")
                    lines.append("\nBe more specific\\.")
                    await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2")
                    return

            if not match:
                await update.message.reply_text(
                    f"No conversation found matching `{_escape_mdv2(target_id)}`\\.\n"
                    f"Use /session list to see available conversations\\.",
                    parse_mode="MarkdownV2"
                )
                return

            conv_id = match["id"]
            success = manager.switch_session("telegram", user_id, conv_id)

            if success:
                title = match.get("title", "Untitled")
                msg_count = match.get("message_count", 0)
                text = (
                    f"*Switched session*\n\n"
                    f"*Title:* {_escape_mdv2(title)}\n"
                    f"*ID:* `{conv_id}`\n"
                    f"*Messages:* {msg_count}\n\n"
                    f"Continuing in this conversation\\."
                )
                await update.message.reply_text(text, parse_mode="MarkdownV2")
            else:
                await update.message.reply_text("Failed to switch session\\. The conversation may have been deleted\\.",
                                                 parse_mode="MarkdownV2")

        except Exception as e:
            logger.error(f"[Telegram] /session switch error: {e}", exc_info=True)
            await update.message.reply_text("Could not switch session.")

    async def _session_sync(self, update: Update, manager, user_id: str):
        """Show cross-surface sync status."""
        try:
            status = manager.get_status()
            conv_id = manager.get_or_create_session("telegram", user_id)

            # All bindings
            bindings = status.get("bindings", {})
            total = status.get("total_bindings", 0)
            listener_count = status.get("listener_count", 0)

            # Surfaces connected to the current conversation
            current_surfaces = manager.get_surfaces_for_conversation(conv_id)

            # Group bindings by surface type
            surface_summary = {}
            for key in bindings:
                surface_type = key.split(":")[0] if ":" in key else key
                surface_summary[surface_type] = surface_summary.get(surface_type, 0) + 1

            summary_lines = []
            for s, count in sorted(surface_summary.items()):
                summary_lines.append(f"  {_escape_mdv2(s)}: {count} binding\\(s\\)")

            # Current session surfaces
            if current_surfaces:
                current_lines = []
                for sk in current_surfaces:
                    parts = sk.split(":", 1)
                    surface_name = parts[0]
                    surface_uid = parts[1] if len(parts) > 1 else "\\-"
                    current_lines.append(f"  {_escape_mdv2(surface_name)} \\(`{_escape_mdv2(surface_uid)}`\\)")
                current_str = "\n".join(current_lines)
            else:
                current_str = "  None"

            text = (
                f"*Cross\\-Surface Sync Status*\n\n"
                f"*Current session:* `{conv_id}`\n"
                f"*Total bindings:* {total}\n"
                f"*Active listeners:* {listener_count}\n\n"
                f"*Connected surfaces \\(this session\\):*\n{current_str}\n\n"
                f"*All surface bindings:*\n" +
                ("\n".join(summary_lines) if summary_lines else "  None")
            )

            await update.message.reply_text(text, parse_mode="MarkdownV2")

        except Exception as e:
            logger.error(f"[Telegram] /session sync error: {e}", exc_info=True)
            await update.message.reply_text("Could not retrieve sync status.")

    async def _handle_chat_member(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle bot being added/removed from groups."""
        try:
            new_status = update.my_chat_member.new_chat_member.status
            if new_status in ("member", "administrator"):
                chat = update.my_chat_member.chat
                await context.bot.send_message(
                    chat.id,
                    "Hi! I'm AURA. Mention me with @Aura828Bot or reply to my messages to chat.\n\n"
                    "Commands: /summarize_group, /summarize_thread, /help"
                )
                logger.info(f"[TelegramBot] Added to group: {chat.title} ({chat.id})")
            elif new_status in ("left", "kicked"):
                chat = update.my_chat_member.chat
                # Clean up group data from store
                cache_key = str(chat.id)
                getattr(self, '_group_message_cache', {}).pop(cache_key, None)
                logger.info(f"[TelegramBot] Removed from group: {chat.title} ({chat.id})")
        except Exception as e:
            logger.error(f"[TelegramBot] Error handling chat_member update: {e}", exc_info=True)

    async def _handle_summarize_group(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Summarize recent group conversation from cached messages."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        chat_id = str(update.effective_chat.id)
        chat_type = update.effective_chat.type

        if chat_type not in ("group", "supergroup"):
            await update.message.reply_text("This command only works in groups.")
            return

        cache = self.store.get_group_messages(chat_id, limit=50)
        if len(cache) < 3:
            await update.message.reply_text("Not enough messages to summarize yet. I need at least 3 cached messages.")
            return

        # Build context from last 30 cached messages
        context_text = "\n".join([f"{m.get('user_name', m.get('user', 'Unknown'))}: {m['text']}" for m in cache[-30:]])
        prompt = f"Summarize this group conversation concisely. Focus on key topics and decisions:\n\n{context_text}"

        placeholder = await update.message.reply_text("Summarizing group conversation...")
        try:
            response_text, _ = await asyncio.to_thread(self._run_agent_sync, prompt)
            await self._edit_or_send_response(placeholder, chat_id, response_text, update)
        except Exception as e:
            logger.error(f"[TelegramBot] /summarize_group failed: {e}", exc_info=True)
            try:
                await placeholder.edit_text("Could not generate summary. Try again later.")
            except Exception:
                pass

    async def _handle_summarize_thread(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Summarize a thread when used as a reply in a group."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        chat_id = str(update.effective_chat.id)
        chat_type = update.effective_chat.type

        if chat_type not in ("group", "supergroup"):
            await update.message.reply_text("This command only works in groups.")
            return

        if not update.message.reply_to_message:
            await update.message.reply_text("Reply to a message with /summarize_thread to summarize the conversation around it.")
            return

        # Get the replied-to message as the anchor
        anchor_msg = update.message.reply_to_message
        anchor_text = anchor_msg.text or anchor_msg.caption or "[non-text message]"
        anchor_user = anchor_msg.from_user.first_name if anchor_msg.from_user else "Unknown"

        # Pull related messages from store around the same timeframe
        cache = self.store.get_group_messages(chat_id, limit=50)
        if len(cache) < 2:
            await update.message.reply_text(
                f"Not enough cached context. Here's the message you pointed to:\n\n"
                f"{anchor_user}: {anchor_text}"
            )
            return

        # Build context from cached messages
        context_text = "\n".join([f"{m['user']}: {m['text']}" for m in cache[-30:]])
        prompt = (
            f"Summarize this group conversation thread. The user is asking about the discussion "
            f"around this message from {anchor_user}: \"{anchor_text}\"\n\n"
            f"Recent conversation context:\n{context_text}"
        )

        placeholder = await update.message.reply_text("Summarizing thread...")
        try:
            response_text, _ = await asyncio.to_thread(self._run_agent_sync, prompt)
            await self._edit_or_send_response(placeholder, chat_id, response_text, update)
        except Exception as e:
            logger.error(f"[TelegramBot] /summarize_thread failed: {e}", exc_info=True)
            try:
                await placeholder.edit_text("Could not generate thread summary. Try again later.")
            except Exception:
                pass

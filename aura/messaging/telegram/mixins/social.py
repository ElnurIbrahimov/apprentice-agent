"""
SocialMixin — stickers, reactions, GIFs, edited messages callback,
              _handle_retry_callback, _try_native_reaction
"""
from __future__ import annotations

import asyncio
import logging
import random

from aura.messaging.telegram.constants import EMOTION_EMOJI, EMOTION_REACTIONS

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import ContextTypes
    try:
        from telegram import ReactionTypeEmoji
        REACTIONS_AVAILABLE = True
    except ImportError:
        REACTIONS_AVAILABLE = False
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    REACTIONS_AVAILABLE = False
    Update = None

logger = logging.getLogger(__name__)


class SocialMixin:
    """Sticker, GIF, reaction, and retry interaction handlers."""

    _EMOTION_TO_REACTION = {
        "joy": "\U0001f44d",         # thumbs up
        "excited": "\U0001f525",     # fire
        "curious": "\U0001f914",     # thinking
        "surprised": "\U0001f62e",   # open mouth
        "sad": "\U0001f622",         # crying
        "frustrated": "\U0001f44e",  # thumbs down
        "grateful": "\u2764\ufe0f",  # red heart
        "empathetic": "\U0001f917",  # hugging
        "confident": "\U0001f60e",   # sunglasses
        "neutral": "\U0001f44d",     # thumbs up
    }

    async def _handle_sticker(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """React when user sends a sticker."""
        if not self._is_user_allowed(update.effective_user.id):
            return
        sticker = update.message.sticker
        emoji = sticker.emoji or "🤔"
        responses = [
            f"Nice {emoji}!",
            f"I see you're feeling {emoji}",
            f"{emoji} right back at you!",
            f"Ooh, {emoji}!",
        ]
        await update.message.reply_text(random.choice(responses))

    async def _maybe_send_reaction(self, chat_id: str, emotion: str, intensity: float):
        """Maybe send a sticker or GIF based on current emotion."""
        if intensity < 0.5 or random.random() > 0.2:
            return

        reaction_info = EMOTION_REACTIONS.get(emotion, EMOTION_REACTIONS["neutral"])

        if random.random() > 0.5:
            await self._send_random_gif(chat_id, reaction_info["gif_queries"])
        else:
            await self._send_emoji_reaction(chat_id, emotion)

    async def _send_random_gif(self, chat_id: str, queries: list):
        """Send a random GIF using Tenor's API."""
        query = random.choice(queries)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = (
                    f"https://tenor.googleapis.com/v2/search"
                    f"?q={query}&limit=5&media_filter=gif"
                    f"&key=AIzaSyAyimkuYQYF_FXVALexPuGQctUWRURdCYQ"
                )
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        if results:
                            gif = random.choice(results)
                            gif_url = gif["media_formats"]["gif"]["url"]
                            await self.bot.send_animation(
                                chat_id=int(chat_id),
                                animation=gif_url,
                            )
        except Exception as e:
            logger.debug(f"GIF reaction failed (non-critical): {e}")

    async def _send_emoji_reaction(self, chat_id: str, emotion: str):
        """Send an emoji as a lightweight reaction message."""
        emoji = EMOTION_EMOJI.get(emotion, "👍")
        try:
            await self.bot.send_message(chat_id=int(chat_id), text=emoji)
        except Exception:
            pass

    async def _try_native_reaction(self, chat_id: str, message_id: int,
                                   emotion: str, intensity: float) -> bool:
        """Try to set a native Telegram reaction on the user's message."""
        if not REACTIONS_AVAILABLE or intensity < 0.4:
            return False
        emoji = self._EMOTION_TO_REACTION.get(emotion, "\U0001f44d")
        try:
            await self.bot.set_message_reaction(
                chat_id=int(chat_id),
                message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
            return True
        except Exception as e:
            logger.debug(f"[Reaction] Native reaction failed: {e}")
            return False

    async def _handle_retry_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle retry_<user_id> and retry_light_<user_id> callbacks."""
        query = update.callback_query
        if not self._is_user_allowed(query.from_user.id):
            await query.answer("Unauthorized", show_alert=True)
            return
        await query.answer("Retrying...")

        data = query.data  # e.g. "retry_12345" or "retry_light_12345"
        user_id = str(query.from_user.id)
        chat_id = str(query.message.chat_id)

        # Retrieve the failed message
        last_input = self._failed_messages.get(user_id, "")
        if not last_input:
            # Fallback: try last exchange from store
            exchange = self.store.get_skill_state(str(query.from_user.id)).get("last_exchange", {})
            last_input = exchange.get("input", "")
        if not last_input:
            await query.message.reply_text("Nothing to retry — I couldn't find the original message.")
            return

        is_light = "retry_light_" in data

        # Show typing
        placeholder = await query.message.reply_text(
            "\U0001f504 Retrying" + (" with lighter model..." if is_light else "...")
        )
        typing_task = asyncio.create_task(self._typing_loop(chat_id))

        try:
            goal = last_input
            if is_light:
                goal = f"[Use a fast, lightweight model] {last_input}"

            response_text, _artifacts = await asyncio.wait_for(
                asyncio.to_thread(self._run_agent_sync, goal),
                timeout=self._AGENT_TIMEOUT,
            )
        except Exception:
            response_text = "Retry also failed. Please try rephrasing your question."
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        # Clear the failed message on success
        if user_id in self._failed_messages:
            del self._failed_messages[user_id]

        action_buttons = self._get_action_buttons(query.message.message_id)
        await self._edit_or_send_response(placeholder, chat_id, response_text, update,
                                          reply_markup=action_buttons)

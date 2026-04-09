"""
CommandsMixin — /start, /help, /status, /mood, /memory, /forget, /image, /code
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
from datetime import datetime
from pathlib import Path

from aura.messaging.telegram.constants import _MAX_OUTPUT_CHARS
from aura.messaging.telegram.parsers import _extract_code_from_message

try:
    from telegram import (
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Update,
    )
    from telegram.constants import ChatAction, ParseMode
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None

logger = logging.getLogger(__name__)


class CommandsMixin:
    """Command handlers: /start, /help, /status, /mood, /memory, /forget, /image, /code"""

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start — interactive onboarding for new users, quick welcome for returning."""

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

        # Check if returning user (has settings already)
        existing = self.store.get_user_settings(str(user.id))
        if existing and existing.get("language"):
            # Returning user — quick welcome
            reply_markup = self._get_reply_keyboard()
            await update.message.reply_text(
                f"Welcome back, {user.first_name}! What's on your mind?",
                reply_markup=reply_markup,
            )
            return

        # === New user onboarding: Step 1 — Language ===
        self.store.set_keyboard_enabled(str(user.id), True)
        await update.message.reply_text(
            f"Hey {user.first_name}! \U0001f44b\n\n"
            f"I'm AURA \u2014 your AI thinking partner.\n\n"
            f"First, what language do you prefer?",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("\U0001f1ec\U0001f1e7 English", callback_data="onboard_lang_en"),
                    InlineKeyboardButton("\U0001f1f7\U0001f1fa \u0420\u0443\u0441\u0441\u043a\u0438\u0439", callback_data="onboard_lang_ru"),
                    InlineKeyboardButton("\U0001f1e6\U0001f1ff Az\u0259rbaycanca", callback_data="onboard_lang_az"),
                ],
            ]),
        )

    async def _handle_onboarding_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle onboard_* callbacks for the multi-step onboarding flow."""
        query = update.callback_query
        await query.answer()
        data = query.data
        user_id = str(query.from_user.id)
        first_name = query.from_user.first_name or "there"

        # --- Step 1 result: Language selected ---
        if data.startswith("onboard_lang_"):
            lang = data.split("_")[-1]
            self.store.set_user_language(user_id, lang)

            lang_names = {"en": "English", "ru": "\u0420\u0443\u0441\u0441\u043a\u0438\u0439", "az": "Az\u0259rbaycan"}
            lang_name = lang_names.get(lang, lang)

            # Step 2 — Capability showcase
            await query.message.edit_text(
                f"\u2705 Language set to {lang_name}.\n\n"
                f"Here's what I can do for you:\n\n"
                f"\U0001f9e0 **Think & Chat** \u2014 I'm not a search engine. I reason, remember context, and have opinions.\n\n"
                f"\U0001f50d **Deep Research** \u2014 /research any topic \u2014 I search the web, synthesize, and cite sources.\n\n"
                f"\U0001f3a8 **Create Images** \u2014 /image describe what you want and I'll generate it.\n\n"
                f"\U0001f4bb **Run Code** \u2014 /code python and I'll execute it live.\n\n"
                f"\U0001f4ca **Compare AI Models** \u2014 /compare a question across 3 models side by side.\n\n"
                f"\U0001f4c4 **Read Documents** \u2014 Send me PDFs, code files, images \u2014 I'll analyze them.\n\n"
                f"\U0001f399 **Voice Messages** \u2014 Send voice, I'll transcribe and respond.\n\n"
                f"Ready to try?",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("\U0001f680 Let's go!", callback_data="onboard_go"),
                        InlineKeyboardButton("\U0001f50d Try research", callback_data="onboard_try_research"),
                    ],
                    [
                        InlineKeyboardButton("\U0001f3a8 Try image gen", callback_data="onboard_try_image"),
                        InlineKeyboardButton("\U0001f4bb Try code", callback_data="onboard_try_code"),
                    ],
                ]),
            )

        # --- Step 2 results: capability demos ---
        elif data == "onboard_go":
            reply_markup = self._get_reply_keyboard()
            await query.message.edit_text(
                f"You're all set, {first_name}! \U0001f389\n\n"
                f"Just talk to me like a friend, or use /help to see all commands.\n\n"
                f"Tip: I have a quick-action keyboard below. Toggle it with /keyboard.",
            )
            await self.bot.send_message(
                chat_id=query.message.chat_id,
                text="What's on your mind?",
                reply_markup=reply_markup,
            )

        elif data == "onboard_try_research":
            await query.message.edit_text(
                "\U0001f50d Send me a /research topic and I'll do a deep dive!\n\n"
                "Example: `/research latest breakthroughs in quantum computing 2026`",
                parse_mode="Markdown",
            )
            reply_markup = self._get_reply_keyboard()
            await self.bot.send_message(
                chat_id=query.message.chat_id,
                text="Try it \u2014 type /research followed by any topic:",
                reply_markup=reply_markup,
            )

        elif data == "onboard_try_image":
            await query.message.edit_text(
                "\U0001f3a8 Send me /image with a description and I'll generate it!\n\n"
                "Example: `/image a cyberpunk city at sunset, neon lights reflecting on wet streets`",
                parse_mode="Markdown",
            )
            reply_markup = self._get_reply_keyboard()
            await self.bot.send_message(
                chat_id=query.message.chat_id,
                text="Try it \u2014 type /image followed by a description:",
                reply_markup=reply_markup,
            )

        elif data == "onboard_try_code":
            await query.message.edit_text(
                "\U0001f4bb Send me /code with Python code and I'll run it!\n\n"
                "Example: `/code print(sum(range(1, 101)))`",
                parse_mode="Markdown",
            )
            reply_markup = self._get_reply_keyboard()
            await self.bot.send_message(
                chat_id=query.message.chat_id,
                text="Try it \u2014 type /code followed by Python code:",
                reply_markup=reply_markup,
            )

    async def _handle_webapp_data(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle data sent from the Telegram Mini App via tg.sendData()."""
        import json
        if not update.effective_message or not update.effective_message.web_app_data:
            return

        user = update.effective_user
        if not user or not self._is_user_allowed(user.id):
            return

        raw = update.effective_message.web_app_data.data
        chat_id = str(update.effective_chat.id)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Treat as plain text command
            data = {"action": "command", "command": raw}

        action = data.get("action", "")

        if action == "command":
            # Mini App sent a /command — execute it as if the user typed it
            command = data.get("command", "").strip()
            if command:
                await self.send_typing_indicator(chat_id)
                # Route through agent if it's not a slash command
                if command.startswith("/"):
                    await update.effective_message.reply_text(
                        f"Running: {command}\n\nType the command directly in chat to execute it."
                    )
                else:
                    await self._run_agent_and_reply(
                        update, command,
                        user_id=str(user.id),
                    )
        elif action == "settings":
            # Mini App sent settings update
            settings = data.get("settings", {})
            uid = str(user.id)
            if "language" in settings:
                self.store.set_user_language(uid, settings["language"])
            if "model" in settings:
                self.store.set_user_model(uid, settings["model"])
            await update.effective_message.reply_text("\u2705 Settings updated from Mini App.")
        else:
            # Unknown action — pass through as chat
            text = data.get("text", raw)
            await self._run_agent_and_reply(update, text, user_id=str(user.id))

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
/session - Show current session info
/session new [title] - Start a new conversation
/session list - List recent conversations
/session <id> - Switch to a conversation by ID
/session sync - Cross-surface sync status
/learn - Learn a skill from our last exchange
/skill list [category] - List all learned skills
/skill search <query> - Search skills by description
/skill info <name> - Detailed skill info
/skill improve <name> - Evolve a skill with GEPA
/skill create <name> - Manually define a new skill

Reminders & Scheduled Tasks:
/remind in 2h Check the deployment
/remind at 17:00 Call the team
/remind tomorrow 9am Review PRs
/schedule every 2h Check CPU usage
/schedule daily at 9am Summarize notifications
/schedule every monday at 10am Weekly summary
/tasks - List active reminders & tasks
/cancel <id> - Cancel a reminder or task

Multi-Agent System:
/agent list - Show available specialists
/agent route <query> - Preview which agent handles a query
/agent <specialist> <task> - Run a specific specialist
/fleet <goal> - Decompose goal and run agents in parallel
/fleet status - Check running fleet status

Inline mode (use @Aura828Bot in any chat):
- Type any question to get a quick answer
- "translate <text>" - Translation
- "explain <topic>" - Explanation
- "summarize <text>" - Summary

Location:
- Share your location to get weather, timezone, sunrise/sunset info
/nearby <query> - Find places near your last shared location

Group Commands (in group chats):
/summarize_group - Summarize recent group conversation
/summarize_thread - Reply to a message to summarize the thread
- Mention @Aura828Bot or reply to my messages to chat in groups

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

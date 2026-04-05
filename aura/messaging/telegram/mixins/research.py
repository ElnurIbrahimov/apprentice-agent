"""
ResearchMixin — /research, /search, /summarize, /model, /compare, _fetch_url_content
"""
from __future__ import annotations

import asyncio
import logging
import re
import time as _time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

try:
    from telegram import Update
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Update = None

logger = logging.getLogger(__name__)


class ResearchMixin:
    """Research, search, summarize and model comparison handlers."""

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

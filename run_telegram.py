#!/usr/bin/env python3
"""
AURA Telegram Bot
=================

Connects AURA agent to Telegram for chat interaction.

Usage:
    python run_telegram.py

Set your bot token:
    export TELEGRAM_BOT_TOKEN="your_token_here"

Or create .env with:
    TELEGRAM_BOT_TOKEN=your_token_here
"""

import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path

# Prevent __pycache__ on server
if os.environ.get("AURA_ENV") == "production":
    sys.dont_write_bytecode = True

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging -- write to file on server, stdout always
_log_handlers = [logging.StreamHandler(sys.stdout)]
_log_dir = Path(os.getenv("AURA_DATA_DIR", "data")).parent / "logs"
if _log_dir.exists():
    _log_handlers.append(logging.FileHandler(_log_dir / "aura_telegram.log", encoding="utf-8"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_log_handlers,
)
logger = logging.getLogger("aura.telegram")

# Reduce noise from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_env():
    """Load environment variables from .env file.

    Existing env vars are NOT overwritten — shell exports take
    precedence over .env values, matching python-dotenv defaults.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()  # override=False by default
    except ImportError:
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('export '):
                        line = line[7:]
                    if "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key not in os.environ:
                            os.environ[key] = value


class TelegramAgentWrapper:
    """
    Wrapper that connects AURA agent to Telegram.

    Routes all messages through the agent's chat/run methods.
    """

    def __init__(self, agent):
        self.agent = agent
        self.aura = getattr(agent, 'aura', None)

        # Expose attributes for Telegram bot compatibility
        if self.aura:
            self.emotion = self.aura.emotion
            self.memory = self.aura.memory
            self.proactive = self.aura.proactive
        else:
            self.emotion = None
            self.memory = None
            self.proactive = None

        # Progress callback for long operations
        self._progress_callback = None

    def set_progress_callback(self, callback):
        """Set callback for progress messages."""
        self._progress_callback = callback

    def _send_progress(self, message: str):
        """Send progress update if callback is set."""
        if self._progress_callback:
            try:
                self._progress_callback(message)
            except Exception:
                pass

    def generate_response(self, user_message: str, chat_id: str = None) -> str:
        """Route ALL messages through agent.run() (full ReAct loop with tools).

        The agent's own fast-path and tool-routing logic decides whether to use
        tools or respond directly — no keyword matching needed here.

        Note: The Telegram bot now calls agent.run() directly via
        _run_agent_sync() instead of this method. This is kept for any
        external callers that still use the wrapper.
        """
        import time
        start_time = time.time()

        try:
            logger.info("agent.run() starting: %s", user_message[:80])
            result = self.agent.run(user_message, timeout_seconds=115)

            if isinstance(result, dict):
                if result.get("timeout"):
                    return "That request took too long. Please try a simpler query."

                response = result.get("response", "")
                if not response:
                    fe = result.get("final_evaluation", {})
                    response = fe.get("progress", "")
                if not response:
                    history = result.get("history", [])
                    if history:
                        last_entry = history[-1]
                        if isinstance(last_entry, dict):
                            response = last_entry.get("result", {}).get("output", str(last_entry))

                elapsed = time.time() - start_time
                logger.info("agent.run() completed in %.1fs", elapsed)
                return response if response else "I processed your request but couldn't find a clear answer."

            return str(result) if result else "No response generated."

        except Exception as e:
            logger.warning("agent.run() failed: %s, falling back to chat()", e)

        # Fallback to chat() if run() fails
        try:
            response = self.agent.chat(user_message)
            elapsed = time.time() - start_time
            logger.info("agent.chat() fallback completed in %.1fs", elapsed)
            return response if response else "I couldn't generate a response."
        except Exception as e:
            logger.error("generate_response failed: %s\n%s", e, traceback.format_exc())
            return f"Sorry, something went wrong: {str(e)[:100]}"

    def get_status(self):
        """Get agent status."""
        status = {
            "version": "AURA Telegram",
            "tools": len(self.agent.tools),
            "mood": {},
            "patterns": {},
            "turns": 0
        }

        if self.aura:
            try:
                aura_status = self.aura.get_status()
                status["mood"] = aura_status.get("mood", {})
                status["patterns"] = aura_status.get("patterns", {})
                status["turns"] = aura_status.get("turns", 0)
            except Exception as e:
                logger.warning("Failed to get AURA status: %s", e)

        return status


async def main():
    load_env()

    # Check for token
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "YOUR_BOT_TOKEN_HERE":
        logger.error("TELEGRAM_BOT_TOKEN not set!")
        print("")
        print("=" * 50)
        print("TELEGRAM_BOT_TOKEN not set!")
        print("=" * 50)
        print("")
        print("To get a token:")
        print("  1. Open Telegram and search for @BotFather")
        print("  2. Send /newbot and follow the instructions")
        print("  3. Copy the token you receive")
        print("")
        print("Then set it:")
        print("  export TELEGRAM_BOT_TOKEN='your_token_here'")
        print("")
        print("Or create a .env file with:")
        print("  TELEGRAM_BOT_TOKEN=your_token_here")
        print("")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  AURA Telegram Bot starting")
    logger.info("=" * 60)

    # Load ApprenticeAgent
    try:
        from aura.agent import ApprenticeAgent
        logger.info("Loading ApprenticeAgent (this may take a moment)...")
        agent = ApprenticeAgent(fast_init=False)
        tools_count = len(agent.tools) if hasattr(agent, 'tools') else 0
        logger.info("ApprenticeAgent loaded with %d tools", tools_count)

        # Wrap agent for Telegram
        wrapped = TelegramAgentWrapper(agent)

        if wrapped.aura:
            try:
                logger.info("AURA: Soul=%s, Mood=%s",
                            wrapped.aura.soul.name,
                            wrapped.aura.emotion.state.mood.value)
            except Exception:
                logger.info("AURA soul loaded (could not read mood)")

    except ImportError as e:
        logger.critical("Failed to import ApprenticeAgent: %s\n%s", e, traceback.format_exc())
        print(f"\nMissing module: {e}")
        print("This usually means a required package is not installed.")
        print("Run: pip install -r requirements.txt")
        return
    except Exception as e:
        logger.critical("Error loading ApprenticeAgent: %s\n%s", e, traceback.format_exc())
        return

    # Initialize Telegram bot
    try:
        from aura.messaging.telegram_bot import TelegramBot
        from aura.messaging.config import TELEGRAM_CONFIG

        TELEGRAM_CONFIG["telegram_token"] = token
        bot = TelegramBot(wrapped, TELEGRAM_CONFIG)

    except ImportError as e:
        logger.critical("Missing telegram library: %s", e)
        print(f"\nError: {e}")
        print("")
        print("Install required packages:")
        print("  pip install python-telegram-bot>=20.0")
        return

    try:
        await bot.start()

        logger.info("AURA Telegram bot is now ALIVE!")
        logger.info("Press Ctrl+C to stop")

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down (keyboard interrupt)...")
    except Exception as e:
        logger.critical("Telegram bot crashed: %s\n%s", e, traceback.format_exc())
    finally:
        try:
            await bot.stop()
        except Exception as e:
            logger.warning("Error during bot shutdown: %s", e)
        logger.info("AURA Telegram bot stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.critical("Fatal error: %s\n%s", e, traceback.format_exc())
        sys.exit(1)

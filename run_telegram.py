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
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Reduce noise from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def load_env():
    """Load environment variables from .env file"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        value = value.strip().strip('"').strip("'")
                        os.environ[key.strip()] = value


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
        """Route message through the agent."""
        import time
        start_time = time.time()

        msg_lower = user_message.lower()

        # Tool detection - certain queries need the full agent loop
        tool_triggers = [
            "search", "look up", "find out", "google", "browse", "open website",
            "what files", "list files", "read file", "create file", "delete file",
            "run code", "execute", "python", "screenshot", "take a picture",
            "download", "fetch", "get the", "check the web", "latest news",
            "current price", "weather", "stock", "bitcoin", "crypto",
            "arxiv", "paper", "research", "pdf", "document"
        ]

        research_triggers = ["deep research", "research thoroughly", "thorough research", "investigate"]
        is_research = any(trigger in msg_lower for trigger in research_triggers)
        needs_tools = any(trigger in msg_lower for trigger in tool_triggers)

        try:
            if needs_tools:
                if is_research:
                    self._send_progress("Researching... This may take up to 60 seconds.")
                    print(f"[RESEARCH] Starting: {user_message[:50]}...")
                else:
                    print(f"[TOOLS] Routing to agent.run(): {user_message[:50]}...")

                result = self.agent.run(user_message, timeout_seconds=90)

                if isinstance(result, dict) and result.get("timeout"):
                    return "That request took too long. Please try a simpler query."

                if isinstance(result, dict):
                    response = result.get("response") or result.get("final_evaluation", {}).get("progress", "")
                    if not response:
                        history = result.get("history", [])
                        if history:
                            last_entry = history[-1]
                            response = last_entry.get("result", {}).get("output", str(last_entry))
                    elapsed = time.time() - start_time
                    print(f"[TOOLS] Completed in {elapsed:.1f}s")
                    return response if response else "I processed your request but couldn't find a clear answer."
                return str(result)
            else:
                response = self.agent.chat(user_message)
                elapsed = time.time() - start_time
                print(f"[CHAT] Completed in {elapsed:.1f}s")
                return response

        except Exception as e:
            print(f"[ERROR] generate_response failed: {e}")
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
            aura_status = self.aura.get_status()
            status["mood"] = aura_status.get("mood", {})
            status["patterns"] = aura_status.get("patterns", {})
            status["turns"] = aura_status.get("turns", 0)

        return status


async def main():
    load_env()

    # Check for token
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "YOUR_BOT_TOKEN_HERE":
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
        return

    print("")
    print("=" * 60)
    print("  AURA Telegram Bot")
    print("=" * 60)
    print("")

    # Load ApprenticeAgent
    try:
        from aura.agent import ApprenticeAgent
        print("Loading ApprenticeAgent (this may take a moment)...")
        agent = ApprenticeAgent(fast_init=False)
        print(f"ApprenticeAgent loaded with {len(agent.tools)} tools")

        # Wrap agent for Telegram
        wrapped = TelegramAgentWrapper(agent)

        if wrapped.aura:
            print(f"AURA: Soul={wrapped.aura.soul.name}, Mood={wrapped.aura.emotion.state.mood.value}")

    except Exception as e:
        print(f"Error loading ApprenticeAgent: {e}")
        import traceback
        traceback.print_exc()
        return

    # Initialize Telegram bot
    try:
        from aura.messaging.telegram_bot import TelegramBot
        from aura.messaging.config import TELEGRAM_CONFIG

        TELEGRAM_CONFIG["telegram_token"] = token
        bot = TelegramBot(wrapped, TELEGRAM_CONFIG)

    except ImportError as e:
        print(f"Error: {e}")
        print("")
        print("Install required packages:")
        print("  pip install python-telegram-bot>=20.0")
        return

    try:
        await bot.start()

        print("")
        print("=" * 60)
        print("  AURA is now ALIVE on Telegram!")
        print("")
        print("  Open Telegram and message your bot.")
        print("  Press Ctrl+C to stop")
        print("=" * 60)
        print("")

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("")
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await bot.stop()
        print("AURA Telegram bot stopped cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

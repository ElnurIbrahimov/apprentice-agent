#!/usr/bin/env python3
"""
Quick start script for AURA Telegram bot.

NOW USES FULL APPRENTICE AGENT with:
- 29 tools (web_search, browser, filesystem, code_executor, etc.)
- 10 cognitive systems (EvoEmo, MirrorMind, CognitiveTheater, etc.)
- AURA's Clawdbot features (soul, heartbeat, humanizer, memory)

Usage:
    python run_telegram.py

Make sure to set your bot token:
    export TELEGRAM_BOT_TOKEN="your_token_here"

Or create a .env file with:
    TELEGRAM_BOT_TOKEN=your_token_here

Get a token from @BotFather on Telegram.
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
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value


class AgentWrapper:
    """
    Wrapper that makes ApprenticeAgent look like AURAEngine for TelegramBot.

    ApprenticeAgent has: .chat(), .aura (AURAEngine inside)
    TelegramBot expects: .generate_response(), .emotion, .memory, .proactive
    """

    def __init__(self, agent):
        self.agent = agent
        # Expose AURA's components through the wrapper
        self.aura = agent.aura  # The AURAEngine inside ApprenticeAgent

        # Expose commonly needed attributes from inner AURA
        if self.aura:
            self.emotion = self.aura.emotion
            self.memory = self.aura.memory
            self.proactive = self.aura.proactive
            self.fast_path = self.aura.fast_path
            self.memory_retriever = self.aura.memory_retriever
            self.llm = self.aura.llm
        else:
            self.emotion = None
            self.memory = None
            self.proactive = None
            self.fast_path = None
            self.memory_retriever = None
            self.llm = None

    def set_progress_callback(self, callback):
        """Set callback for sending progress messages (e.g., to Telegram)."""
        self._progress_callback = callback

    def _send_progress(self, message: str):
        """Send progress update if callback is set."""
        if hasattr(self, '_progress_callback') and self._progress_callback:
            try:
                self._progress_callback(message)
            except:
                pass

    def generate_response(self, user_message: str, chat_id: str = None) -> str:
        """
        Smart routing with timeout protection:
        - Tool queries (search, browse, files, etc.) -> agent.run() with tools
        - Simple conversation -> agent.chat() for fast response
        """
        import time
        start_time = time.time()

        msg_lower = user_message.lower()

        # Detect if message needs tools
        tool_triggers = [
            "search", "look up", "find out", "google", "browse", "open website",
            "what files", "list files", "read file", "create file", "delete file",
            "run code", "execute", "python", "screenshot", "take a picture",
            "download", "fetch", "get the", "check the web", "latest news",
            "current price", "weather", "stock", "bitcoin", "crypto",
            "arxiv", "paper", "research", "pdf", "document"
        ]

        # Deep research triggers - need progress message
        research_triggers = ["deep research", "research thoroughly", "thorough research", "investigate"]
        is_research = any(trigger in msg_lower for trigger in research_triggers)

        needs_tools = any(trigger in msg_lower for trigger in tool_triggers)

        try:
            if needs_tools:
                # Send progress message for research tasks
                if is_research:
                    self._send_progress("🔍 Researching... This may take up to 60 seconds.")
                    print(f"[RESEARCH] Starting deep research: {user_message[:50]}...")
                else:
                    print(f"[TOOLS] Routing to agent.run(): {user_message[:50]}...")

                # Use full agent loop with tools
                result = self.agent.run(user_message, timeout_seconds=90)  # 90 second timeout

                # Check for timeout
                if isinstance(result, dict) and result.get("timeout"):
                    return "That request took too long. Please try a simpler query."

                # Extract response from result dict
                if isinstance(result, dict):
                    response = result.get("response") or result.get("final_evaluation", {}).get("progress", "")
                    if not response:
                        # Try to get from history
                        history = result.get("history", [])
                        if history:
                            last_entry = history[-1]
                            response = last_entry.get("result", {}).get("output", str(last_entry))
                    elapsed = time.time() - start_time
                    print(f"[TOOLS] Completed in {elapsed:.1f}s")
                    return response if response else "I processed your request but couldn't find a clear answer."
                return str(result)
            else:
                # Simple conversation - use chat() for speed
                response = self.agent.chat(user_message)
                elapsed = time.time() - start_time
                print(f"[CHAT] Completed in {elapsed:.1f}s")
                return response
        except Exception as e:
            print(f"[ERROR] generate_response failed: {e}")
            return f"Sorry, something went wrong: {str(e)[:100]}"

    def get_status(self):
        """Get combined status from agent and AURA."""
        status = {
            "version": "4.0 MERGED",
            "soul": "ApprenticeAgent + AURA",
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
    print("=" * 50)
    print("Starting ApprenticeAgent + AURA Telegram Bot")
    print("=" * 50)
    print("")

    # Load FULL ApprenticeAgent (includes AURA + 29 tools + cognitive systems)
    try:
        from apprentice_agent.agent import ApprenticeAgent
        print("Loading ApprenticeAgent (this may take a moment)...")
        agent = ApprenticeAgent(fast_init=False)  # Full init with all tools
        print(f"ApprenticeAgent loaded with {len(agent.tools)} tools")

        # Wrap agent to expose AURA-compatible interface
        wrapped = AgentWrapper(agent)

        if wrapped.aura:
            print(f"AURA inside: Soul={wrapped.aura.soul.name}, Mood={wrapped.aura.emotion.state.mood.value}")

    except Exception as e:
        print(f"Error loading ApprenticeAgent: {e}")
        print("Falling back to AURA-only mode...")

        # Fallback to just AURA if ApprenticeAgent fails
        try:
            from aura.engine import AURAEngine
            wrapped = AURAEngine()
            print("AURA engine loaded (fallback mode - no tools)")
        except Exception as e2:
            print(f"Error loading AURA: {e2}")
            print("Running with minimal engine...")

            class MinimalAura:
                def __init__(self):
                    self.memory = None
                    self.emotion = None
                    self.proactive = None

                def generate_response(self, msg, chat_id=None):
                    return "I'm here but running in minimal mode. Some features unavailable."

                def get_status(self):
                    return {"version": "minimal", "soul": "none", "mood": {}, "patterns": {}, "turns": 0}

            wrapped = MinimalAura()

    # Import and initialize Telegram
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
        print("=" * 50)
        print("ApprenticeAgent + AURA is now running on Telegram!")
        print("")
        print("Features available:")
        print("  - Natural conversation with memory")
        print("  - Web search, browser, code execution")
        print("  - Emotional awareness (EvoEmo)")
        print("  - Multi-perspective reasoning (CognitiveTheater)")
        print("  - Self-critique (MirrorMind)")
        print("  - Proactive messaging (Heartbeat)")
        print("")
        print("Open Telegram and message your bot to start chatting.")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        print("")

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("")
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await bot.stop()
        print("AURA stopped cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

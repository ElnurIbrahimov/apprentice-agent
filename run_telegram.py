#!/usr/bin/env python3
"""
Proto-AGI v5 Telegram Bot - TRUTH SPINE
=======================================

This version ENFORCES verification at every step.

Features:
- TRUTH SPINE: Non-negotiable verification layer
- ARTIFACTS: Physical proof (file hash, stdout, return code, JSON)
- 3-TIER MEMORY: FACT (verified), BELIEF (inferred), SPECULATION (unverified)
- SECURE EXECUTOR: Confirmation required for dangerous operations
- SANDBOX ENFORCEMENT: Not suggestions, enforcement

The Contract:
    ACTION → ARTIFACT → VERIFICATION → MEMORY TIER

Core Principle: "If you can't verify it with an artifact, it's SPECULATION"

Modes:
- idle: Think internally only, no external actions
- assist: Act only in response to user (DEFAULT)
- operate: Autonomous actions under budgets (10/hr, 3 msgs/hr)

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
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value


class ProtoAGIWrapper:
    """
    Wrapper that connects Proto-AGI to Telegram.

    Routes all messages through Proto-AGI's process_input() which:
    - Updates needs (connection satisfied)
    - Stores in memory
    - Recalls relevant context
    - Generates personality-aware response
    """

    def __init__(self, agent):
        self.agent = agent
        self.proto_agi = agent.proto_agi
        self.aura = agent.aura  # Keep AURA reference for compatibility

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
        """
        Route message through Proto-AGI.

        Proto-AGI handles:
        - Need satisfaction (connection +40)
        - Memory storage and recall
        - Personality/emotion-colored response
        - Tool detection happens separately
        """
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
                # Tool-based query - use agent.run() with timeout
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
                # Conversational query - route through Proto-AGI
                if self.proto_agi:
                    response = self.proto_agi.process_input(user_message, chat_id)
                    elapsed = time.time() - start_time
                    print(f"[PROTO-AGI] Completed in {elapsed:.1f}s")
                    return response
                else:
                    # Fallback to regular chat
                    response = self.agent.chat(user_message)
                    elapsed = time.time() - start_time
                    print(f"[CHAT] Completed in {elapsed:.1f}s")
                    return response

        except Exception as e:
            print(f"[ERROR] generate_response failed: {e}")
            return f"Sorry, something went wrong: {str(e)[:100]}"

    def get_status(self):
        """Get combined status from Proto-AGI v5 and agent."""
        status = {
            "version": "5.0 PROTO-AGI-v5-TRUTH-SPINE",
            "soul": "Truth Spine Verification-First Cognition",
            "tools": len(self.agent.tools),
            "mood": {},
            "patterns": {},
            "turns": 0
        }

        if self.proto_agi:
            agi_status = self.proto_agi.get_status()
            status["mode"] = agi_status.get("mode", "assist")
            status["needs"] = agi_status.get("needs", {})
            status["memory"] = agi_status.get("memory", {})
            status["governance"] = agi_status.get("governance", {})
            status["cycle_count"] = agi_status.get("cycle_count", 0)
            status["running"] = agi_status.get("running", False)
            status["agi_version"] = agi_status.get("version", "v3")

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
    print("  PROTO-AGI v5 - TRUTH SPINE VERIFICATION")
    print("=" * 60)
    print("")

    # Load ApprenticeAgent with Proto-AGI
    try:
        from aura.agent import ApprenticeAgent
        print("Loading ApprenticeAgent + Proto-AGI (this may take a moment)...")
        agent = ApprenticeAgent(fast_init=False)
        print(f"ApprenticeAgent loaded with {len(agent.tools)} tools")

        # Wrap agent for Telegram
        wrapped = ProtoAGIWrapper(agent)

        if wrapped.proto_agi:
            status = wrapped.proto_agi.get_status()
            mem = status.get('memory', {})
            gov = status.get('governance', {})
            verifier = status.get('verifier', {})
            print(f"Proto-AGI v5: Mode={status.get('mode', 'assist')}, Cycle={status['cycle_count']}")
            print(f"  Memory: {mem.get('facts', 0)} FACTS, {mem.get('beliefs', 0)} BELIEFS, {mem.get('speculations', 0)} SPECULATIONS")
            print(f"  Verifier: {verifier.get('total_verifications', 0)} checks, {verifier.get('success_rate', 0):.0%} pass rate")
            print(f"  Budget: {gov.get('actions_remaining', 10)}/10 actions, {gov.get('messages_remaining', 3)}/3 messages")
        else:
            print("[WARNING] Proto-AGI not available - using fallback mode")

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

    # Setup Proto-AGI output callback for proactive messages
    if agent.proto_agi:
        def telegram_output_callback(message: str, chat_id: str = None):
            """Send proactive message to Telegram"""
            if chat_id and bot.bot:
                try:
                    # Create async task for sending
                    asyncio.create_task(
                        bot.bot.send_message(chat_id=int(chat_id), text=message)
                    )
                except Exception as e:
                    print(f"[Proto-AGI] Failed to send proactive message: {e}")

        agent.set_proto_agi_output_callback(telegram_output_callback)

    try:
        await bot.start()

        # Start Proto-AGI autonomous loop
        if agent.proto_agi:
            agent.start_proto_agi(cycle_interval=60.0)  # Think every 60 seconds
            print("")
            print("[Proto-AGI] Autonomous loop STARTED (60s interval)")

        print("")
        print("=" * 60)
        print("  Proto-AGI v5 is now ALIVE on Telegram!")
        print("")
        print("  TRUTH SPINE - Non-Negotiable Verification:")
        print("    - ACTION → ARTIFACT → VERIFICATION → MEMORY TIER")
        print("    - FACT: verified with artifact (hash, return code)")
        print("    - BELIEF: inferred but not proven")
        print("    - SPECULATION: unverified (including LLM output)")
        print("")
        print("  \"If you can't verify it with an artifact, it's SPECULATION\"")
        print("")
        print("  MODES: idle | assist (default) | operate")
        print("    - assist: Responds to user only")
        print("    - operate: Autonomous actions (10/hr, 3 msgs/hr)")
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
        # Stop Proto-AGI loop
        if agent.proto_agi:
            agent.stop_proto_agi()
        await bot.stop()
        print("Proto-AGI v5 stopped cleanly")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

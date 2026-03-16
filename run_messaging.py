#!/usr/bin/env python3
"""
Run AURA with all enabled messaging platforms.

Usage:
    python run_messaging.py

Configure platforms in aura/messaging/config.py or via environment variables.
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

logger = logging.getLogger(__name__)


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


async def main():
    load_env()

    print("")
    print("=" * 50)
    print("AURA Messaging System")
    print("=" * 50)
    print("")

    # Import ApprenticeAgent (wraps ALMA + unified memory)
    try:
        from aura.agent import ApprenticeAgent
        print("Loading ApprenticeAgent...")
        agent = ApprenticeAgent(fast_init=True)
        aura = agent  # MessageRouter uses the aura interface
        print(f"ApprenticeAgent loaded with {len(agent.tools)} tools")
    except Exception as e:
        logger.error(f"Error loading ApprenticeAgent: {e}")
        print("Running with minimal AURA...")

        class MinimalAura:
            def __init__(self):
                self.memory = None
                self.emotion = None

            def get_status(self):
                return {"version": "3.0", "soul": "AURA", "mood": {}, "patterns": {}, "turns": 0}

        aura = MinimalAura()

    # Import and initialize router
    try:
        from aura.messaging.router import MessageRouter

        router = MessageRouter(aura)

    except ImportError as e:
        print(f"Error: {e}")
        print("")
        print("Install required packages:")
        print("  pip install python-telegram-bot>=20.0 websockets")
        return

    try:
        await router.start()

        if not router.platforms:
            print("")
            print("No platforms started. Check configuration.")
            return

        print("")
        print("=" * 50)
        print("AURA Messaging is running!")
        print("")
        print(f"Platforms: {', '.join(router.platforms.keys())}")
        print("")
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
        logger.error(f"Error: {e}")
    finally:
        await router.stop()

        # Shutdown AURA
        if hasattr(aura, 'shutdown'):
            aura.shutdown()

        print("AURA messaging stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

"""
Central Message Router for AURA

Routes messages between AURA core and various platforms.
Also handles proactive message distribution.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, List

from .config import load_config, MESSAGING_CONFIG
from .base_platform import BasePlatform, OutgoingMessage

logger = logging.getLogger(__name__)


class MessageRouter:
    """
    Central router that:
    1. Manages all platform connections
    2. Routes incoming messages to AURA
    3. Distributes proactive messages to platforms
    4. Handles cross-platform state
    """

    def __init__(self, aura_engine):
        self.aura = aura_engine
        self.config = load_config()

        self.platforms: Dict[str, BasePlatform] = {}
        self.is_running = False

        # Initialize platforms
        self._init_platforms()

    def _init_platforms(self):
        """Initialize enabled platforms"""

        general = self.config.get("general", {})

        # Telegram
        if general.get("enable_telegram", True):
            try:
                from .telegram_bot import TelegramBot

                telegram_config = self.config.get("telegram", {})
                token = telegram_config.get("telegram_token", "")

                if token and token != "YOUR_BOT_TOKEN_HERE":
                    self.platforms["telegram"] = TelegramBot(self.aura, telegram_config)
                    logger.info("Telegram platform initialized")
                else:
                    logger.warning("Telegram token not configured - skipping Telegram")
                    logger.info("Set TELEGRAM_BOT_TOKEN environment variable or edit config.py")

            except ImportError as e:
                logger.warning(f"Telegram dependencies not installed: {e}")
                logger.info("Run: pip install python-telegram-bot>=20.0")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram: {e}")

        # WhatsApp
        if general.get("enable_whatsapp", False):
            try:
                from .whatsapp_bot import WhatsAppBot

                whatsapp_config = self.config.get("whatsapp", {})
                self.platforms["whatsapp"] = WhatsAppBot(self.aura, whatsapp_config)
                logger.info("WhatsApp platform initialized")

            except ImportError as e:
                logger.warning(f"WhatsApp dependencies not installed: {e}")
                logger.info("Run: pip install websockets")
            except Exception as e:
                logger.error(f"Failed to initialize WhatsApp: {e}")

    async def start(self):
        """Start all platforms"""

        logger.info("Starting Message Router...")
        self.is_running = True

        if not self.platforms:
            logger.warning("No messaging platforms configured!")
            logger.info("")
            logger.info("To enable Telegram:")
            logger.info("  1. Get a token from @BotFather on Telegram")
            logger.info("  2. Set: export TELEGRAM_BOT_TOKEN='your_token'")
            logger.info("  3. Run this script again")
            logger.info("")
            return

        # Start each platform
        for name, platform in self.platforms.items():
            try:
                await platform.start()
                logger.info(f"{name} started")
            except Exception as e:
                logger.error(f"Failed to start {name}: {e}")

        # Start proactive message loop
        if MESSAGING_CONFIG.get("proactive_enabled", True):
            self._proactive_task = asyncio.create_task(self._proactive_loop())

        logger.info("Message Router started!")

    async def stop(self):
        """Stop all platforms"""

        logger.info("Stopping Message Router...")
        self.is_running = False

        for name, platform in self.platforms.items():
            try:
                await platform.stop()
                logger.info(f"{name} stopped")
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")

        logger.info("Message Router stopped.")

    async def send_to_all(self, message: str):
        """Send a message to all active chats on all platforms"""

        for platform in self.platforms.values():
            if hasattr(platform, 'send_to_all_active'):
                await platform.send_to_all_active(message)

    async def send_to_platform(self, platform_name: str, chat_id: str, message: str):
        """Send a message to a specific platform and chat"""

        if platform_name in self.platforms:
            await self.platforms[platform_name].send_message(
                OutgoingMessage(chat_id=chat_id, text=message)
            )

    async def _proactive_loop(self):
        """Background loop for proactive messaging"""

        check_interval = MESSAGING_CONFIG.get("proactive_check_interval", 300)

        while self.is_running:
            try:
                await self._check_proactive_triggers()
            except Exception as e:
                logger.error(f"Error in proactive loop: {e}")

            await asyncio.sleep(check_interval)

    async def _check_proactive_triggers(self):
        """Check if any proactive messages should be sent"""

        now = datetime.now()
        current_hour = now.hour

        # Check quiet hours
        quiet_start = MESSAGING_CONFIG.get("quiet_hours_start", 22)
        quiet_end = MESSAGING_CONFIG.get("quiet_hours_end", 8)

        if quiet_start <= current_hour or current_hour < quiet_end:
            return  # In quiet hours

        # Get proactive messages from AURA if available
        if hasattr(self.aura, 'proactive'):
            try:
                pending = self.aura.proactive.get_pending_notifications()

                for notification in pending:
                    message = notification.message

                    # Send to all platforms
                    await self.send_to_all(message)

            except Exception as e:
                logger.debug(f"Proactive check: {e}")

    def get_all_active_chats(self) -> List[dict]:
        """Get all active chats across all platforms"""

        chats = []
        for name, platform in self.platforms.items():
            if hasattr(platform, 'active_chats'):
                for chat_id, info in platform.active_chats.items():
                    chats.append({
                        "platform": name,
                        "chat_id": chat_id,
                        **info
                    })
        return chats

    def get_status(self) -> dict:
        """Get router status"""
        return {
            "running": self.is_running,
            "platforms": list(self.platforms.keys()),
            "active_chats": len(self.get_all_active_chats()),
            "proactive_enabled": MESSAGING_CONFIG.get("proactive_enabled", True)
        }

"""
Messaging Platform Configuration

Store your tokens and settings here.
IMPORTANT: Don't commit real tokens to git!
"""

import os
from pathlib import Path

# ============ TELEGRAM CONFIG ============

TELEGRAM_CONFIG = {
    # Bot token from @BotFather
    "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE"),

    # SECURITY: Allowed user IDs — REQUIRED for bot to respond.
    # Empty list = deny ALL users (fail-closed). Bot will reject messages
    # and log a warning until you add at least one user ID here.
    # Get your ID by messaging @userinfobot on Telegram.
    # Can also set via env var: TELEGRAM_ALLOWED_USERS=123456789,987654321
    "allowed_users": [
        # "123456789",  # Your Telegram user ID
    ],

    # Admin user IDs (can use admin commands)
    "admin_users": [
        # "123456789",  # Your Telegram user ID
    ],

    # Rate limiting
    "max_messages_per_minute": 20,

    # Proactive messaging settings
    "proactive_enabled": True,
    "morning_greeting_hour": 9,
    "evening_wrapup_hour": 21,
    "quiet_hours_start": 22,
    "quiet_hours_end": 8,
}

# ============ WHATSAPP CONFIG ============

WHATSAPP_CONFIG = {
    # WebSocket URL for Baileys bridge
    "websocket_url": "ws://localhost:3001",

    # WhatsApp session storage path
    "session_path": str(Path("data/messaging/whatsapp_session")),

    # Allowed phone numbers (empty = allow all)
    "allowed_numbers": [
        # "+1234567890",  # Your phone number
    ],

    # Proactive messaging
    "proactive_enabled": True,

    # Rate limiting (WhatsApp is stricter)
    "max_messages_per_minute": 10,
    "min_message_interval_seconds": 3,
}

# ============ GENERAL CONFIG ============

MESSAGING_CONFIG = {
    # Which platforms to enable
    "enable_telegram": True,
    "enable_whatsapp": False,  # Start with Telegram first

    # Unified settings
    "typing_indicator_delay": 1.0,  # Seconds to show typing
    "max_response_length": 4000,    # Max characters per message

    # Proactive messaging
    "proactive_enabled": True,
    "proactive_check_interval": 300,  # 5 minutes

    # Quiet hours (no proactive messages)
    "quiet_hours_start": 22,
    "quiet_hours_end": 8,
}


def load_config():
    """Load config from environment or file"""

    # Try to load from .env file
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")
        except Exception:
            pass

    # Update configs from environment
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        TELEGRAM_CONFIG["telegram_token"] = os.getenv("TELEGRAM_BOT_TOKEN")

    if os.getenv("TELEGRAM_ALLOWED_USERS"):
        TELEGRAM_CONFIG["allowed_users"] = [
            u.strip() for u in os.getenv("TELEGRAM_ALLOWED_USERS").split(",")
        ]

    if os.getenv("TELEGRAM_ADMIN_USERS"):
        TELEGRAM_CONFIG["admin_users"] = [
            u.strip() for u in os.getenv("TELEGRAM_ADMIN_USERS").split(",")
        ]

    if os.getenv("WHATSAPP_WEBSOCKET_URL"):
        WHATSAPP_CONFIG["websocket_url"] = os.getenv("WHATSAPP_WEBSOCKET_URL")

    if os.getenv("ENABLE_WHATSAPP"):
        MESSAGING_CONFIG["enable_whatsapp"] = os.getenv("ENABLE_WHATSAPP").lower() == "true"

    return {
        "telegram": TELEGRAM_CONFIG,
        "whatsapp": WHATSAPP_CONFIG,
        "general": MESSAGING_CONFIG
    }

# Auto-load config from environment on import
load_config()

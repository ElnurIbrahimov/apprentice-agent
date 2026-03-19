#!/usr/bin/env python3
"""
One-shot patch: Fix Telegram bot _is_user_allowed to read env var directly.

Run on server:
    cd /opt/aura && python patch_telegram_auth.py && sudo systemctl restart aura-telegram

Safe to run multiple times (idempotent).
"""
import re, sys

FILE = "aura/messaging/telegram_bot.py"

with open(FILE, "r") as f:
    src = f.read()

# 1. Add 'import os' if missing
if "\nimport os\n" not in src:
    src = src.replace("\nimport logging\n", "\nimport logging\nimport os\n", 1)
    print("[patch] Added 'import os'")

# 2. Replace _is_user_allowed method
OLD_PATTERN = re.compile(
    r"    def _is_user_allowed\(self, user_id: int\) -> bool:.*?(?=\n    def )",
    re.DOTALL,
)

NEW_METHOD = '''    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed — reads os.environ EVERY call (bulletproof)."""
        env_val = os.environ.get("TELEGRAM_ALLOWED_USERS", "")
        allowed = [u.strip() for u in env_val.split(",") if u.strip()] if env_val else []
        if self.allowed_users:
            for u in self.allowed_users:
                if u and u not in allowed:
                    allowed.append(u)
        if not allowed:
            logger.warning(f"[TelegramBot] Rejected user {user_id} — no allowed_users configured.")
            return False
        is_allowed = str(user_id) in allowed
        if not is_allowed:
            logger.warning(f"[TelegramBot] Rejected user {user_id} — not in allowed list {allowed}")
        return is_allowed

'''

if OLD_PATTERN.search(src):
    src = OLD_PATTERN.sub(NEW_METHOD, src)
    print("[patch] Replaced _is_user_allowed method")
elif "os.environ.get(\"TELEGRAM_ALLOWED_USERS\"" in src:
    print("[patch] Already patched — nothing to do")
    sys.exit(0)
else:
    print("[patch] ERROR: Could not find _is_user_allowed method to replace!")
    sys.exit(1)

with open(FILE, "w") as f:
    f.write(src)

print("[patch] Done! Restart the bot: sudo systemctl restart aura-telegram")

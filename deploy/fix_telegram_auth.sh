#!/bin/bash
# ===========================================================================
# Fix: Telegram Bot Auth — Read TELEGRAM_ALLOWED_USERS from env directly
#
# Problem:
#   The config system doesn't pass allowed_users properly, so the bot
#   rejects all messages even when TELEGRAM_ALLOWED_USERS is in .env.
#
# Fix:
#   Patches _is_user_allowed() in telegram_bot.py to read
#   os.environ["TELEGRAM_ALLOWED_USERS"] directly every call,
#   bypassing the broken config pipeline entirely.
#
# Usage (as root on the server — no git pull needed):
#   curl -sL <raw-url> | sudo bash
#   — or —
#   sudo bash /opt/aura/deploy/fix_telegram_auth.sh
# ===========================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()   { echo -e "${GREEN}[FIX]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${CYAN}━━━ $* ━━━${NC}"; }

[ "$(id -u)" -eq 0 ] || error "Run as root: sudo bash $0"

AURA_DIR="/opt/aura"
TARGET="$AURA_DIR/aura/messaging/telegram_bot.py"

[ -f "$TARGET" ] || error "File not found: $TARGET"

# ===========================================================================
# 1. Back up the original file
# ===========================================================================
step "Step 1/4: Backup"

BACKUP="$TARGET.bak.$(date +%Y%m%d_%H%M%S)"
cp "$TARGET" "$BACKUP"
log "Backed up to $BACKUP"

# ===========================================================================
# 2. Patch _is_user_allowed using Python (safe multi-line replace)
# ===========================================================================
step "Step 2/4: Patching _is_user_allowed"

# Verify the method exists
if ! grep -q 'def _is_user_allowed' "$TARGET"; then
    error "_is_user_allowed method not found in $TARGET"
fi

# Check if already patched (idempotent)
if grep -q 'env_val = os.environ.get("TELEGRAM_ALLOWED_USERS"' "$TARGET"; then
    log "Already patched — skipping code change"
else
    # Use Python for safe multi-line replacement
    python3 << 'PYEOF'
import re
import sys

target = "/opt/aura/aura/messaging/telegram_bot.py"

with open(target, "r", encoding="utf-8") as f:
    content = f.read()

# --- Step A: Ensure 'import os' is present ---
if "\nimport os\n" not in content and "\nimport os " not in content:
    # Add it after the first import block (after 'import asyncio' or similar)
    content = content.replace("import asyncio\n", "import asyncio\nimport os\n", 1)
    if "import os" not in content:
        # Fallback: add at very top after docstring
        content = "import os\n" + content
    print("Added 'import os' to imports")
else:
    print("'import os' already present")

# --- Step B: Replace _is_user_allowed method ---
# Match from "def _is_user_allowed" to the line before the next method or blank-line+def
# This regex captures the entire method body
pattern = re.compile(
    r'([ \t]+)def _is_user_allowed\(self, user_id: int\) -> bool:.*?'
    r'return str\(user_id\) in self\.allowed_users',
    re.DOTALL
)

new_method = '''    def _is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed — reads os.environ EVERY call (bulletproof)."""
        env_val = os.environ.get("TELEGRAM_ALLOWED_USERS", "")
        allowed = [u.strip() for u in env_val.split(",") if u.strip()] if env_val else []
        if self.allowed_users:
            for u in self.allowed_users:
                if u and u not in allowed:
                    allowed.append(u)
        if not allowed:
            logger.warning(f"[TelegramBot] Rejected user {user_id} \\u2014 no allowed_users configured.")
            return False
        is_allowed = str(user_id) in allowed
        if not is_allowed:
            logger.warning(f"[TelegramBot] Rejected user {user_id} \\u2014 not in allowed list {allowed}")
        return is_allowed'''

m = pattern.search(content)
if not m:
    print("ERROR: Could not match _is_user_allowed method body.", file=sys.stderr)
    print("The method signature or 'return str(user_id) in self.allowed_users' was not found.", file=sys.stderr)
    print("The file may have been modified in an unexpected way.", file=sys.stderr)
    sys.exit(1)

content = content[:m.start()] + new_method + content[m.end():]
print(f"Replaced _is_user_allowed (chars {m.start()}-{m.end()})")

with open(target, "w", encoding="utf-8") as f:
    f.write(content)

print("File written successfully")
PYEOF

    if [ $? -ne 0 ]; then
        warn "Python patch failed — restoring backup"
        cp "$BACKUP" "$TARGET"
        error "Patch failed. File restored from backup."
    fi
    log "Patched successfully"
fi

# Verify the patch is in place
if grep -q 'env_val = os.environ.get("TELEGRAM_ALLOWED_USERS"' "$TARGET"; then
    log "Verification: patch confirmed in file"
else
    warn "Patch verification failed — restoring backup"
    cp "$BACKUP" "$TARGET"
    error "Patch not detected after write. File restored."
fi

# Quick syntax check
if python3 -c "import py_compile; py_compile.compile('$TARGET', doraise=True)" 2>/dev/null; then
    log "Syntax check: OK"
else
    warn "Syntax check failed — restoring backup"
    cp "$BACKUP" "$TARGET"
    error "Python syntax error in patched file. File restored."
fi

# ===========================================================================
# 3. Clear __pycache__ and restart
# ===========================================================================
step "Step 3/4: Clear cache & restart service"

# Clear all pycache dirs under aura/
find "$AURA_DIR/aura" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
log "Cleared __pycache__ dirs under $AURA_DIR/aura/"

# Also nuke .pyc files for the specific module
find "$AURA_DIR/aura/messaging" -name "*.pyc" -delete 2>/dev/null || true
log "Cleared .pyc files in aura/messaging/"

# Restart the telegram service
if systemctl is-enabled --quiet aura-telegram 2>/dev/null; then
    log "Restarting aura-telegram..."
    systemctl restart aura-telegram
    sleep 3
    if systemctl is-active --quiet aura-telegram; then
        log "aura-telegram.service: RUNNING"
    else
        warn "aura-telegram.service failed to start"
        warn "Check: journalctl -u aura-telegram -n 30"
    fi
else
    warn "aura-telegram.service not found or not enabled"
    warn "If the bot runs differently, restart it manually"
fi

# ===========================================================================
# 4. Check logs for the fix
# ===========================================================================
step "Step 4/4: Checking logs"

sleep 2

echo ""
log "Recent aura-telegram logs (last 20 lines):"
echo "---"
journalctl -u aura-telegram -n 20 --no-pager 2>/dev/null || warn "Could not read journalctl logs"
echo "---"

# Check if there are rejection messages AFTER the restart
RESTART_TIME=$(systemctl show aura-telegram --property=ActiveEnterTimestamp 2>/dev/null | cut -d= -f2)
if [ -n "${RESTART_TIME:-}" ]; then
    REJECTIONS=$(journalctl -u aura-telegram --since "$RESTART_TIME" --no-pager 2>/dev/null | grep -c "Rejected user" || true)
    if [ "${REJECTIONS:-0}" -gt 0 ]; then
        warn "$REJECTIONS rejection(s) found since restart"
        warn "Check TELEGRAM_ALLOWED_USERS in .env — make sure your user ID is listed"
    else
        log "No rejections since restart"
    fi
fi

# Show current env var
ENV_VAL=$(grep "^TELEGRAM_ALLOWED_USERS=" "$AURA_DIR/.env" 2>/dev/null | head -1 || true)
if [ -n "$ENV_VAL" ]; then
    log ".env has: $ENV_VAL"
else
    warn "TELEGRAM_ALLOWED_USERS not found in $AURA_DIR/.env"
    warn "Add it:  echo 'TELEGRAM_ALLOWED_USERS=YOUR_USER_ID' >> $AURA_DIR/.env"
    warn "Then:    systemctl restart aura-telegram"
fi

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "==========================================================================="
echo -e "${GREEN}  Telegram Auth Fix Applied${NC}"
echo "==========================================================================="
echo ""
echo "  What was done:"
echo "    1. Backed up telegram_bot.py"
echo "    2. Patched _is_user_allowed() to read TELEGRAM_ALLOWED_USERS from env"
echo "    3. Cleared __pycache__ and .pyc files"
echo "    4. Restarted aura-telegram service"
echo ""
echo "  How it works now:"
echo "    - Every call reads os.environ['TELEGRAM_ALLOWED_USERS'] directly"
echo "    - Also merges any users from self.allowed_users (config system)"
echo "    - Logs exactly which user was rejected and why"
echo ""
echo "  Test: send a message to your bot on Telegram"
echo "  Monitor: journalctl -u aura-telegram -f"
echo "  Revert:  cp $BACKUP $TARGET && systemctl restart aura-telegram"
echo "==========================================================================="

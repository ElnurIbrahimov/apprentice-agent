#!/bin/bash
# ===========================================================================
# Aura Server Update Script
# Pulls latest code, cleans caches, installs deps, restarts all services.
#
# Usage:
#   sudo bash /opt/aura/deploy/update_server.sh
# ===========================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[AURA]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

[ "$(id -u)" -eq 0 ] || error "Run as root: sudo bash $0"

AURA_DIR="/opt/aura"
cd "$AURA_DIR"

log "Current version:"
git log --oneline -1

log "Pulling latest changes..."
GIT_OUTPUT=$(git pull --ff-only 2>&1) || {
  if echo "$GIT_OUTPUT" | grep -qiE "authentication|403|401|could not read Username"; then
    echo ""
    error "Git authentication failed — the repo is private and needs a token.

    Follow the instructions in: /opt/aura/deploy/setup_git_auth.md

    Quick fix:
      1. Go to https://github.com/settings/tokens
      2. Generate a token with 'repo' scope
      3. Run: sudo -u aura git -C /opt/aura remote set-url origin https://YOUR_TOKEN@github.com/ElnurIbrahimov/apprentice-agent.git
      4. Re-run this script"
  else
    echo "$GIT_OUTPUT"
    error "Git pull failed. Output above may help diagnose the issue."
  fi
}

# ---------------------------------------------------------------------------
# Clean ALL __pycache__ directories to prevent stale bytecode after git pull
# ---------------------------------------------------------------------------
log "Cleaning __pycache__ directories..."
CACHE_COUNT=$(find "$AURA_DIR" -type d -name "__pycache__" | wc -l)
find "$AURA_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$AURA_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$AURA_DIR" -name "*.pyo" -delete 2>/dev/null || true
log "Cleaned $CACHE_COUNT __pycache__ directories and all .pyc/.pyo files"

# ---------------------------------------------------------------------------
# Update Python dependencies
# ---------------------------------------------------------------------------
log "Updating Python dependencies..."
source "$AURA_DIR/venv/bin/activate"
grep -v -E '(comtypes|pyttsx3|pycaw|winotify|screen-brightness-control|sounddevice|mss|pyperclip)' \
  requirements.txt > requirements-server.txt || true
pip install -r requirements-server.txt -q 2>&1 | tail -5
deactivate

log "Fixing ownership..."
chown -R aura:aura "$AURA_DIR"

# ---------------------------------------------------------------------------
# Restart all services (stop first to ensure clean state)
# ---------------------------------------------------------------------------
log "Stopping all AURA services..."
systemctl stop aura-daemon 2>/dev/null || true
systemctl stop aura-telegram 2>/dev/null || true
systemctl stop aura 2>/dev/null || true
sleep 1

log "Starting AURA backend..."
systemctl start aura

# Wait for backend to become healthy before starting dependents
RETRIES=0
MAX_RETRIES=15
while [ $RETRIES -lt $MAX_RETRIES ]; do
  if curl -sf http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
    log "Backend is healthy."
    break
  fi
  RETRIES=$((RETRIES + 1))
  sleep 2
done
if [ $RETRIES -eq $MAX_RETRIES ]; then
  warn "Backend did not become healthy in 30s. Starting dependents anyway."
fi

# Start Telegram bot if enabled
if systemctl is-enabled --quiet aura-telegram 2>/dev/null; then
  systemctl start aura-telegram
  log "Telegram bot started."
fi

# Start daemon if enabled
if systemctl is-enabled --quiet aura-daemon 2>/dev/null; then
  systemctl start aura-daemon
  log "Daemon started."
fi

sleep 2

# ---------------------------------------------------------------------------
# Verify all services
# ---------------------------------------------------------------------------
log "Service status:"
for svc in aura aura-telegram aura-daemon; do
  if systemctl is-active --quiet "$svc" 2>/dev/null; then
    log "  $svc: RUNNING"
  else
    warn "  $svc: NOT RUNNING (check: journalctl -u $svc -n 50)"
  fi
done

log "New version:"
git log --oneline -1

# Quick health check
if curl -sf http://127.0.0.1:8000/api/health 2>/dev/null; then
  log "Health check: OK"
else
  warn "Health check: FAILED"
fi

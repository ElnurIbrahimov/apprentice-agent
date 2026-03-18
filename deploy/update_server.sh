#!/bin/bash
# ===========================================================================
# Aura Server Update Script
# Pulls latest code, installs new dependencies, restarts the service.
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
git pull --ff-only || error "Git pull failed. Resolve conflicts manually."

log "Updating Python dependencies..."
source "$AURA_DIR/venv/bin/activate"
grep -v -E '(comtypes|pyttsx3|pycaw|winotify|screen-brightness-control|sounddevice|mss|pyperclip)' \
  requirements.txt > requirements-server.txt || true
pip install -r requirements-server.txt -q 2>&1 | tail -5
deactivate

log "Fixing ownership..."
chown -R aura:aura "$AURA_DIR"

log "Restarting AURA..."
systemctl restart aura

sleep 3

if systemctl is-active --quiet aura; then
  log "AURA restarted successfully."
  log "New version:"
  git log --oneline -1
else
  error "AURA failed to start. Check: journalctl -u aura -f"
fi

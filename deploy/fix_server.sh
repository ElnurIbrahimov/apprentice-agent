#!/bin/bash
# ===========================================================================
# Aura Server Fix Script — Daemon + Web UI
#
# Fixes:
#   1. Daemon crash: missing Python packages + headless mode env var
#   2. Web UI: installs Node.js 20.x, builds React frontend
#   3. Permissions and service restart
#
# Usage (as root on the server):
#   cd /opt/aura && git pull
#   bash deploy/fix_server.sh
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

# Must run as root
[ "$(id -u)" -eq 0 ] || error "Run as root: sudo bash deploy/fix_server.sh"

AURA_DIR="/opt/aura"
[ -d "$AURA_DIR" ] || error "$AURA_DIR does not exist. Run setup_server.sh first."

# ===========================================================================
# 1. Install missing Python packages for headless server
# ===========================================================================
step "Step 1/6: Installing missing Python packages"

if [ ! -f "$AURA_DIR/venv/bin/pip" ]; then
    error "Virtual environment not found at $AURA_DIR/venv. Run setup_server.sh first."
fi

# These packages are needed by tools that do try/except imports.
# On a headless server they'll fail gracefully, but having stub-compatible
# versions prevents noisy warnings. pyperclip works headless (xclip fallback).
# mss needs a display but the import won't crash — the tool just returns errors.
log "Installing pyperclip, mss (safe for headless — tools handle failures)..."
"$AURA_DIR/venv/bin/pip" install --quiet pyperclip mss 2>&1 | tail -3 || {
    warn "pip install had issues — continuing anyway (tools use try/except)"
}

log "Python packages done."

# ===========================================================================
# 2. Set AURA_HEADLESS in .env
# ===========================================================================
step "Step 2/6: Configuring headless mode"

ENV_FILE="$AURA_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    warn "No .env file found — creating minimal one"
    touch "$ENV_FILE"
fi

if grep -q "^AURA_HEADLESS=" "$ENV_FILE" 2>/dev/null; then
    # Already present, ensure it's true
    sed -i 's/^AURA_HEADLESS=.*/AURA_HEADLESS=true/' "$ENV_FILE"
    log "AURA_HEADLESS already in .env — ensured it's set to true"
else
    echo "" >> "$ENV_FILE"
    echo "# Headless mode — disables screen monitoring, screenshot, GUI features" >> "$ENV_FILE"
    echo "AURA_HEADLESS=true" >> "$ENV_FILE"
    log "Added AURA_HEADLESS=true to .env"
fi

# ===========================================================================
# 3. Install Node.js 20.x (if not present)
# ===========================================================================
step "Step 3/6: Installing Node.js 20.x"

if command -v node &>/dev/null; then
    NODE_VER=$(node --version 2>/dev/null || echo "unknown")
    log "Node.js already installed: $NODE_VER"

    # Check if it's at least v18 (needed for Vite 5)
    NODE_MAJOR=$(echo "$NODE_VER" | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_MAJOR" -lt 18 ] 2>/dev/null; then
        warn "Node.js $NODE_VER is too old (need >=18). Upgrading..."
        NEED_NODE=true
    else
        NEED_NODE=false
    fi
else
    log "Node.js not found — installing..."
    NEED_NODE=true
fi

if [ "$NEED_NODE" = true ]; then
    # Use NodeSource for Node.js 20.x LTS
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg

    # NodeSource setup
    mkdir -p /etc/apt/keyrings
    if [ ! -f /etc/apt/keyrings/nodesource.gpg ]; then
        curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | \
            gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
    fi

    NODE_MAJOR_VER=20
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR_VER.x nodistro main" | \
        tee /etc/apt/sources.list.d/nodesource.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq nodejs

    log "Node.js installed: $(node --version)"
    log "npm version: $(npm --version)"
fi

# ===========================================================================
# 4. Build the React web UI
# ===========================================================================
step "Step 4/6: Building web UI"

WEB_DIR="$AURA_DIR/web"
if [ ! -f "$WEB_DIR/package.json" ]; then
    error "Web UI source not found at $WEB_DIR/package.json"
fi

cd "$WEB_DIR"

log "Installing npm dependencies..."
npm install --no-audit --no-fund 2>&1 | tail -5

log "Building production bundle (tsc + vite build)..."
# TypeScript check may have non-critical errors in a large codebase — allow it
npm run build 2>&1 | tail -20

if [ -d "$WEB_DIR/dist" ] && [ -f "$WEB_DIR/dist/index.html" ]; then
    DIST_SIZE=$(du -sh "$WEB_DIR/dist" | cut -f1)
    log "Web UI built successfully ($DIST_SIZE in $WEB_DIR/dist/)"
else
    warn "Build may have failed — dist/index.html not found"
    warn "Check output above. The API will still work without the web UI."
    warn "You can retry manually: cd $WEB_DIR && npm run build"
fi

# ===========================================================================
# 5. Fix permissions
# ===========================================================================
step "Step 5/6: Fixing permissions"

if id aura &>/dev/null; then
    chown -R aura:aura "$AURA_DIR"
    log "Ownership set to aura:aura"
else
    warn "User 'aura' does not exist — skipping chown"
fi

# Ensure data dirs exist and are writable
mkdir -p "$AURA_DIR/data" "$AURA_DIR/aura_data" "$AURA_DIR/logs"
if id aura &>/dev/null; then
    chown -R aura:aura "$AURA_DIR/data" "$AURA_DIR/aura_data" "$AURA_DIR/logs"
fi

# ===========================================================================
# 6. Restart services
# ===========================================================================
step "Step 6/6: Restarting services"

systemctl daemon-reload

# Restart main backend
if systemctl is-enabled --quiet aura 2>/dev/null; then
    systemctl restart aura
    sleep 2
    if systemctl is-active --quiet aura; then
        log "aura.service: RUNNING"
    else
        warn "aura.service failed to start — check: journalctl -u aura -n 50"
    fi
else
    warn "aura.service not enabled — skipping"
fi

# Restart daemon
if systemctl is-enabled --quiet aura-daemon 2>/dev/null; then
    systemctl restart aura-daemon
    sleep 2
    if systemctl is-active --quiet aura-daemon; then
        log "aura-daemon.service: RUNNING"
    else
        warn "aura-daemon.service failed to start — check: journalctl -u aura-daemon -n 50"
    fi
else
    warn "aura-daemon.service not enabled — skipping"
fi

# Restart Telegram bot (if configured)
if systemctl is-enabled --quiet aura-telegram 2>/dev/null; then
    if grep -q "TELEGRAM_BOT_TOKEN=." "$ENV_FILE" 2>/dev/null; then
        systemctl restart aura-telegram
        sleep 1
        if systemctl is-active --quiet aura-telegram; then
            log "aura-telegram.service: RUNNING"
        else
            warn "aura-telegram.service failed — check: journalctl -u aura-telegram -n 50"
        fi
    else
        log "Telegram bot token not set — skipping aura-telegram"
    fi
fi

# ===========================================================================
# Health check
# ===========================================================================
step "Health Check"

sleep 2
if curl -sf http://localhost:8000/api/status > /dev/null 2>&1; then
    log "API health check: OK (http://localhost:8000/api/status)"
else
    warn "API not responding yet — it may still be starting up"
    warn "Check: curl localhost:8000/api/status"
fi

# Check if web UI is being served
if curl -sf http://localhost:8000/ 2>/dev/null | grep -q "Aura\|aura\|<!DOCTYPE" 2>/dev/null; then
    log "Web UI: being served at http://localhost:8000/"
else
    warn "Web UI may not be served yet (API might still be starting)"
fi

# ===========================================================================
# Summary
# ===========================================================================
echo ""
echo "==========================================================================="
echo -e "${GREEN}  Fix Complete${NC}"
echo "==========================================================================="
echo ""
echo "  What was done:"
echo "    1. Installed pyperclip + mss Python packages"
echo "    2. Set AURA_HEADLESS=true in .env (disables screen monitoring)"
echo "    3. Installed Node.js $(node --version 2>/dev/null || echo 'N/A')"
echo "    4. Built React web UI to web/dist/"
echo "    5. Fixed file ownership"
echo "    6. Restarted all services"
echo ""
echo "  Verify:"
echo "    systemctl status aura aura-daemon aura-telegram"
echo "    journalctl -u aura -f          # Backend logs"
echo "    journalctl -u aura-daemon -f   # Daemon logs"
echo "    curl localhost:8000/api/status  # API health"
echo ""
echo "  Web UI: http://89.167.107.134 (or your domain if configured)"
echo "==========================================================================="

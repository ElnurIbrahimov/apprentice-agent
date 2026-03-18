#!/bin/bash
# ===========================================================================
# Aura Server Setup Script
# Sets up AURA on a fresh Ubuntu 22.04/24.04 server with:
#   - Python 3.12, Nginx reverse proxy, Let's Encrypt SSL
#   - Systemd service (auto-restart, boot-start)
#
# Usage:
#   # On a fresh server as root:
#   curl -sL https://raw.githubusercontent.com/ElnurIbrahimov/apprentice-agent/main/deploy/setup_server.sh | bash
#
#   # Or clone first:
#   git clone https://github.com/ElnurIbrahimov/apprentice-agent.git /opt/aura
#   cd /opt/aura/deploy && bash setup_server.sh
# ===========================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[AURA]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# Must run as root
[ "$(id -u)" -eq 0 ] || error "Run this script as root (sudo bash setup_server.sh)"

AURA_DIR="/opt/aura"
AURA_USER="aura"
AURA_PORT=8000

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
log "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
  software-properties-common \
  git curl wget unzip \
  nginx certbot python3-certbot-nginx \
  build-essential libffi-dev libssl-dev

# Python 3.12 — use deadsnakes PPA if not available in base repos
if ! command -v python3.12 &>/dev/null; then
  log "Adding deadsnakes PPA for Python 3.12..."
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -qq
fi
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev

log "Python version: $(python3.12 --version)"

# ---------------------------------------------------------------------------
# 2. Clone repo (skip if already present)
# ---------------------------------------------------------------------------
if [ -d "$AURA_DIR/.git" ]; then
  log "Aura repo already exists at $AURA_DIR, pulling latest..."
  cd "$AURA_DIR"
  git pull --ff-only || warn "Git pull failed — continuing with existing code"
else
  log "Cloning Aura repository..."
  git clone https://github.com/ElnurIbrahimov/apprentice-agent.git "$AURA_DIR"
  cd "$AURA_DIR"
fi

# ---------------------------------------------------------------------------
# 3. Create aura system user
# ---------------------------------------------------------------------------
if ! id "$AURA_USER" &>/dev/null; then
  log "Creating system user '$AURA_USER'..."
  useradd -r -s /bin/false -d "$AURA_DIR" "$AURA_USER"
fi

# ---------------------------------------------------------------------------
# 4. Virtual environment and dependencies
# ---------------------------------------------------------------------------
log "Setting up Python virtual environment..."
python3.12 -m venv "$AURA_DIR/venv"
source "$AURA_DIR/venv/bin/activate"

log "Installing Python dependencies (this may take a few minutes)..."
pip install --upgrade pip setuptools wheel -q

# Server-only requirements: skip Windows-only packages
# Create a filtered requirements file
grep -v -E '(comtypes|pyttsx3|pycaw|winotify|screen-brightness-control|sounddevice|mss|pyperclip)' \
  "$AURA_DIR/requirements.txt" > "$AURA_DIR/requirements-server.txt" || true

pip install -r "$AURA_DIR/requirements-server.txt" -q 2>&1 | tail -5

deactivate
log "Dependencies installed."

# ---------------------------------------------------------------------------
# 5. Create .env file (if not exists)
# ---------------------------------------------------------------------------
if [ ! -f "$AURA_DIR/.env" ]; then
  log "Creating .env from template..."
  cp "$AURA_DIR/.env.example" "$AURA_DIR/.env"

  # Generate a secure API key
  API_KEY=$(python3.12 -c "import secrets; print(secrets.token_urlsafe(32))")

  # Patch critical security settings
  sed -i "s|^AURA_API_KEY=.*|AURA_API_KEY=${API_KEY}|" "$AURA_DIR/.env"
  sed -i "s|^AURA_API_AUTH_ENABLED=.*|AURA_API_AUTH_ENABLED=true|" "$AURA_DIR/.env"
  sed -i "s|^AURA_ENV=.*|AURA_ENV=production|" "$AURA_DIR/.env"
  # Trust nginx as reverse proxy for correct rate limiting
  echo "AURA_TRUST_PROXY=true" >> "$AURA_DIR/.env"

  log "Generated API key: ${API_KEY}"
  warn "SAVE THIS KEY — you'll need it in the browser extension settings."
  warn "It's also stored in $AURA_DIR/.env"
else
  warn ".env already exists, not overwriting. Verify AURA_API_AUTH_ENABLED=true"
fi

# ---------------------------------------------------------------------------
# 6. Data directories
# ---------------------------------------------------------------------------
mkdir -p "$AURA_DIR/data" "$AURA_DIR/aura_data" "$AURA_DIR/logs"

# ---------------------------------------------------------------------------
# 7. Fix ownership
# ---------------------------------------------------------------------------
chown -R "$AURA_USER:$AURA_USER" "$AURA_DIR"

# ---------------------------------------------------------------------------
# 8. Systemd service
# ---------------------------------------------------------------------------
log "Creating systemd service..."
cat > /etc/systemd/system/aura.service << 'SERVICE'
[Unit]
Description=AURA AI Backend
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=aura
Group=aura
WorkingDirectory=/opt/aura
EnvironmentFile=/opt/aura/.env
ExecStart=/opt/aura/venv/bin/python run_web.py --host 127.0.0.1 --port 8000 --prod
Restart=always
RestartSec=5
StartLimitIntervalSec=60
StartLimitBurst=5

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/opt/aura/data /opt/aura/aura_data /opt/aura/logs
PrivateTmp=yes

# Resource limits
LimitNOFILE=65536
MemoryMax=4G

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=aura

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable aura

# ---------------------------------------------------------------------------
# 9. Nginx reverse proxy
# ---------------------------------------------------------------------------
log "Configuring Nginx..."
cat > /etc/nginx/sites-available/aura << 'NGINX'
# AURA reverse proxy — replace YOUR_DOMAIN with your actual domain
# After replacing, run: certbot --nginx -d YOUR_DOMAIN

server {
    listen 80;
    listen [::]:80;
    server_name YOUR_DOMAIN;

    # Security headers
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;

        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Forward real client info
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts — LLM responses can take a while
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_connect_timeout 10s;

        # Allow large uploads (PDFs, images)
        client_max_body_size 50M;

        # Buffer settings for streaming
        proxy_buffering off;
        proxy_cache off;
    }

    # Health check endpoint (no auth needed by AURA middleware)
    location = /api/status {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        access_log off;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/aura /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Validate nginx config
nginx -t || error "Nginx config test failed"

# ---------------------------------------------------------------------------
# 10. Firewall (ufw)
# ---------------------------------------------------------------------------
if command -v ufw &>/dev/null; then
  log "Configuring firewall..."
  ufw allow 22/tcp    # SSH
  ufw allow 80/tcp    # HTTP (for Let's Encrypt + redirect)
  ufw allow 443/tcp   # HTTPS
  ufw --force enable
  log "Firewall enabled (22, 80, 443 open)"
fi

# ---------------------------------------------------------------------------
# 11. Start services
# ---------------------------------------------------------------------------
log "Starting services..."
systemctl restart nginx
systemctl start aura

# Wait a moment for startup
sleep 3

if systemctl is-active --quiet aura; then
  log "AURA service is running."
else
  warn "AURA service may not have started. Check: journalctl -u aura -f"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "==========================================================================="
echo -e "${GREEN}  AURA Server Setup Complete${NC}"
echo "==========================================================================="
echo ""
echo "  Next steps:"
echo ""
echo "  1. Set your domain in Nginx config:"
echo "     nano /etc/nginx/sites-available/aura"
echo "     Replace YOUR_DOMAIN with your actual domain"
echo "     systemctl reload nginx"
echo ""
echo "  2. Get SSL certificate:"
echo "     certbot --nginx -d YOUR_DOMAIN"
echo ""
echo "  3. Configure your .env file:"
echo "     nano /opt/aura/.env"
echo "     - Set Ollama host (OLLAMA_HOST) if not running locally"
echo "     - Add any API keys you need (TAVILY_API_KEY, etc.)"
echo "     systemctl restart aura"
echo ""
echo "  4. In the browser extension:"
echo "     Settings > Backend Connection"
echo "     - Server URL: https://YOUR_DOMAIN"
echo "     - API Key: (the key shown above / in .env)"
echo ""
echo "  Useful commands:"
echo "     journalctl -u aura -f          # Live logs"
echo "     systemctl restart aura         # Restart"
echo "     systemctl status aura          # Status"
echo "     curl localhost:8000/api/status  # Health check"
echo ""
echo "==========================================================================="

# Deploying AURA to a Remote Server

Run AURA on an always-on server so you can access it from anywhere via the browser extension.

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| vCPU | 2 | 4+ |
| RAM | 4 GB | 8 GB+ |
| Disk | 20 GB | 40 GB+ (for models/data) |
| OS | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |
| Network | Public IP | Domain name + SSL |

AURA uses cloud models via Ollama Pro ($20/month). No local GPU needed — all LLM inference happens in the cloud. You just need:
- `OLLAMA_API_KEY` for cloud models (Kimi, Qwen, MiniMax, DeepSeek, etc.)
- Optionally, a ChatGPT subscription for GPT-5.x models via OAuth

## Quick Deploy (Bare Metal)

SSH into your fresh Ubuntu server as root:

```bash
curl -sL https://raw.githubusercontent.com/ElnurIbrahimov/apprentice-agent/main/deploy/setup_server.sh | bash
```

This script:
1. Installs Python 3.12, Nginx, Certbot
2. Clones the repo to `/opt/aura`
3. Creates a venv and installs dependencies
4. Generates a secure API key
5. Sets up a systemd service (auto-start, auto-restart)
6. Configures Nginx reverse proxy
7. Opens firewall ports (22, 80, 443)

After it finishes, follow the printed "Next steps" to set your domain and get SSL.

## Docker Deploy

```bash
git clone https://github.com/ElnurIbrahimov/apprentice-agent.git /opt/aura
cd /opt/aura

# Create .env from template
cp .env.example .env

# Generate API key and set security settings
API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
sed -i "s|^AURA_API_KEY=.*|AURA_API_KEY=${API_KEY}|" .env
sed -i "s|^AURA_API_AUTH_ENABLED=.*|AURA_API_AUTH_ENABLED=true|" .env
sed -i "s|^AURA_ENV=.*|AURA_ENV=production|" .env
echo "Your API key: $API_KEY"

# Start
cd deploy
docker compose up -d --build

# Logs
docker compose logs -f aura
```

Put Nginx + Certbot on the host (not in Docker) for simplest SSL setup, or uncomment the nginx service in `docker-compose.yml`.

## Oracle Cloud Free Tier (Always Free)

Oracle offers an always-free ARM instance (4 vCPU, 24 GB RAM) — more than enough for AURA.

1. **Create account** at [cloud.oracle.com](https://cloud.oracle.com) (credit card required but never charged for free tier)

2. **Create an instance:**
   - Shape: `VM.Standard.A1.Flex` (ARM)
   - OCPUs: 2-4, Memory: 8-24 GB
   - Image: Ubuntu 22.04 or 24.04 (aarch64)
   - Add your SSH public key

3. **Configure networking:**
   - Go to your instance's VCN > Security Lists > Default
   - Add Ingress Rules:
     - Source: `0.0.0.0/0`, Protocol: TCP, Port: 80
     - Source: `0.0.0.0/0`, Protocol: TCP, Port: 443

4. **SSH in and run setup:**
   ```bash
   ssh ubuntu@YOUR_SERVER_IP
   sudo -i
   curl -sL https://raw.githubusercontent.com/ElnurIbrahimov/apprentice-agent/main/deploy/setup_server.sh | bash
   ```

5. **Point your domain** (DNS A record) to the server's public IP

6. **Get SSL:**
   ```bash
   # Edit nginx config first
   nano /etc/nginx/sites-available/aura
   # Replace YOUR_DOMAIN with your actual domain
   systemctl reload nginx
   certbot --nginx -d yourdomain.com
   ```

Note: Oracle ARM instances use aarch64. PyTorch and some ML packages have ARM wheels available.

## Hetzner Setup

Hetzner Cloud offers cheap VPS starting at ~$4/month.

1. **Create server** at [console.hetzner.cloud](https://console.hetzner.cloud):
   - Type: CX22 (2 vCPU, 4 GB) or CX32 (4 vCPU, 8 GB)
   - Image: Ubuntu 24.04
   - Location: closest to you
   - Add your SSH key

2. **SSH in:**
   ```bash
   ssh root@YOUR_SERVER_IP
   curl -sL https://raw.githubusercontent.com/ElnurIbrahimov/apprentice-agent/main/deploy/setup_server.sh | bash
   ```

3. **Domain + SSL** — same as above (point DNS, run certbot)

## ChatGPT Token Setup for Servers

On a headless server you cannot open a browser for OAuth login. Use one of these methods:

### Method 1: Login locally, copy token to server

1. On your local machine (with a browser):
   ```bash
   aura --login chatgpt
   ```

2. Copy the token file to your server:
   ```bash
   scp ~/.aura/chatgpt_auth.json root@YOUR_SERVER:/opt/aura/.aura/chatgpt_auth.json
   chown aura:aura /opt/aura/.aura/chatgpt_auth.json
   ```

3. The refresh token auto-renews the access token, so this is a one-time setup.

### Method 2: Use the API endpoint

1. Get a refresh token from a local login (check `~/.aura/chatgpt_auth.json` for the `refresh` field).

2. POST it to your server's API:
   ```bash
   curl -X POST https://yourdomain.com/api/auth/chatgpt/token \
     -H "X-API-Key: YOUR_AURA_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"refresh_token": "YOUR_REFRESH_TOKEN"}'
   ```

3. Verify:
   ```bash
   curl https://yourdomain.com/api/auth/chatgpt/status \
     -H "X-API-Key: YOUR_AURA_API_KEY"
   ```

### Method 3: Write the token file directly

```bash
mkdir -p /opt/aura/.aura
cat > /opt/aura/.aura/chatgpt_auth.json << 'EOF'
{
  "access": "",
  "refresh": "YOUR_REFRESH_TOKEN_HERE",
  "expires": 0,
  "account_id": ""
}
EOF
chown aura:aura /opt/aura/.aura/chatgpt_auth.json
```

The empty `access` and `expires: 0` will trigger an automatic refresh on first use.

## SSL/HTTPS with Let's Encrypt

After running the setup script and setting your domain in nginx:

```bash
# Replace YOUR_DOMAIN in nginx config
nano /etc/nginx/sites-available/aura
systemctl reload nginx

# Get certificate (auto-configures nginx for HTTPS + HTTP redirect)
certbot --nginx -d YOUR_DOMAIN

# Auto-renewal is set up automatically. Test it:
certbot renew --dry-run
```

Certbot will:
- Obtain a free SSL certificate
- Configure nginx to serve HTTPS
- Add automatic HTTP -> HTTPS redirect
- Set up auto-renewal (runs twice daily via systemd timer)

## Connecting the Extension

1. Open the AURA browser extension
2. Go to **Settings** (gear icon in the sidebar)
3. Under **Backend Connection**:
   - **Server URL**: `https://yourdomain.com` (your server's HTTPS URL)
   - **API Key**: paste the key from the server's `.env` file
4. Click **Test Connection** to verify
5. Click **Save & Reconnect**

The extension will now connect to your remote server instead of localhost.

## Security Checklist

Before exposing AURA to the internet:

- [ ] `AURA_API_AUTH_ENABLED=true` in `.env`
- [ ] `AURA_API_KEY` is set to a strong random value (the setup script generates one)
- [ ] `AURA_ENV=production` (disables /docs, /redoc, /openapi.json)
- [ ] SSL/HTTPS is configured (never run plain HTTP on public internet)
- [ ] Firewall allows only ports 22, 80, 443
- [ ] Ollama is NOT exposed to the internet (bind to 127.0.0.1 only)
- [ ] SSH uses key auth, not passwords (`PasswordAuthentication no` in sshd_config)
- [ ] API key is entered in the extension settings (not hardcoded anywhere public)
- [ ] `AURA_HEADLESS=true` is set in `.env` (disables screen monitoring on servers)

## CORS Configuration

For the extension to connect from any origin, update `.env`:

```
AURA_CORS_ORIGINS=*
```

The browser extension uses `chrome-extension://` origin which is automatically allowed by AURA's CORS middleware.

## Monitoring and Logs

```bash
# Live service logs
journalctl -u aura -f

# Last 100 lines
journalctl -u aura -n 100

# Service status
systemctl status aura

# All three services
systemctl status aura aura-telegram aura-daemon

# Health check
curl -s https://yourdomain.com/api/status | python3 -m json.tool

# ChatGPT auth status
curl -s -H "X-API-Key: YOUR_KEY" https://yourdomain.com/api/auth/chatgpt/status

# Nginx access logs
tail -f /var/log/nginx/access.log

# Nginx error logs
tail -f /var/log/nginx/error.log

# Docker logs (if using Docker)
docker compose logs -f aura
```

## Updating AURA

### Bare metal
```bash
sudo bash /opt/aura/deploy/update_server.sh
```

### Docker
```bash
cd /opt/aura/deploy
git pull
docker compose up -d --build
```

## Troubleshooting

### AURA won't start

```bash
journalctl -u aura -n 50   # Check logs for errors
cat /opt/aura/.env          # Verify config
/opt/aura/venv/bin/python -c "import fastapi; print('ok')"  # Test deps
```

### WebSocket connection fails

- Check that nginx has `proxy_set_header Upgrade` and `Connection "upgrade"`
- Check browser console for CORS errors
- Verify SSL cert is valid: `curl -v https://yourdomain.com/api/status`

### Extension shows "offline"

- Verify the Server URL in extension settings matches your domain exactly
- Check API key matches the one in `.env`
- Test with: `curl -H "X-API-Key: YOUR_KEY" https://yourdomain.com/api/status`

### Rate limiting (429 errors)

- Default is 300 requests/minute per IP
- Adjust `AURA_API_RATE_LIMIT` in `.env` if needed
- The extension makes multiple requests per interaction (chat + streaming + status)

### Model not found

- Cloud models require `OLLAMA_API_KEY` to be set (Ollama Pro subscription)
- Check available models: `curl -H "X-API-Key: YOUR_KEY" https://yourdomain.com/api/models`
- If a specific cloud model is down, AURA's fallback chains will try alternatives automatically

### ChatGPT auth failures

- Token expired: The refresh token auto-renews, but if the refresh token itself expired (rare), re-login:
  ```bash
  # On local machine with browser:
  aura --login chatgpt
  # Then copy token to server (see ChatGPT Token Setup above)
  ```
- Check status: `curl -H "X-API-Key: YOUR_KEY" https://yourdomain.com/api/auth/chatgpt/status`
- Verify token file exists: `ls -la /opt/aura/.aura/chatgpt_auth.json`

### Ollama connection

- If using Ollama Pro (cloud-only, no local Ollama needed), just set `OLLAMA_API_KEY` in `.env`
- If Ollama runs on a different machine, set `OLLAMA_HOST=http://OLLAMA_IP:11434`
- If on the same machine, default `http://localhost:11434` works

### High memory usage

- Default memory limit is 4 GB (set in systemd service and docker-compose)
- Reduce if needed: edit `/etc/systemd/system/aura.service` and change `MemoryMax`
- AURA uses ~1-2 GB at rest, spikes during heavy memory retrieval or KG operations

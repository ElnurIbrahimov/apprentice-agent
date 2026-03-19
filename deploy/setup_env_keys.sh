#!/bin/bash
# =============================================================
# Aura Server — Set All API Keys
# Run once on the server after initial deploy or when keys change
# Usage: bash /opt/aura/deploy/setup_env_keys.sh
# =============================================================
set -e

ENV_FILE="/opt/aura/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: $ENV_FILE not found. Run setup_server.sh first."
    exit 1
fi

echo "Setting API keys in $ENV_FILE ..."

# --- Ollama Pro API Key (required for all cloud models) ---
sed -i 's|^OLLAMA_API_KEY=.*|OLLAMA_API_KEY=b027c9cba8aa4470be6e1366d0f497fc.8XawzDwgp2dNoIuHulBbEKc3|' "$ENV_FILE"

# --- Tavily API Key (web search) ---
sed -i 's|^TAVILY_API_KEY=.*|TAVILY_API_KEY=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|' "$ENV_FILE"

# --- Brave Search API Key ---
sed -i 's|^BRAVE_API_KEY=.*|BRAVE_API_KEY=BSApF1NIZvHTdh673xKOh2x3dNWJFo0|' "$ENV_FILE"

# --- Firecrawl API Key (web scraping) ---
grep -q "^FIRECRAWL_API_KEY=" "$ENV_FILE" || echo "FIRECRAWL_API_KEY=" >> "$ENV_FILE"
sed -i 's|^FIRECRAWL_API_KEY=.*|FIRECRAWL_API_KEY=fc-7cf680c459fe47a7a95e1f6f21956273|' "$ENV_FILE"

echo ""
echo "Keys set:"
echo "  OLLAMA_API_KEY    = ...$(grep '^OLLAMA_API_KEY=' "$ENV_FILE" | cut -d= -f2 | tail -c 8)"
echo "  TAVILY_API_KEY    = ...$(grep '^TAVILY_API_KEY=' "$ENV_FILE" | cut -d= -f2 | tail -c 8)"
echo "  BRAVE_API_KEY     = ...$(grep '^BRAVE_API_KEY=' "$ENV_FILE" | cut -d= -f2 | tail -c 8)"
echo "  FIRECRAWL_API_KEY = ...$(grep '^FIRECRAWL_API_KEY=' "$ENV_FILE" | cut -d= -f2 | tail -c 8)"
echo ""
echo "Restart to apply: systemctl restart aura"

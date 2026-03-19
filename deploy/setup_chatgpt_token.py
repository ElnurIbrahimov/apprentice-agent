#!/usr/bin/env python3
"""Write ChatGPT auth token to the server. Run on the server:
   python3 /opt/aura/deploy/setup_chatgpt_token.py

NOTE: You can now set the token via the API instead:
   POST /api/auth/chatgpt/set-token  {"refresh": "rt_...", "account_id": "..."}
   Or use the extension Settings panel > ChatGPT Token section.
"""
import json, os

TOKEN_DIR = "/opt/aura/.aura"
TOKEN_FILE = os.path.join(TOKEN_DIR, "chatgpt_auth.json")

token_data = {
    "refresh": "PASTE_YOUR_REFRESH_TOKEN_HERE",
    "expires": 0,
    "account_id": "PASTE_YOUR_ACCOUNT_ID_HERE"
}

os.makedirs(TOKEN_DIR, exist_ok=True)
with open(TOKEN_FILE, "w") as f:
    json.dump(token_data, f, indent=2)

# Fix ownership
os.system("chown -R aura:aura /opt/aura/.aura 2>/dev/null")
print(f"ChatGPT token written to {TOKEN_FILE}")
print("Restart aura: systemctl restart aura")

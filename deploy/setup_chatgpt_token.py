#!/usr/bin/env python3
"""Write ChatGPT auth token to the server. Run on the server:
   python3 /opt/aura/deploy/setup_chatgpt_token.py
"""
import json, os

TOKEN_DIR = "/opt/aura/.aura"
TOKEN_FILE = os.path.join(TOKEN_DIR, "chatgpt_auth.json")

token_data = {
    "refresh": "rt_DjlZSXy1KOCd_xvjvREV3la59K-k2PBvxxO9vamBZ20.t5jtlM1KnwyUyU1l4b7xpWdN4HzbBQhPsQMeiindFBs",
    "expires": 0,
    "account_id": "92dff04b-90a9-4dfa-82eb-f35c6a9c3646"
}

os.makedirs(TOKEN_DIR, exist_ok=True)
with open(TOKEN_FILE, "w") as f:
    json.dump(token_data, f, indent=2)

# Fix ownership
os.system("chown -R aura:aura /opt/aura/.aura 2>/dev/null")
print(f"ChatGPT token written to {TOKEN_FILE}")
print("Restart aura: systemctl restart aura")

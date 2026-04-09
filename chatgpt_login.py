"""ChatGPT OAuth Login — run locally, sends tokens to remote AURA server.

Usage: python chatgpt_login.py
Opens browser, captures OAuth callback, sends refresh token to server.
"""

import base64
import hashlib
import json
import os
import secrets
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs

import requests

# OAuth constants (from OpenAI Codex CLI)
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
CALLBACK_PORT = 1455

# Your AURA server — configure via environment variables
AURA_SERVER = os.environ.get("AURA_SERVER", "https://aura-elnur.duckdns.org")
AURA_API_KEY = os.environ.get("AURA_API_KEY", "")
if not AURA_API_KEY:
    print("ERROR: Set AURA_API_KEY environment variable before running.")
    print("  export AURA_API_KEY='your-key-here'")
    sys.exit(1)


def _generate_pkce():
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


class CallbackHandler(BaseHTTPRequestHandler):
    code = None
    expected_state = ""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)
        state = params.get("state", [None])[0]
        code = params.get("code", [None])[0]

        if state != self.expected_state or not code:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Auth failed - state mismatch or missing code")
            return

        CallbackHandler.code = code
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
            b"<h1>Authenticated!</h1>"
            b"<p>Sending tokens to AURA server... You can close this window.</p>"
            b"</body></html>"
        )

    def log_message(self, *args):
        pass


def main():
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": "codex_cli_rs",
    }
    auth_url = f"{AUTHORIZE_URL}?{urlencode(params)}"

    CallbackHandler.code = None
    CallbackHandler.expected_state = state

    server = HTTPServer(("127.0.0.1", CALLBACK_PORT), CallbackHandler)
    server.timeout = 1

    print(f"\nOpening browser for ChatGPT login...")
    print(f"If it doesn't open, visit:\n{auth_url}\n")
    webbrowser.open(auth_url)

    print("Waiting for authentication...")
    for _ in range(120):
        server.handle_request()
        if CallbackHandler.code:
            break
    server.server_close()

    code = CallbackHandler.code
    if not code:
        print("Authentication timed out.")
        return

    # Exchange code for tokens
    print("Exchanging code for tokens...")
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": REDIRECT_URI,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )

    if resp.status_code != 200:
        print(f"Token exchange failed: {resp.status_code}")
        print(resp.text)
        return

    data = resp.json()
    refresh_token = data.get("refresh_token", "")
    access_token = data.get("access_token", "")

    if not refresh_token:
        print("No refresh token in response!")
        return

    # Extract account ID from access token
    try:
        parts = access_token.split(".")
        payload = parts[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        decoded = json.loads(base64.urlsafe_b64decode(payload))
        auth_claim = decoded.get("https://api.openai.com/auth", {})
        account_id = auth_claim.get("chatgpt_account_id", "")
    except Exception:
        account_id = ""

    print(f"Got tokens! Account: {account_id[:8]}..." if account_id else "Got tokens!")

    # Send refresh token to AURA server
    print(f"\nSending to AURA server at {AURA_SERVER}...")
    resp = requests.post(
        f"{AURA_SERVER}/api/auth/chatgpt/set-token",
        json={"refresh": refresh_token, "account_id": account_id},
        headers={"X-API-Key": AURA_API_KEY},
        timeout=30,
    )

    if resp.status_code == 200:
        result = resp.json()
        if result.get("success"):
            print(f"SUCCESS! {result.get('message', 'Token saved')}")
            if result.get("account_id"):
                print(f"Account: {result['account_id']}")
            print("\nChatGPT models are now available across all AURA surfaces!")
        else:
            print(f"Server error: {result.get('error', 'unknown')}")
    else:
        print(f"Failed to send to server: {resp.status_code}")
        print(resp.text)


if __name__ == "__main__":
    main()

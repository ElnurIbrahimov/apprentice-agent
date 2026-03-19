"""ChatGPT OAuth 2.0 PKCE authentication for AURA.

Uses the same OAuth flow as OpenAI's official Codex CLI to authenticate
with ChatGPT Plus/Pro subscriptions.

INTENDED USE: Personal coding/AI assistance with your own ChatGPT subscription.
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs

import requests

logger = logging.getLogger(__name__)

# OAuth constants (from OpenAI Codex CLI)
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"
CALLBACK_PORT = 1455

# Token storage — resolve path with fallback for server deployments
# On the server, Path.home() may be /root or /home/aura, but the deploy script
# writes tokens to /opt/aura/.aura/. Check the project dir first, then home.
def _resolve_token_file() -> Path:
    """Find the token file, preferring the project-local path on servers."""
    # 1. Check /opt/aura/.aura/ (server deploy path)
    server_path = Path("/opt/aura/.aura/chatgpt_auth.json")
    if server_path.exists():
        return server_path
    # 2. Check home dir (local dev / default)
    home_path = Path.home() / ".aura" / "chatgpt_auth.json"
    if home_path.exists():
        return home_path
    # 3. If neither exists, prefer server path when running on Linux (deployed),
    #    otherwise home path (local dev on Windows/Mac)
    if os.name != "nt" and Path("/opt/aura").is_dir():
        return server_path
    return home_path


def _get_token_file() -> Path:
    """Get the current token file path (re-resolves each call)."""
    return _resolve_token_file()

# Keep module-level TOKEN_FILE for backward compat (used by auth.py status endpoint)
TOKEN_FILE = _resolve_token_file()

# Refresh 5 min before expiry
REFRESH_BUFFER_MS = 5 * 60 * 1000


def _generate_pkce():
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


def _decode_jwt(token: str) -> Optional[dict]:
    """Decode JWT payload without verification (we just need claims)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except Exception:
        return None


def _extract_account_id(access_token: str) -> Optional[str]:
    """Extract ChatGPT account ID from JWT access token."""
    payload = _decode_jwt(access_token)
    if not payload:
        return None
    auth_claim = payload.get("https://api.openai.com/auth", {})
    if isinstance(auth_claim, dict):
        return auth_claim.get("chatgpt_account_id")
    return None


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback on localhost."""

    code: Optional[str] = None
    expected_state: str = ""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        params = parse_qs(parsed.query)
        state = params.get("state", [None])[0]
        code = params.get("code", [None])[0]

        if state != self.expected_state:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"State mismatch")
            return

        if not code:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing authorization code")
            return

        _OAuthCallbackHandler.code = code
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
            b"<h1>Authenticated!</h1>"
            b"<p>You can close this window and return to AURA.</p>"
            b"</body></html>"
        )

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass


def _save_tokens(access: str, refresh: str, expires: int):
    """Save tokens to disk."""
    tf = _get_token_file()
    tf.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "access": access,
        "refresh": refresh,
        "expires": expires,
        "account_id": _extract_account_id(access),
    }
    tf.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        import stat
        tf.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 — owner read/write only
    except (OSError, NotImplementedError):
        pass  # Windows may not support Unix permissions
    logger.info("[CHATGPT_AUTH] Tokens saved to %s", tf)


def load_tokens() -> Optional[dict]:
    """Load tokens from disk. Returns dict with access, refresh, expires, account_id."""
    try:
        tf = _get_token_file()
        if tf.exists():
            data = json.loads(tf.read_text(encoding="utf-8"))
            if data.get("access") or data.get("refresh"):
                return data
    except Exception as e:
        logger.warning(f"[CHATGPT_AUTH] Failed to load tokens: {e}")
    return None


def refresh_token(refresh_token_str: str) -> Optional[dict]:
    """Refresh the access token. Returns updated token dict or None."""
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token_str,
                "client_id": CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error(f"[CHATGPT_AUTH] Token refresh failed: {resp.status_code}")
            return None

        data = resp.json()
        access = data.get("access_token")
        refresh = data.get("refresh_token")
        expires_in = data.get("expires_in")

        if not all([access, refresh, expires_in]):
            logger.error("[CHATGPT_AUTH] Token refresh response missing fields")
            return None

        expires = int(time.time() * 1000) + expires_in * 1000
        _save_tokens(access, refresh, expires)
        return load_tokens()
    except Exception as e:
        logger.error(f"[CHATGPT_AUTH] Token refresh error: {e}")
        return None


def get_valid_token() -> Optional[dict]:
    """Get a valid access token, auto-refreshing if needed."""
    tokens = load_tokens()
    if not tokens:
        return None

    # If we only have a refresh token (set via API), do an immediate refresh
    if not tokens.get("access") or tokens.get("expires", 0) == 0:
        logger.info("[CHATGPT_AUTH] No access token, refreshing from refresh token...")
        tokens = refresh_token(tokens["refresh"])
        return tokens

    now_ms = int(time.time() * 1000)
    if tokens["expires"] - now_ms < REFRESH_BUFFER_MS:
        logger.info("[CHATGPT_AUTH] Token expiring soon, refreshing...")
        tokens = refresh_token(tokens["refresh"])

    return tokens


def login() -> bool:
    """Run the OAuth login flow. Opens browser, waits for callback.

    Returns True on success, False on failure.
    """
    verifier, challenge = _generate_pkce()
    state = secrets.token_hex(16)

    # Build auth URL
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

    # Reset handler state
    _OAuthCallbackHandler.code = None
    _OAuthCallbackHandler.expected_state = state

    # Start local callback server
    try:
        server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _OAuthCallbackHandler)
    except OSError as e:
        logger.error(f"[CHATGPT_AUTH] Cannot bind port {CALLBACK_PORT}: {e}")
        print(f"Error: Port {CALLBACK_PORT} is in use. Close any other auth flows and try again.")
        return False

    server.timeout = 1

    # Open browser
    print(f"\nOpening browser for ChatGPT login...")
    print(f"If the browser doesn't open, visit:\n{auth_url}\n")
    webbrowser.open(auth_url)

    # Wait for callback (up to 120 seconds)
    print("Waiting for authentication...")
    for _ in range(120):
        server.handle_request()
        if _OAuthCallbackHandler.code:
            break

    server.server_close()

    code = _OAuthCallbackHandler.code
    if not code:
        print("Authentication timed out or failed.")
        return False

    # Exchange code for tokens
    print("Exchanging authorization code for tokens...")
    try:
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
            logger.error(f"[CHATGPT_AUTH] Token exchange failed: {resp.status_code} {resp.text}")
            print(f"Token exchange failed: {resp.status_code}")
            return False

        data = resp.json()
        access = data.get("access_token")
        refresh = data.get("refresh_token")
        expires_in = data.get("expires_in")

        if not all([access, refresh, expires_in]):
            print("Token response missing required fields.")
            return False

        expires = int(time.time() * 1000) + expires_in * 1000
        _save_tokens(access, refresh, expires)

        account_id = _extract_account_id(access)
        print(f"\nAuthenticated successfully!")
        if account_id:
            print(f"Account ID: {account_id[:8]}...")
        print("You can now use chatgpt: models in AURA.\n")
        return True

    except Exception as e:
        logger.error(f"[CHATGPT_AUTH] Token exchange error: {e}")
        print(f"Authentication failed: {e}")
        return False


def logout():
    """Remove stored tokens."""
    tf = _get_token_file()
    if tf.exists():
        tf.unlink()
        print("ChatGPT authentication cleared.")
    else:
        print("No ChatGPT authentication found.")


def save_refresh_token(refresh: str, account_id: str = "") -> Path:
    """Save a refresh token directly (no access token yet).

    The access token will be obtained on first use via refresh_token().
    Returns the path where the token was saved.
    """
    tf = _get_token_file()
    tf.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "access": "",
        "refresh": refresh,
        "expires": 0,
        "account_id": account_id,
    }
    tf.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("[CHATGPT_AUTH] Refresh token saved to %s", tf)
    return tf


def is_authenticated() -> bool:
    """Check if we have valid (or refreshable) tokens."""
    return get_valid_token() is not None

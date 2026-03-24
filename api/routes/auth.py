"""Authentication routes for external providers (ChatGPT OAuth, etc.)."""

import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth import require_api_key

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Fallback server-side PKCE verifier store (keyed by OAuth state).
# Used only if aura.auth.chatgpt_oauth.store_pkce_verifier is unavailable.
# Entries are (verifier, timestamp) tuples; evicted after _PKCE_TTL seconds.
import time as _time
_pkce_store: dict[str, tuple[str, float]] = {}
_PKCE_MAX_ENTRIES = 100
_PKCE_TTL = 600  # 10 minutes


def _pkce_store_put(state: str, verifier: str) -> None:
    """Store a PKCE verifier with TTL and bounded size."""
    now = _time.time()
    # Evict expired entries
    expired = [k for k, (_, ts) in _pkce_store.items() if now - ts > _PKCE_TTL]
    for k in expired:
        del _pkce_store[k]
    # Evict oldest if at capacity
    while len(_pkce_store) >= _PKCE_MAX_ENTRIES:
        oldest_key = min(_pkce_store, key=lambda k: _pkce_store[k][1])
        del _pkce_store[oldest_key]
    _pkce_store[state] = (verifier, now)


def _pkce_store_get(state: str) -> str | None:
    """Retrieve and consume a PKCE verifier (one-time use)."""
    entry = _pkce_store.pop(state, None)
    if entry is None:
        return None
    verifier, ts = entry
    if _time.time() - ts > _PKCE_TTL:
        return None  # expired
    return verifier


class ChatGPTTokenRequest(BaseModel):
    refresh: str
    account_id: str = ""


@router.get("/chatgpt/status")
async def chatgpt_status():
    """Check ChatGPT authentication status."""
    try:
        from aura.auth.chatgpt_oauth import is_authenticated, load_tokens
        authenticated = is_authenticated()
        info = {}
        if authenticated:
            tokens = load_tokens()
            if tokens and tokens.get("account_id"):
                info["account_id"] = tokens["account_id"][:8] + "..."
        return {"authenticated": authenticated, **info}
    except ImportError:
        return {"authenticated": False, "error": "auth module not available"}


@router.post("/chatgpt/set-token", dependencies=[Depends(require_api_key)])
async def chatgpt_set_token(body: ChatGPTTokenRequest):
    """Set the ChatGPT refresh token via HTTP POST.

    This allows setting the token from the browser extension or any HTTP client,
    avoiding the problem of long token strings breaking when pasted into a terminal.
    """
    if not body.refresh or not body.refresh.strip():
        return {"success": False, "error": "refresh token is required"}
    try:
        from aura.auth.chatgpt_oauth import save_refresh_token, get_valid_token
        path = save_refresh_token(body.refresh.strip(), body.account_id.strip())

        # Try an immediate refresh to validate the token
        tokens = get_valid_token()
        if tokens and tokens.get("access"):
            acct = tokens.get("account_id", "")
            return {
                "success": True,
                "message": "Token saved and validated",
                "account_id": (acct[:8] + "...") if acct else None,
                "path": str(path),
            }
        else:
            # Token was saved but refresh failed — might still work later
            return {
                "success": True,
                "message": "Token saved but could not validate (refresh failed). It may still work if the token is correct.",
                "path": str(path),
            }
    except Exception as e:
        logging.getLogger(__name__).error("chatgpt set-token failed: %s", e, exc_info=True)
        return {"success": False, "error": "Failed to set token — check server logs"}


@router.post("/chatgpt/login", dependencies=[Depends(require_api_key)])
async def chatgpt_login_url():
    """Get the ChatGPT OAuth login URL (for browser-based login).

    Uses the server-side callback at /api/auth/chatgpt/callback so the OAuth
    flow works even when the server is remote (no localhost needed).
    """
    try:
        from aura.auth.chatgpt_oauth import (
            AUTHORIZE_URL, CLIENT_ID, SCOPE,
            _generate_pkce,
        )
        import secrets
        from urllib.parse import urlencode
        import os

        verifier, challenge = _generate_pkce()
        state = secrets.token_hex(16)

        # Use localhost callback (required by OpenAI's registered redirect URI)
        from aura.auth.chatgpt_oauth import REDIRECT_URI, CALLBACK_PORT
        redirect_uri = REDIRECT_URI

        params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": "codex_cli_rs",
        }
        url = f"{AUTHORIZE_URL}?{urlencode(params)}"

        # Store verifier + redirect_uri server-side keyed by state
        _pkce_store_put(state, verifier)
        # Also store redirect_uri for the token exchange
        _pkce_redirect_store[state] = redirect_uri

        return {
            "url": url,
            "state": state,
            "instructions": "Open the URL in your browser. After login you'll be "
                            "redirected back and authenticated automatically.",
        }
    except Exception as e:
        logging.getLogger(__name__).error("chatgpt login-url failed: %s", e, exc_info=True)
        return {"error": "Failed to generate login URL — check server logs"}


# Store redirect_uri per state for callback exchange
_pkce_redirect_store: dict[str, str] = {}


@router.get("/chatgpt/callback")
async def chatgpt_oauth_callback(code: str = "", state: str = "", error: str = ""):
    """Server-side OAuth callback — exchanges code for tokens automatically."""
    from fastapi.responses import HTMLResponse

    if error:
        return HTMLResponse(
            f"<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
            f"<h1>Authentication Failed</h1><p>{error}</p>"
            f"<p>Close this window and try again.</p></body></html>",
            status_code=400,
        )

    if not code or not state:
        return HTMLResponse(
            "<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
            "<h1>Missing Parameters</h1><p>No authorization code received.</p>"
            "</body></html>",
            status_code=400,
        )

    # Retrieve stored PKCE verifier
    verifier = _pkce_store_get(state)
    if not verifier:
        return HTMLResponse(
            "<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
            "<h1>Session Expired</h1><p>OAuth state not found. Please try logging in again.</p>"
            "</body></html>",
            status_code=400,
        )

    redirect_uri = _pkce_redirect_store.pop(state, "")

    # Exchange authorization code for tokens
    try:
        import requests as http_requests
        from aura.auth.chatgpt_oauth import (
            TOKEN_URL, CLIENT_ID, _save_tokens, _extract_account_id,
        )

        resp = http_requests.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )

        if resp.status_code != 200:
            logging.getLogger(__name__).error(
                "ChatGPT token exchange failed: %s %s", resp.status_code, resp.text
            )
            return HTMLResponse(
                f"<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
                f"<h1>Token Exchange Failed</h1><p>Status {resp.status_code}</p>"
                f"</body></html>",
                status_code=502,
            )

        data = resp.json()
        access = data.get("access_token", "")
        refresh = data.get("refresh_token", "")
        expires_in = data.get("expires_in", 3600)

        if not access:
            return HTMLResponse(
                "<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
                "<h1>No Access Token</h1><p>Response missing access token.</p>"
                "</body></html>",
                status_code=502,
            )

        import time
        expires = int(time.time() * 1000) + expires_in * 1000
        _save_tokens(access, refresh, expires)

        account_id = _extract_account_id(access) or ""
        display_id = (account_id[:8] + "...") if account_id else "unknown"

        return HTMLResponse(
            f"<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
            f"<h1>Authenticated!</h1>"
            f"<p>Account: {display_id}</p>"
            f"<p>ChatGPT models are now available in AURA.</p>"
            f"<p>You can close this window.</p>"
            f"</body></html>"
        )

    except Exception as e:
        logging.getLogger(__name__).error("ChatGPT callback exchange failed: %s", e, exc_info=True)
        return HTMLResponse(
            f"<html><body style='font-family:system-ui;text-align:center;padding:60px'>"
            f"<h1>Error</h1><p>{e}</p></body></html>",
            status_code=500,
        )


@router.post("/chatgpt/logout", dependencies=[Depends(require_api_key)])
async def chatgpt_logout():
    """Remove ChatGPT authentication."""
    try:
        from aura.auth.chatgpt_oauth import logout
        logout()
        return {"success": True}
    except Exception as e:
        logging.getLogger(__name__).error("chatgpt logout failed: %s", e, exc_info=True)
        return {"error": "Logout failed — check server logs"}

"""Authentication routes for external providers (ChatGPT OAuth, etc.)."""

import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth import require_api_key

router = APIRouter(prefix="/api/auth", tags=["auth"])


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


@router.post("/chatgpt/login")
async def chatgpt_login_url():
    """Get the ChatGPT OAuth login URL (for browser-based login).

    The actual login happens in the browser. Call GET /chatgpt/status
    to check if auth completed.
    """
    try:
        from aura.auth.chatgpt_oauth import (
            AUTHORIZE_URL, CLIENT_ID, REDIRECT_URI, SCOPE,
            _generate_pkce, CALLBACK_PORT
        )
        import secrets
        from urllib.parse import urlencode

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
        url = f"{AUTHORIZE_URL}?{urlencode(params)}"

        return {
            "url": url,
            "state": state,
            "verifier": verifier,
            "port": CALLBACK_PORT,
            "instructions": "Open the URL in a browser. After login, the callback "
                            f"server on port {CALLBACK_PORT} will capture the token.",
        }
    except Exception as e:
        logging.getLogger(__name__).error("chatgpt login-url failed: %s", e, exc_info=True)
        return {"error": "Failed to generate login URL — check server logs"}


@router.post("/chatgpt/logout")
async def chatgpt_logout():
    """Remove ChatGPT authentication."""
    try:
        from aura.auth.chatgpt_oauth import logout
        logout()
        return {"success": True}
    except Exception as e:
        logging.getLogger(__name__).error("chatgpt logout failed: %s", e, exc_info=True)
        return {"error": "Logout failed — check server logs"}

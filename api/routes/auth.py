"""Authentication routes for external providers (ChatGPT OAuth, etc.)."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/auth", tags=["auth"])


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
        return {"error": str(e)}


@router.post("/chatgpt/logout")
async def chatgpt_logout():
    """Remove ChatGPT authentication."""
    try:
        from aura.auth.chatgpt_oauth import logout
        logout()
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

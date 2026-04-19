"""API key authentication for Aura routes."""
import logging
import os
import secrets

from fastapi import Header, HTTPException, Request, status

logger = logging.getLogger(__name__)

_API_KEY_ENV = "AURA_API_KEY"
_AUTH_REQUIRED_ENV = "AURA_REQUIRE_AUTH"  # Legacy — prefer AURA_API_AUTH_ENABLED


def _get_configured_key() -> str | None:
    return os.environ.get(_API_KEY_ENV)


def _auth_is_enabled() -> bool:
    """Returns True if API authentication is enabled.

    Canonical flag is AURA_API_AUTH_ENABLED (secure default: true).
    Legacy AURA_REQUIRE_AUTH is accepted as an alias with a one-time
    deprecation warning. If neither is set, auth is ON.
    """
    api_auth = os.environ.get("AURA_API_AUTH_ENABLED")
    if api_auth is not None:
        return api_auth.lower() in ("true", "1", "yes")
    legacy = os.environ.get(_AUTH_REQUIRED_ENV)
    if legacy is not None:
        if not _auth_is_enabled._warned:
            logger.warning(
                "AURA_REQUIRE_AUTH is deprecated; use AURA_API_AUTH_ENABLED"
            )
            _auth_is_enabled._warned = True
        return legacy.lower() in ("true", "1", "yes")
    return True  # Secure default
_auth_is_enabled._warned = False


async def require_api_key(
    request: Request,
    x_api_key: str = Header(default=""),
) -> str:
    """FastAPI dependency: accepts EITHER a valid X-API-Key header OR a
    valid aura_session cookie from the web-login flow."""
    if not _auth_is_enabled():
        return ""

    configured = _get_configured_key()

    if configured is None:
        # SECURITY: Fail CLOSED — auth is enabled but no key is configured.
        logger.error("Auth enabled but no AURA_API_KEY set — blocking all requests")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth is enabled but AURA_API_KEY is not configured. Set it or disable auth.",
        )

    # Path 1: header
    if x_api_key and secrets.compare_digest(x_api_key, configured):
        return x_api_key

    # Path 2: session cookie (web UI login)
    try:
        from api.auth_session import extract_session_username
        if extract_session_username(request.headers):
            return "session"
    except Exception as e:
        logger.debug(f"session cookie check failed: {e}")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "ApiKey"},
    )


def verify_api_key_ws(key: str) -> bool:
    """For WebSocket connections where Header dependency isn't available.

    Uses the same _auth_is_enabled() check as require_api_key and
    APIKeyAuthMiddleware for consistent behavior across all paths.
    """
    if not _auth_is_enabled():
        return True

    configured = _get_configured_key()
    if configured is None:
        return False  # Auth enabled but no key configured — fail closed
    return secrets.compare_digest(key or "", configured)


def verify_websocket_auth(websocket) -> bool:
    """Accept EITHER a valid X-API-Key header, a ?token= query param, OR a
    valid aura_session cookie. Used by WebSocket handlers that can't rely on
    the HTTP middleware.

    The web UI's browser automatically sends the session cookie on the WS
    upgrade (same-origin), so logged-in users need no extra setup.
    The extension continues to pass X-API-Key / ?token=.
    """
    if not _auth_is_enabled():
        return True

    # Header path — works for desktop clients that can set headers
    header_key = websocket.headers.get("X-API-Key") or websocket.query_params.get("token")
    if header_key and verify_api_key_ws(header_key):
        return True

    # Cookie path — browsers send this automatically on same-origin WS
    try:
        from api.auth_session import extract_session_username
        if extract_session_username(websocket.headers):
            return True
    except Exception as e:
        logger.debug(f"WebSocket cookie check failed: {e}")
    return False

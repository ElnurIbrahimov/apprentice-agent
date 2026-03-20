"""API key authentication for Aura routes."""
import os
import secrets
import logging
from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)

_API_KEY_ENV = "AURA_API_KEY"
_AUTH_REQUIRED_ENV = "AURA_REQUIRE_AUTH"


def _get_configured_key() -> str | None:
    return os.environ.get(_API_KEY_ENV)


def _auth_is_required() -> bool:
    """Returns True if AURA_REQUIRE_AUTH=true (default: False for local dev)."""
    return os.environ.get(_AUTH_REQUIRED_ENV, "false").lower() in ("true", "1", "yes")


async def require_api_key(x_api_key: str = Header(default="")) -> str:
    """FastAPI dependency: validates X-API-Key header."""
    # If API-level auth is disabled (AURA_API_AUTH_ENABLED=false), allow all.
    # Consistent with APIKeyAuthMiddleware and verify_api_key_ws.
    api_auth_enabled = os.environ.get("AURA_API_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
    if api_auth_enabled and not os.environ.get("AURA_API_KEY"):
        logger.warning("AURA_API_AUTH_ENABLED is true but no AURA_API_KEY set — auth disabled")
        api_auth_enabled = False
    if not api_auth_enabled:
        return ""

    configured = _get_configured_key()

    if configured is None:
        if _auth_is_required():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "AURA_REQUIRE_AUTH=true but AURA_API_KEY is not set. "
                    "Set AURA_API_KEY environment variable."
                ),
            )
        return ""

    if not secrets.compare_digest(x_api_key or "", configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


def verify_api_key_ws(key: str) -> bool:
    """For WebSocket connections where Header dependency isn't available.

    Checks AURA_API_AUTH_ENABLED first (same flag the APIKeyAuthMiddleware uses)
    so that WebSocket auth is consistent with HTTP auth.  Falls back to
    AURA_REQUIRE_AUTH for backward compatibility.
    """
    # If the API-level auth flag is not enabled, allow all WS connections.
    # This matches APIKeyAuthMiddleware which reads AURA_API_AUTH_ENABLED
    # (defaults to "true" when unset — consistent with require_api_key).
    api_auth_enabled = os.environ.get("AURA_API_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")
    if not api_auth_enabled:
        return True

    configured = _get_configured_key()
    if configured is None:
        return not _auth_is_required()  # Block if auth required, allow in dev mode
    return secrets.compare_digest(key or "", configured)

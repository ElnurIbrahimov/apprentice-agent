"""Web UI login/logout/me endpoints — cookie-based session auth.

Single-user login. Credentials and session secret live in the server .env.
Issues an HTTP-only, Secure, SameSite=Lax cookie on success that the web UI
uses for every subsequent request. The same cookie is also accepted by
APIKeyAuthMiddleware (alongside the existing X-API-Key header path).
"""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel

from api.auth_session import (
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    create_session_token,
    credentials_configured,
    extract_session_username,
    revoke_session_token,
    verify_credentials,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth/web", tags=["web-auth"])


# ---- Rate limit login attempts ---------------------------------------------
# Per-IP sliding window: max _MAX_ATTEMPTS failed logins in _WINDOW_SEC.
_MAX_ATTEMPTS = 10
_WINDOW_SEC = 300
_attempts: dict[str, deque] = defaultdict(deque)
_attempts_lock = Lock()


def _client_ip(request: Request) -> str:
    """Return the client IP, respecting X-Forwarded-Only when behind a proxy."""
    # Only trust X-Forwarded-For when AURA_TRUST_PROXY is enabled (same logic
    # as api.middleware.APIKeyAuthMiddleware).  Direct internet exposure without
    # a reverse proxy makes this header trivially spoofable.
    trust_proxy = os.getenv("AURA_TRUST_PROXY", "").lower() in ("1", "true", "yes")
    fwd = request.headers.get("x-forwarded-for")
    if trust_proxy and fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(ip: str) -> bool:
    """Return True if the IP is still under the limit."""
    now = time.time()
    with _attempts_lock:
        bucket = _attempts[ip]
        # Drop timestamps outside the window
        while bucket and now - bucket[0] > _WINDOW_SEC:
            bucket.popleft()
        return len(bucket) < _MAX_ATTEMPTS


def _record_failure(ip: str) -> None:
    with _attempts_lock:
        _attempts[ip].append(time.time())


# ---- Schemas ---------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class MeResponse(BaseModel):
    authenticated: bool
    username: str = ""
    configured: bool = True


# ---- Routes ----------------------------------------------------------------

@router.post("/login")
async def login(request: Request, response: Response, body: LoginRequest):
    """Validate credentials and set the session cookie."""
    if not credentials_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Web login is not configured on the server. Set "
                "AURA_WEB_USERNAME, AURA_WEB_PASSWORD_SALT, "
                "AURA_WEB_PASSWORD_HASH, and AURA_SESSION_SECRET in .env."
            ),
        )

    ip = _client_ip(request)
    if not _check_rate_limit(ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many login attempts. Try again in {_WINDOW_SEC}s.",
        )

    if not verify_credentials(body.username, body.password):
        _record_failure(ip)
        logger.warning("[web-auth] Failed login from %s (user=%r)", ip, body.username[:32])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    token = create_session_token(body.username)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not mint session token (missing AURA_SESSION_SECRET).",
        )

    # Use Secure cookie behind HTTPS; SameSite=Lax is the right default for
    # same-site web UI use (extension still uses X-API-Key, not this cookie).
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    logger.info("[web-auth] %s logged in from %s", body.username, ip)
    return {"ok": True, "username": body.username}


@router.post("/logout")
async def logout(request: Request, response: Response):
    """Revoke the session token server-side and clear the cookie.
    Always succeeds (idempotent) so clients can't probe for validity."""
    # Pull the token out of the cookie header ourselves — we need the raw
    # string, not just the decoded username.
    cookie_header = request.headers.get("cookie", "")
    for part in cookie_header.split(";"):
        name, _, value = part.strip().partition("=")
        if name == SESSION_COOKIE_NAME and value:
            try:
                revoke_session_token(value)
            except Exception:
                logger.debug("[web-auth] revoke failed", exc_info=True)
            break
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=True,
        samesite="lax",
    )
    return {"ok": True}


@router.get("/me", response_model=MeResponse)
async def me(request: Request):
    """Return the authenticated username, if any. Used by the web UI to
    decide whether to show the login screen on page load."""
    username = extract_session_username(request.headers)
    return MeResponse(
        authenticated=bool(username),
        username=username or "",
        configured=credentials_configured(),
    )

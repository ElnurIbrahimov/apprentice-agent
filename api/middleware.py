"""API middleware for authentication, rate limiting, and security."""

import logging
import os
import re
import secrets
import time
import uuid
from collections import defaultdict
from typing import ClassVar, Dict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# When True, parse X-Forwarded-For to get the real client IP behind a reverse proxy.
# NOTE: X-Forwarded-For can be spoofed if not behind a trusted reverse proxy.
# Only enable this when running behind a trusted proxy (nginx, Cloudflare, etc.).
_trust_proxy = os.environ.get("AURA_TRUST_PROXY", "").lower() in ("true", "1", "yes")


def _get_client_ip(request) -> str:
    """Extract client IP from request, respecting trusted proxy headers.

    When behind a trusted reverse proxy, use the RIGHTMOST IP in
    X-Forwarded-For (the one appended by the proxy itself). The leftmost
    IP is attacker-controlled and must not be trusted for rate limiting.
    """
    if _trust_proxy:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Rightmost IP is the one added by the trusted proxy
            return forwarded_for.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware.

    Validates requests against a configured API key.
    Skips auth for health/status endpoints and when auth is disabled.
    """

    # Endpoints that don't require authentication
    PUBLIC_PATHS: ClassVar[set[str]] = {
        "/",
        "/health",
        "/api/health",
        "/api/health/deep",
        "/api/status",
        "/api/auth/chatgpt/status",
        "/api/auth/chatgpt/login",
        # Web login lives here — must itself be reachable without auth
        "/api/auth/web/login",
        "/api/auth/web/logout",
        "/api/auth/web/me",
        # PWA manifest + service worker must be reachable pre-login
        "/manifest.json",
        "/sw.js",
        "/api/telegram/validate-init",
        "/api/telegram/proactive/action",
        "/api/telegram/memory/browse",
        "/api/telegram/memory/item/get",
        "/api/telegram/memory/item/patch",
        "/api/telegram/memory/item/delete",
        "/api/telegram/memory/item/pin",
        "/api/telegram/memory/stats",
        "/api/telegram/memory/kg/top",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    def __init__(self, app, api_key: str = "", enabled: bool = True):
        super().__init__(app)
        self.api_key = api_key
        if enabled and not api_key:
            logger.warning("[Auth] Auth enabled but no API key configured — all authenticated requests will be rejected. Set AURA_API_KEY.")
            self.enabled = True  # Reject all rather than pass all
        else:
            self.enabled = enabled and bool(api_key)
        if self.enabled:
            logger.info("[Auth] API key authentication enabled")
        else:
            logger.info("[Auth] API key authentication disabled (set AURA_API_KEY and AURA_API_AUTH_ENABLED=true to enable)")

    async def dispatch(self, request: Request, call_next):
        # Re-check at request time so runtime overrides (e.g. tests) work.
        # Delegates to api.auth._auth_is_enabled so legacy AURA_REQUIRE_AUTH
        # stays in sync with the canonical AURA_API_AUTH_ENABLED flag.
        from api.auth import _auth_is_enabled
        if not _auth_is_enabled():
            return await call_next(request)
        if not self.enabled:
            return await call_next(request)

        # Skip auth for WebSocket upgrades — browsers cannot send custom headers
        # on new WebSocket(), so API key auth is handled inside the WS handler
        # itself (see verify_api_key_ws in api/auth.py).
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth for static files
        if request.url.path.startswith("/static") or request.url.path.startswith("/assets"):
            return await call_next(request)

        # Accept EITHER the X-API-Key header (extensions, raw API clients) OR
        # a valid aura_session cookie (web UI after login). Either is enough.
        api_key = request.headers.get("X-API-Key")
        if api_key and secrets.compare_digest(api_key, self.api_key):
            return await call_next(request)

        # Fallback: session cookie from the login page.
        try:
            from api.auth_session import extract_session_username
            if extract_session_username(request.headers):
                return await call_next(request)
        except Exception as e:
            logger.debug(f"[Auth] Session cookie check failed: {e}")

        if api_key:
            client_host = request.client.host if request.client else "unknown"
            logger.warning(f"[Auth] Invalid API key from {client_host}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."}
            )
        return JSONResponse(
            status_code=401,
            content={"detail": "Authentication required. Provide X-API-Key header or log in via the web UI."}
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting per IP address.

    Uses a sliding window counter approach.
    """

    # Paths exempt from rate limiting (health checks, monitoring)
    EXEMPT_PATHS: ClassVar[set[str]] = {
        "/",
        "/health",
        "/api/health",
        "/api/health/deep",
        "/api/status",
        "/api/init",
    }

    def __init__(self, app, requests_per_minute: int = 300, enabled: bool = True):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled and requests_per_minute > 0
        # Track requests: ip -> list of timestamps
        self._requests: Dict[str, list] = defaultdict(list)
        # Track WebSocket connection attempts: ip -> list of timestamps
        self._ws_requests: Dict[str, list] = defaultdict(list)
        self._ws_connections_per_minute = 30  # Lower limit for WS handshakes
        self._cleanup_interval = 300  # Clean old entries every 5 min
        self._last_cleanup = time.time()
        # No lock needed: asyncio is single-threaded and there's no await
        # inside the critical section, so no coroutine can interleave.
        if self.enabled:
            logger.info(f"[RateLimit] Enabled: {requests_per_minute} requests/minute per IP")

    def _cleanup_old_entries(self):
        """Remove entries older than 2 minutes to prevent memory growth."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        cutoff = now - 120
        stale_ips = []
        for ip, timestamps in self._requests.items():
            self._requests[ip] = [t for t in timestamps if t > cutoff]
            if not self._requests[ip]:
                stale_ips.append(ip)
        for ip in stale_ips:
            del self._requests[ip]
        # Also clean WebSocket rate limit entries
        stale_ws_ips = []
        for ip, timestamps in self._ws_requests.items():
            self._ws_requests[ip] = [t for t in timestamps if t > cutoff]
            if not self._ws_requests[ip]:
                stale_ws_ips.append(ip)
        for ip in stale_ws_ips:
            del self._ws_requests[ip]
        self._last_cleanup = now

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        # Rate limit WebSocket handshakes (lower limit than HTTP)
        if request.headers.get("upgrade", "").lower() == "websocket":
            ws_ip = _get_client_ip(request)
            now = time.time()
            ws_window_start = now - 60
            self._ws_requests[ws_ip] = [
                t for t in self._ws_requests[ws_ip] if t > ws_window_start
            ]
            if len(self._ws_requests[ws_ip]) >= self._ws_connections_per_minute:
                retry_after = int(60 - (now - self._ws_requests[ws_ip][0]))
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"WebSocket rate limit exceeded. Try again in {max(1, retry_after)}s."},
                    headers={"Retry-After": str(max(1, retry_after))}
                )
            self._ws_requests[ws_ip].append(now)
            return await call_next(request)

        # Skip rate limiting for health/monitoring endpoints
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        client_ip = _get_client_ip(request)
        now = time.time()

        window_start = now - 60

        # Count requests in the current window
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > window_start
        ]

        if len(self._requests[client_ip]) >= self.requests_per_minute:
            retry_after = int(60 - (now - self._requests[client_ip][0]))
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded. Try again in {max(1, retry_after)}s."},
                headers={"Retry-After": str(max(1, retry_after))}
            )

        self._requests[client_ip].append(now)
        self._cleanup_old_entries()

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every HTTP response.

    Prevents clickjacking, MIME-sniffing, and information leakage.
    """

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID header to every HTTP response.

    If the client already provides an X-Request-ID header, it is reused;
    otherwise a new UUID4 is generated.  The ID is also stored on
    ``request.state.request_id`` so error handlers can include it.
    """

    async def dispatch(self, request: Request, call_next):
        # Skip WebSocket upgrades — headers can't be added to WS responses
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        raw_request_id = request.headers.get("X-Request-ID")
        # Validate: only accept UUID-like or alphanumeric request IDs (max 64 chars)
        if raw_request_id and re.match(r'^[a-zA-Z0-9_\-]{1,64}$', raw_request_id):
            request_id = raw_request_id
        else:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

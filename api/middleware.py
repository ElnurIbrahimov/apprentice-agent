"""API middleware for authentication, rate limiting, and security."""

import os
import time
import uuid
import logging
import secrets
from collections import defaultdict
from typing import Dict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# When True, parse X-Forwarded-For to get the real client IP behind a reverse proxy.
# NOTE: X-Forwarded-For can be spoofed if not behind a trusted reverse proxy.
# Only enable this when running behind a trusted proxy (nginx, Cloudflare, etc.).
_trust_proxy = os.environ.get("AURA_TRUST_PROXY", "").lower() in ("true", "1", "yes")


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware.

    Validates requests against a configured API key.
    Skips auth for health/status endpoints and when auth is disabled.
    """

    # Endpoints that don't require authentication
    PUBLIC_PATHS = {
        "/",
        "/health",
        "/api/health",
        "/api/health/deep",
        "/api/status",
        "/api/auth/chatgpt/status",
        "/api/auth/chatgpt/login",
        "/api/auth/chatgpt/logout",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    def __init__(self, app, api_key: str = "", enabled: bool = False):
        super().__init__(app)
        self.api_key = api_key
        self.enabled = enabled and bool(api_key)
        if self.enabled:
            logger.info("[Auth] API key authentication enabled")
        else:
            logger.info("[Auth] API key authentication disabled (set AURA_API_KEY and AURA_API_AUTH_ENABLED=true to enable)")

    async def dispatch(self, request: Request, call_next):
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

        # Check API key in header only (query params are logged and leaked in referrers)
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing API key. Provide X-API-Key header."}
            )

        if not secrets.compare_digest(api_key, self.api_key):
            client_host = request.client.host if request.client else "unknown"
            logger.warning(f"[Auth] Invalid API key from {client_host}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."}
            )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting per IP address.

    Uses a sliding window counter approach.
    """

    # Paths exempt from rate limiting (health checks, monitoring)
    EXEMPT_PATHS = {"/", "/health", "/api/health", "/api/health/deep", "/api/status", "/api/init"}

    def __init__(self, app, requests_per_minute: int = 300, enabled: bool = True):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.enabled = enabled and requests_per_minute > 0
        # Track requests: ip -> list of timestamps
        self._requests: Dict[str, list] = defaultdict(list)
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
        self._last_cleanup = now

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        # Skip rate limiting for WebSocket upgrades (handled differently)
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        # Skip rate limiting for health/monitoring endpoints
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Support reverse proxy: check X-Forwarded-For when behind a trusted proxy
        if _trust_proxy:
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                # X-Forwarded-For is comma-separated; leftmost is the original client
                client_ip = forwarded_for.split(",")[0].strip()
            else:
                client_ip = request.client.host if request.client else "unknown"
        else:
            client_ip = request.client.host if request.client else "unknown"
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

        import re as _re
        raw_request_id = request.headers.get("X-Request-ID")
        # Validate: only accept UUID-like or alphanumeric request IDs (max 64 chars)
        if raw_request_id and _re.match(r'^[a-zA-Z0-9_\-]{1,64}$', raw_request_id):
            request_id = raw_request_id
        else:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

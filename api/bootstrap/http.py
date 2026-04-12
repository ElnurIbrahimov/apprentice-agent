"""HTTP app configuration helpers for the AURA API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.middleware import (
    APIKeyAuthMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
)

if TYPE_CHECKING:
    import logging

    from fastapi import FastAPI


def _is_production() -> bool:
    return os.environ.get("AURA_ENV") == "production"


def _resolve_cors_origins(logger: "logging.Logger") -> tuple[list[str], str | None]:
    """Load and normalize the configured CORS origins."""

    try:
        from aura.config import Config as config

        cors_origins_str = getattr(config, "API_CORS_ORIGINS", "*")
    except Exception:
        logger.warning(
            "[API] CORS config load failed - defaulting to localhost only "
            "(set API_CORS_ORIGINS explicitly)",
            exc_info=True,
        )
        cors_origins_str = "http://localhost:5173,http://127.0.0.1:5173"

    if cors_origins_str == "*":
        cors_origins = ["*"]
    else:
        cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

    if cors_origins != ["*"]:
        extra_origins: list[str] = []
        for origin in cors_origins:
            if "localhost" in origin:
                extra_origins.append(origin.replace("localhost", "127.0.0.1"))
            elif "127.0.0.1" in origin:
                extra_origins.append(origin.replace("127.0.0.1", "localhost"))
        cors_origins = list(dict.fromkeys([*cors_origins, *extra_origins]))

    cors_origin_regex = r"^chrome-extension://.*$" if cors_origins != ["*"] else None
    return cors_origins, cors_origin_regex


def configure_http_middleware(app: "FastAPI", logger: "logging.Logger") -> None:
    """Attach middleware in the correct order for the API surface."""

    cors_origins, cors_origin_regex = _resolve_cors_origins(logger)

    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestIDMiddleware)

    try:
        from aura.config import Config as config

        api_key = getattr(config, "API_KEY", "")
        auth_enabled = getattr(config, "API_AUTH_ENABLED", bool(api_key))
        app.add_middleware(
            APIKeyAuthMiddleware,
            api_key=api_key,
            enabled=auth_enabled,
        )
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=getattr(config, "API_RATE_LIMIT", 300),
            enabled=True,
        )
    except Exception as exc:
        logger.error(
            "[API] CRITICAL: Auth/rate-limit middleware setup FAILED: %s - "
            "server may be unprotected",
            exc,
            exc_info=True,
        )
        if _is_production():
            raise RuntimeError(
                "Production startup aborted because auth/rate-limit middleware failed to initialize"
            ) from exc

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=cors_origin_regex,
        allow_credentials=cors_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*", "X-API-Key"],
    )


def mount_static_frontend(app: "FastAPI") -> None:
    """Serve the built frontend in production mode."""

    static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "dist")
    is_dev = os.environ.get("AURA_ENV") != "production"
    if not os.path.exists(static_path) or is_dev:
        return

    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(static_path, "assets")),
        name="assets",
    )

    @app.get("/miniapp")
    async def serve_miniapp() -> FileResponse:
        """Serve the Telegram Mini App HTML."""

        miniapp_path = os.path.join(static_path, "miniapp.html")
        if os.path.exists(miniapp_path):
            return FileResponse(miniapp_path)

        src_miniapp = os.path.join(os.path.dirname(static_path), "miniapp.html")
        if os.path.exists(src_miniapp):
            return FileResponse(src_miniapp)

        raise HTTPException(status_code=404, detail="Mini app not found")

    @app.get("/")
    async def serve_index() -> FileResponse:
        """Serve the React app index.html."""

        return FileResponse(os.path.join(static_path, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str) -> FileResponse:
        """Serve SPA assets, falling back to index.html for client routing."""

        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        real_static = os.path.realpath(static_path)
        candidate = os.path.realpath(os.path.join(static_path, full_path))
        if not (candidate.startswith(real_static + os.sep) or candidate == real_static):
            return FileResponse(os.path.join(static_path, "index.html"))

        if os.path.exists(candidate) and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(static_path, "index.html"))

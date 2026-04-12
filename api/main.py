"""FastAPI application entry point for AURA Web API."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI

from api.bootstrap.http import configure_http_middleware, mount_static_frontend
from api.bootstrap.routes import include_loaded_routers, load_route_modules
from api.bootstrap.runtime import create_lifespan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

is_production = os.environ.get("AURA_ENV") == "production"
app = FastAPI(
    title="AURA Web API",
    description="Modern web interface for AURA - Autonomous Universal Reasoning Agent",
    version="1.0.0",
    lifespan=create_lifespan(logger),
    docs_url=None if is_production else "/docs",
    redoc_url=None if is_production else "/redoc",
    openapi_url=None if is_production else "/openapi.json",
)

configure_http_middleware(app, logger)
include_loaded_routers(app, load_route_modules(logger))
mount_static_frontend(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.environ.get("AURA_HOST", "127.0.0.1"),
        port=8000,
        reload=False,
        log_level="info",
    )

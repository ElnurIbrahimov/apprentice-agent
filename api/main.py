"""FastAPI application entry point for AURA Web API."""

import os
import sys
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import chat, status, features, multi_agent, upload, reasoning_tree, introspection
from api.services.agent_service import agent_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track async initialization state
_init_state = {"ready": False, "error": None, "progress": "starting"}


def _background_init():
    """Initialize agent in background thread."""
    global _init_state
    try:
        _init_state["progress"] = "loading agent..."
        logger.info("[API] Background: Starting agent initialization...")
        agent_service.initialize(fast_init=False)  # Full init with all tools
        _init_state["ready"] = True
        _init_state["progress"] = "ready"
        logger.info("[API] Background: Agent initialization complete!")
    except Exception as e:
        _init_state["error"] = str(e)
        _init_state["progress"] = f"error: {e}"
        logger.error(f"[API] Background: Agent initialization failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("[API] Starting AURA Web API...")
    logger.info("[API] Server ready - agent initializing in background...")

    # Start agent initialization in background thread
    init_thread = threading.Thread(target=_background_init, daemon=True)
    init_thread.start()

    # Store init state in app for endpoints to check
    app.state.init_state = _init_state

    yield

    # Shutdown
    logger.info("[API] Shutting down AURA Web API...")


# Create FastAPI app
app = FastAPI(
    title="AURA Web API",
    description="Modern web interface for AURA - Autonomous Universal Reasoning Agent",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(status.router)
app.include_router(features.router)
app.include_router(multi_agent.router)
app.include_router(upload.router)
app.include_router(reasoning_tree.router)
app.include_router(introspection.router)

# Serve static files in production (built React app)
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "dist")
if os.path.exists(static_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(static_path, "assets")), name="assets")

    @app.get("/")
    async def serve_index():
        """Serve the React app index.html."""
        return FileResponse(os.path.join(static_path, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA - fallback to index.html for client-side routing."""
        # Don't intercept API routes - let the API routers handle them
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="API endpoint not found")

        file_path = os.path.join(static_path, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(static_path, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

"""FastAPI application entry point for AURA Web API."""

import os
import sys
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import chat, status, features, multi_agent, upload, reasoning_tree, introspection, proactive, memory, context, conversation_starters, thinking
from api.services.agent_service import agent_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("[API] Starting AURA Web API...")
    logger.info("[API] Initializing agent (this may take a moment)...")

    try:
        agent_service.initialize(fast_init=False)  # Load ALL tools
        logger.info("[API] Agent initialized successfully")
    except Exception as e:
        logger.error(f"[API] Failed to initialize agent: {e}")
        raise

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
app.include_router(proactive.router)
app.include_router(memory.router)
app.include_router(context.router)
app.include_router(conversation_starters.router)
app.include_router(thinking.router)

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

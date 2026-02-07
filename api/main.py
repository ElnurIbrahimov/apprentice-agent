"""FastAPI application entry point for AURA Web API."""

import os
import sys
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocket as StarletteWebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import chat, status, upload, features, multi_agent, reasoning_tree, introspection, proactive, memory, context, conversation_starters, thinking, idle_behaviors
# Lazy-loaded agent_service (import removed - now lazy in routes)
# from api.services.agent_service import agent_service

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

    # Increase thread pool from default 5 to 20.
    # The backend has 20+ polling endpoints all using run_in_executor(None, ...),
    # and chat requests can block for 30-60s waiting for Ollama.
    # With only 5 threads, 3 chat requests + polling = total starvation.
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=20))
    logger.info("[API] Thread pool set to 20 workers")

    # Start agent initialization in background thread
    # This doesn't block the event loop - server can respond to health checks immediately
    try:
        from api.services.agent_service import agent_service
        agent_service.start_background_init()
        logger.info("[API] Agent initialization started in background")
    except Exception as e:
        logger.error(f"[API] Agent initialization failed: {e}")
        logger.warning("[API] Server running without agent - install missing dependencies")

    # Start Gateway Daemon + SystemMonitor in background
    # (runs after agent init finishes, non-blocking)
    async def _start_proactive_system():
        """Wait for agent init, then start the proactive daemon."""
        # Wait up to 60s for agent to be ready
        for _ in range(30):
            await asyncio.sleep(2)
            try:
                from api.services.agent_service import agent_service
                if agent_service.is_ready:
                    break
            except Exception:
                pass
        else:
            logger.warning("[API] Agent not ready after 60s, starting proactive system anyway")

        try:
            from apprentice_agent.proactive.gateway_daemon import get_gateway_daemon
            from apprentice_agent.proactive.monitors.system_monitor import SystemMonitor

            daemon = get_gateway_daemon()

            # Wire the notification callback so messages go to the pending queue
            # AND get logged. The frontend polls get_pending_messages() via API.
            def _on_proactive_message(msg):
                logger.info(f"[Proactive] {msg.action.value}: {msg.content[:80]}...")

            daemon.set_notification_callback(_on_proactive_message)

            # Start the daemon (creates event bus, decision loop)
            await daemon.start()

            # Start SystemMonitor connected to daemon's event bus
            sys_monitor = SystemMonitor(
                event_bus=daemon.event_bus,
                poll_interval=30.0,  # Check system every 30s
            )
            await sys_monitor.start()

            # Store ref for shutdown
            app.state.proactive_daemon = daemon
            app.state.system_monitor = sys_monitor

            logger.info("[API] Proactive system started (Gateway Daemon + SystemMonitor)")
            logger.info("[API] SQLite persistence active for proactive subsystem")
        except Exception as e:
            logger.warning(f"[API] Proactive system failed to start: {e}")

        # Start Idle Presence Engine (sleep scheduling)
        try:
            from api.routes.idle_behaviors import init_idle_presence
            init_idle_presence()
        except Exception as e:
            logger.warning(f"[API] Idle presence init failed: {e}")

    asyncio.create_task(_start_proactive_system())

    yield

    # Shutdown
    logger.info("[API] Shutting down AURA Web API...")

    # Stop proactive system
    try:
        if hasattr(app.state, 'proactive_daemon') and app.state.proactive_daemon:
            await app.state.proactive_daemon.stop()
        if hasattr(app.state, 'system_monitor') and app.state.system_monitor:
            await app.state.system_monitor.stop()
        logger.info("[API] Proactive system stopped")
    except Exception as e:
        logger.warning(f"[API] Proactive shutdown error: {e}")

    # Stop Idle Presence Engine
    try:
        from apprentice_agent.consciousness.idle_presence import get_idle_presence_engine
        get_idle_presence_engine().stop_background_tasks()
    except Exception:
        pass

    # Close proactive persistence database
    try:
        from apprentice_agent.proactive.persistence import get_persistence
        get_persistence().close()
        logger.info("[API] Proactive persistence closed")
    except Exception as e:
        logger.warning(f"[API] Persistence shutdown error: {e}")


# Create FastAPI app
app = FastAPI(
    title="AURA Web API",
    description="Modern web interface for AURA - Autonomous Universal Reasoning Agent",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for development
# NOTE: Starlette 0.50+ CORSMiddleware rejects WebSocket with 403 when
# specific origins are listed. Use wildcard for dev to allow WebSocket.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers - frontend uses 2s stagger + 30s intervals to prevent thread pool exhaustion
app.include_router(chat.router)
app.include_router(status.router)
app.include_router(upload.router)
app.include_router(features.router)
app.include_router(multi_agent.router)
app.include_router(reasoning_tree.router)
app.include_router(introspection.router)
app.include_router(proactive.router)
app.include_router(memory.router)
app.include_router(context.router)
app.include_router(conversation_starters.router)
app.include_router(thinking.router)
app.include_router(idle_behaviors.router)

# Serve static files in production (built React app)
# NOTE: Only mount SPA routes when NOT in dev mode (Vite serves the frontend in dev)
# The catch-all /{full_path:path} route was intercepting WebSocket upgrades
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "dist")
_is_dev = os.environ.get("AURA_ENV") != "production"
if os.path.exists(static_path) and not _is_dev:
    app.mount("/assets", StaticFiles(directory=os.path.join(static_path, "assets")), name="assets")

    @app.get("/")
    async def serve_index():
        """Serve the React app index.html."""
        return FileResponse(os.path.join(static_path, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA - fallback to index.html for client-side routing."""
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
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

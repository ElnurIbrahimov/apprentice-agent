"""FastAPI application entry point for AURA Web API."""
# reload-trigger: 2026-03-01

import os
import sys
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.middleware import APIKeyAuthMiddleware, RateLimitMiddleware, RequestIDMiddleware

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from api.routes import chat, status, upload, features, multi_agent, reasoning_tree, proactive, memory, context, conversation_starters, thinking, idle_behaviors, self_improvement, thinking_mode, tools_new, activity, multi_model, knowledge, search, pdf, transcribe, ocr, image_gen, agent_action, build as build_route, models as models_route, summarize, youtube, math as math_route, research, evolution, artifacts, feed, providers as providers_route, code as code_route, webhooks as webhooks_route, generate as generate_route, share as share_route, hands as hands_route
try:
    from api.routes import reliability as reliability_route
    _reliability_available = True
except Exception as _re:
    logger.warning(f"[API] Reliability routes unavailable: {_re}")
    reliability_route = None
    _reliability_available = False
try:
    from api.routes import auth as auth_route
    _auth_available = True
except Exception as _ae:
    logger.warning(f"[API] Auth routes unavailable: {_ae}")
    auth_route = None
    _auth_available = False
_consciousness_available = False
# Lazy-loaded agent_service (import removed - now lazy in routes)
# from api.services.agent_service import agent_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("[API] Starting AURA Web API...")

    # Increase thread pool from default 5 to 20.
    # The backend has 20+ polling endpoints all using run_in_executor(None, ...),
    # and chat requests can block for 30-60s waiting for Ollama.
    # With only 5 threads, 3 chat requests + polling = total starvation.
    loop = asyncio.get_running_loop()
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
                logger.debug("agent_ready_check_failed", exc_info=True)
        else:
            logger.warning("[API] Agent not ready after 60s, starting proactive system anyway")

        try:
            from aura.proactive.gateway_daemon import get_gateway_daemon
            from aura.proactive.monitors.system_monitor import SystemMonitor

            daemon = get_gateway_daemon()

            # Wire the notification callback so messages go to the pending queue,
            # get logged, AND pushed to connected WebSocket clients in real-time.
            _proactive_loop = asyncio.get_running_loop()

            def _on_proactive_message(msg):
                logger.info(f"[Proactive] {msg.action.value}: {msg.content[:80]}...")
                # Push to all connected WebSocket clients (instant delivery)
                try:
                    from api.routes.chat import broadcast_proactive_message
                    if not _proactive_loop.is_closed():
                        _proactive_loop.call_soon_threadsafe(
                            _proactive_loop.create_task,
                            broadcast_proactive_message(msg)
                        )
                except Exception as e:
                    logger.debug(f"[Proactive] WebSocket push failed: {e}")

            daemon.set_notification_callback(_on_proactive_message)

            # Start the daemon (creates event bus, decision loop)
            await daemon.start()

            # Trigger first intrinsic motivation cycle on startup
            try:
                from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
                loop = asyncio.get_running_loop()
                im = get_intrinsic_motivation()
                await loop.run_in_executor(None, im.run_motivation_cycle)
                logger.info("[API] Intrinsic Motivation engine: first cycle complete")
            except Exception as e:
                logger.warning(f"[API] Motivation engine startup cycle failed: {e}")

            # Start SystemMonitor connected to daemon's event bus
            sys_monitor = SystemMonitor(
                event_bus=daemon.event_bus,
                poll_interval=60.0,  # Check system every 60s (was 30s)
            )
            await sys_monitor.start()

            # Start ScreenMonitor for app/window tracking + Screenpipe OCR
            # Skip on headless servers (no display = no useful screen data)
            _is_headless = bool(os.environ.get("AURA_HEADLESS")) or (
                sys.platform != "win32" and not os.environ.get("DISPLAY")
            )
            screen_monitor = None
            if _is_headless:
                logger.info("[API] ScreenMonitor skipped (headless mode)")
            else:
                try:
                    from aura.proactive.monitors.screen_monitor import ScreenMonitor
                    screen_monitor = ScreenMonitor(
                        event_bus=daemon.event_bus,
                        poll_interval=10.0,  # Was 3.0, reduce CPU/thread pressure
                    )
                    await screen_monitor.start()
                    logger.info("[API] ScreenMonitor started")
                except Exception as e:
                    logger.warning(f"[API] ScreenMonitor failed to start: {e}")

            # Start CalendarMonitor for meeting/event awareness
            calendar_monitor = None
            try:
                from aura.proactive.monitors.calendar_monitor import get_calendar_monitor
                calendar_monitor = get_calendar_monitor(event_bus=daemon.event_bus)
                await calendar_monitor.start()
                logger.info("[API] CalendarMonitor started")
            except Exception as e:
                logger.warning(f"[API] CalendarMonitor failed to start: {e}")

            # Start WorkflowDetector for interruption timing
            # Skip on headless servers (depends on Screenpipe / display)
            workflow_detector = None
            if _is_headless:
                logger.info("[API] WorkflowDetector skipped (headless mode)")
            else:
                try:
                    from aura.proactive.monitors.workflow_detector import get_workflow_detector
                    workflow_detector = get_workflow_detector(event_bus=daemon.event_bus)
                    await workflow_detector.start()
                    logger.info("[API] WorkflowDetector started")
                except Exception as e:
                    logger.warning(f"[API] WorkflowDetector failed to start: {e}")

            # Store refs for shutdown
            app.state.proactive_daemon = daemon
            app.state.system_monitor = sys_monitor
            app.state.screen_monitor = screen_monitor
            app.state.calendar_monitor = calendar_monitor
            app.state.workflow_detector = workflow_detector

            _active = ["GatewayDaemon", "SystemMonitor", "CalendarMonitor"]
            if screen_monitor:
                _active.append("ScreenMonitor")
            if workflow_detector:
                _active.append("WorkflowDetector")
            logger.info("[API] Proactive system started (%s)%s",
                        " + ".join(_active),
                        " [headless]" if _is_headless else "")
            logger.info("[API] SQLite persistence active for proactive subsystem")
        except Exception as e:
            logger.warning(f"[API] Proactive system failed to start: {e}")

        # Start Voice Presence Service
        try:
            from aura.services.voice_presence import get_voice_presence
            voice_svc = get_voice_presence()
            voice_svc.start()
            app.state.voice_presence = voice_svc
            logger.info("[API] VoicePresenceService started (kokoro)")
        except Exception as e:
            logger.warning(f"[API] VoicePresenceService failed to start: {e}")
            app.state.voice_presence = None

        # Global Workspace Engine removed

        # Start Idle Presence Engine (sleep scheduling)
        try:
            from api.routes.idle_behaviors import init_idle_presence
            init_idle_presence()
        except Exception as e:
            logger.warning(f"[API] Idle presence init failed: {e}")

        # Start Self-Improvement Engine
        try:
            from aura.consciousness.self_improvement import get_self_improvement_engine
            get_self_improvement_engine().start()
            logger.info("[API] Self-Improvement Engine started")
        except Exception as e:
            logger.warning(f"[API] Self-Improvement Engine failed to start: {e}")

        # Wire HandManager notification callback for WebSocket + Telegram push
        try:
            from aura.hands.manager import get_hand_manager
            _hands_mgr = get_hand_manager()

            _hands_loop = asyncio.get_running_loop()

            def _on_hand_result(result):
                # Broadcast to WebSocket clients
                try:
                    from api.routes.chat import broadcast_hand_event
                    if not _hands_loop.is_closed():
                        _hands_loop.call_soon_threadsafe(
                            _hands_loop.create_task,
                            broadcast_hand_event(result.to_dict()),
                        )
                except Exception as e:
                    logger.debug(f"[API] Hand WS broadcast failed: {e}")
                # Notify Telegram
                try:
                    from aura.messaging.telegram_bot import notify_hand_result
                    notify_hand_result(result)
                except Exception:
                    pass

            _hands_mgr.set_notify_callback(_on_hand_result)
            logger.info("[API] HandManager notification callback wired")
        except Exception as e:
            logger.debug(f"[API] HandManager callback wiring failed: {e}")

    app.state.proactive_startup_task = asyncio.get_running_loop().create_task(_start_proactive_system())

    yield

    # Shutdown
    logger.info("[API] Shutting down AURA Web API...")

    # Cancel proactive startup task if still running
    try:
        if hasattr(app.state, 'proactive_startup_task') and app.state.proactive_startup_task:
            task = app.state.proactive_startup_task
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.info("[API] Proactive startup task cancelled")
    except Exception as e:
        logger.warning(f"[API] Error cancelling startup task: {e}")

    # Stop proactive system
    try:
        if hasattr(app.state, 'proactive_daemon') and app.state.proactive_daemon:
            await app.state.proactive_daemon.stop()
        if hasattr(app.state, 'system_monitor') and app.state.system_monitor:
            await app.state.system_monitor.stop()
        if hasattr(app.state, 'screen_monitor') and app.state.screen_monitor:
            await app.state.screen_monitor.stop()
        if hasattr(app.state, 'calendar_monitor') and app.state.calendar_monitor:
            await app.state.calendar_monitor.stop()
        if hasattr(app.state, 'workflow_detector') and app.state.workflow_detector:
            await app.state.workflow_detector.stop()
        logger.info("[API] Proactive system stopped")
    except Exception as e:
        logger.warning(f"[API] Proactive shutdown error: {e}")

    # Stop Voice Presence Service
    try:
        if hasattr(app.state, 'voice_presence') and app.state.voice_presence:
            app.state.voice_presence.stop()
            logger.info("[API] VoicePresenceService stopped")
    except Exception as e:
        logger.warning(f"[API] Voice shutdown error: {e}")

    # Global Workspace Engine removed

    # Stop Self-Improvement Engine
    try:
        from aura.consciousness.self_improvement import get_self_improvement_engine
        get_self_improvement_engine().stop()
        logger.info("[API] Self-Improvement Engine stopped")
    except Exception as e:
        logger.warning(f"[API] Self-Improvement Engine stop failed: {e}")

    # Stop Idle Presence Engine
    try:
        from aura.consciousness.idle_presence import get_idle_presence_engine
        get_idle_presence_engine().stop_background_tasks()
        logger.info("[API] Idle Presence Engine stopped")
    except Exception as e:
        logger.warning(f"[API] Idle Presence Engine stop failed: {e}")

    # Memory systems — UnifiedMemory (SQLite) auto-commits, Kuzu auto-persists.
    # A-MEM and KG NetworkX removed in memory consolidation (2026-03-22).

    # Close proactive persistence database
    try:
        from aura.proactive.persistence import get_persistence
        get_persistence().close()
        logger.info("[API] Proactive persistence closed")
    except Exception as e:
        logger.warning(f"[API] Persistence shutdown error: {e}")


# Create FastAPI app
_is_production = os.environ.get("AURA_ENV") == "production"
app = FastAPI(
    title="AURA Web API",
    description="Modern web interface for AURA - Autonomous Universal Reasoning Agent",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    openapi_url=None if _is_production else "/openapi.json",
)

# Configure CORS
# NOTE: Starlette's CORSMiddleware rejects WebSocket upgrade requests with 403
# when specific origins are listed (it validates the Origin header).  Browsers
# always send an Origin header on WebSocket handshakes, and Starlette treats a
# missing/unmatched origin as a CORS violation → 403.
# Fix: In development we force allow_origins=["*"] so WebSocket works.
# In production the explicit list is kept for HTTP but we add a lightweight
# middleware that lets WebSocket upgrades pass through before CORS runs.
try:
    from aura.config import Config as _cfg
    _cors_origins_str = getattr(_cfg, 'API_CORS_ORIGINS', '*')
except Exception:
    logger.debug("cors_config_load_failed_using_default", exc_info=True)
    _cors_origins_str = '*'

# Default to localhost origins instead of wildcard for security.
# Set API_CORS_ORIGINS="*" explicitly to allow all origins.
if _cors_origins_str == "*":
    _cors_origins = ["*"]
else:
    _cors_origins = [o.strip() for o in _cors_origins_str.split(",")]

# When specific origins are configured, also accept 127.0.0.1 variants
# so that localhost:5173 -> 127.0.0.1:8000 WebSocket works regardless.
if _cors_origins != ["*"]:
    _extra = []
    for _o in _cors_origins:
        if "localhost" in _o:
            _extra.append(_o.replace("localhost", "127.0.0.1"))
        elif "127.0.0.1" in _o:
            _extra.append(_o.replace("127.0.0.1", "localhost"))
    _cors_origins = list(dict.fromkeys(_cors_origins + _extra))  # dedupe, preserve order

# Chrome extension origins are dynamic (chrome-extension://<id>) — use regex
_cors_origin_regex = r"^chrome-extension://.*$" if _cors_origins != ["*"] else None

# Request ID middleware — adds X-Request-ID header to every response
app.add_middleware(RequestIDMiddleware)

# API key authentication middleware (disabled by default, enable via env vars)
try:
    from aura.config import Config as _auth_cfg
    _api_key = getattr(_auth_cfg, 'API_KEY', '')
    _auth_enabled = getattr(_auth_cfg, 'API_AUTH_ENABLED', bool(_api_key))
    app.add_middleware(
        APIKeyAuthMiddleware,
        api_key=_api_key,
        enabled=_auth_enabled,
    )
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=getattr(_auth_cfg, 'API_RATE_LIMIT', 300),
        enabled=True,
    )
except Exception as e:
    logger.warning(f"[API] Auth middleware setup skipped: {e}")

# CORS must be added LAST — Starlette runs last-added middleware first (outermost),
# and CORS must wrap everything to handle preflight requests before auth rejects them.
# When origins is wildcard, credentials must be False (browser security requirement).
# Specific origins can safely use credentials=True.
_allow_creds = _cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_origin_regex,
    allow_credentials=_allow_creds,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)

# Include all routers - frontend uses 2s stagger + 30s intervals to prevent thread pool exhaustion
app.include_router(chat.router)
app.include_router(status.public_router)  # Unauthenticated: /api/health only
app.include_router(status.router)
app.include_router(upload.router)
app.include_router(features.router)
app.include_router(multi_agent.router)
app.include_router(reasoning_tree.router)
app.include_router(proactive.router)
app.include_router(memory.router)
app.include_router(context.router)
app.include_router(conversation_starters.router)
app.include_router(thinking.router)
app.include_router(idle_behaviors.router)
# consciousness route removed
app.include_router(self_improvement.router)
app.include_router(thinking_mode.router)
app.include_router(tools_new.router)
app.include_router(activity.router)
app.include_router(multi_model.router)
app.include_router(knowledge.router)
app.include_router(search.router)
app.include_router(pdf.router)
app.include_router(transcribe.router)
app.include_router(ocr.router)
app.include_router(image_gen.router)
app.include_router(agent_action.router)
app.include_router(build_route.router)
app.include_router(models_route.router)
app.include_router(summarize.router)
app.include_router(youtube.router)
app.include_router(math_route.router)
app.include_router(research.router)
app.include_router(evolution.router)
app.include_router(hands_route.router)
app.include_router(artifacts.router)
app.include_router(feed.router)
app.include_router(providers_route.router)
app.include_router(code_route.router)
app.include_router(webhooks_route.router)
app.include_router(generate_route.router)
app.include_router(share_route.router)
if _reliability_available and reliability_route:
    app.include_router(reliability_route.router)
if _auth_available and auth_route:
    app.include_router(auth_route.router)

# /api/health is provided by status.py router

# Serve static files in production (built React app)
# NOTE: Only mount SPA routes when NOT in dev mode (Vite serves the frontend in dev)
# The catch-all /{full_path:path} route was intercepting WebSocket upgrades
static_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "dist")
_is_dev = os.environ.get("AURA_ENV") != "production"
if os.path.exists(static_path) and not _is_dev:
    app.mount("/assets", StaticFiles(directory=os.path.join(static_path, "assets")), name="assets")

    @app.get("/miniapp")
    async def serve_miniapp():
        """Serve the Telegram Mini App HTML."""
        miniapp_path = os.path.join(static_path, "miniapp.html")
        if os.path.exists(miniapp_path):
            return FileResponse(miniapp_path)
        # Fallback: serve from web/ source dir in case build didn't run
        src_miniapp = os.path.join(os.path.dirname(static_path), "miniapp.html")
        if os.path.exists(src_miniapp):
            return FileResponse(src_miniapp)
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Mini app not found")

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

        # Canonicalize to prevent path traversal
        real_static = os.path.realpath(static_path)
        candidate = os.path.realpath(os.path.join(static_path, full_path))
        # Ensure candidate is inside the static directory (separator prevents prefix confusion)
        if not (candidate.startswith(real_static + os.sep) or candidate == real_static):
            return FileResponse(os.path.join(static_path, "index.html"))

        if os.path.exists(candidate) and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(static_path, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.environ.get("AURA_HOST", "127.0.0.1"),
        port=8000,
        reload=False,
        log_level="info"
    )

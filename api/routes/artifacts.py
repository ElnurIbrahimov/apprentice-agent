"""Artifacts live-preview endpoints.

Streams file changes from the CLI agent loop to the extension's Artifacts panel
so users see live previews as the agent builds HTML/React/CSS/SVG/Markdown files.

Endpoints:
    POST   /api/artifacts/preview   — push a preview from any source
    GET    /api/artifacts/latest     — poll the latest preview slot
    WS     /api/artifacts/stream     — real-time push to connected extensions
"""

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel

from api.auth import require_api_key, verify_api_key_ws

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"], dependencies=[Depends(require_api_key)])

# ---------------------------------------------------------------------------
# Previewable file extensions
# ---------------------------------------------------------------------------
PREVIEW_EXTENSIONS = {".html", ".htm", ".tsx", ".jsx", ".css", ".svg", ".md", ".markdown"}


def is_previewable(filename: str) -> bool:
    """Return True if the filename has an extension we can live-preview."""
    if not filename:
        return False
    dot = filename.rfind(".")
    if dot == -1:
        return False
    return filename[dot:].lower() in PREVIEW_EXTENSIONS


def _ext_to_type(filename: str) -> str:
    """Map file extension to artifact type understood by the panel."""
    ext = filename[filename.rfind("."):].lower() if "." in filename else ""
    mapping = {
        ".html": "html", ".htm": "html",
        ".tsx": "react", ".jsx": "react",
        ".css": "css",
        ".svg": "svg",
        ".md": "markdown", ".markdown": "markdown",
    }
    return mapping.get(ext, "html")


# ---------------------------------------------------------------------------
# In-memory preview store (latest per filename) + WebSocket registry
# ---------------------------------------------------------------------------

class ArtifactPreview(BaseModel):
    code: str
    type: str
    filename: str
    timestamp: float


_previews: Dict[str, ArtifactPreview] = {}
_previews_lock = threading.Lock()

_ws_clients: List[WebSocket] = []
_ws_lock = threading.Lock()

# Reference to the running asyncio loop (set when first WS connects)
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _store_preview(filename: str, code: str, file_type: Optional[str] = None) -> ArtifactPreview:
    """Store a preview and return the ArtifactPreview object."""
    t = file_type or _ext_to_type(filename)
    preview = ArtifactPreview(code=code, type=t, filename=filename, timestamp=time.time())
    with _previews_lock:
        _previews[filename] = preview
        # Cap stored previews at 20 to prevent unbounded growth
        if len(_previews) > 20:
            oldest_key = min(_previews, key=lambda k: _previews[k].timestamp)
            del _previews[oldest_key]
    return preview


async def _broadcast_to_clients(payload: dict) -> None:
    """Send JSON to all connected artifact WebSocket clients."""
    with _ws_lock:
        targets = list(_ws_clients)
    dead = []
    for ws in targets:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    if dead:
        with _ws_lock:
            for ws in dead:
                try:
                    _ws_clients.remove(ws)
                except ValueError:
                    pass


def broadcast_event(payload: Dict[str, Any]) -> None:
    """Broadcast an arbitrary event payload to connected artifact clients."""
    global _event_loop
    if _event_loop and not _event_loop.is_closed():
        try:
            _event_loop.call_soon_threadsafe(
                _event_loop.create_task,
                _broadcast_to_clients(payload),
            )
        except RuntimeError:
            pass


def broadcast_artifact(filename: str, code: str, file_type: Optional[str] = None) -> None:
    """Called from the agent loop (sync context) to push a file change.

    This is the main hook — call it whenever a previewable file is written/edited.
    It stores the preview and broadcasts to all connected WebSocket clients.
    """
    preview = _store_preview(filename, code, file_type)
    payload = {
        "type": "artifact_update",
        "filename": preview.filename,
        "code": preview.code,
        "artifact_type": preview.type,
        "timestamp": preview.timestamp,
    }

    broadcast_event(payload)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

class PreviewRequest(BaseModel):
    code: str
    type: Optional[str] = None
    filename: str = "preview.html"


@router.post("/preview")
async def post_preview(req: PreviewRequest):
    """Accept a manual preview push (e.g., from the extension or external tool)."""
    preview = _store_preview(req.filename, req.code, req.type)
    payload = {
        "type": "artifact_update",
        "filename": preview.filename,
        "code": preview.code,
        "artifact_type": preview.type,
        "timestamp": preview.timestamp,
    }
    await _broadcast_to_clients(payload)
    return {"ok": True, "filename": preview.filename, "timestamp": preview.timestamp}


@router.get("/latest")
async def get_latest():
    """Return the latest preview for each tracked file."""
    with _previews_lock:
        items = {k: v.model_dump() for k, v in _previews.items()}
    return {"previews": items}


# ---------------------------------------------------------------------------
# WebSocket endpoint — extension connects here for real-time updates
# ---------------------------------------------------------------------------

@router.websocket("/stream")
async def artifacts_ws(websocket: WebSocket):
    """WebSocket for streaming artifact updates to the extension.

    Protocol:
        Server -> Client: {"type": "artifact_update", "filename": "...", "code": "...",
                           "artifact_type": "html", "timestamp": 1234567890.123}
        Server -> Client: {"type": "snapshot", "previews": {...}}  (on connect)
        Client -> Server: {"type": "ping"}  ->  Server: {"type": "pong"}
    """
    # Auth check
    api_key = websocket.headers.get("X-API-Key", "")
    if not verify_api_key_ws(api_key):
        await websocket.close(code=1008)
        return

    await websocket.accept()

    global _event_loop
    _event_loop = asyncio.get_running_loop()

    with _ws_lock:
        _ws_clients.append(websocket)
    logger.info("[Artifacts WS] Client connected")

    # Send current snapshot so the client can hydrate immediately
    try:
        with _previews_lock:
            snapshot = {k: v.model_dump() for k, v in _previews.items()}
        await websocket.send_json({"type": "snapshot", "previews": snapshot})
    except Exception:
        pass

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info("[Artifacts WS] Client disconnected")
    except Exception as e:
        logger.warning(f"[Artifacts WS] Error: {e}")
    finally:
        with _ws_lock:
            try:
                _ws_clients.remove(websocket)
            except ValueError:
                pass

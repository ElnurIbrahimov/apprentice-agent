"""Chat endpoints with WebSocket streaming support."""

import json
import logging
import asyncio
import os
import time
import threading
from pathlib import Path
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from starlette.responses import StreamingResponse
from api.auth import verify_api_key_ws, require_api_key
from api.utils import safe_error_detail, get_agent_service as _get_agent_service, get_agent, run_sync

from api.models.schemas import (
    ChatRequest, ChatResponse, RunRequest, RunResponse,
    ClearHistoryResponse, MoodState, AttachmentType,
    ConversationSummary, CreateConversationRequest, RenameConversationRequest,
    ConversationResponse, SaveToMemoryResponse,
)

logger = logging.getLogger(__name__)

def _get_conversation_manager():
    """Get ConversationManager with lazy loading."""
    from aura.core.conversation_manager import get_conversation_manager
    return get_conversation_manager()

# Upload directory for file cleanup
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
UPLOAD_DIR_RESOLVED = Path(UPLOAD_DIR).resolve()

router = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[Depends(require_api_key)])

# ---------------------------------------------------------------------------
# WebSocket connection registry for server-push (proactive messages, etc.)
# ---------------------------------------------------------------------------
_active_websockets: List[WebSocket] = []
_ws_lock = threading.Lock()


def register_websocket(ws: WebSocket) -> None:
    """Add a WebSocket to the active connection set."""
    with _ws_lock:
        _active_websockets.append(ws)


def unregister_websocket(ws: WebSocket) -> None:
    """Remove a WebSocket from the active connection set."""
    with _ws_lock:
        try:
            _active_websockets.remove(ws)
        except ValueError:
            pass


async def _broadcast_json(payload: dict) -> None:
    """Send a JSON message to all active WebSocket connections."""
    with _ws_lock:
        targets = list(_active_websockets)
    for ws in targets:
        try:
            await ws.send_json(payload)
        except Exception:
            logger.debug("broadcast_ws_send_failed", exc_info=True)
            unregister_websocket(ws)


# ---------------------------------------------------------------------------
# ConversationManager async listener (registered once)
# ---------------------------------------------------------------------------
_conv_listener_registered = False
_conv_listener_lock = threading.Lock()


def _ensure_conv_listener():
    """Register the ConversationManager -> WebSocket broadcast listener once."""
    global _conv_listener_registered
    if _conv_listener_registered:
        return
    with _conv_listener_lock:
        if _conv_listener_registered:
            return
        try:
            manager = _get_conversation_manager()

            async def _on_conv_event(event):
                payload = {
                    "type": "conv_sync",
                    "event": event.event_type,
                    "conversation_id": event.conversation_id,
                    "surface": event.surface,
                    "data": event.data,
                    "timestamp": event.timestamp,
                }
                await _broadcast_json(payload)

            manager.register_async_listener(_on_conv_event)
            _conv_listener_registered = True
            logger.info("[Chat] ConversationManager async listener registered")
        except Exception as e:
            logger.debug(f"[Chat] ConvManager listener registration deferred: {e}")


async def broadcast_proactive_message(msg) -> None:
    """Push a Gateway Daemon ProactiveMessage to all connected clients.

    Called from the notification callback wired in api/main.py.
    """
    payload = {
        "type": "proactive",
        "content": msg.content,
        "action": msg.action.value if hasattr(msg.action, "value") else str(msg.action),
        "priority": msg.priority.name if hasattr(msg.priority, "name") else str(msg.priority),
        "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, "isoformat") else str(msg.timestamp),
        "metadata": getattr(msg, "metadata", {}),
    }
    await _broadcast_json(payload)


async def broadcast_hand_event(result_dict: dict) -> None:
    """Push a Hand execution result to all connected WebSocket clients.

    Called from HandManager notification callback.
    """
    payload = {
        "type": "hand_event",
        **result_dict,
    }
    await _broadcast_json(payload)


async def broadcast_action_trace(hand_name: str, step: int, description: str) -> None:
    """Push a live step update during Hand execution."""
    import time as _time
    payload = {
        "type": "action_trace",
        "hand": hand_name,
        "step": step,
        "description": description,
        "timestamp": _time.time(),
    }
    await _broadcast_json(payload)


async def broadcast_hand_approval_request(request_dict: dict) -> None:
    """Push a Hand approval request to all connected WebSocket clients."""
    payload = {
        "type": "hand_approval_request",
        **request_dict,
    }
    await _broadcast_json(payload)


async def process_attachments(attachments: List[dict], loop) -> str:
    """Process attachments and return context to prepend to message.

    Args:
        attachments: List of attachment metadata dicts
        loop: Event loop for running sync code

    Returns:
        Context string to prepend to the user message
    """
    context_parts = []

    for attachment in attachments:
        try:
            file_path = attachment.get("path")
            filename = attachment.get("filename", "unknown")
            file_type = attachment.get("type")

            if not file_path:
                continue

            # Resolve relative paths (from upload API) against upload directory
            if not os.path.isabs(file_path):
                file_path = str(UPLOAD_DIR / file_path)

            try:
                file_path_resolved = Path(file_path).resolve(strict=True)
            except OSError:
                logger.warning(f"[Attachments] File not found or inaccessible: {file_path}")
                continue

            try:
                file_path_resolved.relative_to(UPLOAD_DIR_RESOLVED)
            except ValueError:
                logger.warning(f"[Attachments] Path traversal blocked: {file_path}")
                continue

            if file_type == AttachmentType.IMAGE.value or file_type == "image":
                # Use VisionTool to analyze image
                try:
                    from aura.tools.vision import VisionTool
                    vision = VisionTool()
                    result = await loop.run_in_executor(
                        None,
                        lambda fp=file_path, v=vision: v.analyze_image(fp, "Describe this image in detail. What do you see?")
                    )
                    if result.get("success"):
                        description = result.get("description", "")
                        # Format clearly so the chat model knows this is the authoritative analysis
                        context_parts.append(f"=== IMAGE ANALYSIS FOR: {filename} ===\nThe following is a computer vision analysis of the uploaded image:\n\n{description}\n\n=== END IMAGE ANALYSIS ===")
                        logger.info(f"[Attachments] Analyzed image: {filename} - description: {description[:200]}...")
                    else:
                        context_parts.append(f"[Image: {filename}] (Could not analyze: {result.get('error', 'unknown error')})")
                except Exception as e:
                    logger.error(f"[Attachments] Vision error for {filename}: {e}")
                    context_parts.append(f"[Image: {filename}] (Vision analysis unavailable)")

            elif file_type == "archive":
                # Extract and analyze zip project
                try:
                    from api.services.zip_analyzer import analyze_zip
                    zip_context = await loop.run_in_executor(None, lambda fp=file_path: analyze_zip(fp))
                    context_parts.append(zip_context)
                    logger.info(f"[Attachments] Analyzed zip: {filename} ({len(zip_context)} chars)")
                except Exception as e:
                    logger.error(f"[Attachments] Zip analysis error for {filename}: {e}")
                    context_parts.append(f"[Archive: {filename}] (Could not analyze: {safe_error_detail(e, 'analysis failed')})")

            else:
                # Read text/code files
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()

                    # Truncate very large files (50K chars ~ 12K tokens)
                    max_chars = 50000
                    if len(content) > max_chars:
                        content = content[:max_chars] + f"\n\n... (truncated - showing first {max_chars} of {len(content)} characters)"

                    ext = os.path.splitext(filename)[1].lower()
                    if file_type == "code":
                        lang = {'.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                                '.tsx': 'tsx', '.jsx': 'jsx', '.html': 'html', '.css': 'css',
                                '.sh': 'bash', '.yaml': 'yaml', '.yml': 'yaml',
                                '.json': 'json', '.sql': 'sql', '.go': 'go',
                                '.rs': 'rust', '.java': 'java', '.cpp': 'cpp', '.c': 'c'}.get(ext, '')
                        context_parts.append(f"[Code: {filename}]\n```{lang}\n{content}\n```")
                    else:
                        # Documents (.md, .txt, .pdf, etc.) — plain content block, not runnable code
                        context_parts.append(f"[Document: {filename}]\n--- BEGIN DOCUMENT CONTENT (read only, do not execute) ---\n{content}\n--- END DOCUMENT CONTENT ---")
                    logger.info(f"[Attachments] Read file: {filename} ({len(content)} chars)")

                except Exception as e:
                    logger.error(f"[Attachments] Error reading {filename}: {e}")
                    context_parts.append(f"[File: {filename}] (Could not read: {safe_error_detail(e, 'read failed')})")

        except Exception as e:
            logger.error(f"[Attachments] Error processing attachment: {e}")

    return "\n\n".join(context_parts)


def cleanup_attachment_files(attachments: List[dict]):
    """Delete attachment files after processing."""
    # Resolve the allowed upload directory
    _upload_root = os.path.realpath(UPLOAD_DIR)
    for attachment in attachments:
        try:
            file_path = attachment.get("path")
            if not file_path:
                continue
            # Resolve relative paths against upload directory
            if not os.path.isabs(file_path):
                file_path = str(UPLOAD_DIR / file_path)
            real_path = os.path.realpath(file_path)
            # Only delete files that are inside the upload directory
            if not (real_path.startswith(_upload_root + os.sep) or real_path == _upload_root):
                logger.warning(f"[Attachments] Refused to delete file outside upload dir: {file_path}")
                continue
            if os.path.exists(real_path):
                os.remove(real_path)
                logger.info(f"[Attachments] Cleaned up: {real_path}")
        except Exception as e:
            logger.warning(f"[Attachments] Failed to cleanup {file_path}: {e}")


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint.

    Args:
        request: Chat request with message, optional speak flag, and optional model override

    Returns:
        Chat response with agent reply and mood
    """
    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _get_agent_service().chat(request.message, speak=request.speak, model_override=request.model)
                ),
                timeout=180.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out after 180 seconds")

        # === Track assistant response and emotion in ContextHeatmap ===
        try:
            from api.routes.context import track_context_from_message, track_context_from_emotion
            track_context_from_message(result["response"], is_user=False)
            mood_raw = result.get("mood")
            if mood_raw and isinstance(mood_raw, dict) and mood_raw.get("emotion"):
                track_context_from_emotion(mood_raw["emotion"], mood_raw.get("confidence", 50) / 100.0)
        except Exception:
            logger.debug("chat_context_tracking_failed", exc_info=True)

        mood = result.get("mood")
        if mood and isinstance(mood, dict):
            mood = MoodState(**mood)

        return ChatResponse(
            response=result["response"],
            fast_path=result.get("fast_path", False),
            mood=mood,
            model_used=result.get("model_used")
        )

    except Exception as e:
        logger.error(f"[Chat] Error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/sse")
async def chat_sse(request: ChatRequest):
    """SSE streaming chat endpoint.

    Streams response chunks as Server-Sent Events.
    """
    message = request.message
    model = request.model

    # Check if agent is ready before starting stream
    svc = _get_agent_service()
    if not svc.is_ready:
        # Wait up to 30s for initialization
        for _ in range(60):
            if svc.is_ready:
                break
            await asyncio.sleep(0.5)
        if not svc.is_ready:
            async def init_error():
                yield f"data: {json.dumps({'type': 'error', 'error': 'Agent is still initializing. Please try again in a few seconds.'})}\n\n"
            return StreamingResponse(init_error(), media_type="text/event-stream")

    loop = asyncio.get_running_loop()
    chunk_queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    def stream_worker():
        """Run streaming in a separate thread."""
        try:
            for item in _get_agent_service().chat_stream(message, model_override=model, action_mode=None):
                if stop_event.is_set():
                    break
                loop.call_soon_threadsafe(chunk_queue.put_nowait, item)
        except Exception as e:
            loop.call_soon_threadsafe(chunk_queue.put_nowait, {"type": "error", "error": safe_error_detail(e)})
        finally:
            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

    async def event_generator():
        stream_thread = threading.Thread(target=stream_worker, daemon=True)
        stream_thread.start()

        # SSE timeout: 300s for complex operations (web creator uses this path)
        stream_deadline = asyncio.get_running_loop().time() + 300
        try:
            while True:
                remaining = max(0.1, stream_deadline - asyncio.get_running_loop().time())
                try:
                    item = await asyncio.wait_for(chunk_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    logger.warning("[SSE] Stream timed out after 300 seconds")
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Response timed out after 300 seconds.'})}\n\n"
                    return

                if item is None:
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return

                if isinstance(item, str):
                    item = {"type": "chunk", "content": item}

                if item.get("type") == "chunk":
                    yield f"data: {json.dumps({'type': 'chunk', 'content': item.get('content', '')})}\n\n"
                elif item.get("type") == "done":
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return
                elif item.get("type") == "error":
                    yield f"data: {json.dumps({'type': 'error', 'error': item.get('error', 'Unknown error')})}\n\n"
                    return
        except Exception as e:
            logger.error(f"[SSE] Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': safe_error_detail(e)})}\n\n"
        finally:
            stop_event.set()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    """Run agent with a goal (full agent loop).

    Args:
        request: Run request with goal and options

    Returns:
        Run response with completion status and history
    """
    try:
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _get_agent_service().run(
                        goal=request.goal,
                        context=request.context,
                        use_fastpath=request.use_fastpath,
                        max_iterations=request.max_iterations
                    )
                ),
                timeout=300.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out after 300 seconds")

        # === Track goal as user message in ContextHeatmap ===
        try:
            from api.routes.context import track_context_from_message
            track_context_from_message(request.goal, is_user=True)
        except Exception:
            logger.debug("run_context_tracking_failed", exc_info=True)

        mood = result.get("mood")
        if mood and isinstance(mood, dict):
            mood = MoodState(**mood)

        return RunResponse(
            goal=result.get("goal", request.goal),
            completed=result.get("completed", False),
            iterations=result.get("iterations", 0),
            final_evaluation=result.get("final_evaluation"),
            history=result.get("history", []),
            mood=mood
        )

    except Exception as e:
        logger.error(f"[Run] Error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/clear", response_model=ClearHistoryResponse)
async def clear_history() -> ClearHistoryResponse:
    """Clear conversation history."""
    try:
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(None, _get_agent_service().clear_history)

        return ClearHistoryResponse(
            success=success,
            message="History cleared" if success else "Failed to clear history"
        )

    except Exception as e:
        logger.error(f"[Clear] Error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


# =========================================================================
# Multi-Conversation Endpoints
# =========================================================================

@router.get("/conversations/search")
async def search_messages(q: str, limit: int = 20):
    """Full-text search across message content."""
    if not q or len(q) < 2:
        return {"results": [], "query": q}
    if len(q) > 500:
        return {"results": [], "query": q, "error": "query too long (max 500 chars)"}
    limit = min(limit, 100)  # cap to prevent unbounded iteration
    try:
        svc = _get_agent_service()
        if not svc.is_ready:
            return {"results": [], "query": q}

        def _search_sync():
            results = []
            q_lower = q.lower()
            max_convs_scanned = 200
            max_msgs_per_conv = 100
            convs_scanned = 0
            for conv in (svc.list_conversations() or []):
                convs_scanned += 1
                if convs_scanned > max_convs_scanned:
                    break
                msgs = svc._agent.brain.get_conversation_messages(conv['id']) if svc._agent else []
                for msg in (msgs or [])[-max_msgs_per_conv:]:
                    content = msg.get('content', '')
                    if q_lower in content.lower():
                        idx = content.lower().find(q_lower)
                        s = max(0, idx - 60)
                        e = min(len(content), idx + len(q) + 60)
                        snippet = ('...' if s > 0 else '') + content[s:e] + ('...' if e < len(content) else '')
                        results.append({
                            "conversation_id": conv['id'],
                            "conversation_title": conv.get('title', 'Untitled'),
                            "role": msg.get('role', 'unknown'),
                            "snippet": snippet,
                            "timestamp": msg.get('timestamp', 0),
                        })
                        if len(results) >= limit:
                            break
                if len(results) >= limit:
                    break
            results.sort(key=lambda r: r.get('timestamp', 0), reverse=True)
            return results[:limit]

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _search_sync)
        return {"results": results, "query": q}
    except Exception as e:
        logger.error(f"[SearchMessages] {e}")
        return {"results": [], "query": q, "error": safe_error_detail(e)}


@router.get("/conversations")
async def list_conversations():
    """List all conversations with surface activity info."""
    try:
        loop = asyncio.get_running_loop()
        conversations = await loop.run_in_executor(None, _get_agent_service().list_conversations)
        return conversations
    except Exception as e:
        logger.error(f"[Conversations] List error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: CreateConversationRequest = None):
    """Create a new conversation."""
    try:
        title = request.title if request else None
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_agent_service().create_conversation(title)
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return ConversationResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Conversations] Create error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.put("/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, request: RenameConversationRequest):
    """Rename a conversation."""
    from api.utils import validate_id
    validate_id(conversation_id, "conversation_id")
    try:
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            None, lambda: _get_agent_service().rename_conversation(conversation_id, request.title)
        )
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Conversations] Rename error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    from api.utils import validate_id
    validate_id(conversation_id, "conversation_id")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_agent_service().delete_conversation(conversation_id)
        )
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "Not found"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Conversations] Delete error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/conversations/{conversation_id}/switch", response_model=ConversationResponse)
async def switch_conversation(conversation_id: str):
    """Switch to a different conversation."""
    from api.utils import validate_id
    validate_id(conversation_id, "conversation_id")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_agent_service().switch_conversation(conversation_id)
        )
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return ConversationResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Conversations] Switch error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.post("/conversations/{conversation_id}/save-to-memory", response_model=SaveToMemoryResponse)
async def save_conversation_to_memory(conversation_id: str):
    """Save a conversation to AURA's long-term memory."""
    from api.utils import validate_id
    validate_id(conversation_id, "conversation_id")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_agent_service().save_conversation_to_memory(conversation_id)
        )
        return SaveToMemoryResponse(**result)
    except Exception as e:
        logger.error(f"[Conversations] Save to memory error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


# =========================================================================
# Cross-Surface Sync Endpoints (Phase 2)
# =========================================================================

@router.get("/sync/status")
async def get_sync_status():
    """Get ConversationManager sync status."""
    try:
        manager = _get_conversation_manager()
        return manager.get_status()
    except Exception as e:
        logger.debug(f"[SyncStatus] ConversationManager not ready: {e}")
        return {
            "initialized": False,
            "total_bindings": 0,
            "bindings": {},
            "listener_count": 0,
            "current_conversation": None,
        }


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """Get messages for a conversation with surface attribution."""
    from api.utils import validate_id
    validate_id(conversation_id, "conversation_id")
    try:
        manager = _get_conversation_manager()
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(
            None, lambda: manager.get_conversation_messages(conversation_id)
        )
        return {"messages": messages, "conversation_id": conversation_id}
    except RuntimeError:
        # ConversationManager not initialized — fall back to brain
        svc = _get_agent_service()
        if not svc.is_ready:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        loop = asyncio.get_running_loop()
        messages = await loop.run_in_executor(
            None, lambda: svc.agent.brain.get_conversation_messages(conversation_id)
        )
        return {"messages": messages, "conversation_id": conversation_id}
    except Exception as e:
        logger.error(f"[ConvMessages] Error: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@router.websocket("/stream")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat.

    Protocol:
        Client -> Server: {"type": "chat", "message": "Hello"}
        Client -> Server: {"type": "stop"} - Stop current generation
        Server -> Client: {"type": "chunk", "content": "Hi"}
        Server -> Client: {"type": "done", "response": "Hi there!", "mood": {...}}
        Server -> Client: {"type": "stopped"} - Generation was stopped
    """
    # Accept API key via header OR query param (browsers cannot send custom
    # headers on WebSocket connections, so the extension passes it as ?api_key=)
    api_key = websocket.headers.get("X-API-Key", "")
    if not api_key:
        api_key = websocket.query_params.get("api_key", "")
    if not verify_api_key_ws(api_key):
        await websocket.close(code=1008)
        return

    # Reject if too many concurrent WebSocket connections
    with _ws_lock:
        if len(_active_websockets) >= 50:
            await websocket.close(code=1013)
            return

    await websocket.accept()
    register_websocket(websocket)
    _ensure_conv_listener()
    logger.info("[WebSocket] Client connected")

    # Flag to signal stop to the streaming thread
    stop_generation = threading.Event()
    # Track latest attachments for cleanup in the outer finally block
    _last_attachments: List[dict] = []

    # Per-connection WebSocket rate limiting
    _ws_msg_count = 0
    _ws_window_start = time.monotonic()
    WS_RATE_LIMIT = 30  # max messages per minute per connection
    WS_RATE_WINDOW = 60  # seconds

    # Server-initiated keepalive: send pings every 30s to detect dead connections
    async def _ws_keepalive():
        try:
            while True:
                await asyncio.sleep(30)
                await websocket.send_json({"type": "ping"})
        except Exception:
            pass  # Connection closed — keepalive exits silently

    keepalive_task = asyncio.create_task(_ws_keepalive())

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            # Reject oversized messages to prevent memory exhaustion
            if len(data) > 1_000_000:  # 1MB limit
                await websocket.send_json({"type": "error", "error": "Message too large (max 1MB)"})
                continue

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON"
                })
                continue

            # Handle auth message (extension sends this right after connect)
            if msg.get("type") == "auth":
                # Auth is already handled during connection via header/query param.
                # Acknowledge it so the client knows it was received.
                await websocket.send_json({"type": "auth_ok"})
                continue

            # Handle ping/pong for keepalive
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # Handle readiness check from client
            if msg.get("type") == "ready_check":
                svc = _get_agent_service()
                await websocket.send_json({
                    "type": "ready_status",
                    "ready": svc.is_ready,
                    "initializing": getattr(svc, '_initializing', False),
                })
                continue

            # Handle stop request
            if msg.get("type") == "stop":
                logger.info("[WebSocket] Stop requested by client")
                stop_generation.set()
                await websocket.send_json({"type": "stopped"})
                continue

            # Handle hand command (stop/redirect)
            if isinstance(msg, dict) and msg.get("type") == "hand_command":
                hand_name = msg.get("hand", "")
                command = msg.get("command", "")
                new_goal = msg.get("new_goal")
                if command in ("stop", "redirect") and hand_name:
                    try:
                        from aura.hands.manager import get_hand_manager
                        get_hand_manager().send_command(hand_name, command, new_goal)
                        await websocket.send_json({"type": "hand_command_ack", "hand": hand_name, "command": command})
                    except Exception as e:
                        await websocket.send_json({"type": "error", "error": str(e)})
                continue

            # Clear stop flag for new messages
            stop_generation.clear()

            # Allow empty message if attachments are present
            has_attachments = msg.get("attachments") and len(msg.get("attachments", [])) > 0

            if msg.get("type") != "chat" or (not msg.get("message") and not has_attachments):
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid message format. Expected: {type: 'chat', message: '...'} or attachments"
                })
                continue

            # Per-connection rate limiting (only for chat messages)
            _ws_msg_count += 1
            now = time.monotonic()
            if now - _ws_window_start > WS_RATE_WINDOW:
                _ws_msg_count = 1
                _ws_window_start = now
            elif _ws_msg_count > WS_RATE_LIMIT:
                await websocket.send_json({"type": "error", "error": "Rate limit exceeded. Please slow down."})
                continue

            message = msg.get("message", "")
            model_override = msg.get("model")  # Optional model override
            action_mode = msg.get("action_mode")  # Optional action mode for auto-model selection
            conversation_id = msg.get("conversation_id")  # Optional conversation context
            attachments = msg.get("attachments", [])  # Optional attachments
            surface = msg.get("surface", "web")  # Cross-surface sync: which surface sent this
            _last_attachments = attachments  # Track for outer finally cleanup
            logger.debug(f"[WebSocket] Received message: '{message[:50]}...' model={model_override} action_mode={action_mode} conv={conversation_id} attachments={len(attachments)}")

            # Auto-switch conversation if needed (serialized to prevent race conditions)
            if conversation_id:
                from api.utils import validate_id
                try:
                    validate_id(conversation_id, "conversation_id")
                except Exception:
                    conversation_id = None  # Invalid ID — skip switch
            if conversation_id:
                try:
                    svc = _get_agent_service()
                    current_conv = svc.agent.brain.get_current_conversation_id() if svc.is_ready else None
                    if current_conv and current_conv != conversation_id:
                        await asyncio.get_running_loop().run_in_executor(
                            None, lambda: svc.switch_conversation(conversation_id)
                        )
                        logger.info(f"[WebSocket] Auto-switched to conversation: {conversation_id}")
                except Exception as e:
                    logger.warning(f"[WebSocket] Conversation switch failed: {e}")

            # === PHASE 1: Wire real cognitive tracking ===
            # Track user message in ContextHeatmap (real attention data)
            try:
                from api.routes.context import track_context_from_message
                if message and not message.startswith("[FILE_ATTACHMENT_CONTEXT]"):
                    track_context_from_message(message, is_user=True)
            except Exception:
                logger.debug("ws_context_tracking_failed", exc_info=True)

            # Record user activity for IdleBehaviorPanel
            try:
                from api.routes.idle_behaviors import record_user_activity
                record_user_activity()
            except Exception:
                logger.debug("ws_idle_activity_record_failed", exc_info=True)

            # Record interaction for Gateway Daemon (proactive system)
            try:
                from aura.proactive.gateway_daemon import get_gateway_daemon
                daemon = get_gateway_daemon()
                if daemon.state.value == "running":
                    daemon.record_interaction()
            except Exception:
                logger.debug("ws_daemon_interaction_record_failed", exc_info=True)
            if attachments:
                logger.debug(f"[WebSocket] Attachment details: {attachments}")
            logger.info(f"[WebSocket] Received: {message[:50]}..." + (f" (model: {model_override})" if model_override else "") + (f" ({len(attachments)} attachments)" if attachments else ""))

            try:
                loop = asyncio.get_running_loop()

                # Process attachments and prepend context to message
                if attachments:
                    logger.debug(f"[WebSocket] Processing {len(attachments)} attachments...")
                    attachment_context = await process_attachments(attachments, loop)
                    logger.debug(f"[WebSocket] Got attachment context: {len(attachment_context)} chars")
                    logger.debug(f"[WebSocket] Context preview: {attachment_context[:300]}...")
                    logger.info(f"[WebSocket] Attachment context ({len(attachment_context)} chars): {attachment_context[:500]}...")
                    if attachment_context:
                        # Add marker to indicate this is a file review (helps agent skip CognitiveTheater)
                        file_marker = "[FILE_ATTACHMENT_CONTEXT]\n"
                        if message.strip():
                            message = f"{file_marker}{attachment_context}\n\n---\n\nIMPORTANT: Use the analysis above as the authoritative source. Do NOT generate your own image description - use what's provided.\n\nUser request: {message}"
                        else:
                            # No text message, just attachments - summarize the provided analysis
                            message = f"{file_marker}{attachment_context}\n\n---\n\nIMPORTANT: Summarize and discuss the analysis provided above. Do NOT generate your own image description - use what's provided in the IMAGE ANALYSIS sections."
                        logger.info(f"[WebSocket] Final message to agent ({len(message)} chars)")

                # Check if agent is ready before starting stream
                svc = _get_agent_service()
                if not svc.is_ready:
                    # Agent still initializing - wait briefly then notify user
                    await websocket.send_json({
                        "type": "chunk",
                        "content": "AURA is warming up, please wait a moment...\n\n"
                    })
                    # Wait up to 30 seconds for initialization
                    for _ in range(60):
                        if svc.is_ready:
                            break
                        await asyncio.sleep(0.5)
                    if not svc.is_ready:
                        await websocket.send_json({
                            "type": "error",
                            "error": "Agent is still initializing. Please try again in a few seconds."
                        })
                        continue

                # Use streaming for real-time response
                # asyncio.Queue bridges the sync streaming thread and async WebSocket
                # Unlike queue.Queue + run_in_executor, this doesn't burn threads polling
                chunk_queue: asyncio.Queue = asyncio.Queue()
                full_response = ""

                def stream_worker():
                    """Run streaming in a separate thread."""
                    try:
                        for item in _get_agent_service().chat_stream(message, model_override=model_override, action_mode=action_mode):
                            if stop_generation.is_set():
                                logger.info("[WebSocket] Generation stopped by user")
                                break
                            # Thread-safe put into asyncio.Queue
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, item)
                    except Exception as e:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, {"type": "error", "error": safe_error_detail(e)})
                    finally:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

                # Start streaming in background thread
                stream_thread = threading.Thread(target=stream_worker, daemon=True)
                stream_thread.start()

                # Send chunks as they arrive (truly async, no busy-wait)
                # Timeout depends on operation complexity:
                #   - deep_research, swarm, agent: 600s (10 min)
                #   - research, code: 300s (5 min)
                #   - default chat/search: 120s
                _TIMEOUT_BY_MODE = {
                    "deep_research": 600,
                    "swarm": 600,
                    "agent": 600,
                    "research": 300,
                    "code": 300,
                }
                stream_timeout = _TIMEOUT_BY_MODE.get(action_mode, 120)
                stream_deadline = asyncio.get_running_loop().time() + stream_timeout
                while True:
                    if stop_generation.is_set():
                        logger.info("[WebSocket] Breaking loop due to stop request")
                        break

                    try:
                        remaining = max(0.1, stream_deadline - asyncio.get_running_loop().time())
                        item = await asyncio.wait_for(chunk_queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        logger.warning(f"[WebSocket] Stream timed out after {stream_timeout}s (mode={action_mode})")
                        stop_generation.set()
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Response timed out after {stream_timeout} seconds. Try again."
                        })
                        break

                    if item is None:
                        break

                    # Defensive: wrap raw strings as chunk dicts
                    if isinstance(item, str):
                        item = {"type": "chunk", "content": item}

                    if item.get("type") == "chunk":
                        content = item.get("content", "")
                        full_response += content
                        await websocket.send_json({
                            "type": "chunk",
                            "content": content
                        })
                    elif item.get("type") == "done":
                        mood = item.get("mood")
                        mood_dict = None
                        if mood:
                            if hasattr(mood, 'model_dump'):
                                mood_dict = mood.model_dump()
                            elif isinstance(mood, dict):
                                mood_dict = mood

                        # === PHASE 1: Track assistant response in ContextHeatmap ===
                        try:
                            from api.routes.context import track_context_from_message
                            if full_response:
                                track_context_from_message(full_response, is_user=False)
                        except Exception:
                            logger.debug("ws_response_context_tracking_failed", exc_info=True)

                        # Track emotion if mood data available
                        try:
                            from api.routes.context import track_context_from_emotion
                            if mood_dict and mood_dict.get("emotion"):
                                track_context_from_emotion(
                                    mood_dict["emotion"],
                                    mood_dict.get("confidence", 50) / 100.0
                                )
                        except Exception:
                            logger.debug("ws_emotion_tracking_failed", exc_info=True)

                        # Build audio_url for frontend
                        audio_url = None
                        try:
                            from aura.services.voice_presence import get_voice_presence
                            if get_voice_presence()._enabled:
                                audio_url = "/api/voice/synthesize"
                        except Exception:
                            logger.debug("ws_voice_presence_check_failed", exc_info=True)

                        # Track messages in ConversationManager for cross-surface sync
                        try:
                            conv_mgr = _get_conversation_manager()
                            cid = conv_mgr.get_current_conversation_id()
                            if cid and full_response:
                                _clean_ws_msg = message.split("[FILE_ATTACHMENT_CONTEXT]")[0].strip() if "[FILE_ATTACHMENT_CONTEXT]" in message else message
                                conv_mgr.on_message_added(cid, "user", _clean_ws_msg[:500], surface=surface, surface_user=f"{surface}_default")
                                conv_mgr.on_message_added(cid, "assistant", full_response, surface=surface, surface_user=f"{surface}_default")
                        except Exception:
                            logger.debug("ws_conv_manager_tracking_failed", exc_info=True)

                        await websocket.send_json({
                            "type": "done",
                            "response": full_response,
                            "mood": mood_dict,
                            "audio_url": audio_url,
                            "model_used": item.get("model_used"),
                        })
                    elif item.get("type") == "error":
                        await websocket.send_json({
                            "type": "error",
                            "error": item.get("error", "Unknown error")
                        })
                    elif item.get("type") == "tool_status":
                        await websocket.send_json({
                            "type": "tool_status",
                            "tool_name": item.get("tool_name", ""),
                            "tool_action": item.get("tool_action", "")
                        })
                    elif item.get("type") == "citations":
                        await websocket.send_json({
                            "type": "citations",
                            "citations": item.get("citations", [])
                        })
                    elif item.get("type") == "tool_trace":
                        await websocket.send_json(item)
                    elif item.get("type") == "research_progress":
                        await websocket.send_json({
                            "type": "research_progress",
                            "stage": item.get("stage", "search"),
                            "data": item.get("data", {}),
                        })

            except Exception as e:
                logger.error(f"[WebSocket] Processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": safe_error_detail(e)
                })
            finally:
                # Always cleanup attachment files, even on error
                if attachments:
                    cleanup_attachment_files(attachments)

    except WebSocketDisconnect:
        logger.info("[WebSocket] Client disconnected")
        stop_generation.set()
    except asyncio.CancelledError:
        logger.info("[WebSocket] Connection cancelled (server shutdown or task cancellation)")
        stop_generation.set()
        try:
            await websocket.send_json({"type": "error", "error": "Connection cancelled"})
            await websocket.close(code=1001)
        except Exception:
            pass  # Intentional: client already disconnected during cancellation
    except Exception as e:
        logger.error(f"[WebSocket] Connection error: {e}")
        stop_generation.set()
        try:
            await websocket.send_json({
                "type": "error",
                "error": safe_error_detail(e, "Connection error")
            })
            await websocket.close(code=1011)
        except Exception:
            pass  # Intentional: client already gone, nothing to send to
    finally:
        keepalive_task.cancel()
        unregister_websocket(websocket)
        # Safety net: cleanup leftover attachment files if the outer loop
        # broke mid-message before the per-message finally could run.
        if _last_attachments:
            try:
                cleanup_attachment_files(_last_attachments)
            except Exception:
                logger.debug("ws_attachment_cleanup_failed", exc_info=True)

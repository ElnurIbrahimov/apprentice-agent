"""Chat endpoints with WebSocket streaming support."""

import json
import logging
import asyncio
import os
import threading
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import JSONResponse
from api.auth import verify_api_key_ws, require_api_key

from api.models.schemas import (
    ChatRequest, ChatResponse, RunRequest, RunResponse,
    ClearHistoryResponse, WebSocketMessage, MoodState, AttachmentType,
    ConversationSummary, CreateConversationRequest, RenameConversationRequest,
    ConversationResponse, SaveToMemoryResponse,
)

logger = logging.getLogger(__name__)

# Lazy import to avoid blocking event loop at module load
def _get_agent_service():
    """Get agent_service with lazy loading."""
    from api.services.agent_service import agent_service
    return agent_service

# Upload directory for file cleanup
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
UPLOAD_DIR_RESOLVED = Path(UPLOAD_DIR).resolve()

router = APIRouter(prefix="/api/chat", tags=["chat"])

# ---------------------------------------------------------------------------
# WebSocket connection registry for server-push (proactive messages, etc.)
# ---------------------------------------------------------------------------
_active_websockets: List[WebSocket] = []
_ws_lock = threading.Lock()
_ws_async_lock = asyncio.Lock()


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
    async with _ws_async_lock:
        targets = list(_active_websockets)
    for ws in targets:
        try:
            await ws.send_json(payload)
        except Exception:
            unregister_websocket(ws)


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
                        lambda: vision.analyze_image(file_path, "Describe this image in detail. What do you see?")
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
                    zip_context = await loop.run_in_executor(None, lambda: analyze_zip(file_path))
                    context_parts.append(zip_context)
                    logger.info(f"[Attachments] Analyzed zip: {filename} ({len(zip_context)} chars)")
                except Exception as e:
                    logger.error(f"[Attachments] Zip analysis error for {filename}: {e}")
                    context_parts.append(f"[Archive: {filename}] (Could not analyze: {str(e)})")

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
                    context_parts.append(f"[File: {filename}] (Could not read: {str(e)})")

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


@router.post("", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
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
                timeout=120.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out after 120 seconds")

        # === Track assistant response and emotion in ContextHeatmap ===
        try:
            from api.routes.context import track_context_from_message, track_context_from_emotion
            track_context_from_message(result["response"], is_user=False)
            mood_raw = result.get("mood")
            if mood_raw and isinstance(mood_raw, dict) and mood_raw.get("emotion"):
                track_context_from_emotion(mood_raw["emotion"], mood_raw.get("confidence", 50) / 100.0)
        except Exception:
            pass

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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run", response_model=RunResponse, dependencies=[Depends(require_api_key)])
async def run(request: RunRequest) -> RunResponse:
    """Run agent with a goal (full agent loop).

    Args:
        request: Run request with goal and options

    Returns:
        Run response with completion status and history
    """
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _get_agent_service().run(
                goal=request.goal,
                context=request.context,
                use_fastpath=request.use_fastpath,
                max_iterations=request.max_iterations
            )
        )

        # === Track goal as user message in ContextHeatmap ===
        try:
            from api.routes.context import track_context_from_message
            track_context_from_message(request.goal, is_user=True)
        except Exception:
            pass

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
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================================
# Multi-Conversation Endpoints
# =========================================================================

@router.get("/conversations/search")
async def search_messages(q: str, limit: int = 20):
    """Full-text search across message content."""
    if not q or len(q) < 2:
        return {"results": [], "query": q}
    try:
        svc = _get_agent_service()
        if not svc.is_ready:
            return {"results": [], "query": q}
        results = []
        for conv in (svc.list_conversations() or []):
            msgs = svc._agent.brain.get_conversation_messages(conv['id']) if svc._agent else []
            for msg in (msgs or []):
                content = msg.get('content', '')
                if q.lower() in content.lower():
                    idx = content.lower().find(q.lower())
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
        results.sort(key=lambda r: r.get('timestamp', 0), reverse=True)
        return {"results": results[:limit], "query": q}
    except Exception as e:
        logger.error(f"[SearchMessages] {e}")
        return {"results": [], "query": q, "error": str(e)}


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations():
    """List all conversations."""
    try:
        loop = asyncio.get_running_loop()
        conversations = await loop.run_in_executor(None, _get_agent_service().list_conversations)
        return conversations
    except Exception as e:
        logger.error(f"[Conversations] List error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, request: RenameConversationRequest):
    """Rename a conversation."""
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
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/switch", response_model=ConversationResponse)
async def switch_conversation(conversation_id: str):
    """Switch to a different conversation."""
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
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/save-to-memory", response_model=SaveToMemoryResponse)
async def save_conversation_to_memory(conversation_id: str):
    """Save a conversation to AURA's long-term memory (A-MEM)."""
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: _get_agent_service().save_conversation_to_memory(conversation_id)
        )
        return SaveToMemoryResponse(**result)
    except Exception as e:
        logger.error(f"[Conversations] Save to memory error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    # Only accept API key via header, not query params
    api_key = websocket.headers.get("X-API-Key", "")
    if not verify_api_key_ws(api_key):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    register_websocket(websocket)
    logger.info("[WebSocket] Client connected")

    # Flag to signal stop to the streaming thread
    stop_generation = threading.Event()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON"
                })
                continue

            # Handle ping/pong for keepalive
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            # Handle stop request
            if msg.get("type") == "stop":
                logger.info("[WebSocket] Stop requested by client")
                stop_generation.set()
                await websocket.send_json({"type": "stopped"})
                continue

            # Clear stop flag for new messages
            stop_generation.clear()

            # Allow empty message if attachments are present
            has_message = msg.get("message") is not None
            has_attachments = msg.get("attachments") and len(msg.get("attachments", [])) > 0

            if msg.get("type") != "chat" or (not msg.get("message") and not has_attachments):
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid message format. Expected: {type: 'chat', message: '...'} or attachments"
                })
                continue

            message = msg.get("message", "")
            model_override = msg.get("model")  # Optional model override
            action_mode = msg.get("action_mode")  # Optional action mode for auto-model selection
            conversation_id = msg.get("conversation_id")  # Optional conversation context
            attachments = msg.get("attachments", [])  # Optional attachments
            logger.debug(f"[WebSocket] Received message: '{message[:50]}...' model={model_override} action_mode={action_mode} conv={conversation_id} attachments={len(attachments)}")

            # Auto-switch conversation if needed
            if conversation_id:
                try:
                    svc = _get_agent_service()
                    current_conv = svc.agent.brain.get_current_conversation_id() if svc.is_ready else None
                    if current_conv and current_conv != conversation_id:
                        svc.switch_conversation(conversation_id)
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
                pass  # Non-critical, don't break chat

            # Record user activity for IdleBehaviorPanel
            try:
                from api.routes.idle_behaviors import record_user_activity
                record_user_activity()
            except Exception:
                pass

            # Record interaction for Gateway Daemon (proactive system)
            try:
                from aura.proactive.gateway_daemon import get_gateway_daemon
                daemon = get_gateway_daemon()
                if daemon.state.value == "running":
                    daemon.record_interaction()
            except Exception:
                pass
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
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, {"type": "error", "error": str(e)})
                    finally:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

                # Start streaming in background thread
                stream_thread = threading.Thread(target=stream_worker, daemon=True)
                stream_thread.start()

                # Send chunks as they arrive (truly async, no busy-wait)
                # Max 90 seconds for any single response, then timeout
                stream_deadline = asyncio.get_running_loop().time() + 90
                while True:
                    if stop_generation.is_set():
                        logger.info("[WebSocket] Breaking loop due to stop request")
                        break

                    try:
                        remaining = max(0.1, stream_deadline - asyncio.get_running_loop().time())
                        item = await asyncio.wait_for(chunk_queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        logger.warning("[WebSocket] Stream timed out after 90 seconds")
                        stop_generation.set()
                        await websocket.send_json({
                            "type": "error",
                            "error": "Response timed out after 90 seconds. Try again."
                        })
                        break

                    if item is None:
                        break

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
                            pass

                        # Track emotion if mood data available
                        try:
                            from api.routes.context import track_context_from_emotion
                            if mood_dict and mood_dict.get("emotion"):
                                track_context_from_emotion(
                                    mood_dict["emotion"],
                                    mood_dict.get("confidence", 50) / 100.0
                                )
                        except Exception:
                            pass

                        # Build audio_url for frontend
                        audio_url = None
                        try:
                            from aura.services.voice_presence import get_voice_presence
                            if get_voice_presence()._enabled:
                                audio_url = "/api/voice/synthesize"
                        except Exception:
                            pass

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

            except Exception as e:
                logger.error(f"[WebSocket] Processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
            finally:
                # Always cleanup attachment files, even on error
                if attachments:
                    cleanup_attachment_files(attachments)

    except WebSocketDisconnect:
        logger.info("[WebSocket] Client disconnected")
        stop_generation.set()  # Kill any running stream_worker thread
    except Exception as e:
        logger.error(f"[WebSocket] Connection error: {e}")
        stop_generation.set()  # Kill any running stream_worker thread
    finally:
        unregister_websocket(websocket)

"""Chat endpoints with WebSocket streaming support."""

import json
import logging
import asyncio
import os
import queue
import threading
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from api.models.schemas import (
    ChatRequest, ChatResponse, RunRequest, RunResponse,
    ClearHistoryResponse, WebSocketMessage, MoodState, AttachmentType
)
from api.services.agent_service import agent_service

logger = logging.getLogger(__name__)

# Upload directory for file cleanup
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"

router = APIRouter(prefix="/api/chat", tags=["chat"])


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

            if not file_path or not os.path.exists(file_path):
                logger.warning(f"[Attachments] File not found: {file_path}")
                continue

            if file_type == AttachmentType.IMAGE.value or file_type == "image":
                # Use VisionTool to analyze image
                try:
                    from apprentice_agent.tools.vision import VisionTool
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

            else:
                # Read text/code files
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()

                    # Truncate very large files (50K chars ~ 12K tokens)
                    max_chars = 50000
                    if len(content) > max_chars:
                        content = content[:max_chars] + f"\n\n... (truncated - showing first {max_chars} of {len(content)} characters)"

                    file_type_label = "Code" if file_type == "code" else "Document"
                    context_parts.append(f"[{file_type_label}: {filename}]\n```\n{content}\n```")
                    logger.info(f"[Attachments] Read file: {filename} ({len(content)} chars)")

                except Exception as e:
                    logger.error(f"[Attachments] Error reading {filename}: {e}")
                    context_parts.append(f"[File: {filename}] (Could not read: {str(e)})")

        except Exception as e:
            logger.error(f"[Attachments] Error processing attachment: {e}")

    return "\n\n".join(context_parts)


def cleanup_attachment_files(attachments: List[dict]):
    """Delete attachment files after processing.

    Args:
        attachments: List of attachment metadata dicts
    """
    for attachment in attachments:
        try:
            file_path = attachment.get("path")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"[Attachments] Cleaned up: {file_path}")
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
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: agent_service.chat(request.message, speak=request.speak, model_override=request.model)
        )

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


@router.post("/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    """Run agent with a goal (full agent loop).

    Args:
        request: Run request with goal and options

    Returns:
        Run response with completion status and history
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: agent_service.run(
                goal=request.goal,
                context=request.context,
                use_fastpath=request.use_fastpath,
                max_iterations=request.max_iterations
            )
        )

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
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, agent_service.clear_history)

        return ClearHistoryResponse(
            success=success,
            message="History cleared" if success else "Failed to clear history"
        )

    except Exception as e:
        logger.error(f"[Clear] Error: {e}")
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
    await websocket.accept()
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
            attachments = msg.get("attachments", [])  # Optional attachments
            print(f"[WebSocket] Received message: '{message[:50]}...' model={model_override} attachments={len(attachments)}")
            if attachments:
                print(f"[WebSocket] Attachment details: {attachments}")
            logger.info(f"[WebSocket] Received: {message[:50]}..." + (f" (model: {model_override})" if model_override else "") + (f" ({len(attachments)} attachments)" if attachments else ""))

            try:
                loop = asyncio.get_event_loop()

                # Process attachments and prepend context to message
                if attachments:
                    print(f"[WebSocket] Processing {len(attachments)} attachments...")
                    attachment_context = await process_attachments(attachments, loop)
                    print(f"[WebSocket] Got attachment context: {len(attachment_context)} chars")
                    print(f"[WebSocket] Context preview: {attachment_context[:300]}...")
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

                # Use streaming for real-time response
                # Create a queue to communicate between sync streaming and async WebSocket
                chunk_queue = queue.Queue()
                full_response = ""

                def stream_worker():
                    """Run streaming in a separate thread."""
                    try:
                        for item in agent_service.chat_stream(message, model_override=model_override):
                            # Check if stop was requested
                            if stop_generation.is_set():
                                logger.info("[WebSocket] Generation stopped by user")
                                break
                            chunk_queue.put(item)
                    except Exception as e:
                        chunk_queue.put({"type": "error", "error": str(e)})
                    finally:
                        chunk_queue.put(None)  # Signal completion

                # Start streaming in background thread
                stream_thread = threading.Thread(target=stream_worker, daemon=True)
                stream_thread.start()

                # Send chunks as they arrive
                while True:
                    # Check if stop was requested
                    if stop_generation.is_set():
                        logger.info("[WebSocket] Breaking loop due to stop request")
                        break

                    try:
                        # Non-blocking check with small timeout (20ms for smoother streaming)
                        item = await loop.run_in_executor(
                            None,
                            lambda: chunk_queue.get(timeout=0.02)
                        )

                        if item is None:
                            # Stream complete
                            break

                        if item.get("type") == "chunk":
                            content = item.get("content", "")
                            full_response += content
                            await websocket.send_json({
                                "type": "chunk",
                                "content": content
                            })
                        elif item.get("type") == "done":
                            # Send final completion message
                            mood = item.get("mood")
                            mood_dict = None
                            if mood:
                                if hasattr(mood, 'model_dump'):
                                    mood_dict = mood.model_dump()
                                elif isinstance(mood, dict):
                                    mood_dict = mood

                            await websocket.send_json({
                                "type": "done",
                                "response": full_response,
                                "mood": mood_dict
                            })
                        elif item.get("type") == "error":
                            await websocket.send_json({
                                "type": "error",
                                "error": item.get("error", "Unknown error")
                            })

                    except queue.Empty:
                        # No chunk available yet, continue waiting
                        continue

                # Cleanup attachment files after processing
                if attachments:
                    cleanup_attachment_files(attachments)

            except Exception as e:
                logger.error(f"[WebSocket] Processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })

    except WebSocketDisconnect:
        logger.info("[WebSocket] Client disconnected")
    except Exception as e:
        logger.error(f"[WebSocket] Connection error: {e}")

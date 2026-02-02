"""Chat endpoints with WebSocket streaming support."""

import json
import logging
import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from api.models.schemas import (
    ChatRequest, ChatResponse, RunRequest, RunResponse,
    ClearHistoryResponse, WebSocketMessage, MoodState
)
from api.services.agent_service import agent_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint.

    Args:
        request: Chat request with message and optional speak flag

    Returns:
        Chat response with agent reply and mood
    """
    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: agent_service.chat(request.message, speak=request.speak)
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
        Server -> Client: {"type": "chunk", "content": "Hi"}
        Server -> Client: {"type": "done", "response": "Hi there!", "mood": {...}}
    """
    await websocket.accept()
    logger.info("[WebSocket] Client connected")

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

            if msg.get("type") != "chat" or not msg.get("message"):
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid message format. Expected: {type: 'chat', message: '...'}"
                })
                continue

            message = msg["message"]
            logger.info(f"[WebSocket] Received: {message[:50]}...")

            try:
                loop = asyncio.get_event_loop()

                # Use agent.chat() which has direct handlers for search/crypto
                # Run in executor to avoid blocking the event loop
                result = await loop.run_in_executor(
                    None,
                    lambda: agent_service.chat(message, speak=False)
                )

                # Send response as single chunk
                await websocket.send_json({
                    "type": "chunk",
                    "content": result["response"]
                })

                # Send completion message
                mood = result.get("mood")
                mood_dict = None
                if mood:
                    if hasattr(mood, 'model_dump'):
                        mood_dict = mood.model_dump()
                    elif isinstance(mood, dict):
                        mood_dict = mood

                await websocket.send_json({
                    "type": "done",
                    "response": result.get("response", ""),
                    "mood": mood_dict
                })

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

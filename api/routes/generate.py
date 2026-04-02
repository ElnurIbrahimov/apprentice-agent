"""Raw LLM generation endpoint — bypasses the agent pipeline.

Used by WebCreator and Artifacts panels that need direct LLM output
with their own system prompts, without Aura's personality, tool calling,
emotion processing, or action-mode routing interfering.
"""

import json
import logging
import asyncio
import threading
from typing import Optional, List, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from api.auth import require_api_key
from api.utils import safe_error_detail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/generate", tags=["generate"], dependencies=[Depends(require_api_key)])


class HistoryItem(BaseModel):
    """A single message in conversation history."""
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=50_000)


class GenerateRequest(BaseModel):
    """Request body for raw generation."""
    message: str = Field(..., min_length=1, max_length=200_000, description="User message / prompt")
    system_prompt: Optional[str] = Field(default=None, max_length=10_000, description="System prompt for the LLM")
    model: Optional[str] = Field(default=None, description="Model override (None = default)")
    history: Optional[List[HistoryItem]] = Field(default=None, max_length=20, description="Conversation history")
    images: Optional[List[str]] = Field(default=None, max_length=5, description="Base64 images for vision models")


def _get_brain():
    """Get the OllamaBrain instance from agent service."""
    from api.services.agent_service import agent_service
    return agent_service.agent.brain


@router.post("/raw")
async def generate_raw(request: GenerateRequest) -> StreamingResponse:
    """Stream raw LLM output as SSE — no agent pipeline, no personality.

    This endpoint makes a direct Ollama call with the provided system prompt
    and message. Used by WebCreator and Artifacts for clean HTML/code generation.
    """
    message = request.message
    system_prompt = request.system_prompt
    model = request.model
    history = request.history or []
    images = request.images or []

    loop = asyncio.get_running_loop()
    chunk_queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()

    def stream_worker():
        """Run streaming in a separate thread using brain's client directly."""
        try:
            brain = _get_brain()

            # Determine which model and client to use
            target_model = model or brain.model
            client, actual_model = brain._get_client_for_model(target_model)

            # Build messages array
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add conversation history if provided (already validated by Pydantic)
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})

            # Add the current user message (with images if provided for vision models)
            user_msg: dict = {"role": "user", "content": message}
            if images:
                # Strip data URI prefix if present (Ollama expects raw base64)
                clean_images = []
                for img in images:
                    if "," in img:
                        clean_images.append(img.split(",", 1)[1])
                    else:
                        clean_images.append(img)
                user_msg["images"] = clean_images
            messages.append(user_msg)

            logger.info(f"[Generate/Raw] Streaming with model={actual_model}, "
                        f"msgs={len(messages)}, system={'yes' if system_prompt else 'no'}, "
                        f"images={len(images)}")

            # Direct streaming call — no agent, no tools, no personality
            response = client.chat(
                model=actual_model,
                messages=messages,
                stream=True,
            )

            for chunk in response:
                if stop_event.is_set():
                    break
                content = chunk.get("message", {}).get("content", "")
                if content:
                    loop.call_soon_threadsafe(
                        chunk_queue.put_nowait,
                        {"type": "chunk", "content": content}
                    )

        except Exception as e:
            logger.error(f"[Generate/Raw] Error: {e}", exc_info=True)
            loop.call_soon_threadsafe(
                chunk_queue.put_nowait,
                {"type": "error", "error": safe_error_detail(e)[:500]}
            )
        finally:
            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

    async def event_generator():
        stream_thread = threading.Thread(target=stream_worker, daemon=True)
        stream_thread.start()

        # 5 minute timeout for long generations
        stream_deadline = asyncio.get_running_loop().time() + 300
        try:
            while True:
                remaining = max(0.1, stream_deadline - asyncio.get_running_loop().time())
                try:
                    item = await asyncio.wait_for(chunk_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    logger.warning("[Generate/Raw] Stream timed out after 300 seconds")
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Generation timed out after 300 seconds.'})}\n\n"
                    return

                if item is None:
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return

                if item.get("type") == "chunk":
                    yield f"data: {json.dumps({'type': 'chunk', 'content': item.get('content', '')})}\n\n"
                elif item.get("type") == "error":
                    yield f"data: {json.dumps({'type': 'error', 'error': item.get('error', 'Unknown error')})}\n\n"
                    return
        except Exception as e:
            logger.error(f"[Generate/Raw] SSE error: {e}")
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

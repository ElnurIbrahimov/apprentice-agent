"""Pydantic models for API request/response schemas."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    """Message role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """A single chat message."""
    role: MessageRole
    content: str
    timestamp: Optional[float] = None


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str = Field(..., min_length=1, description="User message")
    speak: bool = Field(default=False, description="Enable TTS for response")
    model: Optional[str] = Field(default=None, description="Model override (None = auto)")


class RunRequest(BaseModel):
    """Request body for agent run endpoint."""
    goal: str = Field(..., min_length=1, description="Goal for the agent")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    use_fastpath: Optional[bool] = Field(default=None, description="Force fast-path mode")
    max_iterations: int = Field(default=10, ge=1, le=50, description="Max iterations")


class MoodState(BaseModel):
    """Agent's emotional/mood state."""
    emotion: Optional[str] = None
    confidence: int = Field(default=0, ge=0, le=100)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    arousal: float = Field(default=0.0, ge=-1.0, le=1.0)


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    response: str
    fast_path: bool = False
    mood: Optional[MoodState] = None
    model_used: Optional[str] = None


class RunResponse(BaseModel):
    """Response from agent run endpoint."""
    goal: str
    completed: bool
    iterations: int
    final_evaluation: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = []
    mood: Optional[MoodState] = None


class StatusResponse(BaseModel):
    """Agent status response."""
    online: bool
    model: str
    aura_enabled: bool
    mood: Optional[MoodState] = None
    memory_count: int = 0
    query_count: int = 0
    last_model_used: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "1.0.0"


class WebSocketMessage(BaseModel):
    """WebSocket message format."""
    type: str  # "chat", "chunk", "done", "error"
    content: Optional[str] = None
    message: Optional[str] = None
    response: Optional[str] = None
    mood: Optional[MoodState] = None
    error: Optional[str] = None


class ClearHistoryResponse(BaseModel):
    """Response from clear history endpoint."""
    success: bool
    message: str = "History cleared"

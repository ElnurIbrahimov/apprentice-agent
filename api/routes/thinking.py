"""Thinking About Teaser - Shows what AURA is contemplating."""

import asyncio
import functools
import logging
import random
import time
from collections import deque
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/thinking", tags=["thinking"], dependencies=[Depends(require_api_key)])

# ============================================================================
# Thought Types and Templates
# ============================================================================

class ThoughtType(str, Enum):
    CONNECTING = "connecting"      # Making connections between concepts
    QUESTIONING = "questioning"    # Forming a question
    RECALLING = "recalling"       # Accessing memories
    ANALYZING = "analyzing"       # Breaking down information
    WONDERING = "wondering"       # Curiosity/exploration
    FORMULATING = "formulating"   # Forming a response
    OBSERVING = "observing"       # Noticing patterns


THOUGHT_TEMPLATES = {
    ThoughtType.CONNECTING: [
        "connecting {topic1} with {topic2}...",
        "seeing a pattern between {topic1} and {topic2}",
        "linking {topic1} to something earlier...",
        "this relates to {topic1}...",
    ],
    ThoughtType.QUESTIONING: [
        "wondering about {topic}...",
        "should I ask about {topic}?",
        "curious: what does {topic} mean to you?",
        "forming a question about {topic}...",
    ],
    ThoughtType.RECALLING: [
        "recalling what you said about {topic}...",
        "this reminds me of {topic}...",
        "searching memories for {topic}...",
        "I remember something about {topic}...",
    ],
    ThoughtType.ANALYZING: [
        "analyzing the implications of {topic}...",
        "breaking down {topic}...",
        "considering different angles on {topic}...",
        "examining {topic} more closely...",
    ],
    ThoughtType.WONDERING: [
        "wondering if {topic} is relevant here...",
        "could {topic} be important?",
        "interesting thought about {topic}...",
        "what if {topic}...",
    ],
    ThoughtType.FORMULATING: [
        "formulating a response about {topic}...",
        "preparing thoughts on {topic}...",
        "organizing ideas about {topic}...",
        "crafting a response...",
    ],
    ThoughtType.OBSERVING: [
        "noticing a pattern in {topic}...",
        "observing something about {topic}...",
        "sensing {topic} is important...",
        "picking up on {topic}...",
    ],
}

THOUGHT_ICONS = {
    ThoughtType.CONNECTING: "🔗",
    ThoughtType.QUESTIONING: "❓",
    ThoughtType.RECALLING: "💭",
    ThoughtType.ANALYZING: "🔍",
    ThoughtType.WONDERING: "🤔",
    ThoughtType.FORMULATING: "✍️",
    ThoughtType.OBSERVING: "👁️",
}


# ============================================================================
# Thought State Manager
# ============================================================================

class ActiveThought:
    """Represents an active thought AURA is having."""

    def __init__(
        self,
        thought_type: ThoughtType,
        content: str,
        topics: List[str],
        intensity: float = 0.5,
        source: str = "template",
        is_real: bool = False,
    ):
        self.id = f"thought_{time.time()}"
        self.type = thought_type
        self.content = content
        self.topics = topics
        self.intensity = intensity
        self.source = source  # "brain", "engine", "memory", "dream", "emotion", "tool", "template"
        self.is_real = is_real  # True = from actual cognitive processing
        self.created_at = time.time()
        self.resolved = False
        self.resolution: Optional[str] = None  # "spoke", "dismissed", "merged"

    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "icon": THOUGHT_ICONS.get(self.type, "💭"),
            "content": self.content,
            "topics": self.topics,
            "intensity": round(self.intensity, 2),
            "age_seconds": round(self.age_seconds(), 1),
            "resolved": self.resolved,
            "resolution": self.resolution,
            "source": self.source,
            "is_real": self.is_real,
        }


class ThinkingStateManager:
    """Manages AURA's visible thinking process.

    PHASE 1: Now supports real cognitive thoughts from actual processing
    (brain.py reasoning, memory retrieval, emotion analysis) alongside
    template-based fallback for when nothing real is happening.
    """

    def __init__(self, max_active_thoughts: int = 3):
        self._lock = RLock()
        self._active_thoughts: List[ActiveThought] = []
        self._max_thoughts = max_active_thoughts
        self._thought_history: deque = deque(maxlen=200)
        self._last_thought_time = 0.0
        self._last_real_thought_time = 0.0  # Track when last REAL thought arrived
        self._stats = {
            "total_thoughts": 0,
            "real_thoughts": 0,
            "template_thoughts": 0,
            "thoughts_spoken": 0,
            "thoughts_dismissed": 0,
        }

    def _get_topics_from_context(self) -> List[str]:
        """Get current focus topics from context tracker."""
        try:
            from api.routes.context import get_tracker
            tracker = get_tracker()
            state = tracker.get_focus_state(limit=5)
            return [item["name"] for item in state.get("items", [])]
        except Exception:
            return []

    def record_real_thought(
        self,
        thought_type_str: str,
        content: str,
        intensity: float = 0.6,
        topics: Optional[List[str]] = None,
        source: str = "brain",
    ) -> Optional[ActiveThought]:
        """Record a REAL thought from actual cognitive processing.

        Called from engine.py, brain.py, agent_service.py, memory systems,
        NeuroDream, ALMA, and tool execution when actual reasoning,
        memory retrieval, emotional processing, or tool selection occurs.
        These are genuine cognitive events, not templates.

        Args:
            thought_type_str: One of ThoughtType values
            content: Human-readable description of the thought
            intensity: 0.0-1.0 importance level
            topics: Related topics (auto-detected from context if None)
            source: Origin system - "brain", "engine", "memory", "dream",
                    "emotion", "tool", "agent", "service"
        """
        with self._lock:
            try:
                thought_type = ThoughtType(thought_type_str)
            except ValueError:
                thought_type = ThoughtType.OBSERVING

            if topics is None:
                topics = self._get_topics_from_context()

            thought = ActiveThought(
                thought_type=thought_type,
                content=content,
                topics=topics[:3] if topics else [],
                intensity=min(1.0, intensity),
                source=source,
                is_real=True,
            )

            self._active_thoughts.append(thought)
            self._last_thought_time = time.time()
            self._last_real_thought_time = time.time()
            self._stats["total_thoughts"] += 1
            self._stats["real_thoughts"] += 1

            # Limit active thoughts
            while len(self._active_thoughts) > self._max_thoughts:
                old = self._active_thoughts.pop(0)
                old.resolved = True
                old.resolution = "faded"
                self._thought_history.append(old)

            logger.debug(f"[THINKING] Real thought from {source}: {content[:60]}")
            return thought

    def generate_thought(self, force: bool = False) -> Optional[ActiveThought]:
        """Generate a template-based thought (FALLBACK only).

        Only used when no real cognitive activity has occurred recently.
        Real thoughts from record_real_thought() are always preferred."""
        with self._lock:
            now = time.time()

            # Rate limiting
            if not force and now - self._last_thought_time < 8:
                return None

            # Get context topics
            topics = self._get_topics_from_context()
            if not topics:
                topics = ["the conversation", "what you mentioned", "your question"]

            # Choose thought type based on context
            thought_type = random.choice(list(ThoughtType))

            # Generate content from template
            template = random.choice(THOUGHT_TEMPLATES[thought_type])

            topic = random.choice(topics) if topics else "this"
            if "{topic1}" in template:
                topic2 = topics[1] if len(topics) >= 2 else "earlier context"
                content = template.format(topic1=topics[0] if topics else "this", topic2=topic2)
            else:
                content = template.format(topic=topic)

            # Create thought (marked as template fallback)
            thought = ActiveThought(
                thought_type=thought_type,
                content=content,
                topics=topics[:3],
                intensity=0.3 + random.random() * 0.5,
                source="template",
                is_real=False,
            )

            # Add to active thoughts
            self._active_thoughts.append(thought)
            self._last_thought_time = now
            self._stats["total_thoughts"] += 1
            self._stats["template_thoughts"] += 1

            # Limit active thoughts
            while len(self._active_thoughts) > self._max_thoughts:
                old = self._active_thoughts.pop(0)
                old.resolved = True
                old.resolution = "faded"
                self._thought_history.append(old)

            return thought

    def resolve_thought(self, thought_id: str, resolution: str = "dismissed"):
        """Resolve a thought (spoke, dismissed, merged)."""
        with self._lock:
            for thought in self._active_thoughts:
                if thought.id == thought_id:
                    thought.resolved = True
                    thought.resolution = resolution
                    self._active_thoughts.remove(thought)
                    self._thought_history.append(thought)

                    if resolution == "spoke":
                        self._stats["thoughts_spoken"] += 1
                    else:
                        self._stats["thoughts_dismissed"] += 1
                    break

    def _is_inner_engine_active(self) -> bool:
        """Check if inner thoughts engine recently produced output (<30s ago)."""
        try:
            from api.services.inner_thoughts_engine import get_inner_thoughts_engine
            engine = get_inner_thoughts_engine()
            stats = engine.get_stats()
            last_time = stats.get("last_thought_time", 0)
            return (time.time() - last_time) < 30
        except Exception:
            return False

    def decay_thoughts(self):
        """Decay old thoughts with differential rates for real vs template."""
        with self._lock:
            now = time.time()
            to_remove = []

            for thought in self._active_thoughts:
                age = thought.age_seconds()

                if thought.is_real:
                    # Real thoughts: slower decay, longer lifespan
                    thought.intensity *= 0.985
                    max_age = 90
                else:
                    # Template thoughts: faster decay, shorter lifespan
                    thought.intensity *= 0.95
                    max_age = 30

                # Remove old or faded thoughts
                if age > max_age or thought.intensity < 0.1:
                    thought.resolved = True
                    thought.resolution = "faded"
                    to_remove.append(thought)

            for thought in to_remove:
                self._active_thoughts.remove(thought)
                self._thought_history.append(thought)

    def get_state(self) -> Dict[str, Any]:
        """Get current thinking state for UI."""
        with self._lock:
            self.decay_thoughts()

            # Only fall back to template generation if no real thoughts recently
            # Real thoughts from record_real_thought() are always preferred
            real_thought_age = time.time() - self._last_real_thought_time
            inner_engine_active = self._is_inner_engine_active()

            # Template suppression: require 60s silence, 10% probability,
            # and suppress entirely if inner engine recently produced output
            if (len(self._active_thoughts) < 2
                    and real_thought_age > 60
                    and not inner_engine_active
                    and random.random() < 0.10):
                # No real cognitive activity for 60+ seconds — use template as subtle fallback
                self.generate_thought()

            active = [t.to_dict() for t in self._active_thoughts]
            recent_history = [t.to_dict() for t in list(self._thought_history)[-5:]]

            return {
                "is_thinking": len(active) > 0,
                "active_thoughts": active,
                "thought_count": len(active),
                "recent_history": recent_history,
                "primary_thought": active[0] if active else None,
            }

    def get_teaser(self) -> Optional[Dict[str, Any]]:
        """Get a teaser preview of current thinking."""
        with self._lock:
            if not self._active_thoughts:
                return None

            # Return the most intense thought as teaser
            sorted_thoughts = sorted(
                self._active_thoughts,
                key=lambda t: t.intensity,
                reverse=True
            )

            primary = sorted_thoughts[0]
            return {
                "content": primary.content,
                "type": primary.type.value,
                "icon": THOUGHT_ICONS.get(primary.type, "💭"),
                "intensity": round(primary.intensity, 2),
                "topics": primary.topics,
            }

    def add_thought_from_context(
        self,
        thought_type: ThoughtType,
        topic: str,
        intensity: float = 0.6
    ):
        """Add a thought triggered by agent context."""
        with self._lock:
            template = random.choice(THOUGHT_TEMPLATES[thought_type])
            content = template.format(topic=topic, topic1=topic, topic2="context")

            thought = ActiveThought(
                thought_type=thought_type,
                content=content,
                topics=[topic],
                intensity=intensity,
            )

            self._active_thoughts.append(thought)
            self._stats["total_thoughts"] += 1

            # Limit active thoughts
            while len(self._active_thoughts) > self._max_thoughts:
                old = self._active_thoughts.pop(0)
                old.resolved = True
                old.resolution = "faded"
                self._thought_history.append(old)

    def get_stats(self) -> Dict[str, Any]:
        """Get thinking statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_thoughts": len(self._active_thoughts),
                "history_size": len(self._thought_history),
            }

    def clear(self):
        """Clear all thoughts."""
        with self._lock:
            self._active_thoughts.clear()
            self._thought_history.clear()


# Global manager
_manager = ThinkingStateManager()


def get_manager() -> ThinkingStateManager:
    return _manager


# ============================================================================
# API Models
# ============================================================================

class ThoughtResponse(BaseModel):
    id: str
    type: str
    icon: str
    content: str
    topics: List[str]
    intensity: float
    age_seconds: float
    resolved: bool
    resolution: Optional[str]


class ThinkingStateResponse(BaseModel):
    is_thinking: bool
    active_thoughts: List[ThoughtResponse]
    thought_count: int
    primary_thought: Optional[ThoughtResponse]


class TeaserResponse(BaseModel):
    content: str
    type: str
    icon: str
    intensity: float
    topics: List[str]


class AddThoughtRequest(BaseModel):
    thought_type: str
    topic: str
    intensity: Optional[float] = 0.6


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/state")
async def get_thinking_state(since: Optional[float] = None):
    """Get current thinking state with all active thoughts.

    Args:
        since: Optional timestamp. If provided, returns only thoughts newer than this.
               Enables efficient polling — frontend can skip full fetch when nothing changed.
    """
    manager = get_manager()
    loop = asyncio.get_running_loop()
    state = await loop.run_in_executor(None, manager.get_state)

    # If 'since' is provided, check if anything changed
    if since is not None:
        has_new = any(
            t.get("age_seconds", 999) < (time.time() - since)
            for t in state.get("active_thoughts", [])
        )
        state["has_new_since"] = has_new

    return state


@router.get("/teaser")
async def get_teaser():
    """Get a teaser preview of what AURA is thinking about."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    teaser = await loop.run_in_executor(None, manager.get_teaser)

    if teaser:
        return {"has_teaser": True, "teaser": teaser}
    return {"has_teaser": False, "teaser": None}


@router.post("/generate")
async def generate_thought(force: bool = False):
    """Generate a new thought."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    thought = await loop.run_in_executor(None, functools.partial(manager.generate_thought, force=force))

    if thought:
        return {"generated": True, "thought": thought.to_dict()}
    return {"generated": False, "reason": "Rate limited"}


@router.post("/add")
async def add_thought(request: AddThoughtRequest):
    """Add a specific thought from context."""
    manager = get_manager()

    try:
        thought_type = ThoughtType(request.thought_type)
    except ValueError:
        thought_type = ThoughtType.WONDERING

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        functools.partial(
            manager.add_thought_from_context,
            thought_type=thought_type,
            topic=request.topic,
            intensity=request.intensity or 0.6,
        ),
    )

    return {"status": "added", "topic": request.topic}


@router.post("/resolve/{thought_id}")
async def resolve_thought(thought_id: str, resolution: str = "dismissed"):
    """Resolve a thought."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, functools.partial(manager.resolve_thought, thought_id, resolution))
    return {"status": "resolved", "thought_id": thought_id}


@router.get("/stats")
async def get_stats():
    """Get thinking statistics."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, manager.get_stats)


@router.post("/clear")
async def clear_thoughts():
    """Clear all thoughts."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, manager.clear)
    return {"status": "cleared"}


# ============================================================================
# Integration Helpers
# ============================================================================

def add_thinking_context(thought_type: str, topic: str, intensity: float = 0.6):
    """Helper to add thoughts from agent code."""
    manager = get_manager()
    try:
        t_type = ThoughtType(thought_type)
    except ValueError:
        t_type = ThoughtType.WONDERING
    manager.add_thought_from_context(t_type, topic, intensity)


def record_thought(
    thought_type: str,
    content: str,
    intensity: float = 0.6,
    source: str = "brain",
    topics: Optional[List[str]] = None,
):
    """Convenience helper to record a real thought from anywhere in the codebase.

    This is the preferred way to hook into the thinking system.
    Safe to call even if the thinking system isn't initialized.

    Args:
        thought_type: "connecting", "questioning", "recalling", "analyzing",
                      "wondering", "formulating", "observing"
        content: Human-readable description (e.g., "retrieving 3 memories about python")
        intensity: 0.0-1.0, how important/visible this thought is
        source: Which system generated it - "brain", "engine", "memory",
                "dream", "emotion", "tool", "agent", "service"
        topics: Optional topic tags
    """
    try:
        manager = get_manager()
        manager.record_real_thought(thought_type, content, intensity, topics, source)
    except Exception:
        pass  # Never let thinking system errors break actual processing

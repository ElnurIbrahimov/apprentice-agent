"""Thinking About Teaser - Shows what AURA is contemplating."""

import logging
import random
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from threading import Lock
from enum import Enum

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/thinking", tags=["thinking"])

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
    ):
        self.id = f"thought_{time.time()}"
        self.type = thought_type
        self.content = content
        self.topics = topics
        self.intensity = intensity
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
        }


class ThinkingStateManager:
    """Manages AURA's visible thinking process."""

    def __init__(self, max_active_thoughts: int = 3):
        self._lock = Lock()
        self._active_thoughts: List[ActiveThought] = []
        self._max_thoughts = max_active_thoughts
        self._thought_history: List[ActiveThought] = []
        self._last_thought_time = 0.0
        self._stats = {
            "total_thoughts": 0,
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

    def generate_thought(self, force: bool = False) -> Optional[ActiveThought]:
        """Generate a new thought based on context."""
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

            if "{topic1}" in template and "{topic2}" in template:
                if len(topics) >= 2:
                    content = template.format(topic1=topics[0], topic2=topics[1])
                else:
                    content = template.format(topic1=topics[0], topic2="earlier context")
            else:
                topic = random.choice(topics) if topics else "this"
                content = template.format(topic=topic)

            # Create thought
            thought = ActiveThought(
                thought_type=thought_type,
                content=content,
                topics=topics[:3],
                intensity=0.3 + random.random() * 0.5,
            )

            # Add to active thoughts
            self._active_thoughts.append(thought)
            self._last_thought_time = now
            self._stats["total_thoughts"] += 1

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

    def decay_thoughts(self):
        """Decay old thoughts."""
        with self._lock:
            now = time.time()
            to_remove = []

            for thought in self._active_thoughts:
                age = thought.age_seconds()

                # Decay intensity over time
                thought.intensity *= 0.95

                # Remove very old or faded thoughts
                if age > 30 or thought.intensity < 0.1:
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

            # Maybe generate a new thought
            if len(self._active_thoughts) < 2 and random.random() < 0.3:
                self.generate_thought()

            active = [t.to_dict() for t in self._active_thoughts]
            recent_history = [t.to_dict() for t in self._thought_history[-5:]]

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
async def get_thinking_state():
    """Get current thinking state with all active thoughts."""
    manager = get_manager()
    return manager.get_state()


@router.get("/teaser")
async def get_teaser():
    """Get a teaser preview of what AURA is thinking about."""
    manager = get_manager()
    teaser = manager.get_teaser()

    if teaser:
        return {"has_teaser": True, "teaser": teaser}
    return {"has_teaser": False, "teaser": None}


@router.post("/generate")
async def generate_thought(force: bool = False):
    """Generate a new thought."""
    manager = get_manager()
    thought = manager.generate_thought(force=force)

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

    manager.add_thought_from_context(
        thought_type=thought_type,
        topic=request.topic,
        intensity=request.intensity or 0.6,
    )

    return {"status": "added", "topic": request.topic}


@router.post("/resolve/{thought_id}")
async def resolve_thought(thought_id: str, resolution: str = "dismissed"):
    """Resolve a thought."""
    manager = get_manager()
    manager.resolve_thought(thought_id, resolution)
    return {"status": "resolved", "thought_id": thought_id}


@router.get("/stats")
async def get_stats():
    """Get thinking statistics."""
    manager = get_manager()
    return manager.get_stats()


@router.post("/clear")
async def clear_thoughts():
    """Clear all thoughts."""
    manager = get_manager()
    manager.clear()
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

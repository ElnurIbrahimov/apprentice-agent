"""Spontaneous Conversation Starters - AURA initiates natural conversations."""

import asyncio
import functools
import logging
import random
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from threading import Lock

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversation", tags=["conversation"], dependencies=[Depends(require_api_key)])

# ============================================================================
# Conversation Starter Templates
# ============================================================================

# Templates organized by trigger type
STARTER_TEMPLATES = {
    "focus_topic": [
        "I noticed you've been focused on {topic} lately. Would you like to explore that further?",
        "You seem interested in {topic}. I have some thoughts on that if you'd like to hear them.",
        "I've been thinking about {topic} based on our recent conversations...",
        "{topic} has come up a few times. Is there something specific you're working on?",
        "I'm curious about your work with {topic}. How's it going?",
    ],
    "time_based": {
        "morning": [
            "Good morning! Ready to tackle something interesting today?",
            "Morning! I've been processing our recent conversations. Anything on your mind?",
            "A new day begins. What shall we explore together?",
        ],
        "afternoon": [
            "How's your afternoon going? Need a thinking partner?",
            "Taking a break? I'm here if you want to chat.",
            "Afternoon check-in - anything I can help you think through?",
        ],
        "evening": [
            "Winding down for the day? I enjoyed our conversations today.",
            "Evening reflection time. Anything you'd like to discuss?",
            "It's getting late. Don't forget to take breaks!",
        ],
        "late_night": [
            "Burning the midnight oil? I'm here to help.",
            "Late night session? Let me know if you need anything.",
            "Still working? Remember to rest when you can.",
        ],
    },
    "idle": [
        "It's been quiet for a while. Everything okay?",
        "I'm here whenever you're ready to continue.",
        "Taking a break? Good idea. I'll be here.",
        "Need any help with what you were working on earlier?",
    ],
    "memory_triggered": [
        "I remembered something relevant - {memory}. Does that help?",
        "This reminds me of when we discussed {memory}.",
        "Based on what I know about you, maybe {memory} could be useful here?",
    ],
    "emotional": {
        "stressed": [
            "You seem to have a lot going on. Want to talk it through?",
            "I sense some tension. How can I help?",
            "Remember, I'm here to help lighten the load.",
        ],
        "curious": [
            "I love your curiosity! What are you exploring?",
            "That's an interesting direction. Tell me more!",
            "Your questions are always thought-provoking.",
        ],
        "focused": [
            "You're in the zone! I won't interrupt much.",
            "Deep focus mode - let me know if you need anything.",
            "I'll keep it brief - you seem focused.",
        ],
        "happy": [
            "Great energy today! What's going well?",
            "Love the positive vibes. What's making you happy?",
            "It's nice to see you in good spirits!",
        ],
    },
    "proactive_insights": [
        "I've been analyzing patterns and noticed something interesting...",
        "Based on our conversations, I had a thought you might find useful.",
        "I've been reflecting on what we discussed. Can I share an observation?",
        "Something occurred to me that might be relevant to your work.",
    ],
}


# ============================================================================
# Conversation Starter Manager
# ============================================================================

class ConversationStarterManager:
    """Manages spontaneous conversation initiation."""

    def __init__(
        self,
        min_interval_seconds: float = 300,  # 5 minutes minimum between starters
        max_starters_per_hour: int = 4,
    ):
        self._lock = Lock()
        self._min_interval = min_interval_seconds
        self._max_per_hour = max_starters_per_hour
        self._last_starter_time: Optional[float] = None
        self._starters_this_hour: List[float] = []
        self._pending_starter: Optional[Dict[str, Any]] = None
        self._dismissed_topics: set = set()  # Topics user dismissed
        self._stats = {
            "total_generated": 0,
            "total_delivered": 0,
            "total_dismissed": 0,
        }

    def _get_time_period(self) -> str:
        """Get current time period for time-based starters."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "late_night"

    def _can_generate(self) -> bool:
        """Check if we can generate a new starter."""
        now = time.time()

        # Check minimum interval
        if self._last_starter_time:
            if now - self._last_starter_time < self._min_interval:
                return False

        # Check hourly limit
        hour_ago = now - 3600
        self._starters_this_hour = [t for t in self._starters_this_hour if t > hour_ago]
        if len(self._starters_this_hour) >= self._max_per_hour:
            return False

        return True

    def generate_starter(
        self,
        focus_topics: List[str] = None,
        emotion: str = None,
        idle_seconds: float = 0,
        memories: List[str] = None,
        force: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a conversation starter based on context.

        Args:
            focus_topics: Current focus topics from context tracker
            emotion: Current detected emotion
            idle_seconds: How long user has been idle
            memories: Relevant memories
            force: Generate even if rate limited

        Returns:
            Starter dict with type, content, and metadata, or None
        """
        with self._lock:
            if not force and not self._can_generate():
                return None

            starter = None
            starter_type = None

            # Priority 1: Long idle period (> 10 minutes)
            if idle_seconds > 600:
                template = random.choice(STARTER_TEMPLATES["idle"])
                starter = template
                starter_type = "idle"

            # Priority 2: Strong focus on a topic
            elif focus_topics and len(focus_topics) > 0:
                # Pick a topic not dismissed recently
                available_topics = [t for t in focus_topics if t.lower() not in self._dismissed_topics]
                if available_topics:
                    topic = available_topics[0]
                    template = random.choice(STARTER_TEMPLATES["focus_topic"])
                    starter = template.format(topic=topic)
                    starter_type = "focus_topic"

            # Priority 3: Emotional context
            elif emotion and emotion.lower() in STARTER_TEMPLATES["emotional"]:
                template = random.choice(STARTER_TEMPLATES["emotional"][emotion.lower()])
                starter = template
                starter_type = "emotional"

            # Priority 4: Memory-triggered
            elif memories and len(memories) > 0:
                memory = memories[0][:50]  # Truncate
                template = random.choice(STARTER_TEMPLATES["memory_triggered"])
                starter = template.format(memory=memory)
                starter_type = "memory_triggered"

            # Priority 5: Time-based
            else:
                time_period = self._get_time_period()
                template = random.choice(STARTER_TEMPLATES["time_based"][time_period])
                starter = template
                starter_type = "time_based"

            if starter:
                now = time.time()
                self._last_starter_time = now
                self._starters_this_hour.append(now)
                self._stats["total_generated"] += 1

                result = {
                    "type": starter_type,
                    "content": starter,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "focus_topics": focus_topics[:3] if focus_topics else [],
                        "emotion": emotion,
                        "idle_seconds": idle_seconds,
                    },
                }

                self._pending_starter = result
                return result

            return None

    def get_pending_starter(self) -> Optional[Dict[str, Any]]:
        """Get and clear pending starter."""
        with self._lock:
            starter = self._pending_starter
            if starter:
                self._pending_starter = None
                self._stats["total_delivered"] += 1
            return starter

    def has_pending_starter(self) -> bool:
        """Check if there's a pending starter."""
        return self._pending_starter is not None

    def dismiss_topic(self, topic: str):
        """Mark a topic as dismissed (user not interested)."""
        with self._lock:
            self._dismissed_topics.add(topic.lower())
            self._stats["total_dismissed"] += 1
            # Clear dismissed topics after 1 hour
            if len(self._dismissed_topics) > 20:
                self._dismissed_topics = set(list(self._dismissed_topics)[-10:])

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        with self._lock:
            return {
                **self._stats,
                "pending": self._pending_starter is not None,
                "starters_this_hour": len(self._starters_this_hour),
                "dismissed_topics": len(self._dismissed_topics),
                "min_interval_seconds": self._min_interval,
                "max_per_hour": self._max_per_hour,
            }

    def set_config(self, min_interval: float = None, max_per_hour: int = None):
        """Update configuration."""
        with self._lock:
            if min_interval is not None:
                self._min_interval = max(60, min_interval)  # Min 1 minute
            if max_per_hour is not None:
                self._max_per_hour = max(1, min(12, max_per_hour))


# Global manager instance
_manager = ConversationStarterManager()


def get_manager() -> ConversationStarterManager:
    """Get the global conversation starter manager."""
    return _manager


# ============================================================================
# API Models
# ============================================================================

class StarterResponse(BaseModel):
    type: str
    content: str
    timestamp: str
    metadata: Dict[str, Any]


class GenerateRequest(BaseModel):
    focus_topics: Optional[List[str]] = None
    emotion: Optional[str] = None
    idle_seconds: Optional[float] = 0
    memories: Optional[List[str]] = None
    force: Optional[bool] = False


class ConfigRequest(BaseModel):
    min_interval_seconds: Optional[float] = None
    max_starters_per_hour: Optional[int] = None


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/starter/pending")
async def get_pending_starter():
    """Get pending conversation starter if available."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    starter = await loop.run_in_executor(None, manager.get_pending_starter)

    if starter:
        return {"has_starter": True, "starter": starter}
    return {"has_starter": False, "starter": None}


@router.get("/starter/check")
async def check_pending():
    """Check if there's a pending starter without consuming it."""
    manager = get_manager()
    return {"has_pending": manager.has_pending_starter()}


@router.post("/starter/generate")
async def generate_starter(request: GenerateRequest):
    """Generate a conversation starter based on context."""
    manager = get_manager()
    loop = asyncio.get_running_loop()

    # Try to get context from other systems if not provided
    focus_topics = request.focus_topics
    emotion = request.emotion

    # Auto-fetch from context tracker if not provided
    if not focus_topics:
        try:
            from api.routes.context import get_tracker
            tracker = get_tracker()
            state = await loop.run_in_executor(None, functools.partial(tracker.get_focus_state, limit=5))
            focus_topics = [item["name"] for item in state.get("items", [])]
        except Exception:
            pass

    # Auto-fetch emotion from ALMA if not provided
    if not emotion:
        try:
            from aura.emotion.alma_engine import alma_engine
            if alma_engine:
                state = await loop.run_in_executor(None, alma_engine.get_emotional_state)
                emotion = state.get("dominant_emotion")
        except Exception:
            pass

    starter = await loop.run_in_executor(
        None,
        functools.partial(
            manager.generate_starter,
            focus_topics=focus_topics,
            emotion=emotion,
            idle_seconds=request.idle_seconds or 0,
            memories=request.memories,
            force=request.force or False,
        ),
    )

    if starter:
        return {"generated": True, "starter": starter}
    return {"generated": False, "reason": "Rate limited or no suitable context"}


@router.post("/starter/dismiss")
async def dismiss_topic(topic: str):
    """Dismiss a topic (user not interested)."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, functools.partial(manager.dismiss_topic, topic))
    return {"status": "dismissed", "topic": topic}


@router.get("/starter/stats")
async def get_stats():
    """Get conversation starter statistics."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, manager.get_stats)


@router.post("/starter/config")
async def update_config(request: ConfigRequest):
    """Update conversation starter configuration."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        functools.partial(
            manager.set_config,
            min_interval=request.min_interval_seconds,
            max_per_hour=request.max_starters_per_hour,
        ),
    )
    config = await loop.run_in_executor(None, manager.get_stats)
    return {"status": "updated", "config": config}


# ============================================================================
# Integration with Proactive System
# ============================================================================

async def check_and_generate_starter():
    """
    Called periodically by proactive system to check if we should
    generate a conversation starter.
    """
    manager = get_manager()
    loop = asyncio.get_running_loop()

    # Get current context
    try:
        from api.routes.context import get_tracker
        tracker = get_tracker()
        state = await loop.run_in_executor(None, functools.partial(tracker.get_focus_state, limit=5))
        focus_topics = [item["name"] for item in state.get("items", [])]
    except Exception:
        focus_topics = []

    # Get emotion
    try:
        from aura.emotion.alma_engine import alma_engine
        if alma_engine:
            emotion_state = await loop.run_in_executor(None, alma_engine.get_emotional_state)
            emotion = emotion_state.get("dominant_emotion")
        else:
            emotion = None
    except Exception:
        emotion = None

    # Generate starter
    return await loop.run_in_executor(
        None,
        functools.partial(manager.generate_starter, focus_topics=focus_topics, emotion=emotion),
    )

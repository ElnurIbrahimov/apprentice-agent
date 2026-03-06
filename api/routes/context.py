"""Context Awareness API - Tracks what AURA is currently focused on."""

import asyncio
import functools
import logging
import re
import math
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
from threading import Lock

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/context", tags=["context"])

# ============================================================================
# Focus Item - Represents something AURA is paying attention to
# ============================================================================

class FocusItem:
    """Represents a topic/concept AURA is currently focused on."""

    def __init__(self, name: str, category: str = "topic", initial_weight: float = 1.0):
        self.name = name
        self.category = category  # 'topic', 'entity', 'keyword', 'memory', 'emotion'
        self.weight = initial_weight
        self.last_updated = time.time()
        self.activation_count = 1
        self.sources: List[str] = []  # Where this focus came from

    def boost(self, amount: float = 0.3, source: str = None):
        """Increase focus weight."""
        self.weight = min(1.0, self.weight + amount)
        self.last_updated = time.time()
        self.activation_count += 1
        if source and source not in self.sources:
            self.sources.append(source)

    def decay(self, half_life: float = 120.0):
        """Apply time-based decay to focus weight."""
        age = time.time() - self.last_updated
        decay_factor = math.exp(-0.693 * age / half_life)
        self.weight *= decay_factor
        return self.weight

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "weight": round(self.weight, 3),
            "intensity": self._get_intensity(),
            "last_updated": datetime.fromtimestamp(self.last_updated).isoformat(),
            "activation_count": self.activation_count,
            "sources": self.sources[-3:],  # Last 3 sources
        }

    def _get_intensity(self) -> str:
        """Get intensity label based on weight."""
        if self.weight >= 0.8:
            return "high"
        elif self.weight >= 0.5:
            return "medium"
        elif self.weight >= 0.2:
            return "low"
        return "fading"


# ============================================================================
# Context Awareness Tracker
# ============================================================================

# Common stop words to filter out
STOP_WORDS = {
    "i", "you", "the", "a", "an", "is", "are", "was", "were",
    "it", "this", "that", "what", "how", "why", "when", "where",
    "do", "does", "did", "have", "has", "had", "be", "been",
    "will", "would", "could", "should", "can", "may", "might",
    "to", "for", "of", "in", "on", "at", "by", "with", "about",
    "just", "really", "very", "so", "and", "but", "or", "if",
    "my", "me", "your", "our", "their", "its", "am", "im",
    "hey", "hi", "hello", "thanks", "thank", "please", "yeah",
    "yes", "no", "ok", "okay", "sure", "well", "like", "know",
    "aura", "tell", "show", "help", "want", "need", "think",
    "let", "get", "make", "see", "go", "come", "take", "give",
}

# Category colors for UI
CATEGORY_COLORS = {
    "topic": "#8b5cf6",      # Purple
    "entity": "#3b82f6",     # Blue
    "keyword": "#10b981",    # Green
    "memory": "#f59e0b",     # Amber
    "emotion": "#ec4899",    # Pink
    "action": "#06b6d4",     # Cyan
}


class ContextAwarenessTracker:
    """Tracks AURA's current focus and attention."""

    def __init__(self, max_items: int = 30, decay_interval: float = 30.0):
        self._focus_items: Dict[str, FocusItem] = {}
        self._lock = Lock()
        self._max_items = max_items
        self._decay_interval = decay_interval
        self._last_decay = time.time()
        self._conversation_context: List[str] = []  # Recent conversation topics
        self._stats = {
            "total_activations": 0,
            "topics_tracked": 0,
            "decay_cycles": 0,
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        return keywords

    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entities (capitalized words, tech terms)."""
        # Find capitalized words (potential proper nouns)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        # Find tech terms (camelCase, acronyms, etc.)
        tech_terms = re.findall(r'\b[A-Z]{2,}\b', text)  # Acronyms
        camel_case = re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', text)  # camelCase

        return list(set(entities + tech_terms + camel_case))

    def track_message(self, message: str, is_user: bool = True, source: str = "chat"):
        """Track focus from a chat message."""
        with self._lock:
            self._apply_decay()

            # Extract keywords
            keywords = self._extract_keywords(message)
            for kw in keywords[:10]:  # Limit to top 10
                self._activate_focus(kw, "keyword", 0.2 if is_user else 0.15, source)

            # Extract entities
            entities = self._extract_entities(message)
            for entity in entities[:5]:
                self._activate_focus(entity, "entity", 0.4, source)

            # Track conversation context
            self._conversation_context.append(message[:100])
            if len(self._conversation_context) > 10:
                self._conversation_context.pop(0)

            self._stats["total_activations"] += 1
            self._cleanup()

    def track_topic(self, topic: str, weight: float = 0.5, source: str = "system"):
        """Explicitly track a topic of focus."""
        with self._lock:
            self._apply_decay()
            self._activate_focus(topic, "topic", weight, source)
            self._cleanup()

    def track_memory_access(self, memory_topics: List[str], source: str = "memory"):
        """Track focus when memories are accessed."""
        with self._lock:
            self._apply_decay()
            for topic in memory_topics[:5]:
                # Extract keywords from memory topic
                keywords = self._extract_keywords(topic)
                for kw in keywords[:3]:
                    self._activate_focus(kw, "memory", 0.35, source)
            self._cleanup()

    def track_emotion(self, emotion: str, intensity: float = 0.5, source: str = "alma"):
        """Track emotional focus."""
        with self._lock:
            self._apply_decay()
            self._activate_focus(emotion, "emotion", intensity * 0.5, source)
            self._cleanup()

    def track_action(self, action: str, weight: float = 0.4, source: str = "agent"):
        """Track action/task focus."""
        with self._lock:
            self._apply_decay()
            self._activate_focus(action, "action", weight, source)
            self._cleanup()

    def _activate_focus(self, name: str, category: str, weight: float, source: str):
        """Activate or boost a focus item."""
        key = f"{category}:{name.lower()}"

        if key in self._focus_items:
            self._focus_items[key].boost(weight, source)
        else:
            item = FocusItem(name, category, weight)
            item.sources.append(source)
            self._focus_items[key] = item
            self._stats["topics_tracked"] += 1

    def _apply_decay(self):
        """Apply decay to all focus items."""
        now = time.time()
        if now - self._last_decay < self._decay_interval:
            return

        for item in self._focus_items.values():
            item.decay(half_life=120.0)

        self._last_decay = now
        self._stats["decay_cycles"] += 1

    def _cleanup(self):
        """Remove faded items and limit total count."""
        # Remove items with very low weight
        faded = [k for k, v in self._focus_items.items() if v.weight < 0.05]
        for key in faded:
            del self._focus_items[key]

        # If still too many, remove lowest weight items
        if len(self._focus_items) > self._max_items:
            sorted_items = sorted(
                self._focus_items.items(),
                key=lambda x: x[1].weight,
                reverse=True
            )
            self._focus_items = dict(sorted_items[:self._max_items])

    def get_focus_state(self, limit: int = 15) -> Dict[str, Any]:
        """Get current focus state for UI."""
        with self._lock:
            self._apply_decay()

            # Sort by weight
            sorted_items = sorted(
                self._focus_items.values(),
                key=lambda x: x.weight,
                reverse=True
            )[:limit]

            # Group by category
            by_category: Dict[str, List[dict]] = defaultdict(list)
            for item in sorted_items:
                by_category[item.category].append(item.to_dict())

            # Calculate overall focus intensity
            total_weight = sum(item.weight for item in sorted_items)
            avg_weight = total_weight / max(1, len(sorted_items))

            return {
                "items": [item.to_dict() for item in sorted_items],
                "by_category": dict(by_category),
                "total_focus": round(total_weight, 2),
                "average_intensity": round(avg_weight, 3),
                "active_count": len(sorted_items),
                "category_colors": CATEGORY_COLORS,
            }

    def get_heatmap_data(self) -> Dict[str, Any]:
        """Get data formatted for heatmap visualization."""
        with self._lock:
            self._apply_decay()

            items = []
            for item in self._focus_items.values():
                if item.weight >= 0.1:  # Only include visible items
                    items.append({
                        "name": item.name,
                        "category": item.category,
                        "weight": round(item.weight, 3),
                        "size": self._weight_to_size(item.weight),
                        "color": CATEGORY_COLORS.get(item.category, "#6b7280"),
                        "opacity": min(1.0, item.weight + 0.3),
                    })

            # Sort by weight for consistent display
            items.sort(key=lambda x: x["weight"], reverse=True)

            return {
                "items": items[:20],  # Limit for UI
                "timestamp": datetime.now().isoformat(),
            }

    def _weight_to_size(self, weight: float) -> str:
        """Convert weight to size class for UI."""
        if weight >= 0.7:
            return "xl"
        elif weight >= 0.5:
            return "lg"
        elif weight >= 0.3:
            return "md"
        return "sm"

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        with self._lock:
            return {
                **self._stats,
                "current_items": len(self._focus_items),
                "conversation_depth": len(self._conversation_context),
            }

    def clear(self):
        """Clear all focus tracking."""
        with self._lock:
            self._focus_items.clear()
            self._conversation_context.clear()


# Global tracker instance
_tracker = ContextAwarenessTracker()


def get_tracker() -> ContextAwarenessTracker:
    """Get the global context awareness tracker."""
    return _tracker


# ============================================================================
# API Response Models
# ============================================================================

class FocusItemResponse(BaseModel):
    name: str
    category: str
    weight: float
    intensity: str
    last_updated: str
    activation_count: int
    sources: List[str]


class FocusStateResponse(BaseModel):
    items: List[FocusItemResponse]
    by_category: Dict[str, List[FocusItemResponse]]
    total_focus: float
    average_intensity: float
    active_count: int
    category_colors: Dict[str, str]


class HeatmapItem(BaseModel):
    name: str
    category: str
    weight: float
    size: str
    color: str
    opacity: float


class HeatmapResponse(BaseModel):
    items: List[HeatmapItem]
    timestamp: str


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/focus")
async def get_focus_state(limit: int = 15):
    """Get current focus state showing what AURA is paying attention to."""
    tracker = get_tracker()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(tracker.get_focus_state, limit=min(limit, 30)))


@router.get("/heatmap")
async def get_heatmap():
    """Get heatmap visualization data."""
    tracker = get_tracker()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, tracker.get_heatmap_data)


@router.get("/stats")
async def get_context_stats():
    """Get context tracking statistics."""
    tracker = get_tracker()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, tracker.get_stats)


@router.post("/track/message")
async def track_message(message: str, is_user: bool = True, source: str = "chat"):
    """Track focus from a chat message."""
    tracker = get_tracker()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, functools.partial(tracker.track_message, message, is_user, source))
    return {"status": "tracked", "message_length": len(message)}


@router.post("/track/topic")
async def track_topic(topic: str, weight: float = 0.5, source: str = "manual"):
    """Explicitly track a topic of focus."""
    tracker = get_tracker()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, functools.partial(tracker.track_topic, topic, weight, source))
    return {"status": "tracked", "topic": topic}


@router.post("/track/memory")
async def track_memory_access(topics: List[str], source: str = "memory"):
    """Track focus when memories are accessed."""
    tracker = get_tracker()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, functools.partial(tracker.track_memory_access, topics, source))
    return {"status": "tracked", "topics_count": len(topics)}


@router.post("/track/emotion")
async def track_emotion(emotion: str, intensity: float = 0.5, source: str = "alma"):
    """Track emotional focus."""
    tracker = get_tracker()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, functools.partial(tracker.track_emotion, emotion, intensity, source))
    return {"status": "tracked", "emotion": emotion}


@router.post("/clear")
async def clear_context():
    """Clear all context tracking."""
    tracker = get_tracker()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, tracker.clear)
    return {"status": "cleared"}


# ============================================================================
# Integration Helper for Agent
# ============================================================================

def track_context_from_message(message: str, is_user: bool = True):
    """Helper to track context from agent message processing."""
    tracker = get_tracker()
    tracker.track_message(message, is_user, "agent")


def track_context_from_memory(memory_contents: List[str]):
    """Helper to track context from memory retrieval."""
    tracker = get_tracker()
    tracker.track_memory_access(memory_contents, "memory_recall")


def track_context_from_emotion(emotion: str, intensity: float):
    """Helper to track emotional context."""
    tracker = get_tracker()
    tracker.track_emotion(emotion, intensity, "alma_engine")

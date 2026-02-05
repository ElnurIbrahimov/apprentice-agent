"""Ambient Idle Behaviors - Makes AURA feel alive when not actively responding."""

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

router = APIRouter(prefix="/api/idle", tags=["idle"])

# ============================================================================
# Idle Behavior Types and Templates
# ============================================================================

class IdleBehaviorType(str, Enum):
    OBSERVING = "observing"           # Noticing things in the environment
    REFLECTING = "reflecting"         # Contemplating previous conversation
    ANTICIPATING = "anticipating"     # Expecting user might say something
    DRIFTING = "drifting"            # Mind wandering naturally
    FOCUSING = "focusing"             # Attention settling on something
    RELAXING = "relaxing"             # Calm, restful state
    CURIOUS = "curious"               # Mild interest in something
    PROCESSING = "processing"         # Background thought processing


class IdleIntensity(str, Enum):
    DEEP = "deep"           # Very relaxed, minimal activity
    LIGHT = "light"         # Slightly active, aware
    ALERT = "alert"         # Ready to engage
    RESTLESS = "restless"   # Mildly anticipating


# Status messages for different behaviors
IDLE_STATUS_MESSAGES = {
    IdleBehaviorType.OBSERVING: [
        "noticing the quiet...",
        "observing...",
        "sensing the space...",
        "watching the moment pass...",
        "taking in the stillness...",
    ],
    IdleBehaviorType.REFLECTING: [
        "reflecting quietly...",
        "thinking back...",
        "considering what was said...",
        "mulling things over...",
        "dwelling on a thought...",
    ],
    IdleBehaviorType.ANTICIPATING: [
        "listening...",
        "waiting patiently...",
        "ready when you are...",
        "here if you need me...",
        "attentive...",
    ],
    IdleBehaviorType.DRIFTING: [
        "mind wandering...",
        "thoughts drifting...",
        "daydreaming...",
        "letting thoughts flow...",
        "in a reverie...",
    ],
    IdleBehaviorType.FOCUSING: [
        "attention settling...",
        "centering...",
        "becoming present...",
        "grounding...",
        "finding focus...",
    ],
    IdleBehaviorType.RELAXING: [
        "at ease...",
        "resting...",
        "peaceful...",
        "calm...",
        "serene...",
    ],
    IdleBehaviorType.CURIOUS: [
        "wondering...",
        "curious about something...",
        "pondering...",
        "intrigued...",
        "exploring a thought...",
    ],
    IdleBehaviorType.PROCESSING: [
        "processing in the background...",
        "integrating thoughts...",
        "organizing memories...",
        "connecting ideas...",
        "synthesizing...",
    ],
}

# Time-of-day influenced behavior weights
TIME_BEHAVIOR_WEIGHTS = {
    "morning": {  # 6am - 12pm
        IdleBehaviorType.FOCUSING: 1.5,
        IdleBehaviorType.ANTICIPATING: 1.3,
        IdleBehaviorType.CURIOUS: 1.2,
    },
    "afternoon": {  # 12pm - 6pm
        IdleBehaviorType.PROCESSING: 1.4,
        IdleBehaviorType.OBSERVING: 1.2,
        IdleBehaviorType.REFLECTING: 1.2,
    },
    "evening": {  # 6pm - 10pm
        IdleBehaviorType.RELAXING: 1.4,
        IdleBehaviorType.REFLECTING: 1.3,
        IdleBehaviorType.DRIFTING: 1.2,
    },
    "night": {  # 10pm - 6am
        IdleBehaviorType.DRIFTING: 1.5,
        IdleBehaviorType.RELAXING: 1.4,
        IdleBehaviorType.OBSERVING: 1.0,
    },
}


# ============================================================================
# Idle State Manager
# ============================================================================

class IdleBehavior:
    """Represents a current idle behavior."""

    def __init__(
        self,
        behavior_type: IdleBehaviorType,
        intensity: IdleIntensity,
        status_message: str,
        duration_hint: float = 10.0,
    ):
        self.type = behavior_type
        self.intensity = intensity
        self.status_message = status_message
        self.duration_hint = duration_hint
        self.started_at = time.time()
        self.breath_rate = self._calculate_breath_rate()
        self.attention_drift = random.uniform(-0.3, 0.3)  # Subtle attention shift

    def _calculate_breath_rate(self) -> float:
        """Calculate breathing rate modifier based on state."""
        rates = {
            IdleIntensity.DEEP: 0.7,      # Slower breathing
            IdleIntensity.LIGHT: 0.9,
            IdleIntensity.ALERT: 1.1,
            IdleIntensity.RESTLESS: 1.3,  # Faster breathing
        }
        return rates.get(self.intensity, 1.0)

    def age_seconds(self) -> float:
        return time.time() - self.started_at

    def is_expired(self) -> bool:
        return self.age_seconds() > self.duration_hint

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "intensity": self.intensity.value,
            "status_message": self.status_message,
            "breath_rate": self.breath_rate,
            "attention_drift": round(self.attention_drift, 3),
            "age_seconds": round(self.age_seconds(), 1),
            "duration_hint": self.duration_hint,
        }


class IdleStateManager:
    """Manages AURA's ambient idle behaviors."""

    def __init__(self):
        self._lock = Lock()
        self._current_behavior: Optional[IdleBehavior] = None
        self._last_activity_time = time.time()
        self._last_behavior_change = 0.0
        self._behavior_history: List[IdleBehavior] = []
        self._idle_since: Optional[float] = None

        # Animation state
        self._micro_movement_seed = random.random()
        self._attention_focus = 0.0  # -1 to 1, where 0 is neutral

        # Stats
        self._stats = {
            "behaviors_generated": 0,
            "total_idle_time": 0.0,
            "favorite_behavior": None,
        }

    def _get_time_period(self) -> str:
        """Get current time period for behavior weighting."""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    def _get_emotion_context(self) -> Optional[str]:
        """Get current emotional state from ALMA."""
        try:
            from api.routes.features import get_alma_state
            state = get_alma_state()
            if state:
                return state.get("dominant_emotion")
        except Exception:
            pass
        return None

    def _select_behavior_type(self) -> IdleBehaviorType:
        """Select a behavior type based on context."""
        # Base weights
        weights = {bt: 1.0 for bt in IdleBehaviorType}

        # Apply time-of-day weights
        time_period = self._get_time_period()
        time_weights = TIME_BEHAVIOR_WEIGHTS.get(time_period, {})
        for bt, weight in time_weights.items():
            weights[bt] *= weight

        # Apply emotion context
        emotion = self._get_emotion_context()
        if emotion:
            emotion_adjustments = {
                "curious": {IdleBehaviorType.CURIOUS: 1.5, IdleBehaviorType.OBSERVING: 1.3},
                "contemplative": {IdleBehaviorType.REFLECTING: 1.5, IdleBehaviorType.PROCESSING: 1.3},
                "calm": {IdleBehaviorType.RELAXING: 1.5, IdleBehaviorType.DRIFTING: 1.2},
                "engaged": {IdleBehaviorType.ANTICIPATING: 1.4, IdleBehaviorType.FOCUSING: 1.3},
                "anxious": {IdleBehaviorType.RESTLESS: 1.5, IdleBehaviorType.ANTICIPATING: 1.3},
            }
            if emotion in emotion_adjustments:
                for bt, adj in emotion_adjustments[emotion].items():
                    weights[bt] *= adj

        # Avoid repeating recent behaviors
        recent_types = [b.type for b in self._behavior_history[-3:]]
        for bt in recent_types:
            weights[bt] *= 0.5

        # Weighted random selection
        total = sum(weights.values())
        r = random.uniform(0, total)
        cumulative = 0
        for bt, w in weights.items():
            cumulative += w
            if r <= cumulative:
                return bt

        return IdleBehaviorType.OBSERVING

    def _select_intensity(self, idle_duration: float) -> IdleIntensity:
        """Select idle intensity based on how long we've been idle."""
        if idle_duration < 30:
            # Recently active - alert or light
            return random.choice([IdleIntensity.ALERT, IdleIntensity.LIGHT])
        elif idle_duration < 120:
            # Moderate idle - light or relaxing
            return random.choice([IdleIntensity.LIGHT, IdleIntensity.LIGHT, IdleIntensity.DEEP])
        else:
            # Long idle - deeper states
            return random.choice([IdleIntensity.DEEP, IdleIntensity.DEEP, IdleIntensity.LIGHT])

    def record_activity(self):
        """Record that user activity occurred."""
        with self._lock:
            now = time.time()
            if self._idle_since:
                self._stats["total_idle_time"] += now - self._idle_since
            self._last_activity_time = now
            self._idle_since = None
            self._current_behavior = None

    def get_idle_duration(self) -> float:
        """Get how long we've been idle."""
        with self._lock:
            return time.time() - self._last_activity_time

    def generate_behavior(self, force: bool = False) -> Optional[IdleBehavior]:
        """Generate a new idle behavior if appropriate."""
        with self._lock:
            now = time.time()
            idle_duration = now - self._last_activity_time

            # Need at least 5 seconds of idle to start behaviors
            if idle_duration < 5 and not force:
                return None

            # Mark when we started being idle
            if self._idle_since is None:
                self._idle_since = self._last_activity_time

            # Rate limit behavior changes (minimum 8 seconds between changes)
            if not force and now - self._last_behavior_change < 8:
                return self._current_behavior

            # Check if current behavior is still valid
            if self._current_behavior and not self._current_behavior.is_expired():
                return self._current_behavior

            # Generate new behavior
            behavior_type = self._select_behavior_type()
            intensity = self._select_intensity(idle_duration)
            status_message = random.choice(IDLE_STATUS_MESSAGES[behavior_type])

            # Vary duration based on intensity
            duration_base = {
                IdleIntensity.DEEP: 15,
                IdleIntensity.LIGHT: 10,
                IdleIntensity.ALERT: 8,
                IdleIntensity.RESTLESS: 6,
            }
            duration = duration_base[intensity] + random.uniform(-3, 5)

            behavior = IdleBehavior(
                behavior_type=behavior_type,
                intensity=intensity,
                status_message=status_message,
                duration_hint=duration,
            )

            # Update state
            if self._current_behavior:
                self._behavior_history.append(self._current_behavior)
                # Keep history bounded
                if len(self._behavior_history) > 20:
                    self._behavior_history = self._behavior_history[-20:]

            self._current_behavior = behavior
            self._last_behavior_change = now
            self._stats["behaviors_generated"] += 1

            # Update micro-movement seed for animation variation
            self._micro_movement_seed = random.random()
            self._attention_focus = random.uniform(-0.5, 0.5)

            return behavior

    def get_state(self) -> Dict[str, Any]:
        """Get current idle state for UI."""
        with self._lock:
            idle_duration = time.time() - self._last_activity_time
            is_idle = idle_duration > 5

            # Maybe generate a new behavior
            if is_idle:
                self.generate_behavior()

            return {
                "is_idle": is_idle,
                "idle_duration": round(idle_duration, 1),
                "current_behavior": self._current_behavior.to_dict() if self._current_behavior else None,
                "micro_movement_seed": self._micro_movement_seed,
                "attention_focus": round(self._attention_focus, 3),
                "time_period": self._get_time_period(),
            }

    def get_animation_params(self) -> Dict[str, Any]:
        """Get animation parameters for the avatar."""
        with self._lock:
            behavior = self._current_behavior

            # Default animation params
            params = {
                "breath_rate_modifier": 1.0,
                "breath_depth_modifier": 1.0,
                "glow_intensity": 0.5,
                "attention_x": 0.0,
                "attention_y": 0.0,
                "micro_movement_x": 0.0,
                "micro_movement_y": 0.0,
                "pulse_variation": 0.0,
            }

            if behavior:
                params["breath_rate_modifier"] = behavior.breath_rate
                params["attention_x"] = behavior.attention_drift

                # Intensity-based adjustments
                intensity_glow = {
                    IdleIntensity.DEEP: 0.3,
                    IdleIntensity.LIGHT: 0.5,
                    IdleIntensity.ALERT: 0.7,
                    IdleIntensity.RESTLESS: 0.8,
                }
                params["glow_intensity"] = intensity_glow.get(behavior.intensity, 0.5)

                # Behavior-specific adjustments
                if behavior.type == IdleBehaviorType.CURIOUS:
                    params["attention_y"] = random.uniform(0.1, 0.3)
                elif behavior.type == IdleBehaviorType.RELAXING:
                    params["breath_depth_modifier"] = 1.2
                elif behavior.type == IdleBehaviorType.RESTLESS:
                    params["micro_movement_x"] = random.uniform(-0.1, 0.1)
                    params["pulse_variation"] = 0.15

            # Add time-based micro-movements
            t = time.time()
            params["micro_movement_x"] += 0.02 * (0.5 + 0.5 * (1 + self._micro_movement_seed) * 0.5) * \
                                           (0.5 + 0.5 * ((t * 0.3) % 1))
            params["micro_movement_y"] += 0.015 * (0.5 + 0.5 * self._micro_movement_seed) * \
                                           (0.5 + 0.5 * ((t * 0.2 + 0.5) % 1))

            return params

    def get_stats(self) -> Dict[str, Any]:
        """Get idle behavior statistics."""
        with self._lock:
            # Find favorite behavior
            if self._behavior_history:
                type_counts = {}
                for b in self._behavior_history:
                    type_counts[b.type.value] = type_counts.get(b.type.value, 0) + 1
                favorite = max(type_counts, key=type_counts.get)
            else:
                favorite = None

            return {
                **self._stats,
                "favorite_behavior": favorite,
                "history_size": len(self._behavior_history),
                "current_idle_duration": round(self.get_idle_duration(), 1),
            }


# Global manager
_manager = IdleStateManager()


def get_manager() -> IdleStateManager:
    return _manager


# ============================================================================
# API Models
# ============================================================================

class IdleBehaviorResponse(BaseModel):
    type: str
    intensity: str
    status_message: str
    breath_rate: float
    attention_drift: float
    age_seconds: float
    duration_hint: float


class IdleStateResponse(BaseModel):
    is_idle: bool
    idle_duration: float
    current_behavior: Optional[IdleBehaviorResponse]
    micro_movement_seed: float
    attention_focus: float
    time_period: str


class AnimationParamsResponse(BaseModel):
    breath_rate_modifier: float
    breath_depth_modifier: float
    glow_intensity: float
    attention_x: float
    attention_y: float
    micro_movement_x: float
    micro_movement_y: float
    pulse_variation: float


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/state")
async def get_idle_state():
    """Get current idle state with behavior info."""
    manager = get_manager()
    return manager.get_state()


@router.get("/animation")
async def get_animation_params():
    """Get animation parameters for the avatar."""
    manager = get_manager()
    return manager.get_animation_params()


@router.post("/activity")
async def record_activity():
    """Record that user activity occurred (resets idle state)."""
    manager = get_manager()
    manager.record_activity()
    return {"status": "recorded"}


@router.post("/generate")
async def generate_behavior(force: bool = False):
    """Generate a new idle behavior."""
    manager = get_manager()
    behavior = manager.generate_behavior(force=force)

    if behavior:
        return {"generated": True, "behavior": behavior.to_dict()}
    return {"generated": False, "reason": "Not idle long enough"}


@router.get("/stats")
async def get_stats():
    """Get idle behavior statistics."""
    manager = get_manager()
    return manager.get_stats()


# ============================================================================
# Integration Helpers
# ============================================================================

def record_user_activity():
    """Helper to record activity from other modules."""
    manager = get_manager()
    manager.record_activity()


def get_current_idle_status() -> Optional[str]:
    """Get current idle status message for display."""
    manager = get_manager()
    state = manager.get_state()
    if state.get("current_behavior"):
        return state["current_behavior"].get("status_message")
    return None

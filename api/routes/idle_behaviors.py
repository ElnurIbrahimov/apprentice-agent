"""Ambient Idle Behaviors - Makes AURA feel alive when not actively responding."""

import asyncio
import functools
import logging
import random
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from threading import RLock
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
        self._lock = RLock()
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
            from aura.emotion.alma_engine import alma_engine
            if alma_engine and alma_engine.current_state:
                return alma_engine.current_state.get("dominant_emotion")
        except Exception:
            pass
        return None

    def _get_real_idle_status(self, behavior_type: IdleBehaviorType) -> Optional[str]:
        """Get a REAL status message from actual cognitive systems.

        Queries multiple always-available systems and selects based on
        behavior type for variety. Returns real data almost every time.
        """
        sources = []

        # === Idle Presence Engine (Phase 6D - highest priority genuine activity) ===
        try:
            from aura.consciousness.idle_presence import get_idle_presence_engine
            ipe = get_idle_presence_engine()
            activity_status = ipe.get_current_activity_status()
            if activity_status:
                sources.append(activity_status)
        except Exception:
            pass

        # === ALMA emotional state (always available) ===
        try:
            from aura.emotion.alma_engine import alma_engine
            if alma_engine:
                state = alma_engine.get_emotional_state()
                if state:
                    emotion = state.get("dominant_emotion", "neutral")
                    pad = state.get("pad", {})
                    pleasure = pad.get("pleasure", 0)
                    arousal = pad.get("arousal", 0)
                    neuro = state.get("neuromodulators", {})
                    mood_info = state.get("mood", {})
                    mood_name = mood_info.get("name", "neutral")
                    active_count = len(state.get("active_emotions", []))

                    # Rich emotional status messages
                    if emotion != "neutral":
                        sources.append(f"emotional state: {emotion} (pleasure: {pleasure:+.2f}, arousal: {arousal:+.2f})")
                    if mood_name and mood_name != "neutral":
                        sources.append(f"mood: {mood_name} (drifting naturally)")
                    if active_count > 0:
                        sources.append(f"processing {active_count} active emotion(s)")
                    if neuro:
                        dopa = neuro.get("dopamine", 0.5)
                        sero = neuro.get("serotonin", 0.5)
                        if abs(dopa - 0.5) > 0.1:
                            sources.append(f"dopamine: {dopa:.2f} ({'elevated' if dopa > 0.5 else 'low'}) — {'motivated' if dopa > 0.5 else 'resting'}")
                        if abs(sero - 0.5) > 0.1:
                            sources.append(f"serotonin: {sero:.2f} — {'patient' if sero > 0.5 else 'restless'}")
        except Exception:
            pass

        # === Gateway Daemon stats (always running) ===
        try:
            from aura.proactive.gateway_daemon import get_gateway_daemon
            daemon = get_gateway_daemon()
            stats = daemon.get_stats()
            events = stats.get("events_processed", 0)
            decisions = stats.get("decisions_made", 0)
            uptime = stats.get("uptime_seconds")
            beliefs = stats.get("beliefs", {})
            daemon_state = stats.get("state", "unknown")

            if daemon_state == "running":
                if uptime and uptime > 0:
                    uptime_min = int(uptime / 60)
                    sources.append(f"gateway daemon: active for {uptime_min}m, {events} events, {decisions} decisions")
                if beliefs:
                    user_focus = beliefs.get("user_focus_level", 0.5)
                    user_activity = beliefs.get("user_activity", "unknown")
                    if user_activity != "unknown":
                        sources.append(f"user state: {user_activity} (focus: {user_focus:.1f})")
        except Exception:
            pass

        # === Thinking system stats (always available) ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            stats = tm.get_stats()
            real = stats.get("real_thoughts", 0)
            total = stats.get("total_thoughts", 0)
            active = stats.get("active_thoughts", 0)
            if total > 0:
                sources.append(f"cognitive activity: {real} real thoughts / {total} total ({active} active)")
        except Exception:
            pass

        # === Memory recall tracker ===
        try:
            from api.routes.memory import get_tracker as get_memory_tracker
            mem_tracker = get_memory_tracker()
            stats = mem_tracker.get_stats()
            recalls = stats.get("total_recalls", 0)
            memories = stats.get("total_memories_retrieved", 0)
            if recalls > 0:
                sources.append(f"memory: {recalls} recall cycles, {memories} memories surfaced")
        except Exception:
            pass

        # === Context focus tracker ===
        try:
            from api.routes.context import get_tracker as get_context_tracker
            ctx_tracker = get_context_tracker()
            focus_state = ctx_tracker.get_focus_state(limit=3)
            items = focus_state.get("items", [])
            if items:
                top_topics = [item["name"] for item in items[:2]]
                sources.append(f"attention on: {', '.join(top_topics)}")
        except Exception:
            pass

        # === Event bus stats (via daemon) ===
        try:
            from aura.proactive.gateway_daemon import get_gateway_daemon
            daemon_ref = get_gateway_daemon()
            bus_stats = daemon_ref.event_bus.get_stats()
            published = bus_stats.get("events_published", 0)
            if published > 0:
                sources.append(f"event bus: {published} events flowing")
        except Exception:
            pass

        # === Calendar context ===
        try:
            from aura.proactive.monitors.calendar_monitor import get_calendar_monitor
            cm = get_calendar_monitor()
            next_evt = cm.get_next_event()
            if next_evt:
                mins = int(next_evt.get("minutes_until", 999))
                title = next_evt.get("title", "event")
                if mins < 60:
                    sources.append(f"upcoming: {title} in {mins} minutes")
        except Exception:
            pass

        # === Weather context ===
        try:
            from aura.emotion.alma_engine import alma_engine
            weather = alma_engine._get_weather_context()
            if weather:
                temp = weather.get("temperature_c", "?")
                cond = weather.get("condition", "unknown")
                sources.append(f"weather: {cond}, {temp}°C")
        except Exception:
            pass

        # Select based on behavior type for variety
        if not sources:
            return None

        # Map behavior types to preferred source indices for natural variety
        # Index 0 = idle presence (genuine activity), 1+ = other sources
        type_preferences = {
            IdleBehaviorType.OBSERVING: [0, 1, 5, 6],    # IdlePresence, emotion, context, events
            IdleBehaviorType.REFLECTING: [0, 3, 4, 1],    # IdlePresence, thinking, memory, emotion
            IdleBehaviorType.ANTICIPATING: [2, 0, 5, 1],  # Daemon, IdlePresence, context, emotion
            IdleBehaviorType.DRIFTING: [1, 0, 3, 6],      # Emotion, IdlePresence, thinking, events
            IdleBehaviorType.FOCUSING: [0, 5, 3, 1],      # IdlePresence, context, thinking, emotion
            IdleBehaviorType.RELAXING: [0, 1, 2, 4],      # IdlePresence, emotion, daemon, memory
            IdleBehaviorType.CURIOUS: [0, 6, 4, 3],       # IdlePresence, events, memory, thinking
            IdleBehaviorType.PROCESSING: [0, 2, 3, 4],    # IdlePresence, daemon, thinking, memory
        }

        preferred = type_preferences.get(behavior_type, [0, 1, 2])
        for idx in preferred:
            if idx < len(sources):
                return sources[idx]

        # Fallback: return first available
        return sources[0] if sources else None

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

        # Calendar context: boost ANTICIPATING when meeting is near
        try:
            from aura.proactive.monitors.calendar_monitor import get_calendar_monitor
            cm = get_calendar_monitor()
            next_evt = cm.get_next_event()
            if next_evt:
                mins = next_evt.get("minutes_until", 999)
                if mins < 15:
                    weights[IdleBehaviorType.ANTICIPATING] *= 2.0
                elif mins < 30:
                    weights[IdleBehaviorType.ANTICIPATING] *= 1.5
        except Exception:
            pass

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

            # PHASE 1: Try to get a REAL status from actual cognitive systems
            real_status = self._get_real_idle_status(behavior_type)
            status_message = real_status if real_status else random.choice(IDLE_STATUS_MESSAGES[behavior_type])

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

            # Spontaneous micro-emotions: 15% chance per cycle
            if random.random() < 0.15:
                micro_emotion_map = {
                    IdleBehaviorType.CURIOUS: ("curious", 0.15),
                    IdleBehaviorType.RELAXING: ("calm", 0.1),
                    IdleBehaviorType.REFLECTING: ("thoughtful", 0.1),
                    IdleBehaviorType.OBSERVING: ("engaged", 0.1),
                    IdleBehaviorType.DRIFTING: ("content", 0.08),
                    IdleBehaviorType.ANTICIPATING: ("hope", 0.12),
                    IdleBehaviorType.FOCUSING: ("confident", 0.1),
                    IdleBehaviorType.PROCESSING: ("thoughtful", 0.1),
                }
                micro = micro_emotion_map.get(behavior_type)
                if micro:
                    try:
                        from aura.emotion.alma_engine import alma_engine
                        alma_engine.trigger_emotion(
                            micro[0], intensity=micro[1],
                            trigger=f"idle_{behavior_type.value}_micro"
                        )
                    except Exception:
                        pass

            return behavior

    def get_state(self) -> Dict[str, Any]:
        """Get current idle state for UI."""
        with self._lock:
            idle_duration = time.time() - self._last_activity_time
            is_idle = idle_duration > 5

            # Maybe generate a new behavior
            if is_idle:
                self.generate_behavior()

            state = {
                "is_idle": is_idle,
                "idle_duration": round(idle_duration, 1),
                "current_behavior": self._current_behavior.to_dict() if self._current_behavior else None,
                "micro_movement_seed": self._micro_movement_seed,
                "attention_focus": round(self._attention_focus, 3),
                "time_period": self._get_time_period(),
            }

            # === Phase 6D: Add genuine cognitive load data ===
            try:
                from aura.consciousness.idle_presence import get_idle_presence_engine
                ipe = get_idle_presence_engine()
                load = ipe.compute_cognitive_load()
                state["cognitive_load"] = load.to_dict()
                state["breath_rate_from_load"] = ipe.get_breath_rate_from_load()
                state["glow_from_load"] = ipe.get_glow_from_load()
                state["dream_state"] = {
                    "active": ipe._dream_session_active,
                    "phase": ipe._current_dream_phase,
                }
            except Exception:
                pass

            return state

    def get_animation_params(self) -> Dict[str, Any]:
        """Get animation parameters for the avatar.

        Phase 6D: Breath rate and glow are driven by actual cognitive load
        rather than CSS-only intensity enums.
        """
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
                "cognitive_load": 0.0,
            }

            # === Phase 6D: Drive from real cognitive load ===
            try:
                from aura.consciousness.idle_presence import get_idle_presence_engine
                ipe = get_idle_presence_engine()
                params["breath_rate_modifier"] = ipe.get_breath_rate_from_load()
                params["glow_intensity"] = ipe.get_glow_from_load()
                load = ipe.compute_cognitive_load()
                params["cognitive_load"] = round(load.total_load, 3)

                # Deeper breathing when dreaming
                if ipe._dream_session_active:
                    if ipe._current_dream_phase == "deep":
                        params["breath_depth_modifier"] = 1.3
                    elif ipe._current_dream_phase == "rem":
                        params["pulse_variation"] = 0.12  # Subtle REM pulse
            except Exception:
                # Fallback to behavior-based values
                if behavior:
                    params["breath_rate_modifier"] = behavior.breath_rate
                    intensity_glow = {
                        IdleIntensity.DEEP: 0.3,
                        IdleIntensity.LIGHT: 0.5,
                        IdleIntensity.ALERT: 0.7,
                        IdleIntensity.RESTLESS: 0.8,
                    }
                    params["glow_intensity"] = intensity_glow.get(behavior.intensity, 0.5)

            if behavior:
                params["attention_x"] = behavior.attention_drift

                # Behavior-specific adjustments
                if behavior.type == IdleBehaviorType.CURIOUS:
                    params["attention_y"] = random.uniform(0.1, 0.3)
                elif behavior.type == IdleBehaviorType.RELAXING:
                    params["breath_depth_modifier"] = max(params["breath_depth_modifier"], 1.2)

                # Intensity-specific adjustments
                if behavior.intensity == IdleIntensity.RESTLESS:
                    params["micro_movement_x"] = random.uniform(-0.1, 0.1)
                    params["pulse_variation"] = max(params["pulse_variation"], 0.15)

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


def init_idle_presence() -> None:
    """Initialize Phase 6D idle presence engine and register callbacks.

    Call this after all systems are initialized (e.g., from agent_service startup).
    """
    try:
        from aura.consciousness.idle_presence import get_idle_presence_engine
        ipe = get_idle_presence_engine()
        ipe.register_neurodream_callbacks()
        ipe.start_background_tasks()
        logger.info("[IdleBehaviors] Phase 6D idle presence engine initialized")
    except Exception as e:
        logger.debug(f"[IdleBehaviors] Idle presence init failed: {e}")


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
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, manager.get_state)


@router.get("/animation")
async def get_animation_params():
    """Get animation parameters for the avatar."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, manager.get_animation_params)


@router.post("/activity")
async def record_activity():
    """Record that user activity occurred (resets idle state)."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, manager.record_activity)
    return {"status": "recorded"}


@router.post("/generate")
async def generate_behavior(force: bool = False):
    """Generate a new idle behavior."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    behavior = await loop.run_in_executor(None, functools.partial(manager.generate_behavior, force=force))

    if behavior:
        return {"generated": True, "behavior": behavior.to_dict()}
    return {"generated": False, "reason": "Not idle long enough"}


@router.get("/stats")
async def get_stats():
    """Get idle behavior statistics."""
    manager = get_manager()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, manager.get_stats)


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

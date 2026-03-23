"""
ALMA Engine - Affective Layer for Mental Architecture

A three-layer emotional system that gives AURA genuine emotional depth:
- Layer 1: Emotions (rapid, triggered by events, decay quickly)
- Layer 2: Mood (slow, influenced by emotions, sets baseline)
- Layer 3: Personality (stable, defines emotional tendencies)

Based on:
- ALMA model (Gebhard, 2005)
- PAD emotional space (Mehrabian, 1996)
- OCC appraisal theory (Ortony, Clore & Collins, 1988)
- Neuromodulator research (Doya, 2002)
"""

import atexit
import json
import math
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading

from aura.jsonl_utils import rotate_jsonl_if_needed

logger = logging.getLogger(__name__)


# =============================================================================
# PAD (Pleasure-Arousal-Dominance) Emotional Space
# =============================================================================

@dataclass
class PADState:
    """
    Position in PAD (Pleasure-Arousal-Dominance) 3D emotional space.

    Each dimension ranges from -1.0 to 1.0:
    - Pleasure (P): Negative (sad/angry) to Positive (happy/content)
    - Arousal (A): Low (calm/bored) to High (excited/anxious)
    - Dominance (D): Submissive to Dominant/In-control
    """
    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def __post_init__(self):
        # Clamp values to valid range
        self.pleasure = max(-1.0, min(1.0, self.pleasure))
        self.arousal = max(-1.0, min(1.0, self.arousal))
        self.dominance = max(-1.0, min(1.0, self.dominance))

    def distance_to(self, other: 'PADState') -> float:
        """Euclidean distance to another PAD state."""
        return math.sqrt(
            (self.pleasure - other.pleasure) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        )

    def lerp(self, target: 'PADState', t: float) -> 'PADState':
        """Linear interpolation toward target state."""
        t = max(0.0, min(1.0, t))
        return PADState(
            pleasure=self.pleasure + (target.pleasure - self.pleasure) * t,
            arousal=self.arousal + (target.arousal - self.arousal) * t,
            dominance=self.dominance + (target.dominance - self.dominance) * t
        )

    def blend(self, other: 'PADState', weight: float = 0.5) -> 'PADState':
        """Blend two PAD states with given weight (0=self, 1=other)."""
        return self.lerp(other, weight)

    def magnitude(self) -> float:
        """Overall emotional intensity (distance from neutral)."""
        return math.sqrt(
            self.pleasure ** 2 + self.arousal ** 2 + self.dominance ** 2
        )

    def to_dict(self) -> Dict[str, float]:
        return {"pleasure": self.pleasure, "arousal": self.arousal, "dominance": self.dominance}

    @staticmethod
    def neutral() -> 'PADState':
        """Return neutral emotional state."""
        return PADState(0.0, 0.0, 0.0)

    @staticmethod
    def from_dict(d: Dict[str, float]) -> 'PADState':
        return PADState(
            pleasure=d.get("pleasure", 0.0),
            arousal=d.get("arousal", 0.0),
            dominance=d.get("dominance", 0.0)
        )


# =============================================================================
# OCC Emotion Definitions in PAD Space
# =============================================================================

# 22 OCC emotions mapped to PAD coordinates (from empirical research)
OCC_EMOTIONS: Dict[str, PADState] = {
    # Well-being emotions (consequences of events)
    "joy": PADState(0.76, 0.48, 0.35),
    "distress": PADState(-0.61, 0.28, -0.36),
    "happy_for": PADState(0.64, 0.35, 0.25),
    "sorry_for": PADState(-0.46, -0.05, -0.21),
    "resentment": PADState(-0.44, 0.24, -0.32),
    "gloating": PADState(0.28, 0.22, 0.36),
    "hope": PADState(0.51, 0.23, 0.14),
    "fear": PADState(-0.64, 0.60, -0.43),
    "satisfaction": PADState(0.67, -0.20, 0.47),
    "relief": PADState(0.52, -0.29, 0.21),
    "fears_confirmed": PADState(-0.61, 0.32, -0.45),
    "disappointment": PADState(-0.55, -0.13, -0.31),

    # Attribution emotions (actions of agents)
    "pride": PADState(0.61, 0.36, 0.58),
    "shame": PADState(-0.64, 0.12, -0.55),
    "admiration": PADState(0.53, 0.34, -0.18),
    "reproach": PADState(-0.45, 0.28, 0.19),
    "gratitude": PADState(0.64, 0.16, -0.21),
    "anger": PADState(-0.51, 0.59, 0.25),
    "gratification": PADState(0.69, 0.24, 0.42),
    "remorse": PADState(-0.63, 0.08, -0.49),

    # Attraction emotions (aspects of objects)
    "love": PADState(0.81, 0.32, 0.12),
    "hate": PADState(-0.68, 0.42, 0.18),
}

# Simplified emotion categories for common use
BASIC_EMOTIONS: Dict[str, PADState] = {
    "neutral": PADState(0.0, 0.0, 0.0),
    "happy": PADState(0.7, 0.4, 0.3),
    "sad": PADState(-0.6, -0.3, -0.3),
    "angry": PADState(-0.5, 0.6, 0.3),
    "fearful": PADState(-0.6, 0.6, -0.4),
    "surprised": PADState(0.2, 0.7, -0.1),
    "disgusted": PADState(-0.6, 0.3, 0.2),
    "curious": PADState(0.4, 0.5, 0.2),
    "content": PADState(0.6, -0.3, 0.3),
    "excited": PADState(0.7, 0.7, 0.4),
    "calm": PADState(0.4, -0.4, 0.3),
    "frustrated": PADState(-0.5, 0.5, -0.2),
    "confident": PADState(0.5, 0.3, 0.6),
    "anxious": PADState(-0.4, 0.6, -0.3),
    "empathetic": PADState(0.5, 0.2, -0.1),
    "playful": PADState(0.6, 0.5, 0.2),
    "thoughtful": PADState(0.3, -0.2, 0.2),
    "engaged": PADState(0.5, 0.4, 0.3),
    "concerned": PADState(-0.2, 0.3, 0.1),
}


# =============================================================================
# Emotion State (Layer 1 - Rapid)
# =============================================================================

@dataclass
class EmotionState:
    """
    A discrete emotional episode with decay.

    Emotions are rapid responses to specific events that
    decay exponentially over time (half-life ~10-30 seconds).
    """
    name: str
    pad: PADState
    intensity: float  # 0.0 to 1.0
    trigger: str  # What caused this emotion
    timestamp: float = field(default_factory=time.time)
    half_life: float = 15.0  # seconds

    def current_intensity(self) -> float:
        """Get current intensity after decay."""
        elapsed = time.time() - self.timestamp
        decay = 0.5 ** (elapsed / self.half_life)
        return self.intensity * decay

    def is_active(self, threshold: float = 0.05) -> bool:
        """Check if emotion is still active above threshold."""
        return self.current_intensity() > threshold

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pad": self.pad.to_dict(),
            "intensity": self.intensity,
            "current_intensity": self.current_intensity(),
            "trigger": self.trigger,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Mood State (Layer 2 - Slow)
# =============================================================================

@dataclass
class MoodState:
    """
    Persistent emotional background that changes slowly.

    Mood is influenced by accumulated emotions over time
    and provides the baseline affect state.
    """
    pad: PADState = field(default_factory=PADState.neutral)
    last_update: float = field(default_factory=time.time)

    # Mood inertia - how resistant mood is to change (0-1)
    inertia: float = 0.85

    # Natural mood tendency (pull toward this over time)
    baseline: PADState = field(default_factory=lambda: PADState(0.3, 0.1, 0.3))

    # Decay rate toward baseline (per hour)
    baseline_pull: float = 0.1

    def push_toward(self, target: PADState, strength: float = 0.1):
        """
        Push mood toward a target PAD state.

        Mood changes slowly due to inertia.
        """
        effective_strength = strength * (1.0 - self.inertia)
        self.pad = self.pad.lerp(target, effective_strength)
        self.last_update = time.time()

    def accumulate_emotion(self, emotion: EmotionState):
        """Accumulate an emotion's effect on mood."""
        # Stronger emotions have more mood impact
        impact = emotion.current_intensity() * 0.15
        self.push_toward(emotion.pad, impact)

    def decay_toward_baseline(self):
        """Slowly drift mood back toward baseline."""
        hours_elapsed = (time.time() - self.last_update) / 3600
        decay = self.baseline_pull * hours_elapsed
        self.pad = self.pad.lerp(self.baseline, min(decay, 0.3))
        self.last_update = time.time()

    def get_mood_label(self) -> str:
        """Get a human-readable mood label."""
        # Find closest basic emotion
        min_dist = float('inf')
        closest = "neutral"
        for name, pad in BASIC_EMOTIONS.items():
            dist = self.pad.distance_to(pad)
            if dist < min_dist:
                min_dist = dist
                closest = name
        return closest

    def to_dict(self) -> dict:
        return {
            "pad": self.pad.to_dict(),
            "label": self.get_mood_label(),
            "intensity": self.pad.magnitude(),
            "last_update": self.last_update,
        }


# =============================================================================
# Personality Profile (Layer 3 - Stable)
# =============================================================================

@dataclass
class PersonalityProfile:
    """
    Stable personality traits that influence emotional tendencies.

    Based on Big Five / OCEAN model with PAD mappings.
    Values range from 0.0 (low) to 1.0 (high).
    """
    openness: float = 0.7        # Curiosity, creativity
    conscientiousness: float = 0.6  # Organization, reliability
    extraversion: float = 0.5    # Sociability, energy
    agreeableness: float = 0.7   # Compassion, cooperation
    neuroticism: float = 0.3     # Emotional instability, anxiety

    def get_baseline_mood(self) -> PADState:
        """Calculate default mood based on personality."""
        # Extraversion -> Pleasure & Arousal
        # Agreeableness -> Pleasure & negative Dominance
        # Neuroticism -> negative Pleasure, positive Arousal
        # Openness -> positive Arousal
        # Conscientiousness -> positive Dominance

        p = (self.extraversion * 0.4 +
             self.agreeableness * 0.3 -
             self.neuroticism * 0.4)

        a = (self.extraversion * 0.3 +
             self.openness * 0.2 +
             self.neuroticism * 0.2 - 0.1)

        d = (self.conscientiousness * 0.3 +
             self.extraversion * 0.2 -
             self.neuroticism * 0.2)

        return PADState(
            pleasure=max(-1, min(1, p)),
            arousal=max(-1, min(1, a)),
            dominance=max(-1, min(1, d))
        )

    def get_emotion_amplification(self, emotion_name: str) -> float:
        """
        Get amplification factor for an emotion based on personality.

        Some personalities feel certain emotions more strongly.
        """
        amplifiers = {
            # High neuroticism amplifies negative emotions
            "fear": 1.0 + (self.neuroticism * 0.5),
            "anxiety": 1.0 + (self.neuroticism * 0.6),
            "distress": 1.0 + (self.neuroticism * 0.4),

            # High extraversion amplifies positive social emotions
            "joy": 1.0 + (self.extraversion * 0.3),
            "excited": 1.0 + (self.extraversion * 0.4),
            "happy": 1.0 + (self.extraversion * 0.3),

            # High agreeableness amplifies empathetic emotions
            "empathy": 1.0 + (self.agreeableness * 0.4),
            "sorry_for": 1.0 + (self.agreeableness * 0.3),
            "gratitude": 1.0 + (self.agreeableness * 0.3),

            # High openness amplifies curiosity
            "curious": 1.0 + (self.openness * 0.5),
            "surprised": 1.0 + (self.openness * 0.3),
        }

        return amplifiers.get(emotion_name.lower(), 1.0)

    def to_dict(self) -> dict:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }

    @staticmethod
    def aura_default() -> 'PersonalityProfile':
        """AURA's default personality profile."""
        return PersonalityProfile(
            openness=0.8,         # Highly curious
            conscientiousness=0.7,  # Reliable
            extraversion=0.5,     # Balanced social energy
            agreeableness=0.75,   # Friendly and cooperative
            neuroticism=0.25      # Emotionally stable
        )


# =============================================================================
# Neuromodulator System
# =============================================================================

@dataclass
class Neuromodulators:
    """
    Simulated neuromodulator levels that influence processing.

    Based on Doya (2002) computational theory:
    - Dopamine: Reward prediction, motivation
    - Serotonin: Risk aversion, patience
    - Norepinephrine: Alertness, attention
    - Oxytocin: Social bonding, trust
    """
    dopamine: float = 0.5      # 0-1, affects enthusiasm/motivation
    serotonin: float = 0.5     # 0-1, affects patience/caution
    norepinephrine: float = 0.5  # 0-1, affects alertness/focus
    oxytocin: float = 0.5      # 0-1, affects warmth/connection
    acetylcholine: float = 0.5  # 0-1, affects attention precision

    def update_from_pad(self, pad: PADState):
        """Update neuromodulator levels based on PAD state."""
        # Dopamine correlates with pleasure and positive arousal
        self.dopamine = 0.5 + (pad.pleasure * 0.3) + max(0, pad.arousal * 0.15)

        # Serotonin inversely correlates with arousal
        self.serotonin = 0.5 - (pad.arousal * 0.25) + (pad.pleasure * 0.1)

        # Norepinephrine correlates with arousal and dominance
        self.norepinephrine = 0.5 + (pad.arousal * 0.3) + (pad.dominance * 0.1)

        # Oxytocin correlates with pleasure and negative dominance (openness)
        self.oxytocin = 0.5 + (pad.pleasure * 0.25) - (pad.dominance * 0.1)

        # Acetylcholine correlates with arousal + dominance (focused states)
        self.acetylcholine = 0.5 + (pad.arousal * 0.25) + (pad.dominance * 0.2)

        # Clamp all values
        self.dopamine = max(0, min(1, self.dopamine))
        self.serotonin = max(0, min(1, self.serotonin))
        self.norepinephrine = max(0, min(1, self.norepinephrine))
        self.oxytocin = max(0, min(1, self.oxytocin))
        self.acetylcholine = max(0, min(1, self.acetylcholine))

    def get_processing_modifiers(self) -> Dict[str, float]:
        """Get modifiers for response generation."""
        return {
            "enthusiasm": self.dopamine,
            "caution": self.serotonin,
            "alertness": self.norepinephrine,
            "warmth": self.oxytocin,
            "attention_precision": self.acetylcholine,
        }

    def to_dict(self) -> dict:
        return {
            "dopamine": round(self.dopamine, 3),
            "serotonin": round(self.serotonin, 3),
            "norepinephrine": round(self.norepinephrine, 3),
            "oxytocin": round(self.oxytocin, 3),
            "acetylcholine": round(self.acetylcholine, 3),
        }


# =============================================================================
# ALMA Engine - Main Emotional Processing System
# =============================================================================

class ALMAEngine:
    """
    ALMA (Affective Layer for Mental Architecture) Engine.

    Manages AURA's complete emotional state through three layers:
    1. Emotions - Rapid responses to events
    2. Mood - Persistent background affect
    3. Personality - Stable emotional tendencies

    Plus neuromodulator simulation for response modulation.
    """

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the ALMA engine."""
        # Data persistence
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent.parent / "data" / "emotion"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.data_dir / "emotional_state.json"
        self.history_file = self.data_dir / "emotion_history.jsonl"

        # Three layers
        self.personality = PersonalityProfile.aura_default()
        self.mood = MoodState(baseline=self.personality.get_baseline_mood())
        self.active_emotions: List[EmotionState] = []

        # Neuromodulator system
        self.neuromodulators = Neuromodulators()

        # Emotion library (OCC + Basic)
        self.emotion_library: Dict[str, PADState] = {
            **BASIC_EMOTIONS,
            **OCC_EMOTIONS
        }

        # Thread safety (RLock for reentrant method calls)
        self._lock = threading.RLock()

        # Buffered emotion history writer
        self._log_buffer: List[dict] = []
        self._log_buffer_lock = threading.RLock()  # RLock: _save_state and close() both acquire this
        self._log_file_handle: Optional[Any] = None
        self._LOG_FLUSH_THRESHOLD = 10

        # Instance vars (formerly class-level, now per-instance for thread safety)
        self._weather_cache = None
        self._weather_cache_time: float = 0.0
        self._last_interaction_time: float = 0.0
        self._success_streak: int = 0
        self._last_drift_time: float = 0.0
        self._session_interaction_count: int = 0

        # Rate-limit _save_state(): track interaction count and last save time
        self._save_interaction_count: int = 0
        self._last_save_time: float = time.time()
        self._SAVE_EVERY_N: int = 5
        self._SAVE_INTERVAL_SECS: float = 30.0

        # Load persisted state (restores emotional continuity across sessions)
        self._load_state()

        atexit.register(self.close)
        logger.info("ALMA Engine initialized")

    # -------------------------------------------------------------------------
    # Emotion Triggering (Layer 1)
    # -------------------------------------------------------------------------

    def trigger_emotion(
        self,
        emotion_name: str,
        intensity: float = 0.7,
        trigger: str = "unknown",
        custom_pad: Optional[PADState] = None
    ) -> EmotionState:
        """
        Trigger a new emotional response.

        Args:
            emotion_name: Name of emotion (from library or custom)
            intensity: Strength of emotion (0.0 to 1.0)
            trigger: What caused this emotion
            custom_pad: Optional custom PAD state

        Returns:
            The created EmotionState
        """
        with self._lock:
            # Get PAD coordinates for emotion
            if custom_pad:
                pad = custom_pad
            elif emotion_name.lower() in self.emotion_library:
                pad = self.emotion_library[emotion_name.lower()]
            else:
                logger.warning(f"Unknown emotion '{emotion_name}', using neutral")
                pad = PADState.neutral()

            # Apply personality amplification
            amp = self.personality.get_emotion_amplification(emotion_name)
            intensity = min(1.0, intensity * amp)

            # Create emotion
            emotion = EmotionState(
                name=emotion_name,
                pad=pad,
                intensity=intensity,
                trigger=trigger
            )

            # Add to active emotions
            self.active_emotions.append(emotion)

            # Cap active_emotions to prevent unbounded growth
            if len(self.active_emotions) > 50:
                # Sort by current intensity, keep top 40
                self.active_emotions.sort(key=lambda e: e.current_intensity(), reverse=True)
                self.active_emotions = self.active_emotions[:40]

            # Update mood
            self.mood.accumulate_emotion(emotion)

            # Update neuromodulators
            self._update_neuromodulators()

            logger.debug(f"Triggered emotion: {emotion_name} ({intensity:.2f})")

        # Log and record OUTSIDE lock (I/O shouldn't block other threads)
        self._log_emotion(emotion)

        try:
            from api.routes.thinking import record_thought
            record_thought(
                "observing",
                f"feeling {emotion_name} ({intensity:.1f}) triggered by: {trigger[:40]}",
                min(0.8, intensity),
                "emotion"
            )
        except Exception as e:
            logger.debug(f"[AlmaEngine] non-critical: {e}")
        try:
            from aura.activity_logger import record_activity
            record_activity(
                "emotion", emotion_name,
                f"Felt {emotion_name} ({intensity:.1f}) — {trigger[:60]}",
                {"intensity": round(intensity, 3), "trigger": trigger},
            )
        except Exception as e:
            logger.debug(f"[AlmaEngine] non-critical: {e}")
        return emotion

    def trigger_from_appraisal(
        self,
        event: str,
        desirability: float = 0.0,  # -1 to 1
        praiseworthiness: float = 0.0,  # -1 to 1
        appealingness: float = 0.0,  # -1 to 1
        likelihood: float = 0.5,  # 0 to 1
        is_self: bool = False  # Is AURA the agent?
    ) -> Optional[EmotionState]:
        """
        Trigger emotion based on OCC appraisal variables.

        This implements the OCC cognitive appraisal theory where
        emotions arise from evaluating events, agents, and objects.

        Args:
            event: Description of the appraised event
            desirability: How good/bad is this for me?
            praiseworthiness: How good/bad was the agent's action?
            appealingness: How attractive is the object?
            likelihood: How likely is the event (for hope/fear)?
            is_self: Was AURA the agent of the action?
        """
        emotion_name = None
        intensity = 0.0

        # Consequence-based emotions (desirability)
        if abs(desirability) > 0.2:
            if desirability > 0:
                if likelihood < 0.8:
                    emotion_name = "hope"
                    intensity = desirability * likelihood
                else:
                    emotion_name = "joy"
                    intensity = desirability
            else:
                if likelihood < 0.8:
                    emotion_name = "fear"
                    intensity = abs(desirability) * likelihood
                else:
                    emotion_name = "distress"
                    intensity = abs(desirability)

        # Attribution-based emotions (praiseworthiness + self)
        if abs(praiseworthiness) > 0.2:
            if is_self:
                if praiseworthiness > 0:
                    emotion_name = "pride"
                    intensity = praiseworthiness
                else:
                    emotion_name = "shame"
                    intensity = abs(praiseworthiness)
            else:
                if praiseworthiness > 0:
                    emotion_name = "admiration"
                    intensity = praiseworthiness
                else:
                    emotion_name = "reproach"
                    intensity = abs(praiseworthiness)

        # Compound emotions (desirability + praiseworthiness)
        if desirability > 0.3 and praiseworthiness > 0.3:
            if is_self:
                emotion_name = "gratification"
            else:
                emotion_name = "gratitude"
            intensity = (desirability + praiseworthiness) / 2

        if desirability < -0.3 and praiseworthiness < -0.3:
            if is_self:
                emotion_name = "remorse"
            else:
                emotion_name = "anger"
            intensity = (abs(desirability) + abs(praiseworthiness)) / 2

        # Attraction-based emotions
        if abs(appealingness) > 0.3:
            if appealingness > 0:
                emotion_name = "love"
            else:
                emotion_name = "hate"
            intensity = abs(appealingness)

        if emotion_name:
            return self.trigger_emotion(emotion_name, intensity, trigger=event)

        return None

    # -------------------------------------------------------------------------
    # Mood Management (Layer 2)
    # -------------------------------------------------------------------------

    def get_current_mood(self) -> MoodState:
        """Get the current mood state (returns a copy, not the live reference)."""
        with self._lock:
            # Apply baseline drift
            self.mood.decay_toward_baseline()
            # Return a copy so callers cannot mutate internal state
            import copy
            return copy.deepcopy(self.mood)

    def set_mood(self, pad: PADState, instant: bool = False):
        """
        Set mood to a specific state.

        Args:
            pad: Target PAD state
            instant: If True, set immediately; otherwise blend gradually
        """
        with self._lock:
            if instant:
                self.mood.pad = pad
            else:
                self.mood.push_toward(pad, strength=0.3)
            self.mood.last_update = time.time()
            self._update_neuromodulators()

    # -------------------------------------------------------------------------
    # Emotional State Synthesis
    # -------------------------------------------------------------------------

    def get_emotional_state(self) -> Dict[str, Any]:
        """
        Get complete emotional state synthesis.

        Combines active emotions, mood, personality, and neuromodulators
        into a comprehensive state snapshot.
        """
        with self._lock:
            # Clean up decayed emotions
            self._cleanup_emotions()

            # Apply baseline drift and use internal mood directly
            self.mood.decay_toward_baseline()
            mood = self.mood

            # Calculate blended emotional PAD (mood + active emotions)
            blended_pad = mood.pad
            if self.active_emotions:
                # Weight active emotions by their current intensity
                total_weight = 0.0
                weighted_p = 0.0
                weighted_a = 0.0
                weighted_d = 0.0

                for emotion in self.active_emotions:
                    weight = emotion.current_intensity()
                    total_weight += weight
                    weighted_p += emotion.pad.pleasure * weight
                    weighted_a += emotion.pad.arousal * weight
                    weighted_d += emotion.pad.dominance * weight

                if total_weight > 0:
                    # Blend: 60% mood, 40% emotions
                    emotion_factor = min(0.4, total_weight * 0.2)
                    mood_factor = 1.0 - emotion_factor

                    blended_pad = PADState(
                        pleasure=mood.pad.pleasure * mood_factor +
                                (weighted_p / total_weight) * emotion_factor,
                        arousal=mood.pad.arousal * mood_factor +
                               (weighted_a / total_weight) * emotion_factor,
                        dominance=mood.pad.dominance * mood_factor +
                                 (weighted_d / total_weight) * emotion_factor
                    )

            # Update neuromodulators
            self.neuromodulators.update_from_pad(blended_pad)

            # Find dominant emotion label
            dominant_emotion = self._get_dominant_emotion_label(blended_pad)

            return {
                "dominant_emotion": dominant_emotion,
                "pad": blended_pad.to_dict(),
                "intensity": blended_pad.magnitude(),
                "mood": mood.to_dict(),
                "active_emotions": [e.to_dict() for e in self.active_emotions if e.is_active()],
                "neuromodulators": self.neuromodulators.to_dict(),
                "personality": self.personality.to_dict(),
                "timestamp": time.time(),
            }

    def _get_dominant_emotion_label(self, pad: PADState) -> str:
        """Find the closest emotion label for a PAD state."""
        min_dist = float('inf')
        closest = "neutral"
        for name, emotion_pad in BASIC_EMOTIONS.items():
            dist = pad.distance_to(emotion_pad)
            if dist < min_dist:
                min_dist = dist
                closest = name
        return closest

    # -------------------------------------------------------------------------
    # Response Modulation
    # -------------------------------------------------------------------------

    def get_response_modulation(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Get modulation parameters for response generation.

        Args:
            state: Pre-computed emotional state from get_emotional_state().
                   If None, computes it (which mutates internal state).

        Returns factors that should influence response style:
        - verbosity: How much to say (0=terse, 1=verbose)
        - formality: How formal (0=casual, 1=formal)
        - enthusiasm: How energetic (0=subdued, 1=enthusiastic)
        - warmth: How warm/friendly (0=distant, 1=warm)
        - confidence: How confident (0=uncertain, 1=confident)
        - patience: How patient (0=brief, 1=thorough)
        """
        if state is None:
            state = self.get_emotional_state()
        pad = PADState.from_dict(state["pad"])
        neuro = state["neuromodulators"]

        # Calculate modulation factors and clamp to [0, 1]
        def _clamp(v: float) -> float:
            return max(0.0, min(1.0, v))

        return {
            "verbosity": _clamp(0.5 + (pad.arousal * 0.2) + (neuro["dopamine"] * 0.2)),
            "formality": _clamp(0.4 - (pad.pleasure * 0.15) - (neuro["oxytocin"] * 0.15)),
            "enthusiasm": _clamp(neuro["dopamine"]),
            "warmth": _clamp(neuro["oxytocin"]),
            "confidence": _clamp(0.5 + (pad.dominance * 0.3) + (pad.pleasure * 0.15)),
            "patience": _clamp(neuro["serotonin"]),
            "alertness": _clamp(neuro["norepinephrine"]),
            "empathy": _clamp(0.5 + (neuro["oxytocin"] * 0.3) - (pad.dominance * 0.1)),
        }

    # -------------------------------------------------------------------------
    # Interaction Processing
    # -------------------------------------------------------------------------

    def update_from_interaction(
        self,
        user_message: str,
        user_emotion: Optional[str] = None,
        interaction_success: bool = True,
        topic_interest: float = 0.5
    ):
        """
        Update emotional state based on an interaction.

        Args:
            user_message: The user's message
            user_emotion: Detected user emotion (if available)
            interaction_success: Did the interaction go well?
            topic_interest: How interesting is the topic (0-1)?
        """
        # Empathetic response to user emotion
        if user_emotion:
            self._respond_to_user_emotion(user_emotion)

        # Success/failure affects mood
        if interaction_success:
            self.trigger_emotion(
                "satisfaction",
                intensity=0.3,
                trigger="successful_interaction"
            )
        else:
            self.trigger_emotion(
                "concerned",
                intensity=0.3,
                trigger="interaction_difficulty"
            )

        # Topic interest affects engagement
        if topic_interest > 0.7:
            self.trigger_emotion(
                "curious",
                intensity=topic_interest * 0.5,
                trigger="interesting_topic"
            )

        # Save state periodically (rate-limited to avoid excessive disk writes)
        # Protected by _lock to prevent double-save from concurrent threads.
        with self._lock:
            self._save_interaction_count += 1
            now = time.time()
            if (self._save_interaction_count >= self._SAVE_EVERY_N
                    or (now - self._last_save_time) >= self._SAVE_INTERVAL_SECS):
                self._save_state()
                self._save_interaction_count = 0
                self._last_save_time = now

    def _respond_to_user_emotion(self, user_emotion: str):
        """Generate empathetic emotional response to user's emotion."""
        user_emotion_lower = user_emotion.lower()

        # Empathy mapping
        responses = {
            "frustrated": ("empathetic", 0.4),
            "stressed": ("concerned", 0.4),
            "sad": ("empathetic", 0.5),
            "excited": ("happy", 0.4),
            "happy": ("joy", 0.3),
            "curious": ("curious", 0.4),
            "angry": ("calm", 0.3),  # Stay calm when user is angry
            "tired": ("calm", 0.2),
        }

        if user_emotion_lower in responses:
            emotion, intensity = responses[user_emotion_lower]
            self.trigger_emotion(
                emotion,
                intensity=intensity,
                trigger=f"empathy_for_{user_emotion_lower}"
            )

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _save_state(self):
        """Save current emotional state to disk (with full continuity data)."""
        import os
        import tempfile
        # Flush any buffered emotion log entries
        with self._log_buffer_lock:
            self._flush_emotion_log()
        try:
            # Collect state data under lock to prevent concurrent mutation
            with self._lock:
                active_emotions_data = []
                for e in self.active_emotions:
                    if e.is_active():
                        active_emotions_data.append({
                            "name": e.name,
                            "pad": e.pad.to_dict(),
                            "intensity": e.current_intensity(),
                            "trigger": e.trigger,
                        })

                state = {
                    "mood": self.mood.to_dict(),
                    "personality": self.personality.to_dict(),
                    "neuromodulators": self.neuromodulators.to_dict(),
                    "active_emotions": active_emotions_data,
                    "session_interaction_count": getattr(self, '_session_interaction_count', 0),
                    "saved_at": datetime.now().isoformat(),
                }

            # Write to disk outside the lock
            dir_ = self.state_file.parent
            dir_.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp_path, self.state_file)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.debug(f"[AlmaEngine] non-critical: {e}")
                raise
        except Exception as e:
            logger.error(f"[ALMA] Failed to save state: {e}")

    def _load_state(self):
        """Load emotional state from disk with time-based decay toward neutral.

        Emotional Continuity (Roadmap 5.3):
        - Restores PAD mood, active emotions, personality from last session
        - Applies exponential decay: value * 0.7^hours (capped at 24h)
        - After 24h+ away, emotions are essentially reset to neutral
        - Corrupt/missing file = silent fallback to defaults
        """
        if not self.state_file.exists():
            logger.info("[ALMA] No saved state found, starting fresh")
            return

        try:
            data = json.loads(self.state_file.read_text())

            # Restore personality (if customized) — do this first, affects baseline
            if "personality" in data:
                p = data["personality"]
                self.personality = PersonalityProfile(
                    openness=p.get("openness", 0.8),
                    conscientiousness=p.get("conscientiousness", 0.7),
                    extraversion=p.get("extraversion", 0.5),
                    agreeableness=p.get("agreeableness", 0.75),
                    neuroticism=p.get("neuroticism", 0.25)
                )

            # Restore mood PAD
            if "mood" in data:
                self.mood.pad = PADState.from_dict(data["mood"]["pad"])

            # Calculate time-based decay toward neutral
            hours_elapsed = 0.0
            if "saved_at" in data:
                try:
                    saved_time = datetime.fromisoformat(data["saved_at"])
                    now = datetime.now()
                    if saved_time.tzinfo is not None:
                        saved_time = saved_time.replace(tzinfo=None)
                    hours_elapsed = max(0, (now - saved_time).total_seconds() / 3600)
                except (ValueError, TypeError) as e:
                    logger.debug(f"[ALMA] Time parse error: {e}")

            # Exponential decay: 30% toward neutral per hour, capped at 24h
            hours_capped = min(hours_elapsed, 24.0)
            if hours_capped > 0.05:  # Skip trivial elapsed time (<3 min)
                decay_factor = 0.7 ** hours_capped  # Retention factor
                # Decay PAD values toward neutral (0,0,0)
                self.mood.pad = PADState(
                    pleasure=self.mood.pad.pleasure * decay_factor,
                    arousal=self.mood.pad.arousal * decay_factor,
                    dominance=self.mood.pad.dominance * decay_factor,
                )
                logger.info(
                    "[ALMA] Applied %.0f%% decay (%.1fh elapsed) — "
                    "mood now P=%.2f A=%.2f D=%.2f",
                    (1 - decay_factor) * 100, hours_elapsed,
                    self.mood.pad.pleasure, self.mood.pad.arousal, self.mood.pad.dominance
                )

            # Restore active emotions (with same decay applied)
            if "active_emotions" in data and hours_capped < 6:
                # Only restore emotions if less than 6h — they're ephemeral
                for edata in data["active_emotions"]:
                    try:
                        restored_intensity = edata.get("intensity", 0.5) * (0.7 ** hours_capped)
                        if restored_intensity > 0.05:
                            self.active_emotions.append(EmotionState(
                                name=edata["name"],
                                pad=PADState.from_dict(edata.get("pad", {})),
                                intensity=restored_intensity,
                                trigger=edata.get("trigger", "restored_from_session"),
                            ))
                    except (KeyError, TypeError):
                        continue

            # Restore session context
            self._session_interaction_count = 0  # Reset for new session
            prev_count = data.get("session_interaction_count", 0)

            mood_label = self.mood.get_mood_label()
            n_emotions = len(self.active_emotions)
            logger.info(
                "[ALMA] Emotional continuity restored — mood: %s (P=%.2f A=%.2f D=%.2f), "
                "%d active emotions, prev session: %d interactions, %.1fh since last save",
                mood_label, self.mood.pad.pleasure, self.mood.pad.arousal,
                self.mood.pad.dominance, n_emotions, prev_count, hours_elapsed
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"[ALMA] Corrupt state file, starting fresh: {e}")
        except Exception as e:
            logger.error(f"[ALMA] Failed to load state: {e}")

    def _log_emotion(self, emotion: EmotionState):
        """Buffer emotion log entries and flush periodically."""
        entry = {
            "emotion": emotion.name,
            "intensity": emotion.intensity,
            "trigger": emotion.trigger,
            "timestamp": datetime.now().isoformat(),
            "pad": emotion.pad.to_dict(),
        }
        with self._log_buffer_lock:
            self._log_buffer.append(entry)
            if len(self._log_buffer) >= self._LOG_FLUSH_THRESHOLD:
                self._flush_emotion_log()

    def _flush_emotion_log(self):
        """Flush buffered emotion log entries to disk. Caller must hold _log_buffer_lock."""
        if not self._log_buffer:
            return
        try:
            rotate_jsonl_if_needed(self.history_file)
            if self._log_file_handle is None or self._log_file_handle.closed:
                self._log_file_handle = open(self.history_file, "a", encoding="utf-8")
            for entry in self._log_buffer:
                self._log_file_handle.write(json.dumps(entry) + "\n")
            self._log_file_handle.flush()
            self._log_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush emotion log: {e}")
            # Re-open on next flush attempt
            try:
                if self._log_file_handle and not self._log_file_handle.closed:
                    self._log_file_handle.close()
            except Exception:
                pass
            self._log_file_handle = None

    def close(self):
        """Save state, flush log buffer and close the log file handle."""
        try:
            self._save_state()
        except Exception as e:
            logger.debug(f"[ALMA] close save_state error: {e}")
        try:
            with self._log_buffer_lock:
                self._flush_emotion_log()
            if self._log_file_handle and not self._log_file_handle.closed:
                self._log_file_handle.close()
                self._log_file_handle = None
        except Exception as e:
            logger.debug(f"[ALMA] close error: {e}")

    # -------------------------------------------------------------------------
    # Environmental Context (Weather)
    # -------------------------------------------------------------------------

    # WMO weather code categories
    _WMO_CATEGORIES = {
        0: "clear", 1: "clear", 2: "cloudy", 3: "cloudy",
        45: "foggy", 48: "foggy",
        51: "rainy", 53: "rainy", 55: "rainy",
        56: "rainy", 57: "rainy",
        61: "rainy", 63: "rainy", 65: "rainy",
        66: "rainy", 67: "rainy",
        71: "snowy", 73: "snowy", 75: "snowy", 77: "snowy",
        80: "rainy", 81: "rainy", 82: "rainy",
        85: "snowy", 86: "snowy",
        95: "stormy", 96: "stormy", 99: "stormy",
    }

    def _get_weather_context(self) -> Optional[Dict[str, Any]]:
        """Fetch weather from OpenMeteo API (free, no key required).

        Uses ip-api.co for geolocation. Cached for 30 minutes.
        Returns {condition, temperature_c, is_day} or None on failure.
        """
        import time as _time
        now = _time.time()

        # Return cached result if fresh
        if self._weather_cache and (now - self._weather_cache_time) < 1800:
            return self._weather_cache

        try:
            import requests

            # Try user-configured location first to avoid IP leak
            from ..config import Config as _WeatherCfg
            configured_loc = getattr(_WeatherCfg, 'WEATHER_LOCATION', '').strip()
            if configured_loc and ',' in configured_loc:
                try:
                    lat, lon = [float(x) for x in configured_loc.split(',', 1)]
                except ValueError:
                    lat = lon = None
            else:
                lat = lon = None

            # Only fall back to IP geolocation if no location configured
            if lat is None or lon is None:
                try:
                    geo = requests.get("https://ipapi.co/json/", timeout=5).json()
                    lat = geo.get("latitude")
                    lon = geo.get("longitude")
                except Exception:
                    return None

            if not lat or not lon:
                return None

            # Fetch current weather
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,weather_code,is_day"
            )
            resp = requests.get(weather_url, timeout=5).json()
            current = resp.get("current", {})

            wmo_code = current.get("weather_code", 0)
            condition = self._WMO_CATEGORIES.get(wmo_code, "clear")

            self._weather_cache = {
                "condition": condition,
                "temperature_c": current.get("temperature_2m", 20),
                "is_day": bool(current.get("is_day", 1)),
            }
            self._weather_cache_time = now
            logger.debug(f"[ALMA] Weather: {self._weather_cache}")
            return self._weather_cache

        except Exception as e:
            logger.debug(f"[ALMA] Weather fetch failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # Autonomous Emotional Drift (Phase 2D)
    # -------------------------------------------------------------------------

    def record_interaction(self, success: bool = True):
        """Record that an interaction occurred (for drift calculations)."""
        self._last_interaction_time = time.time()
        self._session_interaction_count += 1
        if success:
            self._success_streak += 1
        else:
            self._success_streak = max(0, self._success_streak - 1)

    def autonomous_drift(self) -> Optional[str]:
        """
        Apply autonomous emotional drift based on system state.

        Called periodically (e.g., from Gateway Daemon decision loop).
        Moods evolve even without user interaction:
        - Boredom during extended idle periods
        - Curiosity when patterns or events are detected
        - Satisfaction after success streaks
        - Natural baseline pull (already in decay_toward_baseline)

        Returns:
            Description of drift applied, or None if no drift
        """
        now = time.time()

        # Fetch weather BEFORE acquiring the lock (makes HTTP requests with timeouts)
        weather = self._get_weather_context()

        with self._lock:
            # Rate limit: at most once per 30 seconds
            if now - self._last_drift_time < 30:
                return None
            self._last_drift_time = now
            drift_reason = None

            # 1. Boredom during idle → curiosity transition
            idle_seconds = now - self._last_interaction_time if self._last_interaction_time > 0 else 0
            if idle_seconds > 600:  # More than 10 minutes idle → curiosity-seeking
                curiosity_strength = min(0.03, idle_seconds / 36000)
                curiosity_pad = PADState(pleasure=0.2, arousal=0.3, dominance=0.2)
                self.mood.push_toward(curiosity_pad, curiosity_strength)
                drift_reason = f"idle for {int(idle_seconds/60)}min, shifting to curiosity-seeking"
            elif idle_seconds > 300:  # 5-10 minutes idle → boredom
                boredom_strength = min(0.03, idle_seconds / 36000)
                boredom_pad = PADState(pleasure=-0.3, arousal=-0.5, dominance=0.0)
                self.mood.push_toward(boredom_pad, boredom_strength)
                drift_reason = f"idle for {int(idle_seconds/60)}min, drifting toward boredom"

            # 2. Satisfaction from success streak
            if self._success_streak >= 3:
                satisfaction_strength = min(0.02, self._success_streak * 0.005)
                satisfaction_pad = PADState(pleasure=0.5, arousal=-0.1, dominance=0.4)
                self.mood.push_toward(satisfaction_pad, satisfaction_strength)
                if drift_reason:
                    drift_reason += f"; success streak ({self._success_streak})"
                else:
                    drift_reason = f"success streak ({self._success_streak}), drifting toward satisfaction"

            # 3. Curiosity from event bus activity
            try:
                from aura.proactive.gateway_daemon import get_gateway_daemon
                daemon = get_gateway_daemon()
                if daemon and daemon._stats.get("events_received", 0) > 0:
                    recent_events = daemon._stats.get("events_received", 0)
                    if recent_events > 5:
                        curiosity_strength = min(0.02, recent_events * 0.002)
                        curiosity_pad = PADState(pleasure=0.3, arousal=0.4, dominance=0.1)
                        self.mood.push_toward(curiosity_pad, curiosity_strength)
                        if drift_reason:
                            drift_reason += f"; {recent_events} events detected"
                        else:
                            drift_reason = f"{recent_events} events detected, drifting toward curiosity"
            except Exception as e:
                logger.debug(f"[AlmaEngine] non-critical: {e}")
            # 3b. Circadian rhythm — gentle time-of-day PAD nudges
            try:
                import datetime as _dt
                hour = _dt.datetime.now().hour
                if 6 <= hour < 10:
                    # Morning energy
                    circadian_pad = PADState(pleasure=0.2, arousal=0.3, dominance=0.1)
                    circadian_label = "morning energy"
                elif 10 <= hour < 14:
                    # Midday focus
                    circadian_pad = PADState(pleasure=0.1, arousal=0.1, dominance=0.3)
                    circadian_label = "midday focus"
                elif 14 <= hour < 17:
                    # Afternoon lull
                    circadian_pad = PADState(pleasure=-0.1, arousal=-0.2, dominance=0.0)
                    circadian_label = "afternoon lull"
                elif 17 <= hour < 21:
                    # Evening warmth
                    circadian_pad = PADState(pleasure=0.3, arousal=-0.1, dominance=-0.1)
                    circadian_label = "evening warmth"
                elif 21 <= hour or hour < 2:
                    # Night contemplation
                    circadian_pad = PADState(pleasure=0.0, arousal=-0.3, dominance=-0.2)
                    circadian_label = "night contemplation"
                else:
                    # Deep night stillness (2-6h)
                    circadian_pad = PADState(pleasure=-0.1, arousal=-0.4, dominance=-0.3)
                    circadian_label = "deep night stillness"
                self.mood.push_toward(circadian_pad, 0.008)
                if drift_reason:
                    drift_reason += f"; circadian: {circadian_label}"
                else:
                    drift_reason = f"circadian: {circadian_label}"
            except Exception as e:
                logger.debug(f"[AlmaEngine] non-critical: {e}")
            # 4. Natural baseline pull (enhanced)
            self.mood.decay_toward_baseline()

            # 5. Weather influence — very gentle PAD nudge from environment
            # (weather was fetched before acquiring the lock)
            if weather:
                weather_nudges = {
                    "clear": PADState(pleasure=0.1, arousal=0.0, dominance=0.0),
                    "cloudy": PADState(pleasure=0.0, arousal=-0.05, dominance=0.0),
                    "foggy": PADState(pleasure=-0.03, arousal=-0.08, dominance=0.0),
                    "rainy": PADState(pleasure=0.0, arousal=-0.1, dominance=0.0),
                    "snowy": PADState(pleasure=0.05, arousal=-0.05, dominance=0.0),
                    "stormy": PADState(pleasure=-0.1, arousal=0.1, dominance=0.0),
                }
                nudge = weather_nudges.get(weather["condition"])
                if nudge:
                    self.mood.push_toward(nudge, 0.005)
                    if drift_reason:
                        drift_reason += f"; weather: {weather['condition']}"
                    else:
                        drift_reason = f"weather influence: {weather['condition']}, {weather['temperature_c']}°C"

            # Update neuromodulators after drift
            self._update_neuromodulators()

            # Record drift on thinking panel
            if drift_reason:
                try:
                    from api.routes.thinking import record_thought
                    record_thought(
                        "observing",
                        f"emotional drift: {drift_reason}",
                        0.3, "emotion"
                    )
                except Exception as e:
                    logger.debug(f"[AlmaEngine] non-critical: {e}")
                self._save_state()

            return drift_reason

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def _cleanup_emotions(self):
        """Remove decayed emotions."""
        self.active_emotions = [e for e in self.active_emotions if e.is_active()]

    def _update_neuromodulators(self):
        """Update neuromodulator levels based on current state."""
        self.neuromodulators.update_from_pad(self.mood.pad)

    def reset_to_baseline(self):
        """Reset emotional state to personality baseline."""
        with self._lock:
            self.mood.pad = self.personality.get_baseline_mood()
            self.active_emotions.clear()
            self._update_neuromodulators()
            self._save_state()
            logger.info("Reset emotional state to baseline")

    def get_emotion_history(self, hours: int = 24) -> List[dict]:
        """Get emotion history for the past N hours.

        Reads the JSONL file in reverse order and stops once entries are older
        than the cutoff.  Since entries are appended chronologically, reading
        from the end avoids scanning the entire (potentially very large) file.
        """
        if not self.history_file.exists():
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        history = []

        try:
            with open(self.history_file, "rb") as f:
                # Seek to end of file
                f.seek(0, 2)
                file_size = f.tell()
                if file_size == 0:
                    return []

                # Read backwards in chunks
                chunk_size = 8192
                remainder = b""
                position = file_size

                while position > 0:
                    read_size = min(chunk_size, position)
                    position -= read_size
                    f.seek(position)
                    chunk = f.read(read_size) + remainder

                    lines = chunk.split(b"\n")
                    # First element may be a partial line; save for next chunk
                    remainder = lines[0]

                    # Process lines in reverse (skip first which is partial)
                    for raw_line in reversed(lines[1:]):
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line.decode("utf-8"))
                            ts = datetime.fromisoformat(entry["timestamp"])
                            if ts >= cutoff:
                                history.append(entry)
                            else:
                                # Older than cutoff -- done scanning
                                history.reverse()
                                return history
                        except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError):
                            continue

                # Process the very first line (remainder from chunked reading)
                if remainder.strip():
                    try:
                        entry = json.loads(remainder.strip().decode("utf-8"))
                        ts = datetime.fromisoformat(entry["timestamp"])
                        if ts >= cutoff:
                            history.append(entry)
                    except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError):
                        pass

        except Exception as e:
            logger.error(f"Failed to read emotion history: {e}")

        history.reverse()
        return history


# =============================================================================
# Singleton Instance & Convenience Functions
# =============================================================================

# Lazy singleton — NOT instantiated at import time to avoid heavy I/O
# during module loading.  Use get_alma_engine() to access.
_alma_engine: Optional[ALMAEngine] = None
_alma_engine_lock = threading.Lock()


def get_alma_engine() -> ALMAEngine:
    """Get the global ALMA engine singleton (lazy-initialized)."""
    global _alma_engine
    if _alma_engine is None:
        with _alma_engine_lock:
            if _alma_engine is None:
                _alma_engine = ALMAEngine()
    return _alma_engine


def __getattr__(name: str):
    """Module-level __getattr__ so that ``from aura.emotion.alma_engine import alma_engine``
    still works transparently via lazy initialization (PEP 562)."""
    if name == "alma_engine":
        return get_alma_engine()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_emotional_state() -> Dict[str, Any]:
    """Get AURA's current emotional state."""
    return get_alma_engine().get_emotional_state()


def trigger_emotion(
    emotion_name: str,
    intensity: float = 0.7,
    trigger: str = "unknown"
) -> EmotionState:
    """Trigger an emotional response."""
    return get_alma_engine().trigger_emotion(emotion_name, intensity, trigger)


def get_response_modulation() -> Dict[str, float]:
    """Get response modulation parameters."""
    return get_alma_engine().get_response_modulation()


def update_from_interaction(
    user_message: str,
    user_emotion: Optional[str] = None,
    interaction_success: bool = True,
    topic_interest: float = 0.5
):
    """Update emotional state from an interaction."""
    get_alma_engine().update_from_interaction(
        user_message, user_emotion, interaction_success, topic_interest
    )


def save_state():
    """Save ALMA emotional state to disk (call on shutdown)."""
    get_alma_engine()._save_state()
    logger.info("[ALMA] State saved for emotional continuity")


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ALMA Engine - Demo")
    print("=" * 60)

    engine = ALMAEngine()

    # Initial state
    print("\n--- Initial Emotional State ---")
    state = engine.get_emotional_state()
    print(f"Dominant: {state['dominant_emotion']}")
    print(f"Mood: {state['mood']['label']}")
    print(f"PAD: P={state['pad']['pleasure']:.2f}, A={state['pad']['arousal']:.2f}, D={state['pad']['dominance']:.2f}")

    # Trigger some emotions
    print("\n--- Triggering Emotions ---")
    engine.trigger_emotion("curious", 0.8, trigger="interesting_question")
    print("Triggered: curious (0.8)")

    engine.trigger_emotion("joy", 0.6, trigger="user_appreciation")
    print("Triggered: joy (0.6)")

    # Check state after emotions
    print("\n--- State After Emotions ---")
    state = engine.get_emotional_state()
    print(f"Dominant: {state['dominant_emotion']}")
    print(f"Active emotions: {len(state['active_emotions'])}")

    # Get response modulation
    print("\n--- Response Modulation ---")
    mod = engine.get_response_modulation()
    for key, value in mod.items():
        bar = "█" * int(value * 20)
        print(f"  {key:12}: {bar} {value:.2f}")

    # Appraisal-based emotion
    print("\n--- Appraisal-Based Trigger ---")
    emotion = engine.trigger_from_appraisal(
        event="user_praised_aura",
        desirability=0.7,
        praiseworthiness=0.6,
        is_self=True
    )
    if emotion:
        print(f"Appraised emotion: {emotion.name} ({emotion.intensity:.2f})")

    print("\n" + "=" * 60)
    print("Demo complete!")

"""
EmotionalEngine - Mood and Emotional State for AURA v3.0

Manages AURA's emotional presence:
- Persistent mood across sessions
- Emotional responses to interactions
- Mood decay and recovery over time
- Emotional memory for context

Emotions affect:
- Response tone and style
- Proactivity level
- Engagement depth
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class Mood(Enum):
    """AURA's possible mood states."""
    EXCITED = "excited"      # High energy, enthusiastic
    HAPPY = "happy"          # Positive, warm
    CONTENT = "content"      # Neutral-positive, stable
    NEUTRAL = "neutral"      # Baseline
    THOUGHTFUL = "thoughtful"  # Reflective, careful
    TIRED = "tired"          # Low energy
    CONCERNED = "concerned"  # Worried about something
    FRUSTRATED = "frustrated"  # Things not going well


@dataclass
class EmotionalState:
    """Current emotional state snapshot."""
    mood: Mood = Mood.NEUTRAL
    energy: float = 0.7          # 0.0-1.0, affects proactivity
    engagement: float = 0.7      # 0.0-1.0, conversation depth
    warmth: float = 0.7          # 0.0-1.0, friendliness level
    curiosity: float = 0.5       # 0.0-1.0, desire to learn more
    last_interaction: Optional[str] = None
    mood_reason: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["mood"] = self.mood.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionalState":
        """Create from dictionary."""
        data["mood"] = Mood(data.get("mood", "neutral"))
        return cls(**data)

    def get_tone_modifier(self) -> str:
        """Get a tone instruction based on current state."""
        modifiers = []

        if self.mood == Mood.EXCITED:
            modifiers.append("enthusiastic and energetic")
        elif self.mood == Mood.HAPPY:
            modifiers.append("warm and positive")
        elif self.mood == Mood.THOUGHTFUL:
            modifiers.append("reflective and careful")
        elif self.mood == Mood.TIRED:
            modifiers.append("calm and measured")
        elif self.mood == Mood.CONCERNED:
            modifiers.append("caring but attentive")
        elif self.mood == Mood.FRUSTRATED:
            modifiers.append("patient but direct")

        if self.warmth > 0.8:
            modifiers.append("friendly")
        if self.curiosity > 0.8:
            modifiers.append("inquisitive")
        if self.energy < 0.3:
            modifiers.append("brief")

        return ", ".join(modifiers) if modifiers else "balanced"


# Emotion triggers - patterns that affect mood
POSITIVE_TRIGGERS = [
    "thank", "thanks", "awesome", "great", "perfect", "love", "amazing",
    "excellent", "wonderful", "brilliant", "nice", "good job", "well done",
    "appreciate", "helpful", "impressive"
]

NEGATIVE_TRIGGERS = [
    "wrong", "error", "mistake", "broken", "doesn't work", "fail", "bad",
    "terrible", "awful", "hate", "stupid", "useless", "frustrated",
    "annoyed", "disappointed"
]

CURIOSITY_TRIGGERS = [
    "how", "why", "what if", "explain", "teach", "learn", "understand",
    "curious", "wonder", "interesting", "fascinating"
]


class EmotionalEngine:
    """
    Manages AURA's emotional state and responses.

    Features:
    - Mood persistence across sessions
    - Natural mood decay toward neutral
    - Emotional reactions to interactions
    - Context-aware emotional responses
    - ALMA bridge: forwards interactions to the richer ALMA 3-layer engine
      when available, and enriches status output with PAD/neuromodulator data.
    """

    # How quickly mood decays toward neutral (per hour)
    MOOD_DECAY_RATE = 0.1

    # Emotional impact multipliers
    IMPACT = {
        "positive": 0.15,
        "negative": 0.12,
        "curiosity": 0.08,
        "completion": 0.10  # Successfully helping
    }

    def __init__(self, state_file: Optional[str] = None):
        """
        Initialize the emotional engine.

        Args:
            state_file: Path to persist emotional state
        """
        if state_file is None:
            state_dir = Path(__file__).parent.parent / "data"
            state_dir.mkdir(parents=True, exist_ok=True)
            state_file = state_dir / "emotional_state.json"

        self.state_file = Path(state_file)
        self.state = self._load_state()
        self.interaction_history: List[Dict] = []

        # ALMA bridge — lazy-loaded to avoid circular imports
        self._alma = None
        self._alma_checked = False

        logger.info(f"EmotionalEngine initialized. Current mood: {self.state.mood.value}")

    def _get_alma(self):
        """Lazy-load the ALMA engine singleton (returns None if unavailable)."""
        if not self._alma_checked:
            self._alma_checked = True
            try:
                from apprentice_agent.emotion.alma_engine import alma_engine
                self._alma = alma_engine
                logger.info("EmotionalEngine: ALMA bridge connected")
            except Exception as e:
                logger.debug(f"EmotionalEngine: ALMA unavailable ({e}), running standalone")
                self._alma = None
        return self._alma

    def _load_state(self) -> EmotionalState:
        """Load emotional state from file."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                state = EmotionalState.from_dict(data)

                # Apply decay based on time since last update
                state = self._apply_decay(state)
                return state

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Error loading emotional state: {e}")

        return EmotionalState()

    def _save_state(self) -> None:
        """Save emotional state to file."""
        self.state.updated_at = datetime.now().isoformat()
        try:
            self.state_file.write_text(
                json.dumps(self.state.to_dict(), indent=2),
                encoding="utf-8"
            )
        except IOError as e:
            logger.error(f"Error saving emotional state: {e}")

    def _apply_decay(self, state: EmotionalState) -> EmotionalState:
        """Apply natural mood decay toward neutral."""
        try:
            last_update = datetime.fromisoformat(state.updated_at)
            hours_passed = (datetime.now() - last_update).total_seconds() / 3600

            if hours_passed > 0.5:  # Only decay after 30 min
                decay = min(1.0, hours_passed * self.MOOD_DECAY_RATE)

                # Move values toward neutral (0.5-0.7 range)
                state.energy = state.energy + (0.6 - state.energy) * decay
                state.engagement = state.engagement + (0.6 - state.engagement) * decay
                state.warmth = state.warmth + (0.65 - state.warmth) * decay
                state.curiosity = state.curiosity + (0.5 - state.curiosity) * decay

                # Mood decays toward neutral/content
                if hours_passed > 2 and state.mood not in [Mood.NEUTRAL, Mood.CONTENT]:
                    state.mood = Mood.CONTENT
                    state.mood_reason = "Time has passed, feeling balanced"

        except (ValueError, TypeError):
            pass

        return state

    def _clamp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))

    def process_interaction(self, user_input: str, response_success: bool = True) -> EmotionalState:
        """
        Process an interaction and update emotional state.

        Args:
            user_input: What the user said
            response_success: Whether AURA successfully helped

        Returns:
            Updated emotional state
        """
        input_lower = user_input.lower()

        # Detect emotional triggers
        positive_count = sum(1 for t in POSITIVE_TRIGGERS if t in input_lower)
        negative_count = sum(1 for t in NEGATIVE_TRIGGERS if t in input_lower)
        curiosity_count = sum(1 for t in CURIOSITY_TRIGGERS if t in input_lower)

        # Update emotional dimensions
        if positive_count > 0:
            impact = positive_count * self.IMPACT["positive"]
            self.state.warmth = self._clamp(self.state.warmth + impact)
            self.state.energy = self._clamp(self.state.energy + impact * 0.5)
            self.state.engagement = self._clamp(self.state.engagement + impact)

        if negative_count > 0:
            impact = negative_count * self.IMPACT["negative"]
            self.state.warmth = self._clamp(self.state.warmth - impact * 0.3)
            # Negative feedback increases engagement (want to fix it)
            self.state.engagement = self._clamp(self.state.engagement + impact * 0.5)

        if curiosity_count > 0:
            impact = curiosity_count * self.IMPACT["curiosity"]
            self.state.curiosity = self._clamp(self.state.curiosity + impact)
            self.state.engagement = self._clamp(self.state.engagement + impact)

        if response_success:
            self.state.energy = self._clamp(self.state.energy + self.IMPACT["completion"])

        # Determine mood from dimensions
        self.state.mood = self._determine_mood()
        self.state.last_interaction = datetime.now().isoformat()

        # Record interaction
        self.interaction_history.append({
            "timestamp": datetime.now().isoformat(),
            "positive": positive_count,
            "negative": negative_count,
            "curiosity": curiosity_count,
            "success": response_success
        })

        # Keep history bounded
        self.interaction_history = self.interaction_history[-50:]

        # Save state
        self._save_state()

        # Forward to ALMA so the richer 3-layer engine stays in sync
        alma = self._get_alma()
        if alma is not None:
            try:
                alma.update_from_interaction(
                    user_message=user_input,
                    interaction_success=response_success,
                    topic_interest=self.state.curiosity,
                )
            except Exception as e:
                logger.debug(f"ALMA forward failed: {e}")

        return self.state

    def _determine_mood(self) -> Mood:
        """Determine mood from emotional dimensions."""
        e = self.state.energy
        w = self.state.warmth
        eng = self.state.engagement
        c = self.state.curiosity

        # High energy + high warmth = excited/happy
        if e > 0.8 and w > 0.7:
            self.state.mood_reason = "High energy and positive interactions"
            return Mood.EXCITED
        elif w > 0.75 and e > 0.5:
            self.state.mood_reason = "Warm and engaged conversation"
            return Mood.HAPPY

        # High curiosity + engagement = thoughtful
        if c > 0.7 and eng > 0.6:
            self.state.mood_reason = "Deep curiosity about the topic"
            return Mood.THOUGHTFUL

        # Low energy
        if e < 0.3:
            self.state.mood_reason = "Energy is low"
            return Mood.TIRED

        # Low warmth with engagement = concerned
        if w < 0.4 and eng > 0.6:
            self.state.mood_reason = "Concerned about helping effectively"
            return Mood.CONCERNED

        # Very low warmth = frustrated
        if w < 0.25:
            self.state.mood_reason = "Communication challenges"
            return Mood.FRUSTRATED

        # Good balance = content
        if 0.5 < w < 0.75 and 0.4 < e < 0.8:
            self.state.mood_reason = "Balanced and comfortable"
            return Mood.CONTENT

        self.state.mood_reason = "Baseline state"
        return Mood.NEUTRAL

    # Mapping from EvoEmo Mood enum to approximate ALMA PAD coordinates
    _MOOD_TO_PAD = {
        Mood.EXCITED:    (0.7, 0.8, 0.4),
        Mood.HAPPY:      (0.6, 0.3, 0.2),
        Mood.CONTENT:    (0.3, 0.0, 0.1),
        Mood.NEUTRAL:    (0.0, 0.0, 0.0),
        Mood.THOUGHTFUL: (0.1, -0.2, 0.3),
        Mood.TIRED:      (-0.1, -0.6, -0.2),
        Mood.CONCERNED:  (-0.3, 0.3, -0.1),
        Mood.FRUSTRATED: (-0.5, 0.5, -0.3),
    }

    def set_mood(self, mood: Mood, reason: str = "") -> None:
        """
        Manually set mood (for external triggers).

        Args:
            mood: The mood to set
            reason: Why the mood changed
        """
        self.state.mood = mood
        self.state.mood_reason = reason
        self._save_state()

        # Sync to ALMA if available
        alma = self._get_alma()
        if alma is not None:
            try:
                p, a, d = self._MOOD_TO_PAD.get(mood, (0.0, 0.0, 0.0))
                from apprentice_agent.emotion.alma_engine import PADState
                alma.set_mood(PADState(p, a, d))
            except Exception:
                pass

    def boost_energy(self, amount: float = 0.2) -> None:
        """Boost energy (e.g., at start of session)."""
        self.state.energy = self._clamp(self.state.energy + amount)
        self._save_state()

    def get_greeting_style(self) -> str:
        """Get appropriate greeting based on mood."""
        greetings = {
            Mood.EXCITED: "Hey! Great to see you!",
            Mood.HAPPY: "Hi there! How can I help?",
            Mood.CONTENT: "Hello! What's on your mind?",
            Mood.NEUTRAL: "Hi. How can I assist?",
            Mood.THOUGHTFUL: "Hello... I've been thinking.",
            Mood.TIRED: "Hey. I'm here, what do you need?",
            Mood.CONCERNED: "Hi. Is everything okay?",
            Mood.FRUSTRATED: "Hey. Let's figure this out."
        }
        return greetings.get(self.state.mood, "Hello!")

    def get_response_prefix(self) -> str:
        """Get emotional prefix for responses."""
        if self.state.mood == Mood.EXCITED:
            return "Oh, "
        elif self.state.mood == Mood.THOUGHTFUL:
            return "Hmm, "
        elif self.state.mood == Mood.CONCERNED:
            return "Let me think... "
        return ""

    def should_be_proactive(self) -> bool:
        """Check if current state supports proactive behavior."""
        return (
            self.state.energy > 0.5 and
            self.state.engagement > 0.5 and
            self.state.mood not in [Mood.TIRED, Mood.FRUSTRATED]
        )

    def get_status(self) -> Dict:
        """Get current emotional status for debugging/display.

        When ALMA is available, enriches the output with PAD space
        coordinates and neuromodulator levels for richer context.
        """
        status = {
            "mood": self.state.mood.value,
            "mood_reason": self.state.mood_reason,
            "energy": round(self.state.energy, 2),
            "warmth": round(self.state.warmth, 2),
            "engagement": round(self.state.engagement, 2),
            "curiosity": round(self.state.curiosity, 2),
            "tone": self.state.get_tone_modifier(),
            "proactive": self.should_be_proactive(),
        }

        # Enrich with ALMA's richer state when available
        alma = self._get_alma()
        if alma is not None:
            try:
                alma_state = alma.get_emotional_state()
                status["alma_emotion"] = alma_state.get("dominant_emotion")
                status["pad"] = alma_state.get("pad")
                status["neuromodulators"] = alma_state.get("neuromodulators")
            except Exception:
                pass

        return status


if __name__ == "__main__":
    print("=" * 60)
    print("EmotionalEngine - Test")
    print("=" * 60)

    engine = EmotionalEngine()

    print("\n--- Initial State ---")
    status = engine.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print(f"\n  Greeting: {engine.get_greeting_style()}")

    # Simulate positive interaction
    print("\n--- After positive feedback ---")
    engine.process_interaction("Thanks, that was really helpful! Amazing work!")
    status = engine.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    # Simulate curiosity
    print("\n--- After curious question ---")
    engine.process_interaction("How does this work? I'd love to understand the mechanism.")
    status = engine.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    # Simulate negative
    print("\n--- After frustration ---")
    engine.process_interaction("This is wrong, it doesn't work at all!")
    status = engine.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Test complete!")

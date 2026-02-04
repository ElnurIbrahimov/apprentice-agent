"""
ALMA Tool - Emotional Intelligence Interface for AURA

Tool #22: Access and control AURA's emotional state through the ALMA engine.

This tool allows:
- Querying current emotional state
- Triggering emotional responses
- Getting response style modulation
- Resetting to baseline emotional state
"""

import logging
from typing import Optional, Dict, Any

# Import ALMA components
from ..emotion.alma_engine import (
    alma_engine,
    PADState,
    BASIC_EMOTIONS,
    OCC_EMOTIONS,
)
from ..emotion.integration import (
    get_emotional_tone_modifier,
    get_mood_emoji,
    get_emotional_summary,
    get_full_emotional_debug,
    process_user_message,
)

logger = logging.getLogger(__name__)


class ALMATool:
    """
    ALMA Emotional Intelligence Tool.

    Provides interface for querying and modifying AURA's emotional state.
    """

    def __init__(self):
        self.name = "alma"
        self.description = "Access AURA's emotional intelligence system"
        self.engine = alma_engine

    def get_current_state(self) -> Dict[str, Any]:
        """Get AURA's current emotional state."""
        state = self.engine.get_emotional_state()
        return {
            "success": True,
            "emotion": state["dominant_emotion"],
            "mood": state["mood"]["label"],
            "intensity": round(state["intensity"], 2),
            "pad": {
                "pleasure": round(state["pad"]["pleasure"], 2),
                "arousal": round(state["pad"]["arousal"], 2),
                "dominance": round(state["pad"]["dominance"], 2),
            },
            "emoji": get_mood_emoji(),
            "summary": get_emotional_summary(),
        }

    def trigger(
        self,
        emotion: str,
        intensity: float = 0.7,
        reason: str = "tool_trigger"
    ) -> Dict[str, Any]:
        """
        Trigger an emotional response.

        Args:
            emotion: Name of emotion (e.g., 'joy', 'curious', 'excited')
            intensity: Strength of emotion (0.0 to 1.0)
            reason: Why this emotion was triggered

        Returns:
            Result with new emotional state
        """
        # Validate emotion
        all_emotions = set(BASIC_EMOTIONS.keys()) | set(OCC_EMOTIONS.keys())
        if emotion.lower() not in all_emotions:
            return {
                "success": False,
                "error": f"Unknown emotion: {emotion}",
                "available": list(BASIC_EMOTIONS.keys()),
            }

        # Trigger emotion
        emotion_state = self.engine.trigger_emotion(
            emotion_name=emotion,
            intensity=min(1.0, max(0.0, intensity)),
            trigger=reason
        )

        return {
            "success": True,
            "triggered": emotion,
            "intensity": round(emotion_state.intensity, 2),
            "current_intensity": round(emotion_state.current_intensity(), 2),
            "new_state": self.get_current_state(),
        }

    def get_modulation(self) -> Dict[str, Any]:
        """Get response style modulation parameters."""
        mod = self.engine.get_response_modulation()
        return {
            "success": True,
            "modulation": {k: round(v, 2) for k, v in mod.items()},
            "tone_prompt": get_emotional_tone_modifier(),
        }

    def reset(self) -> Dict[str, Any]:
        """Reset emotional state to personality baseline."""
        self.engine.reset_to_baseline()
        return {
            "success": True,
            "message": "Emotional state reset to baseline",
            "new_state": self.get_current_state(),
        }

    def get_history(self, hours: int = 24) -> Dict[str, Any]:
        """Get emotion history for the past N hours."""
        history = self.engine.get_emotion_history(hours=hours)
        return {
            "success": True,
            "count": len(history),
            "hours": hours,
            "emotions": history[-20:],  # Last 20 entries
        }

    def set_mood(
        self,
        pleasure: float = 0.0,
        arousal: float = 0.0,
        dominance: float = 0.0,
        instant: bool = False
    ) -> Dict[str, Any]:
        """
        Set mood to specific PAD coordinates.

        Args:
            pleasure: -1.0 to 1.0 (negative to positive)
            arousal: -1.0 to 1.0 (calm to excited)
            dominance: -1.0 to 1.0 (submissive to dominant)
            instant: Set immediately vs. blend gradually

        Returns:
            Result with new mood state
        """
        pad = PADState(
            pleasure=max(-1.0, min(1.0, pleasure)),
            arousal=max(-1.0, min(1.0, arousal)),
            dominance=max(-1.0, min(1.0, dominance))
        )

        self.engine.set_mood(pad, instant=instant)

        return {
            "success": True,
            "set_to": pad.to_dict(),
            "instant": instant,
            "new_state": self.get_current_state(),
        }

    def appraise(
        self,
        event: str,
        desirability: float = 0.0,
        praiseworthiness: float = 0.0,
        appealingness: float = 0.0,
        likelihood: float = 1.0,
        is_self: bool = False
    ) -> Dict[str, Any]:
        """
        Trigger emotion through cognitive appraisal (OCC model).

        This is a more sophisticated way to trigger emotions based on
        evaluating events rather than directly naming emotions.

        Args:
            event: Description of the event
            desirability: How good/bad is this? (-1 to 1)
            praiseworthiness: How good/bad was the action? (-1 to 1)
            appealingness: How attractive is the object? (-1 to 1)
            likelihood: How likely is this event? (0 to 1)
            is_self: Was AURA the agent?

        Returns:
            Result with appraised emotion
        """
        emotion = self.engine.trigger_from_appraisal(
            event=event,
            desirability=desirability,
            praiseworthiness=praiseworthiness,
            appealingness=appealingness,
            likelihood=likelihood,
            is_self=is_self
        )

        if emotion:
            return {
                "success": True,
                "appraised_emotion": emotion.name,
                "intensity": round(emotion.intensity, 2),
                "event": event,
                "new_state": self.get_current_state(),
            }
        else:
            return {
                "success": True,
                "appraised_emotion": None,
                "message": "No emotion triggered from appraisal",
                "event": event,
            }

    def get_debug(self) -> Dict[str, Any]:
        """Get full debug information about emotional state."""
        return {
            "success": True,
            **get_full_emotional_debug(),
        }

    def list_emotions(self) -> Dict[str, Any]:
        """List all available emotions."""
        return {
            "success": True,
            "basic_emotions": list(BASIC_EMOTIONS.keys()),
            "occ_emotions": list(OCC_EMOTIONS.keys()),
            "total": len(BASIC_EMOTIONS) + len(OCC_EMOTIONS),
        }

    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an ALMA action.

        Actions:
        - state / status: Get current emotional state
        - trigger <emotion> [intensity]: Trigger an emotion
        - modulation / style: Get response modulation
        - reset: Reset to baseline
        - history [hours]: Get emotion history
        - mood <p> <a> <d>: Set mood via PAD
        - appraise <event>: Cognitive appraisal
        - debug: Full debug info
        - emotions / list: List available emotions
        """
        action_lower = action.lower().strip()

        # State / Status
        if action_lower in ["state", "status", "current", "how are you"]:
            return self.get_current_state()

        # Trigger emotion
        if action_lower.startswith("trigger"):
            parts = action_lower.split()
            if len(parts) >= 2:
                emotion = parts[1]
                intensity = float(parts[2]) if len(parts) > 2 else 0.7
                reason = kwargs.get("reason", "user_trigger")
                return self.trigger(emotion, intensity, reason)
            return {"success": False, "error": "Usage: trigger <emotion> [intensity]"}

        # Modulation / Style
        if action_lower in ["modulation", "style", "mod", "tone"]:
            return self.get_modulation()

        # Reset
        if action_lower in ["reset", "baseline", "clear"]:
            return self.reset()

        # History
        if action_lower.startswith("history"):
            parts = action_lower.split()
            hours = int(parts[1]) if len(parts) > 1 else 24
            return self.get_history(hours)

        # Set mood
        if action_lower.startswith("mood"):
            p = kwargs.get("pleasure", kwargs.get("p", 0.0))
            a = kwargs.get("arousal", kwargs.get("a", 0.0))
            d = kwargs.get("dominance", kwargs.get("d", 0.0))
            instant = kwargs.get("instant", False)
            return self.set_mood(p, a, d, instant)

        # Appraise
        if action_lower.startswith("appraise"):
            event = kwargs.get("event", action_lower.replace("appraise", "").strip())
            return self.appraise(
                event=event,
                desirability=kwargs.get("desirability", 0.0),
                praiseworthiness=kwargs.get("praiseworthiness", 0.0),
                appealingness=kwargs.get("appealingness", 0.0),
                likelihood=kwargs.get("likelihood", 1.0),
                is_self=kwargs.get("is_self", False)
            )

        # Debug
        if action_lower in ["debug", "full", "inspect"]:
            return self.get_debug()

        # List emotions
        if action_lower in ["emotions", "list", "available"]:
            return self.list_emotions()

        # Default: show state
        return self.get_current_state()


# Singleton instance
alma_tool = ALMATool()


# Convenience functions
def get_emotional_state() -> Dict[str, Any]:
    """Get AURA's current emotional state."""
    return alma_tool.get_current_state()


def trigger_emotion(emotion: str, intensity: float = 0.7) -> Dict[str, Any]:
    """Trigger an emotional response."""
    return alma_tool.trigger(emotion, intensity)


def get_response_modulation() -> Dict[str, Any]:
    """Get response style modulation."""
    return alma_tool.get_modulation()


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("ALMA Tool - Test")
    print("=" * 60)

    tool = ALMATool()

    # Get current state
    print("\n--- Current State ---")
    state = tool.execute("state")
    print(f"Emotion: {state['emotion']}")
    print(f"Mood: {state['mood']}")
    print(f"Summary: {state['summary']}")

    # Trigger emotion
    print("\n--- Triggering Joy ---")
    result = tool.execute("trigger joy 0.8")
    print(f"Triggered: {result['triggered']}")
    print(f"New emotion: {result['new_state']['emotion']}")

    # Get modulation
    print("\n--- Response Modulation ---")
    mod = tool.execute("modulation")
    for key, value in mod["modulation"].items():
        print(f"  {key}: {value}")

    # List emotions
    print("\n--- Available Emotions ---")
    emotions = tool.execute("list")
    print(f"Basic: {', '.join(emotions['basic_emotions'][:5])}...")
    print(f"OCC: {', '.join(emotions['occ_emotions'][:5])}...")

    print("\n" + "=" * 60)
    print("Test complete!")

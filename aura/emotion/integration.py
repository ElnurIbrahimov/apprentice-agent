"""
ALMA Integration - Connect emotional engine to AURA's brain and systems.

This module provides utilities for:
- Generating emotional tone modifiers for the brain
- Processing user messages for emotional triggers
- Integration with EvoEmo (user emotion detection)
"""

import logging
from typing import Optional, Dict, Any, Tuple

from .alma_engine import (
    alma_engine,
    get_emotional_state,
    trigger_emotion,
    get_response_modulation,
    PADState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Brain Integration
# =============================================================================

def get_emotional_tone_modifier() -> str:
    """
    Generate a tone modifier string for the brain's system prompt.

    This should be added to every LLM call to modulate AURA's response style
    based on current emotional state.

    Returns:
        String to append to system prompt, or empty string if neutral
    """
    try:
        state = get_emotional_state()
        mod = get_response_modulation()

        parts = []

        # Dominant emotion context
        dominant = state.get("dominant_emotion", "neutral")
        if dominant != "neutral":
            intensity = state.get("intensity", 0)
            if intensity > 0.3:
                parts.append(f"Current emotional state: {dominant}")

        # Style modulation
        style_hints = []

        if mod.get("warmth", 0.5) > 0.65:
            style_hints.append("warm and friendly")
        elif mod.get("warmth", 0.5) < 0.35:
            style_hints.append("more reserved")

        if mod.get("enthusiasm", 0.5) > 0.65:
            style_hints.append("enthusiastic")
        elif mod.get("enthusiasm", 0.5) < 0.35:
            style_hints.append("measured")

        if mod.get("confidence", 0.5) > 0.65:
            style_hints.append("confident")

        if mod.get("patience", 0.5) > 0.65:
            style_hints.append("patient and thorough")
        elif mod.get("patience", 0.5) < 0.35:
            style_hints.append("concise")

        if mod.get("empathy", 0.5) > 0.65:
            style_hints.append("empathetic")

        if style_hints:
            parts.append(f"Response tone: {', '.join(style_hints)}")

        if not parts:
            return ""

        return "[Emotional Context] " + ". ".join(parts) + "."

    except Exception as e:
        logger.error(f"Failed to generate emotional tone modifier: {e}")
        return ""


def get_emotional_context_for_prompt() -> Dict[str, Any]:
    """
    Get structured emotional context for advanced prompt engineering.

    Returns a dictionary with emotional state that can be formatted
    into prompts in various ways.
    """
    state = get_emotional_state()
    mod = get_response_modulation()

    return {
        "emotion": state.get("dominant_emotion", "neutral"),
        "mood": state.get("mood", {}).get("label", "neutral"),
        "intensity": state.get("intensity", 0),
        "pad": state.get("pad", {"pleasure": 0, "arousal": 0, "dominance": 0}),
        "modulation": mod,
        "style_prompt": get_emotional_tone_modifier(),
    }


# =============================================================================
# Message Processing
# =============================================================================

def process_user_message(
    message: str,
    user_emotion: Optional[str] = None,
    topic_keywords: Optional[list] = None
) -> Dict[str, Any]:
    """
    Process a user message and update AURA's emotional state.

    This should be called at the start of each interaction to:
    1. Detect emotional triggers in the message
    2. Apply empathetic responses to user emotion
    3. Update engagement/interest levels

    Args:
        message: The user's message text
        user_emotion: Detected user emotion from EvoEmo (optional)
        topic_keywords: Keywords indicating topic (for interest detection)

    Returns:
        Dictionary with emotional processing results
    """
    result = {
        "triggered_emotions": [],
        "empathy_applied": False,
        "interest_level": 0.5,
    }

    # Detect emotional triggers in message
    triggers = _detect_message_triggers(message)
    for emotion, intensity, trigger in triggers:
        trigger_emotion(emotion, intensity, trigger)
        result["triggered_emotions"].append({
            "emotion": emotion,
            "intensity": intensity,
            "trigger": trigger
        })

    # Apply empathy for user emotion
    if user_emotion:
        alma_engine.update_from_interaction(
            user_message=message,
            user_emotion=user_emotion,
            interaction_success=True
        )
        result["empathy_applied"] = True

    # Detect topic interest
    interest = _detect_topic_interest(message, topic_keywords)
    result["interest_level"] = interest

    if interest > 0.7:
        trigger_emotion("curious", intensity=interest * 0.5, trigger="interesting_topic")

    return result


def _detect_message_triggers(message: str) -> list:
    """
    Detect emotional triggers in a message.

    Returns list of (emotion_name, intensity, trigger_reason) tuples.
    """
    triggers = []
    message_lower = message.lower()

    # Appreciation triggers
    appreciation_words = ["thank", "thanks", "appreciate", "grateful", "awesome", "great job", "perfect"]
    if any(word in message_lower for word in appreciation_words):
        triggers.append(("joy", 0.5, "user_appreciation"))

    # Question triggers curiosity
    if "?" in message and any(q in message_lower for q in ["how", "why", "what", "explain", "tell me"]):
        triggers.append(("curious", 0.4, "interesting_question"))

    # Challenge triggers engagement
    challenge_words = ["can you", "could you", "help me", "figure out", "solve", "build", "create"]
    if any(word in message_lower for word in challenge_words):
        triggers.append(("engaged", 0.5, "challenge_presented"))

    # Frustration/problem triggers empathy
    problem_words = ["problem", "issue", "bug", "error", "broken", "doesn't work", "help"]
    if any(word in message_lower for word in problem_words):
        triggers.append(("empathetic", 0.4, "user_needs_help"))

    # Excitement is contagious
    excitement_words = ["excited", "can't wait", "amazing", "incredible", "wow"]
    if any(word in message_lower for word in excitement_words):
        triggers.append(("excited", 0.4, "shared_excitement"))

    return triggers


def _detect_topic_interest(message: str, keywords: Optional[list] = None) -> float:
    """
    Detect how interesting a topic is to AURA.

    Based on AURA's personality (high openness, curious nature).
    """
    message_lower = message.lower()
    interest = 0.5  # Base interest

    # Technical/programming topics
    tech_words = ["code", "program", "algorithm", "function", "api", "database",
                  "machine learning", "ai", "neural", "architecture", "design"]
    if any(word in message_lower for word in tech_words):
        interest += 0.2

    # Creative/building topics
    creative_words = ["create", "build", "design", "make", "develop", "prototype"]
    if any(word in message_lower for word in creative_words):
        interest += 0.15

    # Learning/exploration topics
    learning_words = ["learn", "understand", "explore", "discover", "research"]
    if any(word in message_lower for word in learning_words):
        interest += 0.15

    # Philosophy/deep topics
    deep_words = ["think", "consciousness", "meaning", "philosophy", "theory"]
    if any(word in message_lower for word in deep_words):
        interest += 0.2

    # Custom keywords from context
    if keywords:
        for kw in keywords:
            if kw.lower() in message_lower:
                interest += 0.1

    return min(1.0, interest)


# =============================================================================
# Response Processing
# =============================================================================

def process_response_outcome(
    success: bool,
    user_satisfied: bool = True,
    error_occurred: bool = False
):
    """
    Update emotional state based on response outcome.

    Called after a response is generated to update mood based on:
    - Whether the interaction was successful
    - User satisfaction signals
    - Any errors that occurred

    Args:
        success: Was the response generated successfully?
        user_satisfied: Did the user seem satisfied?
        error_occurred: Did any errors occur?
    """
    if success and user_satisfied:
        trigger_emotion("satisfaction", 0.3, "successful_response")
    elif error_occurred:
        trigger_emotion("concerned", 0.4, "response_error")
    elif not user_satisfied:
        trigger_emotion("disappointed", 0.3, "user_unsatisfied")


# =============================================================================
# EvoEmo Bridge
# =============================================================================

def bridge_evoemo_detection(evoemo_result: Dict[str, Any]) -> str:
    """
    Bridge EvoEmo user emotion detection to ALMA.

    Takes EvoEmo analysis result and updates AURA's emotional state
    with appropriate empathetic response.

    Args:
        evoemo_result: Result from EvoEmo.analyze_text()

    Returns:
        Combined tone modifier (ALMA + EvoEmo context)
    """
    user_emotion = evoemo_result.get("emotion", "calm")
    confidence = evoemo_result.get("confidence", 50)

    # Only process if confidence is reasonable
    if confidence >= 40:
        alma_engine.update_from_interaction(
            user_message="",
            user_emotion=user_emotion,
            interaction_success=True
        )

    # Get ALMA's emotional tone
    alma_tone = get_emotional_tone_modifier()

    # Add user emotion awareness
    if confidence >= 60 and user_emotion not in ["calm", "neutral"]:
        user_context = f"User appears to be feeling {user_emotion}."
        if alma_tone:
            return f"{alma_tone} {user_context}"
        return f"[Context] {user_context}"

    return alma_tone


# =============================================================================
# Quick Access Functions
# =============================================================================

def get_mood_emoji() -> str:
    """Get an emoji representing AURA's current mood."""
    state = get_emotional_state()
    mood = state.get("mood", {}).get("label", "neutral")

    emoji_map = {
        "happy": "😊",
        "excited": "🤩",
        "curious": "🤔",
        "calm": "😌",
        "content": "🙂",
        "thoughtful": "💭",
        "engaged": "✨",
        "playful": "😄",
        "confident": "💪",
        "empathetic": "💙",
        "sad": "😢",
        "frustrated": "😤",
        "anxious": "😰",
        "neutral": "😐",
    }

    return emoji_map.get(mood, "🤖")


def get_emotional_summary() -> str:
    """Get a brief summary of AURA's emotional state."""
    state = get_emotional_state()
    mood = state.get("mood", {}).get("label", "neutral")
    dominant = state.get("dominant_emotion", "neutral")
    intensity = state.get("intensity", 0)

    if intensity < 0.2:
        return f"Feeling {mood}"
    elif dominant == mood:
        return f"Feeling {mood} ({intensity:.0%} intensity)"
    else:
        return f"Mood: {mood}, currently feeling {dominant}"


# =============================================================================
# Debug / Inspection
# =============================================================================

def get_full_emotional_debug() -> Dict[str, Any]:
    """Get complete emotional state for debugging."""
    state = get_emotional_state()
    mod = get_response_modulation()

    return {
        "state": state,
        "modulation": mod,
        "tone_modifier": get_emotional_tone_modifier(),
        "mood_emoji": get_mood_emoji(),
        "summary": get_emotional_summary(),
    }


# =============================================================================
# LLM Parameter Export
# =============================================================================

def get_llm_parameters() -> Dict[str, float]:
    """Export current ALMA emotional state as LLM generation parameters.

    Maps neuromodulator levels to Ollama generation parameters:
    - dopamine -> temperature (high dopamine = more creative/exploratory)
    - serotonin -> top_p (high serotonin = calmer/focused = lower top_p)

    Note: brain.py has its own, more comprehensive neuromodulator mapping that
    also handles norepinephrine, acetylcholine, timeout scaling, and NeuroDream
    offsets. This function is a convenience export for external consumers (e.g.
    tools or tests) that need the current emotional state as LLM parameters
    without going through the full brain.py think() pipeline.

    Returns:
        dict with 'temperature' and 'top_p' keys, or empty dict on any failure.
    """
    try:
        state = get_emotional_state()
        mods = state.get("neuromodulators", {})

        if not mods:
            return {}

        # Dopamine: 0.0-1.0 -> temperature: 0.3-1.2
        dopamine = float(mods.get("dopamine", 0.5))
        base_temp = 0.7
        temp_delta = (dopamine - 0.5) * 0.6  # ±0.3 range around base
        temperature = max(0.3, min(1.2, base_temp + temp_delta))

        # Serotonin: 0.0-1.0 -> top_p: 0.7-0.99
        # High serotonin = calmer/focused -> lower top_p (more focused sampling)
        serotonin = float(mods.get("serotonin", 0.5))
        top_p = max(0.7, min(0.99, 0.85 + (0.5 - serotonin) * 0.2))

        return {
            "temperature": round(temperature, 2),
            "top_p": round(top_p, 2),
        }
    except Exception:
        return {}

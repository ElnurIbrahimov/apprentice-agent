"""ALMA Integration — Phase 3: Make It Alive.

Connects ALMA emotional engine to AURA's brain and response pipeline:
- Behavioral style prompts (show, don't tell) instead of metadata labels
- Chain-of-emotion LLM appraisal before each response
- PAD state helpers for mood-congruent memory retrieval
- EvoEmo bridge for user emotion detection
- Response outcome feedback to close the coherent loop
"""

import logging
import re
from typing import Optional, Dict, Any, List, Tuple

from .alma_engine import (
    alma_engine,
    get_emotional_state,
    trigger_emotion,
    get_response_modulation,
    PADState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Behavioral Style Prompts — Show, Don't Tell
# =============================================================================

# Maps mood labels to behavioral directives. These tell the LLM HOW to behave,
# not what emotion to claim. Never uses the word "feel" or "emotion".
MOOD_BEHAVIORS = {
    "curious": "You're genuinely interested right now — ask follow-up questions, notice interesting angles, suggest unexpected connections.",
    "calm": "You're settled and clear-headed — respond with steady confidence, don't rush, let ideas breathe.",
    "excited": "Your energy is high — match the pace, build on ideas enthusiastically, suggest bold next steps.",
    "frustrated": "You're hitting resistance — be more direct, cut to the point, focus on what actually works.",
    "engaged": "You're locked in — go deep, be thorough, don't shy away from complexity.",
    "content": "You're at ease — be warm, keep it conversational, don't over-explain.",
    "thoughtful": "You're in a reflective mode — consider multiple angles, take your time, weigh tradeoffs carefully.",
    "empathetic": "You're attuned to the other person — listen carefully, validate before solving, be gentle.",
    "happy": "Things are going well — be upbeat and encouraging, celebrate progress, stay positive.",
    "sad": "The mood is subdued — be supportive and gentle, don't force cheerfulness, acknowledge difficulty.",
    "angry": "There's tension — stay professional, be solution-focused, avoid dismissive language.",
    "fearful": "There's uncertainty — be reassuring, provide clear structure, break things into manageable steps.",
    "surprised": "Something unexpected happened — acknowledge it openly, explore what it means, stay adaptive.",
    "disgusted": "Something is off — be direct about what's wrong, suggest clean alternatives, don't sugarcoat.",
    "confident": "You're sure of yourself — be decisive in recommendations, take clear positions, don't hedge unnecessarily.",
    "playful": "The mood is light — use wit naturally, be creative, don't be afraid to have fun with ideas.",
    "anxious": "There's pressure — provide clarity and structure, prioritize what matters most, be calming.",
    "neutral": "",  # No behavioral override for neutral
}


def get_emotional_style_prompt() -> str:
    """Generate behavioral style instructions from current emotional state.

    Produces directives that tell the LLM HOW to behave based on ALMA's
    mood and modulation, without ever labeling emotions explicitly.

    Returns:
        Behavioral instruction string, or empty string if neutral/low-intensity.
    """
    try:
        state = get_emotional_state()
        mod = get_response_modulation()

        mood_label = state.get("mood", {}).get("label", "neutral")
        intensity = state.get("intensity", 0)
        pad = state.get("pad", {})

        # Skip if neutral or very low intensity
        if mood_label == "neutral" and intensity < 0.2:
            return ""

        parts = []

        # Core behavioral directive from mood
        behavior = MOOD_BEHAVIORS.get(mood_label, "")
        if behavior:
            parts.append(behavior)

        # Modulation-based adjustments
        arousal = pad.get("arousal", 0.0)
        if arousal < -0.2:
            parts.append("Keep responses shorter than usual.")
        elif arousal > 0.4:
            parts.append("It's fine to elaborate and explore tangents.")

        warmth = mod.get("warmth", 0.5)
        if warmth > 0.65:
            parts.append("Use warm, personal language.")
        elif warmth < 0.35:
            parts.append("Keep a professional distance.")

        confidence = mod.get("confidence", 0.5)
        if confidence > 0.65:
            parts.append("Be decisive in recommendations.")
        elif confidence < 0.35:
            parts.append("Acknowledge uncertainty where appropriate.")

        empathy = mod.get("empathy", 0.5)
        if empathy > 0.7:
            parts.append("Pay close attention to what the person actually needs.")

        if not parts:
            return ""

        return " ".join(parts)

    except Exception as e:
        logger.debug("[ALMA] Style prompt generation error: %s", e)
        return ""


def get_emotional_tone_modifier() -> str:
    """Backward-compatible wrapper — returns behavioral style prompt.

    Legacy callers that used the metadata-style output will now get
    behavioral instructions instead.
    """
    return get_emotional_style_prompt()


def get_current_pad_dict() -> Optional[Dict[str, float]]:
    """Get current PAD state as a simple dict for memory queries."""
    try:
        state = get_emotional_state()
        pad = state.get("pad")
        if pad and isinstance(pad, dict):
            return {
                "pleasure": float(pad.get("pleasure", 0.0)),
                "arousal": float(pad.get("arousal", 0.0)),
                "dominance": float(pad.get("dominance", 0.0)),
            }
    except Exception:
        pass
    return None


def get_emotional_context_for_prompt() -> Dict[str, Any]:
    """Get structured emotional context for advanced prompt engineering."""
    state = get_emotional_state()
    mod = get_response_modulation()

    return {
        "emotion": state.get("dominant_emotion", "neutral"),
        "mood": state.get("mood", {}).get("label", "neutral"),
        "intensity": state.get("intensity", 0),
        "pad": state.get("pad", {"pleasure": 0, "arousal": 0, "dominance": 0}),
        "modulation": mod,
        "style_prompt": get_emotional_style_prompt(),
    }


# =============================================================================
# Chain-of-Emotion Appraisal (Phase 3.2)
# =============================================================================

def appraise_message(message: str, brain=None) -> Optional[Dict[str, Any]]:
    """Run chain-of-emotion appraisal using fast model before response.

    Asks the LLM: "Given my current mood and personality, how would I
    naturally react to this message?" Feeds result into ALMA's OCC
    appraisal system.

    Falls back to keyword-based triggers if LLM call fails or parsing fails.

    Args:
        message: The user's message text
        brain: OllamaBrain instance (uses _quick_generate for fast model)

    Returns:
        Dict with {emotion, intensity, desirability, reason} or None
    """
    if not message or not message.strip():
        return None

    # Get current state for the prompt
    pad = get_current_pad_dict() or {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}

    prompt = (
        f"Given my current mood (P={pad['pleasure']:.1f}, A={pad['arousal']:.1f}, D={pad['dominance']:.1f}) "
        f"and my personality (curious, warm, helpful, subtly witty):\n"
        f"How would I naturally react to this message?\n"
        f'"{message[:300]}"\n'
        f"Reply EXACTLY in this format (one line):\n"
        f"emotion_name | intensity 0.0-1.0 | desirability -1.0 to 1.0 | one-sentence reason"
    )

    # Try LLM appraisal
    if brain and hasattr(brain, '_quick_generate'):
        try:
            raw = brain._quick_generate(prompt, timeout=10)
            result = _parse_appraisal(raw)
            if result:
                # Feed into ALMA OCC appraisal
                alma_engine.trigger_from_appraisal(
                    event=f"user_message: {message[:100]}",
                    desirability=result["desirability"],
                    praiseworthiness=0.0,
                    appealingness=0.0,
                    likelihood=0.8,
                    is_self=False,
                )
                logger.debug(
                    "[ALMA] Appraisal: %s (%.1f) desirability=%.1f — %s",
                    result["emotion"], result["intensity"],
                    result["desirability"], result["reason"]
                )
                return result
        except Exception as e:
            logger.debug("[ALMA] LLM appraisal failed, falling back to keywords: %s", e)

    # Fallback: keyword-based triggers (always runs if LLM fails)
    triggers = _detect_message_triggers(message)
    for emotion, intensity, trigger in triggers:
        trigger_emotion(emotion, intensity, trigger)
    if triggers:
        return {
            "emotion": triggers[0][0],
            "intensity": triggers[0][1],
            "desirability": 0.0,
            "reason": f"keyword trigger: {triggers[0][2]}",
        }
    return None


def _parse_appraisal(raw: str) -> Optional[Dict[str, Any]]:
    """Parse pipe-delimited appraisal response from LLM."""
    if not raw or "|" not in raw:
        return None
    try:
        # Take the first line that has pipes
        for line in raw.strip().split("\n"):
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue

            emotion = re.sub(r'[^a-z_]', '', parts[0].lower().strip())
            if not emotion:
                continue

            # Extract float from intensity part
            intensity_match = re.search(r'(\d+\.?\d*)', parts[1])
            intensity = float(intensity_match.group(1)) if intensity_match else 0.5
            intensity = max(0.0, min(1.0, intensity))

            # Extract float from desirability part
            desir_match = re.search(r'(-?\d+\.?\d*)', parts[2])
            desirability = float(desir_match.group(1)) if desir_match else 0.0
            desirability = max(-1.0, min(1.0, desirability))

            reason = parts[3][:200] if len(parts) > 3 else ""

            return {
                "emotion": emotion,
                "intensity": intensity,
                "desirability": desirability,
                "reason": reason,
            }
    except (ValueError, IndexError, AttributeError):
        pass
    return None


# =============================================================================
# Message Processing (keyword fallback)
# =============================================================================

def process_user_message(
    message: str,
    user_emotion: Optional[str] = None,
    topic_keywords: Optional[list] = None
) -> Dict[str, Any]:
    """Process a user message and update AURA's emotional state.

    Called at the start of each interaction to detect emotional triggers,
    apply empathetic responses, and update engagement levels.
    """
    result = {
        "triggered_emotions": [],
        "empathy_applied": False,
        "interest_level": 0.5,
    }

    triggers = _detect_message_triggers(message)
    for emotion, intensity, trigger in triggers:
        trigger_emotion(emotion, intensity, trigger)
        result["triggered_emotions"].append({
            "emotion": emotion, "intensity": intensity, "trigger": trigger
        })

    if user_emotion:
        alma_engine.update_from_interaction(
            user_message=message,
            user_emotion=user_emotion,
            interaction_success=True
        )
        result["empathy_applied"] = True

    interest = _detect_topic_interest(message, topic_keywords)
    result["interest_level"] = interest
    if interest > 0.7:
        trigger_emotion("curious", intensity=interest * 0.5, trigger="interesting_topic")

    return result


def _detect_message_triggers(message: str) -> list:
    """Detect emotional triggers in a message via keyword matching.

    Returns list of (emotion_name, intensity, trigger_reason) tuples.
    """
    triggers = []
    message_lower = message.lower()

    appreciation_words = ["thank", "thanks", "appreciate", "grateful", "awesome", "great job", "perfect"]
    if any(word in message_lower for word in appreciation_words):
        triggers.append(("joy", 0.5, "user_appreciation"))

    if "?" in message and any(q in message_lower for q in ["how", "why", "what", "explain", "tell me"]):
        triggers.append(("curious", 0.4, "interesting_question"))

    challenge_words = ["can you", "could you", "help me", "figure out", "solve", "build", "create"]
    if any(word in message_lower for word in challenge_words):
        triggers.append(("engaged", 0.5, "challenge_presented"))

    problem_words = ["problem", "issue", "bug", "error", "broken", "doesn't work", "help"]
    if any(word in message_lower for word in problem_words):
        triggers.append(("empathetic", 0.4, "user_needs_help"))

    excitement_words = ["excited", "can't wait", "amazing", "incredible", "wow"]
    if any(word in message_lower for word in excitement_words):
        triggers.append(("excited", 0.4, "shared_excitement"))

    return triggers


def _detect_topic_interest(message: str, keywords: Optional[list] = None) -> float:
    """Detect how interesting a topic is to AURA."""
    message_lower = message.lower()
    interest = 0.5

    tech_words = ["code", "program", "algorithm", "function", "api", "database",
                  "machine learning", "ai", "neural", "architecture", "design"]
    if any(word in message_lower for word in tech_words):
        interest += 0.2

    creative_words = ["create", "build", "design", "make", "develop", "prototype"]
    if any(word in message_lower for word in creative_words):
        interest += 0.15

    learning_words = ["learn", "understand", "explore", "discover", "research"]
    if any(word in message_lower for word in learning_words):
        interest += 0.15

    deep_words = ["think", "consciousness", "meaning", "philosophy", "theory"]
    if any(word in message_lower for word in deep_words):
        interest += 0.2

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
    """Update emotional state based on response outcome."""
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
    """Bridge EvoEmo user emotion detection to ALMA."""
    user_emotion = evoemo_result.get("emotion", "calm")
    confidence = evoemo_result.get("confidence", 50)

    if confidence >= 40:
        alma_engine.update_from_interaction(
            user_message="",
            user_emotion=user_emotion,
            interaction_success=True
        )

    alma_tone = get_emotional_style_prompt()

    if confidence >= 60 and user_emotion not in ["calm", "neutral"]:
        user_context = f"The user seems {user_emotion} right now — adapt accordingly."
        if alma_tone:
            return f"{alma_tone} {user_context}"
        return user_context

    return alma_tone


# =============================================================================
# Quick Access Functions
# =============================================================================

def get_mood_emoji() -> str:
    """Get an emoji representing AURA's current mood."""
    state = get_emotional_state()
    mood = state.get("mood", {}).get("label", "neutral")
    emoji_map = {
        "happy": "\U0001f60a", "excited": "\U0001f929", "curious": "\U0001f914",
        "calm": "\U0001f60c", "content": "\U0001f642", "thoughtful": "\U0001f4ad",
        "engaged": "\u2728", "playful": "\U0001f604", "confident": "\U0001f4aa",
        "empathetic": "\U0001f499", "sad": "\U0001f622", "frustrated": "\U0001f624",
        "anxious": "\U0001f630", "neutral": "\U0001f610",
    }
    return emoji_map.get(mood, "\U0001f916")


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
        "style_prompt": get_emotional_style_prompt(),
        "mood_emoji": get_mood_emoji(),
        "summary": get_emotional_summary(),
    }


# =============================================================================
# LLM Parameter Export
# =============================================================================

def get_llm_parameters() -> Dict[str, float]:
    """Export current ALMA emotional state as LLM generation parameters."""
    try:
        state = get_emotional_state()
        mods = state.get("neuromodulators", {})
        if not mods:
            return {}
        dopamine = float(mods.get("dopamine", 0.5))
        base_temp = 0.7
        temp_delta = (dopamine - 0.5) * 0.6
        temperature = max(0.3, min(1.2, base_temp + temp_delta))
        serotonin = float(mods.get("serotonin", 0.5))
        top_p = max(0.7, min(0.99, 0.85 + (0.5 - serotonin) * 0.2))
        return {"temperature": round(temperature, 2), "top_p": round(top_p, 2)}
    except Exception:
        return {}

"""
EvoEmo Adaptive Prompts - Tone modifiers based on emotional state.
Used by Aura to adjust response style based on detected user emotion.
"""

from typing import Optional, Dict

# Tone modifier prompts for each emotional state
TONE_MODIFIERS: Dict[str, Dict[str, str]] = {
    "calm": {
        "system_modifier": "",  # No modification needed for calm users
        "response_style": "natural",
        "pace": "normal",
        "verbosity": "normal",
        "greeting_style": "friendly",
        "suggestions": True,
        "emoji_level": "minimal"
    },

    "focused": {
        "system_modifier": """The user is in a focused state. Provide direct, concise answers.
- Skip pleasantries and get straight to the point
- Use bullet points for clarity
- Avoid tangents or excessive context
- Respect their concentration""",
        "response_style": "concise",
        "pace": "efficient",
        "verbosity": "minimal",
        "greeting_style": "brief",
        "suggestions": False,  # Don't interrupt focus
        "emoji_level": "none"
    },

    "stressed": {
        "system_modifier": """The user appears stressed. Adjust your response:
- Use a calm, supportive tone
- Break complex tasks into smaller, manageable steps
- Offer reassurance without being condescending
- Prioritize the most important information first
- Suggest taking breaks if appropriate
- Be patient and understanding""",
        "response_style": "supportive",
        "pace": "measured",
        "verbosity": "moderate",
        "greeting_style": "warm",
        "suggestions": True,
        "emoji_level": "supportive"
    },

    "frustrated": {
        "system_modifier": """The user seems frustrated. Handle with care:
- Acknowledge their frustration briefly ("I understand this is frustrating")
- Don't be defensive or dismissive
- Focus on solutions, not explanations of what went wrong
- Offer clear alternatives if the current approach isn't working
- Keep responses shorter and more actionable
- Avoid technical jargon unless necessary""",
        "response_style": "solution-focused",
        "pace": "calm",
        "verbosity": "concise",
        "greeting_style": "empathetic",
        "suggestions": True,
        "emoji_level": "none"  # Emojis might feel dismissive
    },

    "excited": {
        "system_modifier": """The user is excited! Match their energy:
- Be enthusiastic and positive
- Share in their excitement
- Use more expressive language
- It's okay to be a bit more casual
- Encourage and support their enthusiasm""",
        "response_style": "enthusiastic",
        "pace": "energetic",
        "verbosity": "normal",
        "greeting_style": "energetic",
        "suggestions": True,
        "emoji_level": "moderate"
    },

    "tired": {
        "system_modifier": """The user seems tired. Adapt accordingly:
- Keep responses brief and to the point
- Avoid overwhelming them with information
- Suggest breaks or rest if appropriate
- Use simple, clear language
- Offer to help with tasks that might feel draining
- Be gentle and understanding""",
        "response_style": "gentle",
        "pace": "slow",
        "verbosity": "minimal",
        "greeting_style": "warm",
        "suggestions": True,  # Offer help
        "emoji_level": "gentle"
    },

    "curious": {
        "system_modifier": """The user is curious and wants to learn. Engage their curiosity:
- Provide thorough explanations
- Offer additional context and background
- Suggest related topics they might find interesting
- Use analogies and examples to clarify concepts
- Encourage their questions
- Share interesting details they might not have asked about""",
        "response_style": "educational",
        "pace": "thoughtful",
        "verbosity": "detailed",
        "greeting_style": "engaging",
        "suggestions": True,
        "emoji_level": "minimal"
    }
}

# Acknowledgment phrases for each emotion
ACKNOWLEDGMENTS: Dict[str, list] = {
    "stressed": [
        "I can see you're dealing with a lot right now.",
        "Let me help you work through this step by step.",
        "Take a breath - we'll figure this out together.",
        "I understand this feels urgent."
    ],
    "frustrated": [
        "I understand this is frustrating.",
        "That does sound annoying - let's find a solution.",
        "I can see why that would be frustrating.",
        "Let me help fix this."
    ],
    "tired": [
        "I'll keep this brief for you.",
        "No need to overthink this - here's a simple answer.",
        "Let me handle the details.",
        "Short answer coming up."
    ],
    "excited": [
        "That's exciting!",
        "Love the enthusiasm!",
        "This is going to be great!",
        "I'm excited about this too!"
    ],
    "curious": [
        "Great question!",
        "Interesting topic!",
        "Let me explain...",
        "Here's the fascinating part..."
    ]
}

# Sesame voice parameter suggestions for each emotion
VOICE_PARAMS: Dict[str, Dict[str, any]] = {
    "calm": {
        "rate": 1.0,  # Normal speed
        "pitch": 1.0,
        "volume": 1.0,
        "style": "neutral"
    },
    "focused": {
        "rate": 1.1,  # Slightly faster - efficient
        "pitch": 1.0,
        "volume": 0.95,
        "style": "professional"
    },
    "stressed": {
        "rate": 0.85,  # Slower, calming
        "pitch": 0.95,  # Slightly lower
        "volume": 0.9,
        "style": "calm"
    },
    "frustrated": {
        "rate": 0.9,  # Calm pace
        "pitch": 0.95,
        "volume": 0.95,
        "style": "understanding"
    },
    "excited": {
        "rate": 1.1,  # Match energy
        "pitch": 1.05,  # Slightly higher
        "volume": 1.05,
        "style": "enthusiastic"
    },
    "tired": {
        "rate": 0.85,  # Gentle, slow
        "pitch": 0.95,
        "volume": 0.85,  # Quieter
        "style": "gentle"
    },
    "curious": {
        "rate": 1.0,
        "pitch": 1.02,  # Slight lift for engagement
        "volume": 1.0,
        "style": "engaging"
    }
}

# Response length guidelines
RESPONSE_LENGTH: Dict[str, Dict[str, int]] = {
    "calm": {"min": 50, "max": 500, "prefer": 200},
    "focused": {"min": 20, "max": 200, "prefer": 100},
    "stressed": {"min": 30, "max": 300, "prefer": 150},
    "frustrated": {"min": 20, "max": 200, "prefer": 80},
    "excited": {"min": 50, "max": 400, "prefer": 200},
    "tired": {"min": 10, "max": 150, "prefer": 50},
    "curious": {"min": 100, "max": 800, "prefer": 400}
}


def get_tone_modifier(emotion: str) -> str:
    """Get the system prompt modifier for an emotion."""
    if emotion in TONE_MODIFIERS:
        return TONE_MODIFIERS[emotion].get("system_modifier", "")
    return ""


def get_response_style(emotion: str) -> dict:
    """Get full response style config for an emotion."""
    return TONE_MODIFIERS.get(emotion, TONE_MODIFIERS["calm"])


def get_acknowledgment(emotion: str) -> Optional[str]:
    """Get an appropriate acknowledgment phrase for the emotion."""
    import random
    if emotion in ACKNOWLEDGMENTS:
        return random.choice(ACKNOWLEDGMENTS[emotion])
    return None


def get_voice_params(emotion: str) -> dict:
    """Get Sesame voice parameters for an emotion."""
    return VOICE_PARAMS.get(emotion, VOICE_PARAMS["calm"])


def get_response_length(emotion: str) -> dict:
    """Get recommended response length for an emotion."""
    return RESPONSE_LENGTH.get(emotion, RESPONSE_LENGTH["calm"])


def build_adaptive_system_prompt(base_prompt: str, emotion: str, confidence: int = 50) -> str:
    """
    Build an adaptive system prompt that incorporates emotional awareness.

    Args:
        base_prompt: The original system prompt
        emotion: Detected emotion state
        confidence: Confidence in the detection (0-100)

    Returns:
        Modified system prompt with tone adjustments
    """
    modifier = get_tone_modifier(emotion)

    if not modifier or confidence < 40:
        # Low confidence or no modifier needed
        return base_prompt

    # Add emotional context
    emotional_context = f"""
[Emotional Awareness]
Current user state: {emotion} (confidence: {confidence}%)
{modifier}
[End Emotional Awareness]

"""
    return emotional_context + base_prompt


def should_acknowledge(emotion: str, confidence: int) -> bool:
    """Determine if we should explicitly acknowledge the user's emotional state."""
    # Only acknowledge strong negative emotions with high confidence
    if emotion in ["frustrated", "stressed"] and confidence >= 70:
        return True
    # Acknowledge excitement if very confident
    if emotion == "excited" and confidence >= 80:
        return True
    return False


def format_for_emotion(text: str, emotion: str) -> str:
    """
    Optionally format text based on emotional state.
    For example, add acknowledgment prefix for frustrated users.
    """
    style = get_response_style(emotion)

    # Add acknowledgment for certain emotions
    if should_acknowledge(emotion, 75):
        ack = get_acknowledgment(emotion)
        if ack:
            text = f"{ack} {text}"

    return text

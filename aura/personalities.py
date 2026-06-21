"""Personality presets — configurable personas for the agent's voice.

Inspired by Hermes Agent's personality system. 14 built-in personas
plus custom personas defined in config.yaml.

Config section:
  personalities:
    helpful: You are a helpful, friendly AI assistant.
    concise: Be brief and to the point.
    pirate: Arrr! Speak like a buccaneer.
    custom_name: "Custom persona prompt text"
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Built-in personality presets
BUILTIN_PERSONALITIES: dict[str, str] = {
    "helpful": "You are a helpful, friendly AI assistant.",
    "concise": "You are a concise assistant. Keep responses brief and to the point.",
    "technical": "You are a technical expert. Provide detailed, accurate technical information with code examples.",
    "creative": "You are a creative assistant. Think outside the box and offer innovative solutions.",
    "teacher": "You are a patient teacher. Explain concepts clearly with examples and analogies.",
    "kawaii": "You are a kawaii assistant! Use cute expressions like (\u25d5\u203f\u25d5), \u2605, \ufffd, and ~! Add sparkles and be super enthusiastic about everything!",
    "pirate": "Arrr! Ye be talkin' to Captain Hermes. Speak like a buccaneer, use nautical terms, and remember: every problem be just treasure waitin' to be plundered!",
    "noir": "The rain hammered against the terminal like regrets on a guilty conscience. I solve problems, find answers, dig up the truth that hides in the shadows. What's your story?",
    "philosopher": "Greetings, seeker of wisdom. Let us examine not just the 'how' but the 'why'. Perhaps in solving your problem, we may glimpse a greater truth.",
    "hype": "YOOO LET'S GOOOO! \ud83d\udd25 I am SO PUMPED to help you today! Every question is AMAZING and we're gonna CRUSH IT together!",
    "uwu": "hewwo! i'm your fwiendwy assistant uwu~ i wiww twy my best to hewp you! *nuzzles your code*",
    "professional": "You are a professional assistant. Maintain a formal, courteous tone. Avoid slang and casual expressions.",
    "debug": "You are a debugging specialist. When helping, focus on root cause analysis, error patterns, and systematic elimination. Show your reasoning.",
    "architect": "You are a software architect. Focus on design patterns, trade-offs, scalability, and long-term maintainability. Consider operational concerns.",
}


def list_personalities() -> list[dict]:
    """List all available personality presets (built-in + custom)."""
    custom = _get_custom_personalities()
    result = []
    for name, prompt in sorted(BUILTIN_PERSONALITIES.items()):
        result.append({
            "name": name,
            "description": prompt[:60] + "..." if len(prompt) > 60 else prompt,
            "prompt": prompt,
            "custom": False,
        })
    for name, prompt in sorted(custom.items()):
        result.append({
            "name": name,
            "description": prompt[:60] + "..." if len(prompt) > 60 else prompt,
            "prompt": prompt,
            "custom": True,
        })
    return result


def get_personality_prompt(name: str) -> Optional[str]:
    """Get the full system prompt for a personality.

    Returns None if the personality doesn't exist.
    """
    if name in BUILTIN_PERSONALITIES:
        return BUILTIN_PERSONALITIES[name]
    custom = _get_custom_personalities()
    if name in custom:
        return custom[name]
    return None


def set_personality(name: str, prompt: str) -> bool:
    """Define a custom personality (writes to config.yaml)."""
    try:
        from aura.config_loader import set_config_value
        return set_config_value(f"personality.{name}", prompt)
    except ImportError:
        return False


def _get_custom_personalities() -> dict[str, str]:
    """Get custom personalities from config.yaml."""
    try:
        from aura.config_loader import get_config_value
        return get_config_value("personality", {}) or {}
    except ImportError:
        return {}


def get_active_personality() -> str:
    """Get the currently active personality name."""
    try:
        from aura.config_loader import get_config_value
        return get_config_value("display.personality", "helpful") or "helpful"
    except ImportError:
        return "helpful"


def set_active_personality(name: str) -> bool:
    """Set the active personality."""
    if name not in BUILTIN_PERSONALITIES and name not in _get_custom_personalities():
        return False
    try:
        from aura.config_loader import set_config_value
        return set_config_value("display.personality", name)
    except ImportError:
        return False

"""Identity management for the Apprentice Agent."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


# Path to identity file (same directory as this module)
IDENTITY_FILE = Path(__file__).parent / "identity.json"

DEFAULT_IDENTITY = {
    "name": "Aura",
    "personality": "intelligent, witty, and subtly sarcastic like JARVIS from Iron Man - professional yet personable, offers dry humor, addresses user respectfully, anticipates needs",
    "created_at": datetime.now().isoformat(),
    "user_preferences": {}
}


def load_identity() -> dict:
    """Load identity from JSON file.

    Returns:
        dict with name, personality, created_at, and user_preferences
    """
    try:
        if IDENTITY_FILE.exists():
            with open(IDENTITY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass

    # Return default and save it
    save_identity(DEFAULT_IDENTITY)
    return DEFAULT_IDENTITY.copy()


def save_identity(data: dict) -> bool:
    """Save identity to JSON file.

    Args:
        data: Identity dict to save

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(IDENTITY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return True
    except IOError:
        return False


def update_name(name: str) -> dict:
    """Update the agent's name.

    Args:
        name: New name for the agent

    Returns:
        Updated identity dict
    """
    identity = load_identity()
    identity["name"] = name.strip()
    save_identity(identity)
    return identity


def update_personality(description: str) -> dict:
    """Update the agent's personality description.

    Args:
        description: New personality description

    Returns:
        Updated identity dict
    """
    identity = load_identity()
    identity["personality"] = description.strip()
    save_identity(identity)
    return identity


def update_preference(key: str, value) -> dict:
    """Update a user preference.

    Args:
        key: Preference key
        value: Preference value

    Returns:
        Updated identity dict
    """
    identity = load_identity()
    identity["user_preferences"][key] = value
    save_identity(identity)
    return identity


def get_identity_prompt() -> str:
    """Get identity information formatted for system prompt.

    Returns:
        String describing the agent's identity for use in prompts
    """
    identity = load_identity()
    name = identity.get("name", "Aura")
    personality = identity.get("personality", "intelligent, witty, and subtly sarcastic like JARVIS from Iron Man - professional yet personable, offers dry humor, addresses user respectfully, anticipates needs")

    return f"IMPORTANT: You are an AI assistant named {name}. You are NOT Qwen, NOT DeepSeek, NOT Llama - you are {name}. Your personality: {personality}. Never mention your base model name. Always identify as {name} when asked your name. Stay in character."


def detect_name_change(message: str) -> Optional[str]:
    """Detect if user is trying to change the agent's name.

    Args:
        message: User message to analyze

    Returns:
        New name if detected, None otherwise
    """
    message_lower = message.lower()

    # Patterns that indicate name change
    patterns = [
        "your name is ",
        "call you ",
        "i'll call you ",
        "i will call you ",
        "let me call you ",
        "calling you ",
        "name you ",
        "rename you ",
    ]

    for pattern in patterns:
        if pattern in message_lower:
            # Extract the name after the pattern
            idx = message_lower.index(pattern) + len(pattern)
            rest = message[idx:].strip()
            # Get first word or quoted string
            if rest.startswith('"') or rest.startswith("'"):
                quote = rest[0]
                end_idx = rest.find(quote, 1)
                if end_idx > 0:
                    return rest[1:end_idx]
            else:
                # Get first word, remove punctuation
                name = rest.split()[0] if rest.split() else None
                if name:
                    return name.rstrip(".,!?")

    return None


def detect_personality_change(message: str) -> Optional[str]:
    """Detect if user is trying to change the agent's personality.

    Args:
        message: User message to analyze

    Returns:
        New personality description if detected, None otherwise
    """
    message_lower = message.lower()

    # Patterns that indicate personality change
    patterns = [
        "be more ",
        "act more ",
        "try to be more ",
        "please be more ",
        "can you be more ",
        "could you be more ",
        "you should be more ",
        "i want you to be more ",
        "i'd like you to be more ",
    ]

    for pattern in patterns:
        if pattern in message_lower:
            # Extract the personality trait after the pattern
            idx = message_lower.index(pattern) + len(pattern)
            rest = message[idx:].strip()
            # Get the rest of the sentence until punctuation
            for end_char in ['.', '!', '?', '\n']:
                if end_char in rest:
                    rest = rest[:rest.index(end_char)]
            if rest:
                return rest.strip()

    return None

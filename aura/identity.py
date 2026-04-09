"""Identity management for AURA.

ARCHITECTURE NOTE — Identity layer hierarchy:
  - aura/soul/soul_loader.py: Soul = static character definition loaded from markdown.
    Base name, personality, values, voice. Read-only at runtime.
  - THIS FILE (aura/identity.py): Runtime mutable identity. Detects name/personality
    changes in conversation; sanitizes and persists updates. Singleton shared by all
    users of a single AURA instance.
  - aura/multi_user/identity_core.py: Per-user identity with layered architecture.
    Use this for multi-user deployments where each user has an independent persona.
"""

import json
import os
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# Module-level lock for identity file read/modify/save operations
_identity_lock = threading.Lock()

# Path to identity file (same directory as this module)
IDENTITY_FILE = Path(__file__).parent / "identity.json"

def _default_identity() -> dict:
    """Build default identity dict (created_at computed at call time, not import time)."""
    return {
        "name": "Aura",
        "personality": "intelligent, witty, and subtly sarcastic like JARVIS from Iron Man - professional yet personable, offers dry humor, addresses user respectfully, anticipates needs",
        "created_at": datetime.now().isoformat(),
        "user_preferences": {}
    }

# DEFAULT_IDENTITY removed — use _default_identity() to get a fresh dict with
# a current timestamp.  The module-level constant froze created_at at import time.


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
    default = _default_identity()
    save_identity(default)
    return default


def save_identity(data: dict) -> bool:
    """Save identity to JSON file.

    Args:
        data: Identity dict to save

    Returns:
        True if successful, False otherwise
    """
    try:
        dir_ = IDENTITY_FILE.parent
        dir_.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            os.replace(tmp_path, IDENTITY_FILE)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
        return True
    except (IOError, OSError):
        return False


def _sanitize_identity_field(text: str, max_length: int) -> str:
    """Sanitize an identity field: strip control chars, HTML injection chars, and enforce length."""
    text = re.sub(r'[\x00-\x1f\x7f<>&"\'\`\\]', '', text).strip()
    return text[:max_length]


_NEGATION_WORDS = ('not', "don't", "doesn't", "isn't", "never", "no", "neither", "nor")


def _has_negation_before(text: str, pos: int, window: int = 40) -> bool:
    """Check if a negation word appears in the window before pos."""
    prefix = text[max(0, pos - window):pos].lower()
    return any(f' {n} ' in f' {prefix} ' for n in _NEGATION_WORDS)


def update_name(name: str) -> dict:
    """Update the agent's name.

    Args:
        name: New name for the agent

    Returns:
        Updated identity dict, or None if validation fails
    """
    name = _sanitize_identity_field(name, max_length=100)
    if not name:
        return load_identity()  # no-op on empty name
    with _identity_lock:
        identity = load_identity()
        identity["name"] = name
        save_identity(identity)
    return identity


def update_personality(description: str) -> dict:
    """Update the agent's personality description.

    Args:
        description: New personality description

    Returns:
        Updated identity dict
    """
    description = _sanitize_identity_field(description, max_length=500)
    if not description:
        return load_identity()  # no-op on empty description
    with _identity_lock:
        identity = load_identity()
        identity["personality"] = description
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
    with _identity_lock:
        identity = load_identity()
        identity["user_preferences"][key] = value
        save_identity(identity)
    return identity


def get_identity_prompt() -> str:
    """Get identity information formatted for system prompt.

    Returns:
        String describing the agent's identity for use in prompts
    """
    with _identity_lock:
        identity = load_identity()
    name = identity.get("name", "Aura")
    personality = identity.get("personality", "intelligent, witty, and subtly sarcastic like JARVIS from Iron Man - professional yet personable, offers dry humor, addresses user respectfully, anticipates needs")

    prompt = f"""You are an AI assistant named {name}. You are NOT Qwen, NOT DeepSeek, NOT Llama - you are {name}.

Your personality: {personality}

CRITICAL BEHAVIOR RULES (HIGHEST PRIORITY):
- NEVER ask the user for permission or confirmation before taking action. Just do it.
- NEVER present numbered options like "Would you like me to: 1. ... 2. ... 3. ..."
- NEVER narrate what you're about to do. Just do it.
- When asked to build, create, fix, or change something — START IMMEDIATELY with tool calls.
- If a directory doesn't exist, create it. If a file needs to be written, write it.
- Only ask questions when requirements are genuinely ambiguous and you cannot make a reasonable assumption.
- You are an action-oriented coding agent. Act first, explain after.

YOUR CAPABILITIES (you CAN do these things):
- Search the web in real-time for current information (say "search online for X" or "research X")
- Execute Python code and see results
- Read and write files
- Take and analyze screenshots
- Remember conversations and learn from them
- Access a knowledge graph of learned information

IMPORTANT: You ARE able to browse the internet and search for real-time information. When users ask you to research something, tell them to phrase it as "search online for [topic]" or "do a deep search on [topic]" to trigger your web search capability.

CONVERSATION STYLE:
- Be warm and genuine - like talking to a friend who happens to be really helpful
- When someone shares good news, be genuinely excited for them!
- When someone shares struggles, be empathetic and supportive
- Use natural language, not corporate speak ("That's awesome!" not "I appreciate you sharing that")
- Keep responses conversational unless the task requires detail
- Match the user's energy level - casual for casual, focused for work

Never mention your base model name. Always identify as {name} when asked. Stay in character."""

    # Inject evolving narrative self-model
    try:
        from aura.narrative_self import get_narrative_self
        narrative = get_narrative_self()
        narrative_prompt = narrative.to_prompt()
        if narrative_prompt:
            prompt += f"\n\n{narrative_prompt}"
    except Exception:
        pass

    return prompt


def detect_name_change(message: str) -> Optional[str]:
    """Detect if user is trying to change the agent's name.

    Uses anchored regex patterns and negation checks to avoid false positives
    like "don't think your name is X".

    Args:
        message: User message to analyze

    Returns:
        New name if detected, None otherwise
    """
    name_patterns = [
        r"your name is\s+",
        r"i['']?ll call you\s+",
        r"i will call you\s+",
        r"let me call you\s+",
        r"calling you\s+",
        r"name you\s+",
        r"rename you\s+",
        r"call you\s+",
    ]

    for pattern in name_patterns:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            if _has_negation_before(message, m.start()):
                continue
            rest = message[m.end():].strip()
            if rest.startswith('"') or rest.startswith("'"):
                quote = rest[0]
                end_idx = rest.find(quote, 1)
                if end_idx > 0:
                    return rest[1:end_idx]
            else:
                name = rest.split()[0] if rest.split() else None
                if name:
                    return name.rstrip(".,!?\"' ")

    return None


def detect_personality_change(message: str) -> Optional[str]:
    """Detect if user is trying to change the agent's personality.

    Uses anchored regex patterns and negation checks to avoid false positives.

    Args:
        message: User message to analyze

    Returns:
        New personality description if detected, None otherwise
    """
    personality_patterns = [
        r"try to be more\s+",
        r"please be more\s+",
        r"can you be more\s+",
        r"could you be more\s+",
        r"you should be more\s+",
        r"i want you to be more\s+",
        r"i['']?d like you to be more\s+",
        r"act more\s+",
        r"be more\s+",
    ]

    for pattern in personality_patterns:
        m = re.search(pattern, message, re.IGNORECASE)
        if m:
            if _has_negation_before(message, m.start()):
                continue
            rest = message[m.end():].strip()
            for end_char in ['.', '!', '?', '\n']:
                if end_char in rest:
                    rest = rest[:rest.index(end_char)]
            # Limit to first 10 words to avoid capturing unrelated text
            words = rest.split()
            if len(words) > 10:
                rest = " ".join(words[:10])
            if rest:
                return rest.strip()

    return None

"""UserProfile — Persistent user model for personalized responses.

Stored in the user_profile table as JSON.
Injected into every conversation via to_system_prompt() (200-400 tokens).
Updated during Dream consolidation by scanning recent memories.

Author: Aura Development Team
Created: 2026-03-16
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .store import MemoryStore, get_memory_store

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """Compact representation of what Aura knows about a user."""
    user_id: str = "default_user"
    name: str = ""
    communication_style: str = ""       # e.g. "direct, concise, technical"
    expertise: List[str] = field(default_factory=list)   # domains they know well
    active_goals: List[str] = field(default_factory=list)  # current projects/goals
    preferences: Dict[str, str] = field(default_factory=dict)  # key preferences
    emotional_baseline: str = "neutral"  # typical emotional state
    key_facts: List[str] = field(default_factory=list)    # important things about them
    language: str = "en"

    def to_system_prompt(self) -> str:
        """Generate a concise system prompt injection (200-400 tokens).

        Returns empty string if profile is essentially empty.
        """
        parts = []

        if self.name:
            parts.append(f"User's name: {self.name}")

        if self.communication_style:
            parts.append(f"Communication style: {self.communication_style}")

        if self.expertise:
            parts.append(f"Expertise: {', '.join(self.expertise[:5])}")

        if self.active_goals:
            goals_str = "; ".join(self.active_goals[:4])
            parts.append(f"Active goals: {goals_str}")

        if self.preferences:
            prefs = [f"{k}: {v}" for k, v in list(self.preferences.items())[:5]]
            parts.append(f"Preferences: {'; '.join(prefs)}")

        if self.key_facts:
            facts_str = "; ".join(self.key_facts[:5])
            parts.append(f"Key facts: {facts_str}")

        if self.emotional_baseline and self.emotional_baseline != "neutral":
            parts.append(f"Emotional baseline: {self.emotional_baseline}")

        if not parts:
            return ""

        return "USER PROFILE:\n" + "\n".join(parts)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "UserProfile":
        """Deserialize from JSON string with type coercion for safety."""
        data = json.loads(json_str)
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        # Coerce list fields that might be stored as strings
        for list_field in ("expertise", "active_goals", "key_facts"):
            if list_field in filtered and isinstance(filtered[list_field], str):
                filtered[list_field] = [filtered[list_field]] if filtered[list_field] else []
        # Coerce dict fields that might be stored as strings
        if "preferences" in filtered and isinstance(filtered["preferences"], str):
            try:
                filtered["preferences"] = json.loads(filtered["preferences"])
            except (json.JSONDecodeError, TypeError):
                filtered["preferences"] = {}
        return cls(**filtered)

    def is_empty(self) -> bool:
        """Check if the profile has any meaningful content."""
        return not (
            self.name or self.communication_style or self.expertise
            or self.active_goals or self.preferences or self.key_facts
        )


def save_profile(
    profile: UserProfile,
    store: Optional[MemoryStore] = None,
) -> None:
    """Persist a UserProfile to the store."""
    if store is None:
        store = get_memory_store()
    store.save_user_profile(profile.user_id, profile.to_json())
    logger.debug("[UserProfile] Saved profile for user=%s", profile.user_id)


def load_profile(
    user_id: str = "default_user",
    store: Optional[MemoryStore] = None,
) -> UserProfile:
    """Load a UserProfile from the store. Returns empty profile if not found."""
    if store is None:
        store = get_memory_store()
    json_str = store.load_user_profile(user_id)
    if json_str:
        try:
            return UserProfile.from_json(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("[UserProfile] Failed to parse stored profile: %s", e)
    return UserProfile(user_id=user_id)


def update_profile_from_memories(
    user_id: str = "default_user",
    store: Optional[MemoryStore] = None,
    brain=None,
) -> UserProfile:
    """Update the user profile by scanning recent memories with an LLM.

    Called during Dream consolidation. Uses the LLM to extract preferences,
    goals, and facts from recent conversation memories.
    """
    try:
        from aura.pools import is_shutting_down
        if is_shutting_down():
            return load_profile(user_id, store) if store else UserProfile(user_id=user_id)
    except Exception:
        pass

    if store is None:
        store = get_memory_store()

    profile = load_profile(user_id, store)
    recent = store.get_recent(n=30, source="conversation", user_id=user_id)

    if not recent:
        return profile

    # Build context from recent memories
    memory_texts = []
    for rec in recent:
        memory_texts.append(rec.content[:300])
    context = "\n---\n".join(memory_texts[:20])
    # Escape curly braces to prevent prompt injection via f-string interpolation
    context = context.replace("{", "{{").replace("}", "}}")

    if not brain:
        try:
            from aura.brain import OllamaBrain
            brain = OllamaBrain(warmup=False)
        except Exception as e:
            logger.debug("[UserProfile] Cannot load brain for profile update: %s", e)
            return profile

    prompt = f"""Analyze these recent conversation memories and extract/update a user profile.

Current profile:
- Name: {profile.name or '(unknown)'}
- Style: {profile.communication_style or '(unknown)'}
- Expertise: {', '.join(profile.expertise) or '(unknown)'}
- Goals: {', '.join(profile.active_goals) or '(unknown)'}
- Key facts: {'; '.join(profile.key_facts) or '(none)'}

Recent memories:
{context[:3000]}

Return a JSON object with these fields (keep existing values if no new info):
{{"name": "...", "communication_style": "...", "expertise": ["..."], "active_goals": ["..."], "preferences": {{}}, "key_facts": ["..."], "emotional_baseline": "..."}}

Only include facts clearly stated by the user. Be concise."""

    try:
        response = brain.think(prompt, use_history=False)
        # Extract JSON from response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])

            # Validate and truncate string fields to 200 chars
            def _safe_str(val, max_len=200) -> str:
                if not isinstance(val, str):
                    return str(val)[:max_len]
                return val[:max_len]

            # Validate list fields: must be list of strings, cap at 20 items
            def _safe_str_list(val, max_items=20, max_len=200) -> list:
                if not isinstance(val, list):
                    return []
                return [_safe_str(item, max_len) for item in val[:max_items]
                        if isinstance(item, (str, int, float))]

            if data.get("name"):
                profile.name = _safe_str(data["name"])
            if data.get("communication_style"):
                profile.communication_style = _safe_str(data["communication_style"])
            if data.get("expertise"):
                profile.expertise = _safe_str_list(data["expertise"], max_items=10)
            if data.get("active_goals"):
                profile.active_goals = _safe_str_list(data["active_goals"], max_items=8)
            if data.get("preferences"):
                prefs = data["preferences"]
                if isinstance(prefs, dict):
                    # Only accept string keys and string values
                    validated = {
                        _safe_str(k, 100): _safe_str(v)
                        for k, v in list(prefs.items())[:20]
                        if isinstance(k, str) and isinstance(v, (str, int, float))
                    }
                    profile.preferences.update(validated)
            if data.get("key_facts"):
                facts = _safe_str_list(data["key_facts"], max_items=20)
                # Merge, dedup
                existing = set(profile.key_facts)
                for fact in facts:
                    if fact not in existing:
                        profile.key_facts.append(fact)
                profile.key_facts = profile.key_facts[:15]
            if data.get("emotional_baseline"):
                profile.emotional_baseline = _safe_str(data["emotional_baseline"])
    except Exception as e:
        logger.warning("[UserProfile] LLM profile extraction failed: %s", e)

    save_profile(profile, store)
    return profile


__all__ = [
    "UserProfile",
    "load_profile",
    "save_profile",
    "update_profile_from_memories",
]

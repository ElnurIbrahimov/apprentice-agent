"""Shared helpers for AgentService."""

from __future__ import annotations

import logging
import re
import threading

logger = logging.getLogger(__name__)

_truth_spine_instance = None
_truth_spine_lock = threading.Lock()
_memory_tier = None


def filter_skill_json(text: str) -> str:
    """Remove leaked skill-learning prompts/JSON from user-facing responses."""
    text = re.sub(
        r"Analyze these successful interactions and extract a reusable skill\..*?"
        r"Respond ONLY with the JSON,? no other text\.?",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'\{[^{}]*"name"\s*:.*?"trigger_patterns"\s*:.*?"procedure"\s*:.*?\}',
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"Create a skill definition with:.*?Respond in this exact JSON format:",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_truth_spine():
    """Get or create the VerifiedMemory singleton in a thread-safe way."""
    global _truth_spine_instance, _memory_tier
    if _truth_spine_instance is None:
        with _truth_spine_lock:
            if _truth_spine_instance is None:
                try:
                    from aura.truth_spine import MemoryTier, VerifiedMemory

                    _truth_spine_instance = VerifiedMemory()
                    _memory_tier = MemoryTier
                except Exception as exc:
                    logger.warning("[TruthSpine] Init failed: %s", exc)
                    return None
    return _truth_spine_instance


class MemoryTierProxy:
    """Lazy proxy so callers can use MemoryTier values before initialization."""

    @property
    def FACT(self):
        return _memory_tier.FACT if _memory_tier else None

    @property
    def BELIEF(self):
        return _memory_tier.BELIEF if _memory_tier else None

    @property
    def SPECULATION(self):
        return _memory_tier.SPECULATION if _memory_tier else None


MemoryTier = MemoryTierProxy()


def record_thought(
    thought_type: str,
    content: str,
    intensity: float = 0.6,
    source: str = "service",
) -> None:
    """Record a thought event if the thinking API is available."""
    try:
        from api.routes.thinking import record_thought as route_record_thought

        route_record_thought(thought_type, content, intensity, source)
    except Exception:
        logger.debug("record_thought_failed", exc_info=True)

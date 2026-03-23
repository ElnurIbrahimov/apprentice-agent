"""Thought recording helper — shared between agent.py and narrative.py."""
import logging

logger = logging.getLogger(__name__)


def record_thought(thought_type: str, content: str, intensity: float = 0.6, source: str = "agent"):
    """Record a real thought event. Safe to call even if thinking system isn't ready."""
    try:
        from api.routes.thinking import record_thought as _api_record
        _api_record(thought_type, content, intensity, source)
    except Exception:
        logger.debug("[thought_recorder] non-critical: thought not recorded")

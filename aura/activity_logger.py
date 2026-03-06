"""Fire-and-forget activity logging. Call from anywhere in the aura package."""
import logging

logger = logging.getLogger(__name__)


def record_activity(
    category: str,
    event_type: str,
    summary: str,
    payload: dict = None,
    duration_ms: int = None,
) -> None:
    """Write one event to the activity log. Never raises."""
    try:
        from aura.proactive.persistence import get_persistence
        get_persistence().log_activity(category, event_type, summary,
                                       payload, duration_ms)
    except Exception as e:
        logger.debug(f"[ActivityLogger] {e}")

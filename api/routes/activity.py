"""Activity Timeline API."""
import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, Depends

from api.auth import require_api_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/activity", tags=["activity"], dependencies=[Depends(require_api_key)])


def _get_persistence():
    from aura.proactive.persistence import get_persistence
    return get_persistence()


@router.get("/events")
async def get_activity_events(
    limit: int = 100,
    after: float = 0.0,
    before: Optional[float] = None,
    categories: Optional[str] = None,
):
    """
    Return activity events newer than `after` (unix timestamp), newest first.
    Optional `before` for paginating older events.
    categories = comma-separated list, e.g. "tool,memory,emotion"
    """
    try:
        cat_list = [c.strip() for c in categories.split(",")] \
                   if categories else None
        loop = asyncio.get_running_loop()
        events = await loop.run_in_executor(
            None,
            lambda: _get_persistence().get_activity_events(
                limit, after, cat_list, before
            )
        )
        return {"events": events, "count": len(events)}
    except Exception as e:
        logger.error(f"[Activity] {e}")
        return {"events": [], "count": 0}

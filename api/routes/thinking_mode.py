"""API routes for System 1/2 explicit thinking-mode control."""

import logging
from fastapi import APIRouter, Depends

from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/thinking-mode", tags=["thinking-mode"], dependencies=[Depends(require_api_key)])


@router.get("/state")
async def get_thinking_mode_state():
    """Get current thinking mode, cognitive load, and effective decision info."""
    try:
        from aura.thinking_mode import get_thinking_mode_manager
        tmm = get_thinking_mode_manager()
        return {"success": True, **tmm.get_state()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/set")
async def set_thinking_mode(body: dict):
    """Set thinking mode to auto/system1/system2.

    Body: {"mode": "auto" | "system1" | "system2"}
    """
    try:
        from aura.thinking_mode import get_thinking_mode_manager, ThinkingMode
        tmm = get_thinking_mode_manager()

        mode_str = body.get("mode", "auto").lower()
        mode_map = {
            "auto": ThinkingMode.AUTO,
            "system1": ThinkingMode.SYSTEM1,
            "s1": ThinkingMode.SYSTEM1,
            "system2": ThinkingMode.SYSTEM2,
            "s2": ThinkingMode.SYSTEM2,
        }
        mode = mode_map.get(mode_str)
        if mode is None:
            return {"success": False, "error": f"Unknown mode: {mode_str}"}

        tmm.mode = mode
        return {"success": True, "mode": tmm.mode.value}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/reset-load")
async def reset_cognitive_load():
    """Reset the cognitive load tracker."""
    try:
        from aura.thinking_mode import get_thinking_mode_manager
        tmm = get_thinking_mode_manager()
        tmm.cognitive_load.reset()
        return {"success": True, "message": "Cognitive load tracker reset"}
    except Exception as e:
        return {"success": False, "error": str(e)}

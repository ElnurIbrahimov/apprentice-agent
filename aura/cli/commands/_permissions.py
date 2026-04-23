from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_permission_manager(agent: Any):
    from ..context import get_ctx

    ctx = get_ctx()
    if ctx and ctx.permissions:
        return ctx.permissions
    return getattr(agent, "permissions", None)


def confirm_action(
    agent: Any,
    action_key: str,
    args: dict[str, Any],
    *,
    fallback_prompt: str | None = None,
    allow_empty: bool = False,
) -> bool:
    """Route command-layer approvals through the permission manager when possible.

    Fail-closed: if the permission manager raises, deny the action instead of
    letting the exception propagate to a broad caller-level except that would
    otherwise silently grant permission.
    """
    pm = get_permission_manager(agent)
    if pm:
        try:
            return bool(pm.check(action_key, args))
        except Exception:
            logger.debug(
                "permission_check_raised action=%r; denying by default",
                action_key, exc_info=True,
            )
            return False

    if fallback_prompt is None:
        # Fail-closed: with no PM and no prompt to ask the user, deny by
        # default. Any caller that intentionally wants auto-grant must
        # supply an explicit prompt or a real permission manager.
        return False

    try:
        response = input(fallback_prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    if response in ("y", "yes"):
        return True
    if allow_empty and response == "":
        return True
    return False

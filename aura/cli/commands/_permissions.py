from __future__ import annotations

from typing import Any


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
    """Route command-layer approvals through the permission manager when possible."""
    pm = get_permission_manager(agent)
    if pm:
        return bool(pm.check(action_key, args))

    if fallback_prompt is None:
        return True

    try:
        response = input(fallback_prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    if response in ("y", "yes"):
        return True
    if allow_empty and response == "":
        return True
    return False

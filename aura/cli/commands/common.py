"""Shared helpers for slash-command handlers.

Keeps per-command modules lean and prevents drift (each file redoing
tilde expansion / path normalization with slightly different bugs).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

TIER_STABLE = "stable"
TIER_BETA = "beta"
TIER_EXPERIMENTAL = "experimental"

_COMMAND_REGISTRY: list[
    tuple[str, str, Callable[..., Any], list[str], str]
] = []


def command(
    name: str,
    description: str,
    aliases: list[str] | None = None,
    tier: str = TIER_STABLE,
    examples: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a slash-command handler in the central registry.

    Usage::

        @command("/help", "Show help")
        def handle_help(agent, arg, context):
            ...

    The decorated function is returned unchanged so it can still be imported
    and called directly. Examples are stored on the function as
    ``func.__aura_examples__`` so ``handle_help`` can show them via
    ``/help <cmd>`` without keeping a separate map.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _COMMAND_REGISTRY.append((name, description, func, aliases or [], tier))
        func.__aura_examples__ = list(examples) if examples else []  # type: ignore[attr-defined]
        return func
    return decorator


def resolve_user_path(raw: str) -> Path:
    """Normalize a user-supplied path.

    Expands ``~`` to the user's home directory and resolves any leading
    ``$VAR`` style env refs. Returns a ``Path`` object; the caller decides
    whether to call ``.resolve()`` (often unwise for non-existent files).

    On Windows ``Path("~/foo.py")`` returns a literal ``~`` folder, which
    confuses every downstream tool. Routing everything through this helper
    stops that bug class.
    """
    import os
    if not raw:
        return Path("")
    # expandvars first so `$HOME/foo` works even when the shell already expanded it
    expanded = os.path.expandvars(raw)
    return Path(expanded).expanduser()


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

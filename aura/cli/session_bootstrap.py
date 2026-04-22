"""Shared session bootstrap used by chat mode and one-shot agentic mode.

Centralizes the cwd + gather_context + AURA.md config + tier resolution + router
construction block, plus a `build_permission_manager()` helper that used to be
duplicated between oneshot.py and chat_session.py. Callers inject their own
confirm callback (Rich prompt for chat, JSON-mode-aware closure for oneshot).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from aura.core.context import gather_context, get_aura_md_config
from aura.core.router import ModelRouter

logger = logging.getLogger(__name__)


@dataclass
class SessionBootstrap:
    project_root: str
    project_context: str
    aura_config: dict
    tier: str
    budget: Optional[float]
    router: ModelRouter
    model: Optional[str]
    display_model: str


def build_session_bootstrap(args: Any, brain: Any = None) -> SessionBootstrap:
    """Build the shared setup for a chat or one-shot session.

    Args:
        args: Parsed argparse Namespace. Reads .tier, .budget, .model.
        brain: Optional brain reference. If supplied, `brain._model_override`
            is consulted as a fallback for the model choice (chat mode relies
            on this, one-shot does not).
    """
    project_root = os.getcwd()

    try:
        project_context = gather_context(project_root)
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("gather_project_context_failed", exc_info=True)
        project_context = f"Working directory: {project_root}"

    try:
        aura_config = get_aura_md_config(project_root) or {}
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("get_aura_md_config_failed", exc_info=True)
        aura_config = {}

    tier = getattr(args, "tier", None) or aura_config.get("tier", "balanced")
    budget = getattr(args, "budget", None) or aura_config.get("budget")
    router = ModelRouter(tier=tier, budget_usd=budget)

    brain_override = getattr(brain, "_model_override", None) if brain is not None else None
    model = getattr(args, "model", None) or brain_override or aura_config.get("model") or None
    display_model = model or f"auto-route ({tier})"

    return SessionBootstrap(
        project_root=project_root,
        project_context=project_context,
        aura_config=aura_config,
        tier=tier,
        budget=budget,
        router=router,
        model=model,
        display_model=display_model,
    )


def build_permission_manager(
    *,
    aura_config: Optional[dict] = None,
    trust: bool = False,
    default_mode: str = "careful",
    confirm_callback: Optional[Callable[[str, str], str]] = None,
) -> Any:
    """Construct the shared PermissionManager used by chat, oneshot, and fast paths.

    Args:
        aura_config: Parsed AURA.md dict. If provided, ``apply_aura_md_overrides``
            runs after the mode is set.
        trust: True → mode is forced to ``full_auto`` regardless of ``default_mode``.
        default_mode: Mode applied when trust is False. Chat mode uses
            ``auto_edit``, oneshot/fast paths use ``careful``.
        confirm_callback: Callable ``(tool_name, description) -> str`` returning one
            of ``'allow_once' | 'allow_session' | 'allow_always' | 'deny'``. Attached
            only if not None.
    """
    from aura.core.permissions import PermissionManager

    pm = PermissionManager()
    pm.set_mode("full_auto" if trust else default_mode)
    if aura_config:
        pm.apply_aura_md_overrides(aura_config)
    if confirm_callback is not None:
        pm.set_confirm_callback(confirm_callback)
    return pm

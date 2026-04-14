"""Shared session bootstrap used by chat mode and one-shot agentic mode.

Centralizes the cwd + gather_context + AURA.md config + tier resolution + router
construction block that was duplicated between oneshot.py and chat_session.py.

Does NOT construct the PermissionManager: chat mode and one-shot use different
permission UIs (rich `_cli_confirm` method vs. plain closure), different default
modes (`auto_edit` vs `careful`), and different callback attachment points. Each
caller builds its own permission manager; the bootstrap only supplies everything
up to that point.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

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

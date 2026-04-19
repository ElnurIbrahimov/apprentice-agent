"""/shadow slash command — run the current prompt against two models in parallel."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from aura.cli.display import console
except ImportError:
    from rich.console import Console
    console = Console()


def handle_shadow(agent, arg, context) -> Optional[str]:
    """`/shadow <prompt>` — dispatch against 2 models in parallel, show diff."""
    prompt = (arg or "").strip()
    if not prompt:
        console.print("  Usage: /shadow <prompt>  (runs primary + best alternative in parallel)")
        return None

    try:
        from aura.routing.dispatcher import dispatch
        from aura.core.shadow_mode import render_shadow_result, run_shadow
    except Exception as e:
        console.print(f"  [red]Shadow mode unavailable:[/] {e}")
        return None

    d = dispatch(prompt)
    if not d.fallback_models:
        console.print(f"  [yellow]Only one candidate for this task class ({d.task_class}) — nothing to shadow[/]")
        return None

    primary = d.model
    shadow = d.fallback_models[0]
    console.print(
        f"  [dim]dispatching {primary} + {shadow} (task={d.task_class}, conf={d.confidence:.2f})[/]"
    )

    try:
        result = run_shadow(agent.brain, prompt, primary, shadow)
        render_shadow_result(result)
    except Exception as e:
        console.print(f"  [red]Shadow run failed:[/] {e}")
        return None

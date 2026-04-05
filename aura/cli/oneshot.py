from __future__ import annotations

import os
import sys
from typing import Any, NoReturn


def run_agentic_oneshot(agent: Any, prompt: str, args: Any, bridge: Any = None) -> NoReturn:
    from .display import show_banner, console
    from aura.core.agentic_loop import run_agentic
    from aura.core.context import gather_context, get_aura_md_config
    from aura.core.router import ModelRouter
    from aura.core.permissions import PermissionManager

    show_banner()

    project_root = os.getcwd()

    context = gather_context(project_root)
    aura_config = get_aura_md_config(project_root)

    tier = args.tier or aura_config.get("tier", "balanced")
    budget = args.budget or aura_config.get("budget")
    router = ModelRouter(tier=tier, budget_usd=budget)
    model = args.model or aura_config.get("model") or None
    display_model = model or f"auto-route ({tier})"

    permissions = PermissionManager()
    if aura_config:
        permissions.apply_aura_md_overrides(aura_config)

    def _confirm(tool_name: str, description: str) -> bool | str:
        console.print(f"\n  [yellow]Permission required:[/yellow]")
        console.print(f"    [bold]{tool_name}[/bold]")
        for line in description.split("\n"):
            console.print(f"    {line}")
        try:
            resp = console.input("    [bold]Allow? [y/n/always]: [/bold]").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if resp == "always":
            return "always"
        return resp in ("y", "yes")

    permissions.set_confirm_callback(_confirm)
    if args.trust:
        permissions.set_trust_mode(True)

    console.print(f"  [dim]Model: {display_model} | Tier: {tier}[/dim]")
    console.print()

    try:
        result = run_agentic(
            brain=agent.brain,
            prompt=prompt,
            project_root=project_root,
            permissions=permissions,
            model_override=model,
            max_iterations=args.max_iterations,
            budget_usd=budget,
            context=context,
            trust_mode=args.trust,
            aura_config=aura_config,
            router=router,
        )
    except KeyboardInterrupt:
        console.print("\n  [red]Aborted.[/red]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n  [red]Agent execution failed: {e}[/red]")
        sys.exit(1)
    finally:
        if bridge:
            bridge.stop()

    try:
        stats = agent.brain.get_session_stats()
        cost = stats.get("cost_usd", 0.0)
    except (AttributeError, TypeError, KeyError):
        cost = 0.0
    console.print(f"\n  [dim]{result['iterations']} iterations, {result['tool_calls']} tool calls, ${cost:.4f}[/dim]")

    sys.exit(0 if result.get("success") else 1)

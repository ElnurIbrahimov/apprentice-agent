from __future__ import annotations

import os
import sys
from typing import Any, NoReturn


def run_agentic_oneshot(agent: Any, prompt: str, args: Any, bridge: Any = None) -> NoReturn:
    from aura.core.agentic_loop import run_agentic
    from aura.core.permissions import PermissionManager

    from .display import console, show_banner
    from .session_bootstrap import build_session_bootstrap

    show_banner()

    boot = build_session_bootstrap(args, brain=getattr(agent, "brain", None))

    permissions = PermissionManager()
    permissions.set_mode("careful")
    if boot.aura_config:
        permissions.apply_aura_md_overrides(boot.aura_config)

    def _confirm(tool_name: str, description: str) -> bool | str:
        console.print("\n  [yellow]Permission required:[/yellow]")
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
        permissions.set_mode("full_auto")

    console.print(f"  [dim]Model: {boot.display_model} | Tier: {boot.tier}[/dim]")
    console.print()

    try:
        result = run_agentic(
            brain=agent.brain,
            prompt=prompt,
            project_root=boot.project_root,
            permissions=permissions,
            model_override=boot.model,
            max_iterations=args.max_iterations,
            budget_usd=boot.budget,
            context=boot.project_context,
            trust_mode=args.trust,
            aura_config=boot.aura_config,
            router=boot.router,
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

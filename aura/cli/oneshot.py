from __future__ import annotations

import os
import sys
from typing import Any, NoReturn


def run_agentic_oneshot(agent: Any, prompt: str, args: Any, bridge: Any = None) -> NoReturn:
    from aura.core.agentic_loop import run_agentic
    from aura.core.permissions import PermissionManager

    from .display import console, show_banner
    from .pipe_mode import StreamingJSONEmitter
    from .session_bootstrap import build_session_bootstrap

    # When caller asked for JSON output, suppress the banner and rich prompts
    # so stdout stays valid JSONL for scripted consumers.
    json_mode = getattr(args, "format", "text") == "json"

    if not json_mode:
        show_banner()

    boot = build_session_bootstrap(args, brain=getattr(agent, "brain", None))

    permissions = PermissionManager()
    permissions.set_mode("careful")
    if boot.aura_config:
        permissions.apply_aura_md_overrides(boot.aura_config)

    def _confirm(tool_name: str, description: str) -> bool | str:
        if json_mode:
            # In JSON mode we can't prompt — deny by default. Callers that need
            # tool calls should pass --trust or set AURA.md permissions to auto.
            return False
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

    # Wire event callbacks for JSONL streaming output.
    on_chunk = on_tool_start = on_tool_call = on_response = None
    emitter: StreamingJSONEmitter | None = None
    if json_mode:
        emitter = StreamingJSONEmitter()
        def on_chunk(text: str) -> None:
            if text:
                emitter.emit_chunk(text)
        def on_tool_start(tool: str, tool_args: dict) -> None:
            emitter.emit_tool_start(tool, tool_args)
        def on_tool_call(tool: str, tool_args: dict, result: Any) -> None:
            emitter.emit_tool_result(tool, tool_args, result)
        # Suppress the default console.print of each response iteration.
        def on_response(text: str, iteration: int) -> None:
            return None

    if not json_mode:
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
            on_chunk=on_chunk,
            on_tool_start=on_tool_start,
            on_tool_call=on_tool_call,
            on_response=on_response,
        )
    except KeyboardInterrupt:
        if not json_mode:
            console.print("\n  [red]Aborted.[/red]")
        sys.exit(130)
    except Exception as e:
        if json_mode and emitter is not None:
            emitter.emit({"type": "error", "message": str(e)})
        else:
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

    if json_mode and emitter is not None:
        emitter.emit_final(
            response=result.get("response", ""),
            model=result.get("model", ""),
            cost=cost,
            iterations=result.get("iterations", 0),
            tool_calls=result.get("tool_calls", 0),
            success=bool(result.get("success")),
        )
    else:
        console.print(f"\n  [dim]{result['iterations']} iterations, {result['tool_calls']} tool calls, ${cost:.4f}[/dim]")

    sys.exit(0 if result.get("success") else 1)

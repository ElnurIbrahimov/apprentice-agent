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

    mode = getattr(args, "mode", "chat")
    if mode == "debate":
        if not json_mode:
            show_banner()
        from .debate_mode import run_debate
        try:
            run_debate(agent.brain, prompt)
        except KeyboardInterrupt:
            sys.exit(130)
        except Exception as e:
            console.print(f"\n  [red]Debate failed: {e}[/red]")
            sys.exit(1)
        finally:
            if bridge:
                bridge.stop()
        sys.exit(0)

    if mode == "chain":
        if not json_mode:
            show_banner()
        from .chain_mode import parse_chain, run_chain
        steps = parse_chain(prompt)
        if not steps:
            console.print("[red]Chain mode requires 'step1 -> step2 -> step3' syntax.[/red]")
            sys.exit(1)
        try:
            result = run_chain(agent.brain, steps)
        except KeyboardInterrupt:
            sys.exit(130)
        except Exception as e:
            console.print(f"\n  [red]Chain failed: {e}[/red]")
            sys.exit(1)
        finally:
            if bridge:
                bridge.stop()
        # Print each step output for scriptability.
        total = len(result.step_results)
        for step_result in result.step_results:
            console.print(f"\n[bold cyan]── Step {step_result['step']}/{total} ──[/bold cyan]")
            console.print(f"[dim]prompt: {step_result['prompt']} (model: {step_result['model']})[/dim]")
            console.print(step_result.get("response", ""))
        sys.exit(0 if result.success else 1)

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
        from .permissions_dialog import request_permission
        return request_permission(console, tool_name, description)

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
        # Preview recalled memories for the current prompt (read-only, same
        # query the agentic loop is about to run). Shows users that Aura has
        # long-term recall for the context it's answering in.
        try:
            from aura.core.agentic_loop_support import _recall_memories
            _recall_memories(prompt)
            n = getattr(_recall_memories, "last_count", 0)
            top = getattr(_recall_memories, "last_top", "")
            if n:
                snippet = (top[:60] + "...") if len(top) > 60 else top
                console.print(f"  [dim cyan]\u25ce recalled {n} memories (top: {snippet!r})[/dim cyan]")
        except Exception:
            pass
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

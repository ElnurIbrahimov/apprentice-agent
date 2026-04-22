from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from typing import Any, NoReturn

logger = logging.getLogger(__name__)


def _install_wallclock_timeout(seconds: int) -> None:
    """Install a hard wall-clock timeout that signals the main thread.

    Uses threading.Timer so it works cross-platform (signal.SIGALRM is
    Unix-only). On fire, we signal the main thread with SIGINT so atexit
    handlers, session saves, and finally blocks all get to run — unlike
    ``os._exit(124)`` which skipped them and lost in-flight conversation
    history. The main thread's top-level handler treats SIGINT during
    oneshot as "cancel the run and exit 124".
    """
    if seconds <= 0:
        return

    def _kill():
        try:
            sys.stderr.write(f"\n[aura exec] hard timeout after {seconds}s — aborting\n")
            sys.stderr.flush()
        except Exception:
            pass
        # Graceful path: raise KeyboardInterrupt on the main thread so
        # finally blocks run. Fall back to os._exit only if the interrupt
        # fails (shouldn't normally).
        try:
            import _thread
            _thread.interrupt_main()
            # Give the main thread a moment to unwind; os._exit as belt-and-
            # suspenders if it doesn't.
            import time as _t
            _t.sleep(2.0)
        except Exception:
            pass
        os._exit(124)

    t = threading.Timer(seconds, _kill)
    t.daemon = True
    t.start()


def run_agentic_oneshot(agent: Any, prompt: str, args: Any, bridge: Any = None) -> NoReturn:
    from aura.core.agentic_loop import run_agentic

    from .display import console, show_banner
    from .pipe_mode import StreamingJSONEmitter
    from .session_bootstrap import build_permission_manager, build_session_bootstrap

    # When caller asked for JSON output, suppress the banner and rich prompts
    # so stdout stays valid JSONL for scripted consumers.
    json_mode = getattr(args, "format", "text") == "json"
    quiet_mode = bool(getattr(args, "quiet", False))
    output_failures = bool(getattr(args, "output_failures", False))
    exec_timeout = int(getattr(args, "exec_timeout", 0) or 0)

    if exec_timeout > 0:
        _install_wallclock_timeout(exec_timeout)

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

    # Emitter is created up front so _confirm can reach it without a forward ref.
    emitter: StreamingJSONEmitter | None = StreamingJSONEmitter() if json_mode else None

    def _confirm(tool_name: str, description: str) -> str:
        if json_mode:
            # In JSON mode we can't prompt — announce the denial so scripted
            # consumers see why the agent stopped touching tools. Callers that
            # need tool calls should pass --trust or set AURA.md permissions
            # to auto.
            if emitter is not None:
                emitter.emit_permission_denied(tool_name, description)
            return "deny"
        from .permissions_dialog import request_permission
        return request_permission(console, tool_name, description)

    permissions = build_permission_manager(
        aura_config=boot.aura_config,
        trust=bool(getattr(args, "trust", False)),
        default_mode="careful",
        confirm_callback=_confirm,
    )

    # Wire event callbacks for JSONL streaming output.
    on_chunk = on_tool_start = on_tool_call = on_response = None
    if json_mode:
        assert emitter is not None  # narrow for type-checkers
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

    if not json_mode and not quiet_mode:
        console.print(f"  [dim]Model: {boot.display_model} | Tier: {boot.tier}[/dim]")
        # Preview recalled memories for the current prompt (read-only, same
        # query the agentic loop is about to run). Shows users that Aura has
        # long-term recall for the context it's answering in.
        try:
            from aura.core.agentic_loop_support import _recall_memories
            recall = _recall_memories(prompt)
            if recall.count:
                snippet = (recall.top[:60] + "...") if len(recall.top) > 60 else recall.top
                console.print(f"  [dim cyan]\u25ce recalled {recall.count} memories (top: {snippet!r})[/dim cyan]")
        except Exception as e:
            logger.debug(f"[Oneshot] Memory preview recall failed: {e}")
        console.print()

    # --resume sets agent._resume_session_id in main._handle_resume(). Consume
    # it here so the prior session's message history is loaded before the
    # prompt runs — previously this attribute was silently discarded in the
    # non-interactive path.
    _resume_id = getattr(agent, "_resume_session_id", None)
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
            resume_session_id=_resume_id,
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
    elif quiet_mode:
        response_text = result.get("response", "").strip()
        if response_text:
            print(response_text)
    else:
        console.print(f"\n  [dim]{result['iterations']} iterations, {result['tool_calls']} tool calls, ${cost:.4f}[/dim]")

    if output_failures:
        failures = result.get("tool_failures") or []
        if failures:
            import json as _json
            for f in failures:
                sys.stderr.write(_json.dumps({"type": "tool_failure", **f}) + "\n")

    sys.exit(0 if result.get("success") else 1)

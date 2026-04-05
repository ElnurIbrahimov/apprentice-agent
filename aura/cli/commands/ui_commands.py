import sys
import logging
from typing import Optional

from ..context import get_ctx

logger = logging.getLogger(__name__)


def handle_model(agent, arg, context) -> Optional[str]:
    ctx = get_ctx()
    if not arg:
        from ..display import console, show_info
        from ..model_picker import pick_model
        current = agent.brain._model_override or "auto"
        choice = pick_model(console, current)
        if choice is not None:
            if choice == "auto":
                agent.brain.set_model_override(None)
                if ctx and ctx.agentic_loop:
                    ctx.agentic_loop.model_override = None
                show_info("Model override cleared. Using auto-selection.")
            else:
                agent.brain.set_model_override(choice)
                if ctx and ctx.agentic_loop:
                    ctx.agentic_loop.model_override = choice
                show_info(f"Model locked to: {choice}")
    elif arg.lower() == "auto":
        agent.brain.set_model_override(None)
        if ctx and ctx.agentic_loop:
            ctx.agentic_loop.model_override = None
        print("Model override cleared. Using auto-selection.")
    else:
        agent.brain.set_model_override(arg)
        if ctx and ctx.agentic_loop:
            ctx.agentic_loop.model_override = arg
        print(f"Model locked to: {arg}")


def handle_theme(agent, arg, context) -> Optional[str]:
    from ..themes import set_theme as _set_theme, get_theme as _get_theme, list_themes as _list_themes, save_theme_preference as _save_pref
    from ..display import console as _theme_console
    if arg:
        if _set_theme(arg.strip()):
            _save_pref(arg.strip())
            _theme_console.print(f"[green]Theme set to: {arg.strip()}[/green]")
        else:
            _theme_console.print(f"[red]Unknown theme: {arg.strip()}[/red]")
            _theme_console.print(f"[dim]Available: {', '.join(_list_themes())}[/dim]")
    else:
        current = _get_theme().name
        available = _list_themes()
        _theme_console.print(f"[bold]Current theme:[/bold] {current}")
        _theme_console.print(f"[bold]Available:[/bold] {', '.join(available)}")


def handle_mood(agent, arg, context) -> Optional[str]:
    from ..display import console as _mood_console
    from ..mood_display import render_mood_detail
    try:
        from aura.emotion.alma_engine import get_alma_engine
        engine = get_alma_engine()
        state = engine.get_emotional_state() if engine else {}
    except Exception:  # Catch-all: mood engine is cosmetic, must not crash /mood command
        logger.debug("mood_command_state_failed", exc_info=True)
        state = {}
    if state:
        render_mood_detail(_mood_console, state)
    else:
        _mood_console.print("[dim]Emotional state not available.[/dim]")


def handle_speak(agent, arg, context) -> Optional[str]:
    if arg:
        agent._speak(arg)
        print(f"[Spoke: {arg}]")
    else:
        print("Usage: /speak <text to speak>")


def handle_trust(agent, arg, context) -> Optional[str]:
    ctx = get_ctx()
    if not ctx or not ctx.permissions:
        from aura.core.permissions import PermissionManager
        print("  Permissions not available outside chat mode.")
        return

    if arg and arg.strip().lower() == "off":
        ctx.permissions.set_trust_mode(False)
        print("  Trust mode disabled — tool calls require approval.")
    else:
        ctx.permissions.set_trust_mode(True)
        print("  Trust mode enabled — all tool calls auto-approved.")


def handle_help(agent, arg, context) -> Optional[str]:
    """Show help. Use /help <command> for detailed help on a specific command."""
    from ..display import show_help, console

    if not arg or not arg.strip():
        show_help()
        return None

    # Show help for specific command
    cmd = arg.strip()
    if not cmd.startswith("/"):
        cmd = f"/{cmd}"

    from . import COMMAND_REGISTRY
    handler = COMMAND_REGISTRY.get(cmd)

    if handler:
        doc = handler.__doc__ or "No description available."
        console.print(f"\n  [bold cyan]{cmd}[/bold cyan]")
        for line in doc.strip().split("\n"):
            console.print(f"  {line.strip()}")

        # Built-in examples
        EXAMPLES = {
            "/model": ["/model              -- open interactive picker",
                       "/model qwen3:8b     -- lock to specific model",
                       "/model auto         -- return to auto-routing"],
            "/copy": ["/copy               -- copy last response",
                      "/copy code          -- copy first code block",
                      "/copy code 2        -- copy 2nd code block"],
            "/fleet": ["/fleet build auth   -- decompose and run in parallel"],
            "/chain": ["/chain step1 -> step2 -> step3",
                       "/chain list         -- show saved chains"],
            "/test": ["/test               -- run configured test command",
                      "/test --fix         -- auto-fix on failure"],
            "/shell": ["/shell ls -la       -- run shell command"],
            "/grep": ["/grep TODO          -- search for pattern",
                      "/grep -t py error   -- search .py files"],
            "/research": ["/research start topic  -- begin research mode",
                          "/research stop         -- end research mode"],
            "/debate": ["/debate Should we use X or Y?"],
        }

        if cmd in EXAMPLES:
            console.print(f"\n  [dim]Examples:[/dim]")
            for ex in EXAMPLES[cmd]:
                console.print(f"    [dim]{ex}[/dim]")
        console.print()
    else:
        console.print(f"  Unknown command: {cmd}")
        from difflib import get_close_matches
        known = list(COMMAND_REGISTRY.keys())
        matches = get_close_matches(cmd, known, n=3, cutoff=0.5)
        if matches:
            console.print(f"  Did you mean: {', '.join(matches)}?")
        console.print()

    return None


def handle_quit(agent, arg, context) -> Optional[str]:
    ctx = get_ctx()
    hook_mgr = ctx.hook_manager if ctx else None
    if hook_mgr:
        from ..hooks import HookEvent as _HE
        hook_mgr.fire(_HE.SESSION_END, {"reason": "quit_command"})
    print("Goodbye!")
    sys.exit(0)


def handle_tasks(agent, arg, context) -> Optional[str]:
    from ..display import console as _tasks_console
    ctx = get_ctx()
    bg_manager = ctx.bg_manager if ctx else None
    if bg_manager:
        from ..background import render_tasks_table
        render_tasks_table(_tasks_console, bg_manager.list_tasks())
    else:
        _tasks_console.print("[dim]No background tasks.[/dim]")

import logging
import sys
from typing import Optional

from ..context import get_ctx
from ..display import console
from .common import command, TIER_BETA, TIER_STABLE

logger = logging.getLogger(__name__)


def _set_model(ctx, agent, choice: Optional[str]) -> None:
    """Route the model override through ChatSession.apply_model_override when available,
    falling back to the two-mirror write for non-interactive entry points (ACP, MCP)
    that build a CLIContext without a ChatSession."""
    if ctx and getattr(ctx, "chat_session", None):
        ctx.chat_session.apply_model_override(choice)
        return
    agent.brain.set_model_override(choice)
    if ctx and ctx.agentic_loop:
        ctx.agentic_loop.model_override = choice


@command("/model",    "View/set model (auto, <name>)",                  tier=TIER_STABLE,
          examples=[
              "/model                        -- open interactive picker",
              "/model qwen3:8b               -- lock to specific model",
              "/model auto                   -- return to auto-routing",
          ])
def handle_model(agent, arg, context) -> Optional[str]:
    ctx = get_ctx()
    if not arg:
        from ..display import show_info
        from ..model_picker import pick_model
        current = (ctx.chat_session.current_model
                   if ctx and getattr(ctx, "chat_session", None)
                   else (agent.brain._model_override or "auto"))
        choice = pick_model(console, current)
        if choice is not None:
            if choice == "auto":
                _set_model(ctx, agent, None)
                show_info("Model override cleared. Using auto-selection.")
            else:
                _set_model(ctx, agent, choice)
                show_info(f"Model locked to: {choice}")
    elif arg.lower() == "auto":
        _set_model(ctx, agent, None)
        console.print("[green]Model override cleared. Using auto-selection.[/green]")
    else:
        _set_model(ctx, agent, arg)
        console.print(f"[green]Model locked to: [cyan]{arg}[/cyan][/green]")


@command("/theme",    "Switch color theme",                             tier=TIER_STABLE)
def handle_theme(agent, arg, context) -> Optional[str]:
    from ..display import console as _theme_console
    from ..themes import get_theme as _get_theme
    from ..themes import list_themes as _list_themes
    from ..themes import save_theme_preference as _save_pref
    from ..themes import set_theme as _set_theme
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


@command("/mood",     "Show emotional state",                             tier=TIER_BETA)
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


@command("/speak",    "Text-to-speech",         aliases=["/say"],                tier=TIER_BETA)
def handle_speak(agent, arg, context) -> Optional[str]:
    if arg:
        agent._speak(arg)
        console.print(f"[dim][Spoke: {arg}][/dim]")
    else:
        console.print("[yellow]Usage: /speak <text to speak>[/yellow]")


@command("/trust",    "Enable trust mode (auto-approve all tools)",     tier=TIER_STABLE)
def handle_trust(agent, arg, context) -> Optional[str]:
    ctx = get_ctx()
    if not ctx or not ctx.permissions:
        console.print("  [yellow]Permissions not available outside chat mode.[/yellow]")
        return

    if arg and arg.strip().lower() == "off":
        ctx.permissions.set_trust_mode(False)
        console.print("  [yellow]Trust mode disabled — tool calls require approval.[/yellow]")
    else:
        ctx.permissions.set_trust_mode(True)
        console.print("  [green]Trust mode enabled — all tool calls auto-approved.[/green]")


@command("/help",     "Show help",                                        tier=TIER_STABLE)
def handle_help(agent, arg, context) -> Optional[str]:
    """Show help. Use /help <command> for detailed help on a specific command.
    `/help all` also shows experimental commands."""
    from ..display import console, show_help

    arg_stripped = (arg or "").strip().lower()
    if not arg_stripped:
        show_help()
        return None
    if arg_stripped in ("all", "--all", "-a"):
        show_help(show_experimental=True)
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

        # Examples are declared at the @command() site via the
        # ``examples=[...]`` kwarg and stored on the function as
        # ``__aura_examples__``. This is the single source of truth
        # (no separate EXAMPLES dict to drift from the registry).
        examples = getattr(handler, "__aura_examples__", []) or []
        if examples:
            console.print("\n  [dim]Examples:[/dim]")
            for ex in examples:
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


@command("/quit",     "Exit AURA",              aliases=["/exit"],               tier=TIER_STABLE)
def handle_quit(agent, arg, context) -> Optional[str]:
    ctx = get_ctx()
    hook_mgr = ctx.hook_manager if ctx else None
    if hook_mgr:
        from ..hooks import HookEvent as _HE
        # wait=True: process is about to exit, async hooks would get killed
        # mid-run when sys.exit tears down the bg pool.
        hook_mgr.fire(_HE.SESSION_END, {"reason": "quit_command"}, wait=True)
    console.print("[dim]Goodbye![/dim]")
    sys.exit(0)


@command("/routing",  "Show/set routing preference",                      tier=TIER_BETA)
def handle_routing(agent, arg, context) -> Optional[str]:
    """Show current neural routing status and the last ModelRouter decision."""
    from ..display import console as _routing_console

    # Section 1: conversation-aware neural routing preference
    try:
        from aura.routing.router import get_router
        router = get_router()

        conv_id = "default"
        try:
            ctx = get_ctx()
            if ctx and hasattr(ctx, 'agentic_loop'):
                conv_id = getattr(ctx.agentic_loop, '_conversation_id', None) or 'default'
        except Exception:
            pass

        profile = router.conversations.get_profile(conv_id)

        _routing_console.print()
        _routing_console.print("[bold cyan]  Routing Status[/bold cyan]")
        _routing_console.print(f"  Preference: {getattr(agent.brain, '_routing_preference', 'balanced')}")
        _routing_console.print(f"  Conversation: {profile.turn_count} turns, {profile.total_tokens} tokens")
        _routing_console.print(f"  Code mode: {'ON' if profile.in_code_mode else 'off'}")
        _routing_console.print(f"  Complexity trend: {profile.complexity_trend:+.2f}")
        _routing_console.print(f"  Last model: {profile.last_model or 'none'}")
        _routing_console.print(f"  Regens: {profile.regen_count}, Switches: {profile.model_switches}")
    except Exception as e:
        from ..display import console as _err_console
        _err_console.print(f"  [red]Neural routing info unavailable: {e}[/red]")

    # Section 2: last ModelRouter tier/category decision (from agentic_loop)
    try:
        ctx = get_ctx()
        model_router = None
        if ctx and ctx.agentic_loop is not None:
            model_router = getattr(ctx.agentic_loop, "_router", None)
        if model_router is not None and hasattr(model_router, "last_decision"):
            decision = model_router.last_decision()
            if decision:
                _routing_console.print()
                _routing_console.print("[bold cyan]  Last ModelRouter Decision[/bold cyan]")
                _routing_console.print(f"  Category:   {decision.get('category', '?')}")
                _routing_console.print(f"  Confidence: {decision.get('confidence', 0):.3f}")
                _routing_console.print(f"  Tier:       {decision.get('tier', '?')}")
                _routing_console.print(f"  Model:      {decision.get('model', '?')}")
                snippet = decision.get('prompt_snippet', '')
                if snippet:
                    _routing_console.print(f"  Prompt:     [dim]{snippet}[/dim]")
            else:
                _routing_console.print()
                _routing_console.print("[dim]  ModelRouter: no decision recorded yet.[/dim]")
    except Exception as e:
        _routing_console.print(f"  [red]ModelRouter info unavailable: {e}[/red]")
    _routing_console.print()


@command("/tasks",    "Show background tasks",                            tier=TIER_BETA)
def handle_tasks(agent, arg, context) -> Optional[str]:
    from ..display import console as _tasks_console
    ctx = get_ctx()
    bg_manager = ctx.bg_manager if ctx else None
    if bg_manager:
        from ..background import render_tasks_table
        render_tasks_table(_tasks_console, bg_manager.list_tasks())
    else:
        _tasks_console.print("[dim]No background tasks.[/dim]")

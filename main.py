#!/usr/bin/env python3
"""Main entry point for the Apprentice Agent."""

import os
os.environ["TQDM_DISABLE"] = "1"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="comtypes")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pycaw")
warnings.filterwarnings("ignore", message="urllib3.*charset_normalizer")
warnings.filterwarnings("ignore", message="Revert to STA COM")

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="AURA - Autonomous Universal Reasoning Agent",
        prog="aura"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("init", help="Create AURA.md in current project")
    subparsers.add_parser("doctor", help="Check Ollama, models, dependencies")
    subparsers.add_parser("config", help="Show current configuration")
    subparsers.add_parser("models", help="List available models with routing roles")
    sub_commit = subparsers.add_parser("commit", help="Smart commit with AI-generated message")
    sub_commit.add_argument("--all", "-a", action="store_true", help="Stage all changes")
    subparsers.add_parser("cost", help="Show session cost breakdown")
    subparsers.add_parser("mcp-serve", help="Run as MCP server (JSON-RPC over stdio)")
    sub_exec = subparsers.add_parser("exec", help="Non-interactive agent execution")
    sub_exec.add_argument("prompt", nargs="?", default=None, help="Prompt to execute")
    sub_ide = subparsers.add_parser("ide", help="IDE integration setup")
    sub_ide.add_argument("action", nargs="?", default="setup", choices=["setup"], help="Action (default: setup)")

    # Positional prompt for one-shot agentic mode
    parser.add_argument(
        "goal",
        nargs="*",
        help="One-shot agentic prompt (e.g., aura 'fix the login bug')"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start in interactive chat mode"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum iterations for the agentic loop (default: 50)"
    )
    parser.add_argument(
        "--dream",
        action="store_true",
        help="Run dream mode to consolidate memories and generate insights"
    )
    parser.add_argument(
        "--dream-date",
        type=str,
        default=None,
        help="Date to analyze in dream mode (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--no-fastpath",
        action="store_true",
        help="Disable fast-path for simple queries (always use full agent loop)"
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Start in voice conversation mode (uses microphone and speaker)"
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="Enable text-to-speech for agent responses in chat mode"
    )
    parser.add_argument(
        "--no-barge-in",
        action="store_true",
        help="Disable barge-in detection in voice mode (use blocking TTS)"
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="pick",
        default=None,
        help="Resume a previous session ('last' for most recent, or pick from list)"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="Non-interactive: run prompt and exit (supports stdin piping)"
    )
    parser.add_argument(
        "--login",
        type=str,
        metavar="PROVIDER",
        help="Authenticate with a provider (e.g., 'chatgpt')"
    )
    parser.add_argument(
        "--logout",
        type=str,
        metavar="PROVIDER",
        help="Remove authentication for a provider (e.g., 'chatgpt')"
    )
    # Agentic CLI flags
    parser.add_argument(
        "--tier",
        type=str,
        choices=["local", "balanced", "max"],
        default="balanced",
        help="Model routing tier (default: balanced)"
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Maximum session cost in USD (e.g., --budget 2.0)"
    )
    parser.add_argument(
        "--trust",
        action="store_true",
        help="Trust mode: auto-approve all tool calls (no prompts)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Use a specific model (e.g., --model deepseek-r1:8b)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format for non-interactive mode (default: text)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose mode: expand all tool output sections"
    )

    args = parser.parse_args()

    # Handle auth commands (no agent needed)
    if args.login:
        if args.login.lower() == "chatgpt":
            from aura.auth.chatgpt_oauth import login
            sys.exit(0 if login() else 1)
        else:
            print(f"Unknown provider: {args.login}. Available: chatgpt")
            sys.exit(1)

    if args.logout:
        if args.logout.lower() == "chatgpt":
            from aura.auth.chatgpt_oauth import logout
            logout()
            sys.exit(0)
        else:
            print(f"Unknown provider: {args.logout}. Available: chatgpt")
            sys.exit(1)

    # Handle MCP server (lightweight, no agent needed)
    if args.command == "mcp-serve":
        from aura.core.mcp_server import main as mcp_main
        mcp_main()
        sys.exit(0)

    # Handle exec subcommand — non-interactive agent execution
    if args.command == "exec":
        exec_prompt = getattr(args, "prompt", None)
        if not exec_prompt:
            print("Usage: aura exec 'your prompt here'")
            sys.exit(1)
        args.prompt = exec_prompt
        # Fall through to the non-interactive --prompt path below

    # Handle subcommands that don't need the full agent
    elif args.command:
        from aura.core.commands import handle_subcommand
        sys.exit(handle_subcommand(args.command, args))

    # Heavy imports deferred until after argparse (so --help is instant)
    from aura import ApprenticeAgent
    from aura.config import Config
    from aura.dream import run_dream_mode

    # Handle dream mode first (doesn't need agent)
    if args.dream:
        result = run_dream_mode(args.dream_date)
        sys.exit(0 if result.get("success") else 1)

    try:
        agent = ApprenticeAgent()
    except Exception as e:
        print(f"\n[AURA] Failed to initialize agent: {e}")
        print("Check that Ollama is running (ollama serve) and your config is valid.")
        sys.exit(1)
    agent.max_iterations = args.max_iterations
    agent.use_fastpath = not args.no_fastpath

    # Handle session resume — try agentic sessions first, fall back to brain conversations
    if args.resume:
        from aura.core.session import AgenticSession as _SessionCheck
        _ses = _SessionCheck()
        agentic_sessions = _ses.list_sessions()
        brain_conversations = agent.brain.list_conversations()

        if args.resume == "last":
            if agentic_sessions:
                latest = agentic_sessions[0]
                agent._resume_session_id = latest["id"]
                print(f"  Resuming: {latest.get('title', 'Untitled')} ({latest.get('message_count', 0)} messages)")
            elif brain_conversations:
                latest = brain_conversations[0]
                agent.brain.switch_conversation(latest["id"])
                print(f"  Resumed (legacy): {latest.get('title', 'Untitled')}")
            else:
                print("No previous sessions found.")
        elif args.resume == "pick" or args.resume:
            all_sessions = []
            for s in agentic_sessions:
                s["_source"] = "agentic"
                all_sessions.append(s)
            for c in brain_conversations:
                c["_source"] = "brain"
                all_sessions.append(c)
            all_sessions.sort(key=lambda x: x.get("updated_at", 0), reverse=True)

            if not all_sessions:
                print("No previous sessions found.")
            else:
                print("\n  Recent sessions:\n")
                for i, s in enumerate(all_sessions[:10], 1):
                    title = s.get("title", "Untitled")[:50]
                    msgs = s.get("message_count", 0)
                    src = s.get("_source", "?")
                    print(f"    {i}. {title} ({msgs} msgs) [{src}]")
                print()
                try:
                    choice = input("  Pick a session (number): ").strip()
                    idx = int(choice) - 1
                    if 0 <= idx < len(all_sessions[:10]):
                        picked = all_sessions[idx]
                        if picked["_source"] == "agentic":
                            agent._resume_session_id = picked["id"]
                        else:
                            agent.brain.switch_conversation(picked["id"])
                        print(f"  Resuming: {picked.get('title', 'Untitled')}")
                    else:
                        print("  Invalid choice, starting new session.")
                except (ValueError, EOFError, KeyboardInterrupt):
                    print("  Starting new session.")

    # Read piped stdin if available (for composability)
    from aura.cli.pipe_mode import PipeOutput, is_pipe_mode, read_piped_input, EXIT_SUCCESS, EXIT_ERROR

    if not args.prompt and not sys.stdin.isatty():
        piped = read_piped_input()
        if piped:
            args.prompt = piped

    # Non-interactive mode: run prompt, print response, exit
    if args.prompt:
        pipe = PipeOutput(format=args.format)
        prompt = args.prompt
        result = agent.run(prompt)
        response = result.get("response", "")
        model_used = result.get("model", "")
        if response:
            pipe.result({"response": response, "model": model_used})
        sys.exit(EXIT_SUCCESS if result.get("success", True) else EXIT_ERROR)

    if args.voice:
        run_voice_mode(agent, enable_barge_in=not args.no_barge_in)
    elif args.goal:
        # One-shot agentic mode: aura "fix the login bug"
        prompt = " ".join(args.goal) if isinstance(args.goal, list) else args.goal
        run_agentic_oneshot(agent, prompt, args)
    else:
        # Default: interactive chat mode (just type 'aura' to start)
        run_chat_mode(agent, speak=args.speak, trust=args.trust, model=args.model, verbose=args.verbose)


def run_voice_mode(agent, enable_barge_in: bool = True):
    """Run the agent in voice conversation mode."""
    from aura.tools.voice import VoiceConversation  # lazy import
    conversation = VoiceConversation(agent, whisper_model="base", enable_barge_in=enable_barge_in)
    conversation.start()


def run_agentic_oneshot(agent, prompt: str, args):
    """Run a one-shot agentic task using the structured tool-calling loop."""
    from aura.cli.display import show_banner, console
    from aura.core.agentic_loop import run_agentic
    from aura.core.context import gather_context, get_aura_md_config
    from aura.core.router import ModelRouter
    from aura.core.permissions import PermissionManager

    show_banner()

    project_root = os.getcwd()

    # Gather project context
    context = gather_context(project_root)
    aura_config = get_aura_md_config(project_root)

    # Setup router
    tier = aura_config.get("tier", args.tier)
    budget = args.budget or aura_config.get("budget")
    router = ModelRouter(tier=tier, budget_usd=budget)
    # If user explicitly set a model, lock to it. Otherwise let router pick per-step.
    model = args.model or aura_config.get("model") or None
    display_model = model or f"auto-route ({tier})"

    # Setup permissions
    permissions = PermissionManager()
    if aura_config:
        permissions.apply_aura_md_overrides(aura_config)

    # Permission confirm callback
    def _confirm(tool_name, description):
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

    # Show final stats
    stats = agent.brain.get_session_stats()
    console.print(f"\n  [dim]{result['iterations']} iterations, {result['tool_calls']} tool calls, ${stats['cost_usd']:.4f}[/dim]")

    sys.exit(0 if result.get("success") else 1)


def _rewind_picker(cp_mgr, console):
    """Show checkpoint picker and restore selected checkpoint. Returns True if restored."""
    import time as _time
    cps = cp_mgr.list_checkpoints()
    if not cps:
        console.print("[dim]No checkpoints available[/dim]")
        return False
    console.print("\n[bold]Rewind to checkpoint:[/bold]")
    for i, cp in enumerate(cps[:10]):
        ts = _time.strftime("%H:%M:%S", _time.localtime(cp["timestamp"]))
        files = ", ".join(
            f["original_path"].split("/")[-1].split("\\")[-1]
            for f in cp["files"]
        )
        console.print(f"  {i+1}. [{ts}] {cp['label']} ({files})")
    console.print("  0. Cancel")
    try:
        choice = input("\nSelect checkpoint: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    if choice.isdigit() and 0 < int(choice) <= min(10, len(cps)):
        selected = cps[int(choice) - 1]
        if cp_mgr.restore(selected["id"]):
            files = ", ".join(
                f["original_path"].split("/")[-1].split("\\")[-1]
                for f in selected["files"]
            )
            console.print(f"[green]Restored: {files}[/green]")
            return True
        else:
            console.print("[red]Restore failed.[/red]")
            return False
    return False


def run_chat_mode(agent, speak: bool = False, trust: bool = False, model: str = None, verbose: bool = False):
    """Interactive CLI — agentic loop with status bar, model picker, tool calling."""
    from aura.cli.display import (
        console, show_banner, show_response,
        show_error, show_info, show_status_bar, show_help,
        show_welcome_info, show_tool_call,
    )
    from aura.cli.input import (
        create_session, get_input,
        SIGNAL_MODEL_PICK, SIGNAL_CLEAR_SCREEN, SIGNAL_NEW_SESSION,
        SIGNAL_COMMAND_PALETTE, SIGNAL_OPEN_EDITOR, SIGNAL_REWIND, SIGNAL_CYCLE_PERMS,
    )
    from aura.cli.model_picker import pick_model, update_model_roles_from_config
    from aura.cli.context_bar import estimate_messages_tokens, get_context_limit, build_context_breakdown, estimate_tokens
    from aura.cli.permissions_ui import cycle_permission_mode, get_mode_description, PermissionMode
    from aura.cli.checkpoint import CheckpointManager
    from aura.core.agentic_loop import AgenticLoop
    from aura.core.session import AgenticSession
    from aura.core.permissions import PermissionManager
    from aura.core.context import gather_context, get_aura_md_config

    # Load saved theme on startup
    from aura.cli.themes import load_theme_preference, set_theme, get_theme, list_themes, save_theme_preference
    saved_theme = load_theme_preference()
    set_theme(saved_theme)

    show_banner()
    show_welcome_info(agent)

    # Detect project type for status bar
    _project_type = ""
    try:
        from aura.tools.project_context import detect_and_load_context
        ctx = detect_and_load_context(".")
        _project_type = ctx.get("project_type", "") if isinstance(ctx, dict) else ""
    except Exception:
        pass

    # Gather project context for the agentic system prompt
    project_root = os.getcwd()
    project_context = ""
    aura_config = {}
    try:
        project_context = gather_context(project_root)
        aura_config = get_aura_md_config(project_root)
    except Exception:
        pass

    # Build permission manager with CLI confirmation
    permissions = PermissionManager()

    def _cli_confirm(tool_name: str, description: str) -> bool:
        console.print(f"\n  [yellow]Permission required:[/yellow] {tool_name}")
        if description:
            for line in description.split("\n")[:10]:
                console.print(f"    {line}", highlight=False)
        try:
            response = input("    Allow? (y/n/always): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if response == "always":
            permissions.set_trust_mode(True)
            return True
        return response in ("y", "yes")

    permissions.set_confirm_callback(_cli_confirm)
    if trust:
        permissions.set_trust_mode(True)

    # Apply AURA.md permission overrides if present
    if aura_config:
        permissions.apply_aura_md_overrides(aura_config)

    # Create session persistence
    agentic_session = AgenticSession()
    agentic_session.new(project_root=project_root, model=agent.brain._model_override or "auto")
    import atexit
    atexit.register(agentic_session.save)

    # Create persistent agentic loop (maintains conversation history)
    # If user explicitly set a model, lock to it. Otherwise let router pick per-step.
    from aura.core.router import ModelRouter
    explicit_model = model or agent.brain._model_override or aura_config.get("model") or None
    chat_tier = aura_config.get("tier", "balanced")
    chat_router = ModelRouter(tier=chat_tier, budget_usd=aura_config.get("budget"))
    agentic = AgenticLoop(
        brain=agent.brain,
        project_root=project_root,
        permissions=permissions,
        model_override=explicit_model,
        max_iterations=aura_config.get("max_iterations", 25),
        budget_usd=aura_config.get("budget"),
        context=project_context,
        session=agentic_session,
        aura_config=aura_config,
        router=chat_router,
    )
    # Initialize checkpoint manager
    try:
        checkpoint_mgr = CheckpointManager()
    except (OSError, PermissionError):
        checkpoint_mgr = None
    if checkpoint_mgr:
        agentic._checkpoint_mgr = checkpoint_mgr
        agentic.executor._checkpoint_mgr = checkpoint_mgr

    # Initialize permission mode
    current_perm_mode = "careful"
    if trust:
        current_perm_mode = "full_auto"

    # Initialize Phase 3 modules: background manager, research context, hooks
    try:
        from aura.cli.background import BackgroundManager, notify_completion, create_background_indicator
    except ImportError:
        BackgroundManager = None
        notify_completion = None
        create_background_indicator = lambda *a, **k: ""

    try:
        from aura.cli.research_mode import ResearchContext, create_research_indicator
    except ImportError:
        ResearchContext = None
        create_research_indicator = lambda *a, **k: ""

    try:
        from aura.cli.hooks import HookManager, HookEvent
    except ImportError:
        HookManager = None
        HookEvent = None

    try:
        from aura.cli.mood_display import create_mood_indicator
    except ImportError:
        create_mood_indicator = lambda *a, **k: ""

    bg_manager = BackgroundManager() if BackgroundManager else None
    if bg_manager and notify_completion:
        bg_manager.set_completion_callback(notify_completion)
    agent._bg_manager = bg_manager

    research_ctx = ResearchContext() if ResearchContext else None
    agent._research_ctx = research_ctx

    hook_mgr = HookManager() if HookManager else None
    agent._hook_manager = hook_mgr
    if hook_mgr and aura_config:
        hook_mgr.load_from_config(aura_config)
        hook_mgr.load_builtin_hooks(aura_config)

    # Initialize disclosure manager for progressive tool output
    if verbose:
        from aura.cli.disclosure import DisclosureManager
        from aura.cli import display as _display_mod
        _display_mod._disclosure = DisclosureManager(default_expanded=True)

    # Fire session_start hooks
    if hook_mgr:
        hook_mgr.fire(HookEvent.SESSION_START, {"project_root": project_root})

    # Initialize mid-turn steering queue
    from aura.cli.steering import SteeringQueue
    steering = SteeringQueue()

    # Store on agent so /clear and /trust can access it
    agent._agentic_loop = agentic
    agent._agentic_permissions = permissions
    agent._agentic_session = agentic_session

    # Resume agentic session if requested
    resume_id = getattr(agent, '_resume_session_id', None)
    if resume_id:
        if agentic.load_session(resume_id):
            agentic_session.load(resume_id)  # Sync the session object too
            show_info(f"Session restored ({len(agentic._conversation_history)} messages)")
        delattr(agent, '_resume_session_id')

    # Status bar state
    _current_model = explicit_model or "auto"
    _session_title = ""
    _msg_count = 0
    _token_used = 0
    _token_limit = get_context_limit(_current_model)

    _mood_cache = {"state": {}, "ts": 0.0}

    def _phase3_indicators():
        """Build Phase 3 status bar indicators."""
        import time as _t
        _bg_ind = create_background_indicator(bg_manager) if bg_manager else ""
        _res_ind = create_research_indicator(research_ctx) if research_ctx else ""
        _mood_ind = ""
        now = _t.time()
        if now - _mood_cache["ts"] > 5.0:
            try:
                from aura.emotion.alma_engine import get_alma_engine
                _engine = get_alma_engine()
                _state = _engine.get_emotional_state() if _engine else {}
                _mood_cache["state"] = _state
                _mood_cache["ts"] = now
            except Exception:
                pass
        if _mood_cache["state"]:
            _mood_ind = create_mood_indicator(_mood_cache["state"])
        return _bg_ind, _res_ind, _mood_ind

    def _show_bar(**kwargs):
        """Show status bar with Phase 3 indicators."""
        bg_ind, res_ind, mood_ind = _phase3_indicators()
        show_status_bar(
            bg_indicator=bg_ind,
            research_indicator=res_ind,
            mood_indicator=mood_ind,
            **kwargs,
        )

    _show_bar(
        model=_current_model, project_type=_project_type,
        session_title=_session_title, message_count=_msg_count,
        token_used=_token_used, token_limit=_token_limit,
        permission_mode=current_perm_mode,
    )

    if speak:
        show_info("Voice output enabled")

    # Initialize model picker roles from config
    update_model_roles_from_config()

    session = create_session()

    _pending_follow_up = None
    _follow_up_depth = 0
    _MAX_FOLLOW_UP_DEPTH = 3

    while True:
        if _pending_follow_up:
            user_input = _pending_follow_up
            _pending_follow_up = None
            show_info(f"Follow-up: {user_input[:60]}...")
        else:
            _follow_up_depth = 0  # reset depth on manual input
            user_input = get_input(session)

        if user_input is None:
            if hook_mgr:
                hook_mgr.fire(HookEvent.SESSION_END, {"reason": "user_exit"})
            console.print("\n[dim]Goodbye.[/dim]\n")
            break

        # Handle keybinding signals
        if user_input == SIGNAL_CLEAR_SCREEN:
            console.clear()
            _show_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
                token_used=_token_used, token_limit=_token_limit,
                permission_mode=current_perm_mode,
            )
            continue
        elif user_input == SIGNAL_NEW_SESSION:
            # Save current session first
            if hasattr(agentic, 'session') and agentic.session:
                agentic.session.save()
            agentic._conversation_history.clear()
            if checkpoint_mgr:
                checkpoint_mgr.clear()
            _msg_count = 0
            _token_used = 0
            console.print("[dim]New session started[/dim]")
            _show_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
                token_used=_token_used, token_limit=_token_limit,
                permission_mode=current_perm_mode,
            )
            continue
        elif user_input == SIGNAL_COMMAND_PALETTE:
            console.print("[dim]Command palette — type / for commands[/dim]")
            continue
        elif user_input == SIGNAL_OPEN_EDITOR:
            import tempfile, subprocess as _sp
            editor = os.environ.get("EDITOR", "notepad" if os.name == "nt" else "nano")
            with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
                f.write("")
                tmp_path = f.name
            try:
                _sp.call([editor, tmp_path])
                user_input = Path(tmp_path).read_text().strip()
            except (FileNotFoundError, OSError) as e:
                console.print(f"[red]Editor failed: {e}[/red]")
                user_input = ""
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            if not user_input:
                continue
            # Fall through to normal processing with the editor content
        elif user_input == SIGNAL_CYCLE_PERMS:
            current_perm_mode = cycle_permission_mode(current_perm_mode)
            console.print(f"[dim]{get_mode_description(current_perm_mode)}[/dim]")
            if current_perm_mode == "full_auto":
                permissions.set_trust_mode(True)
            elif current_perm_mode == "plan":
                permissions.set_trust_mode(False)
            else:
                permissions.set_trust_mode(False)
            _show_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
                token_used=_token_used, token_limit=_token_limit,
                permission_mode=current_perm_mode,
            )
            continue
        elif user_input == SIGNAL_REWIND:
            if checkpoint_mgr:
                _rewind_picker(checkpoint_mgr, console)
            else:
                console.print("[dim]No checkpoint manager available[/dim]")
            continue

        # Handle Alt+M model picker
        if user_input == SIGNAL_MODEL_PICK:
            _current_model = agent.brain._model_override or "auto"
            choice = pick_model(console, _current_model)
            if choice:
                if choice == "auto":
                    agent.brain.set_model_override(None)
                    _current_model = "auto"
                    agentic.model_override = None
                    show_info("Model set to auto-routing")
                else:
                    agent.brain.set_model_override(choice)
                    _current_model = choice
                    agentic.model_override = choice
                    show_info(f"Model set to {choice}")
            _token_limit = get_context_limit(_current_model)
            _show_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
                token_used=_token_used, token_limit=_token_limit,
                permission_mode=current_perm_mode,
            )
            continue

        if not user_input:
            continue

        # Background mode: & prefix submits task to background
        if user_input.startswith("& ") or (user_input.startswith("&") and len(user_input) > 1 and user_input[1] != " "):
            bg_prompt = user_input[2:].strip() if user_input.startswith("& ") else user_input[1:].strip()
            if not bg_prompt:
                console.print("[dim]Usage: & <prompt>[/dim]")
                continue
            # NOTE: Use brain.think() for background tasks — agentic.run() is not thread-safe
            def _bg_task_fn(prompt):
                try:
                    response = agent.brain.think(prompt)
                    if isinstance(response, dict):
                        response = response.get("response", response.get("content", str(response)))
                    return {"success": True, "response": response or "", "iterations": 1}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            task = bg_manager.submit(bg_prompt, _bg_task_fn)
            if task:
                console.print(f"[cyan]Background task started: {task.id}[/cyan]")
            else:
                console.print("[red]Too many background tasks running.[/red]")
            continue

        # Signal activity to daemon (if running)
        try:
            import socket, json as _json
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                s.connect(("127.0.0.1", 19733))
                s.send((_json.dumps({"type": "activity"}) + "\n").encode())
        except Exception:
            pass

        # Handle ? for help
        if user_input.strip() == "?":
            show_help()
            continue

        if user_input.startswith("/"):
            handle_command(agent, user_input, speak=speak)
            _current_model = agent.brain._model_override or "auto"
            _token_used = estimate_messages_tokens(agentic._conversation_history)
            _token_limit = get_context_limit(_current_model)
            _show_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
                token_used=_token_used, token_limit=_token_limit,
                permission_mode=current_perm_mode,
            )
            continue

        # Run agentic loop directly (no threading — permission prompts need
        # to be visible on the main thread, and think_with_tools has its own
        # 120s timeout so it won't hang forever)
        show_info("Thinking...")
        try:
            def _on_tool(name, args, _result):
                # Fire pre_tool_call hooks before display
                if hook_mgr:
                    hook_mgr.fire(HookEvent.PRE_TOOL_CALL, {
                        "tool_name": name,
                        "tool_args": str(args)[:500],
                    })
                desc = args.get("path") or args.get("pattern") or args.get("query") or ""
                if not desc and "command" in args:
                    desc = args["command"][:60]
                show_tool_call(name, str(desc))
                # Fire post_tool_call hooks
                if hook_mgr:
                    hook_mgr.fire(HookEvent.POST_TOOL_CALL, {
                        "tool_name": name,
                        "tool_args": str(args)[:500],
                    })
                # Fire post_edit hooks for edit/write tools
                if hook_mgr and name in ("edit_file", "write_file"):
                    hook_mgr.fire(HookEvent.POST_EDIT, {
                        "tool_name": name,
                        "file_path": args.get("path", args.get("file_path", "")),
                    })

            result = agentic.run(
                user_input,
                on_tool_call=_on_tool,
                steering_queue=steering,
            )
        except KeyboardInterrupt:
            steering.clear()
            show_info("Interrupted.")
            continue
        except Exception as exc:
            show_error(str(exc))
            continue

        if result is None:
            show_error("No response received.")
            continue

        response_text = result.get("response", "")
        model_used = result.get("model", _current_model)

        show_response(response_text, model=model_used)

        # Check for queued follow-up from steering
        follow_up = steering.pop_follow_up()
        if follow_up and _follow_up_depth < _MAX_FOLLOW_UP_DEPTH:
            _pending_follow_up = follow_up
            _follow_up_depth += 1
            # Don't continue — the top of the loop will pick up _pending_follow_up
        elif follow_up:
            show_info("Max auto-follow-up depth reached, dropping follow-up.")

        # Update status bar
        _msg_count += 1
        _current_model = agent.brain._model_override or "auto"
        _token_used = estimate_messages_tokens(agentic._conversation_history)
        _token_limit = get_context_limit(_current_model)
        cost_usd = 0.0
        try:
            stats = agent.brain.get_session_stats()
            cost_usd = stats.get("cost_usd", 0.0)
        except Exception:
            pass
        _show_bar(
            model=_current_model, project_type=_project_type,
            session_title=_session_title, message_count=_msg_count,
            cost_usd=cost_usd,
            token_used=_token_used, token_limit=_token_limit,
            permission_mode=current_perm_mode,
        )

        # Fire post_response hooks
        if hook_mgr:
            hook_mgr.fire(HookEvent.POST_RESPONSE, {
                "response": response_text[:500] if response_text else "",
                "model": model_used,
            })

        if speak and response_text:
            try:
                agent._speak(response_text)
            except Exception:
                pass


def handle_command(agent, command: str, speak: bool = False):
    """Handle special commands in chat mode."""
    from aura.cli.display import show_help
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/quit" or cmd == "/exit":
        _hook_mgr = getattr(agent, '_hook_manager', None)
        if _hook_mgr:
            from aura.cli.hooks import HookEvent as _HE
            _hook_mgr.fire(_HE.SESSION_END, {"reason": "quit_command"})
        print("Goodbye!")
        sys.exit(0)
    elif cmd == "/help" or cmd == "?":
        show_help()
        return
    elif cmd == "/goal":
        if arg:
            result = agent.run(arg)
            print_result(result)
        else:
            print("Usage: /goal <your goal>")
    elif cmd == "/recall":
        if arg:
            memories = agent.recall_memories(arg)
            print(f"\nRecalled {len(memories)} memories:")
            for m in memories:
                print(f"  - {m.get('content', str(m))[:100]}...")
        else:
            print("Usage: /recall <query>")
    elif cmd == "/clear":
        agent.brain.clear_history()
        if hasattr(agent, '_agentic_loop'):
            agent._agentic_loop.clear_history()
        print("Conversation history cleared.")
    elif cmd == "/speak" or cmd == "/say":
        if arg:
            agent._speak(arg)
            print(f"[Spoke: {arg}]")
        else:
            print("Usage: /speak <text to speak>")
    elif cmd == "/model":
        if not arg:
            from aura.cli.display import console, show_info
            from aura.cli.model_picker import pick_model
            current = agent.brain._model_override or "auto"
            choice = pick_model(console, current)
            if choice is not None:
                if choice == "auto":
                    agent.brain.set_model_override(None)
                    if hasattr(agent, '_agentic_loop'):
                        agent._agentic_loop.model_override = None
                    show_info("Model override cleared. Using auto-selection.")
                else:
                    agent.brain.set_model_override(choice)
                    if hasattr(agent, '_agentic_loop'):
                        agent._agentic_loop.model_override = choice
                    show_info(f"Model locked to: {choice}")
        elif arg.lower() == "auto":
            agent.brain.set_model_override(None)
            if hasattr(agent, '_agentic_loop'):
                agent._agentic_loop.model_override = None
            print("Model override cleared. Using auto-selection.")
        else:
            agent.brain.set_model_override(arg)
            if hasattr(agent, '_agentic_loop'):
                agent._agentic_loop.model_override = arg
            print(f"Model locked to: {arg}")
    elif cmd == "/compact":
        focus = arg if arg else None
        print("Compacting conversation history...")
        summary = agent.brain.compact_history(focus=focus)
        if summary:
            print(f"Compacted. Summary: {summary[:200]}...")
        else:
            print("Nothing to compact (history too short).")
    elif cmd == "/plan":
        if arg:
            from aura.cli.plan_mode import parse_plan_from_llm, render_plan, PLAN_GENERATION_PROMPT, StepStatus
            from aura.cli.display import console as _plan_console, show_thinking, show_info as _plan_info, show_tool_call as _plan_tool
            _plan_info("Generating plan...")
            prompt = PLAN_GENERATION_PROMPT.format(task=arg)
            try:
                response = agent.brain.think(prompt)
                if isinstance(response, dict):
                    response = response.get("response", response.get("content", str(response)))
            except Exception as e:
                print(f"  Error generating plan: {e}")
                return

            plan = parse_plan_from_llm(response)
            render_plan(_plan_console, plan)

            # TODO: implement plan editing (deferred)
            _plan_console.print("\n[dim]Execute this plan? (y/n)[/dim]")
            try:
                choice = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return

            if choice in ("y", "yes"):
                # Execute step by step via agentic loop
                _agentic = getattr(agent, '_agentic_loop', None)
                for step in plan.steps:
                    step.status = StepStatus.RUNNING
                    render_plan(_plan_console, plan)
                    try:
                        if _agentic:
                            def _on_plan_tool(name, args, _result):
                                desc = args.get("path") or args.get("pattern") or args.get("query") or ""
                                if not desc and "command" in args:
                                    desc = args["command"][:60]
                                _plan_tool(name, str(desc))
                            result = _agentic.run(step.description, on_tool_call=_on_plan_tool)
                        else:
                            result = agent.run(step.description)
                        if result.get("success"):
                            step.status = StepStatus.DONE
                            step.result = result.get("response", "")[:200]
                        else:
                            step.status = StepStatus.FAILED
                            step.error = result.get("error", result.get("response", "Step failed"))[:200]
                    except Exception as e:
                        step.status = StepStatus.FAILED
                        step.error = str(e)[:200]

                    if step.status == StepStatus.FAILED:
                        render_plan(_plan_console, plan)
                        _plan_console.print("[yellow]Step failed. Continue? (y/n)[/yellow]")
                        try:
                            cont = input("> ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            break
                        if cont not in ("y", "yes"):
                            break

                render_plan(_plan_console, plan)
                _plan_console.print("[green]Plan execution complete.[/green]")
            return
        else:
            print("Usage: /plan <task description>")
    elif cmd == "/browse":
        if not arg:
            print("Usage: /browse <url> | /browse search <query> | /browse text | /browse screenshot | /browse click <selector> | /browse links")
        else:
            _handle_browse_command(agent, arg)
    elif cmd == "/grep":
        _handle_grep_command(agent, arg)
    elif cmd == "/search" or cmd == "/find":
        _handle_search_command(agent, arg)
    elif cmd == "/edit":
        _handle_edit_command(agent, arg)
    elif cmd == "/project":
        _handle_project_command(agent, arg)
    elif cmd == "/shell" or cmd == "/bash" or cmd == "/run":
        _handle_shell_command(agent, arg)
    elif cmd == "/agent":
        _handle_agent_command(agent, arg)
    elif cmd == "/evolve":
        _handle_evolve_command(agent, arg)
    elif cmd == "/fleet":
        task = arg.strip()
        if not task:
            from aura.cli.display import console as _fleet_console
            _fleet_console.print("[dim]Usage: /fleet <task description>[/dim]")
            return
        from aura.cli.fleet import (
            FleetRun, FleetExecutor, parse_decomposition,
            render_fleet_dashboard, DECOMPOSITION_PROMPT,
        )
        from aura.cli.display import console as _fleet_console
        # Decompose the task
        prompt = DECOMPOSITION_PROMPT.format(task=task)
        response = agent.brain.think(prompt)
        if isinstance(response, dict):
            response = response.get("response", response.get("content", str(response)))
        subtasks = parse_decomposition(response)
        if not subtasks:
            _fleet_console.print("[red]Could not decompose task into sub-tasks.[/red]")
            return
        fleet = FleetRun(goal=task, tasks=subtasks)
        render_fleet_dashboard(_fleet_console, fleet)
        _fleet_console.print(f"\n[dim]Execute {len(subtasks)} tasks in parallel? (y/n)[/dim]")
        try:
            choice = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if choice not in ("y", "yes"):
            return
        # NOTE: Use brain.think() for fleet sub-tasks — agentic.run() is not thread-safe
        def _fleet_task_fn(prompt):
            try:
                response = agent.brain.think(prompt)
                if isinstance(response, dict):
                    response = response.get("response", response.get("content", str(response)))
                return {"success": True, "response": response or "", "iterations": 1}
            except Exception as e:
                return {"success": False, "error": str(e)}
        executor = FleetExecutor(max_workers=3)
        executor.run(fleet, _fleet_task_fn, on_update=lambda f: render_fleet_dashboard(_fleet_console, f))
        render_fleet_dashboard(_fleet_console, fleet)
        return

    elif cmd == "/tasks":
        from aura.cli.display import console as _tasks_console
        _bg_mgr = getattr(agent, '_bg_manager', None)
        if _bg_mgr:
            from aura.cli.background import render_tasks_table
            render_tasks_table(_tasks_console, _bg_mgr.list_tasks())
        else:
            _tasks_console.print("[dim]No background tasks.[/dim]")
        return

    elif cmd == "/research":
        from aura.cli.display import console as _res_console
        _research_ctx = getattr(agent, '_research_ctx', None)
        if _research_ctx is None:
            from aura.cli.research_mode import ResearchContext
            _research_ctx = ResearchContext()
            agent._research_ctx = _research_ctx
        topic = arg.strip()
        if not topic:
            if _research_ctx.is_active:
                _research_ctx.stop()
                _res_console.print("[dim]Research mode ended.[/dim]")
            else:
                _res_console.print("[dim]Usage: /research <topic>[/dim]")
            return
        _research_ctx.start(topic)
        _res_console.print(f"[magenta]Research mode: {topic}[/magenta]")
        return

    elif cmd == "/sources":
        from aura.cli.display import console as _src_console
        _research_ctx = getattr(agent, '_research_ctx', None)
        if _research_ctx and _research_ctx.is_active:
            from aura.cli.research_mode import render_sources
            render_sources(_src_console, _research_ctx)
        else:
            _src_console.print("[dim]No research session active.[/dim]")
        return

    elif cmd == "/export" and arg.strip().startswith("research"):
        from aura.cli.display import console as _exp_console
        import re as _re_export
        _research_ctx = getattr(agent, '_research_ctx', None)
        if _research_ctx and _research_ctx.is_active:
            md = _research_ctx.export_markdown()
            safe_topic = _re_export.sub(r'[^\w\-]', '_', _research_ctx.topic)[:30]
            out_path = Path(f"research_{safe_topic}.md")
            out_path.write_text(md)
            _exp_console.print(f"[green]Exported to {out_path}[/green]")
        else:
            _exp_console.print("[dim]No active research session to export.[/dim]")
        return

    elif cmd == "/mood":
        from aura.cli.display import console as _mood_console
        from aura.cli.mood_display import render_mood_detail
        try:
            from aura.emotion.alma_engine import get_alma_engine
            engine = get_alma_engine()
            state = engine.get_emotional_state() if engine else {}
        except Exception:
            state = {}
        if state:
            render_mood_detail(_mood_console, state)
        else:
            _mood_console.print("[dim]Emotional state not available.[/dim]")
        return

    elif cmd == "/hook":
        from aura.cli.display import console as _hook_console
        _hook_mgr = getattr(agent, '_hook_manager', None)
        if _hook_mgr is None:
            from aura.cli.hooks import HookManager
            _hook_mgr = HookManager()
            agent._hook_manager = _hook_mgr
        from aura.cli.hooks import render_hooks_table, HookEvent
        sub = arg.strip().split(None, 1)
        if not sub or sub[0] == "list":
            render_hooks_table(_hook_console, _hook_mgr.list_hooks())
        elif sub[0] == "add" and len(sub) > 1:
            parts_h = sub[1].split(None, 1)
            if len(parts_h) == 2:
                try:
                    _hook_mgr.add(parts_h[0], parts_h[1])
                    _hook_console.print(f"[green]Hook added: {parts_h[0]} -> {parts_h[1]}[/green]")
                except ValueError as e:
                    _hook_console.print(f"[red]{e}[/red]")
            else:
                _hook_console.print("[dim]Usage: /hook add <event> <command>[/dim]")
        elif sub[0] == "remove" and len(sub) > 1:
            if _hook_mgr.remove(sub[1]):
                _hook_console.print(f"[green]Hook removed: {sub[1]}[/green]")
            else:
                _hook_console.print(f"[red]Hook not found: {sub[1]}[/red]")
        return
    elif cmd == "/sessions":
        # Show agentic sessions (full tool-call history) + legacy brain conversations
        from aura.core.session import AgenticSession as _SesCmd
        _ses = _SesCmd()
        agentic_sessions = _ses.list_sessions()
        brain_conversations = agent.brain.list_conversations()

        parts_arg = arg.split(maxsplit=1) if arg else []
        subcmd = parts_arg[0].lower() if parts_arg else "list"

        if subcmd == "delete" and len(parts_arg) > 1:
            target = parts_arg[1]
            if _ses.delete(target):
                print(f"  Deleted session: {target}")
            else:
                print(f"  Session not found: {target}")
        elif subcmd == "new":
            if hasattr(agent, '_agentic_session'):
                agent._agentic_session.save()
            new_ses = _SesCmd()
            new_ses.new(project_root=os.getcwd())
            if hasattr(agent, '_agentic_loop'):
                agent._agentic_loop.session = new_ses
                agent._agentic_loop.clear_history()
            agent._agentic_session = new_ses
            print("  Started new session.")
        else:
            if not agentic_sessions and not brain_conversations:
                print("  No sessions found.")
                return
            # Use smart session picker if available
            from aura.cli.session_picker import pick_session
            from aura.cli.display import console as _ses_console
            current_sid = ""
            if hasattr(agent, '_agentic_session') and agent._agentic_session:
                current_sid = getattr(agent._agentic_session, 'session_id', "") or ""
            all_sessions = agentic_sessions + brain_conversations
            result = pick_session(_ses_console, all_sessions, current_sid)
            if result and "__action__" not in result:
                # Load selected session
                sid = result.get("id", "")
                if sid and hasattr(agent, '_agentic_loop'):
                    if agent._agentic_loop.load_session(sid):
                        print(f"  Switched to session: {result.get('title', 'Untitled')}")
                    else:
                        print(f"  Failed to load session: {sid}")
            elif result and result.get("__action__") == "delete":
                target_session = result.get("session", {})
                target_id = target_session.get("id", "")
                if target_id and _ses.delete(target_id):
                    print(f"  Deleted session: {target_session.get('title', target_id)}")
                else:
                    print(f"  Failed to delete session.")
    elif cmd == "/theme":
        from aura.cli.themes import set_theme as _set_theme, get_theme as _get_theme, list_themes as _list_themes, save_theme_preference as _save_pref
        from aura.cli.display import console as _theme_console
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
        return
    elif cmd == "/trust":
        if hasattr(agent, '_agentic_permissions'):
            agent._agentic_permissions.set_trust_mode(True)
        else:
            from aura.core.permissions import PermissionManager
            agent._agentic_permissions = PermissionManager()
            agent._agentic_permissions.set_trust_mode(True)
        print("  Trust mode enabled — all tool calls auto-approved.")
    elif cmd == "/context":
        if hasattr(agent, '_agentic_loop'):
            from aura.cli.display import console as _ctx_console
            from aura.cli.context_bar import estimate_messages_tokens as _est_msgs, get_context_limit as _get_lim, build_context_breakdown as _build_bd, estimate_tokens as _est_tok
            from rich.panel import Panel as _CtxPanel
            _al = agent._agentic_loop
            _tok_used = _est_msgs(_al._conversation_history)
            _tok_limit = _get_lim(agent.brain._model_override or "default")
            _sys_tokens = 0
            try:
                _sys_tokens = _est_tok(_al._build_system_prompt(""))
            except Exception:
                pass
            _ctx_console.print(_CtxPanel(
                _build_bd(_sys_tokens, _tok_used, 0, _tok_limit),
                title="[bold cyan]Context Window[/bold cyan]",
                border_style="cyan",
            ))
        else:
            print("  Context tracking not available.")
    elif cmd == "/rewind":
        if hasattr(agent, '_agentic_loop') and hasattr(agent._agentic_loop, '_checkpoint_mgr'):
            from aura.cli.display import console as _rw_console
            _rewind_picker(agent._agentic_loop._checkpoint_mgr, _rw_console)
        else:
            print("  No checkpoint manager available.")
    elif cmd == "/cost":
        stats = agent.brain.get_session_stats()
        print(f"\n  Session Cost:")
        print(f"    Input tokens:  {stats['input_tokens']:,}")
        print(f"    Output tokens: {stats['output_tokens']:,}")
        print(f"    Total tokens:  {stats['total_tokens']:,}")
        print(f"    Estimated cost: ${stats['cost_usd']:.4f}")
        print(f"    Queries: {stats['queries']}")
        print()
    elif cmd == "/undo":
        if hasattr(agent, '_agentic_loop'):
            tool_exec = agent._agentic_loop.executor
            if arg:
                result = tool_exec.code_edit.rollback(arg)
                if result.get("success"):
                    print(f"  Rolled back: {result.get('restored', arg)}")
                else:
                    print(f"  Error: {result.get('error', 'Unknown error')}")
            else:
                backups = tool_exec.code_edit._last_backups
                if backups:
                    last_path = list(backups.keys())[-1]
                    result = tool_exec.code_edit.rollback(last_path)
                    if result.get("success"):
                        print(f"  Rolled back: {last_path}")
                    else:
                        print(f"  Error: {result.get('error', 'Unknown error')}")
                else:
                    print("  No edits to undo (no .bak files)")
        else:
            print("  No active agentic loop.")
    elif cmd == "/diff":
        import subprocess as _sp
        try:
            diff_args = ["git", "diff"]
            if arg:
                diff_args.extend(arg.split())
            result = _sp.run(diff_args, capture_output=True, text=True, cwd=os.getcwd(), timeout=10)
            if result.stdout:
                print(result.stdout[:5000])
            else:
                print("  No changes.")
        except Exception as e:
            print(f"  Error: {e}")
    elif cmd == "/git":
        if not arg:
            print("Usage: /git <command> (e.g., /git status, /git log, /git diff)")
        else:
            import subprocess as _sp
            try:
                result = _sp.run(
                    ["git"] + arg.split(),
                    capture_output=True, text=True, cwd=os.getcwd(), timeout=15,
                )
                output = result.stdout or result.stderr
                print(output[:5000] if output else "  (no output)")
            except Exception as e:
                print(f"  Error: {e}")
    elif cmd == "/mcp":
        if hasattr(agent, '_agentic_loop') and hasattr(agent._agentic_loop, '_mcp_client'):
            mgr = agent._agentic_loop._mcp_client
            if not mgr.connections:
                print("  No MCP servers connected. Configure in AURA.md under mcp_servers:")
            else:
                for name, conn in mgr.connections.items():
                    print(f"  {name}: {len(conn.tools)} tools")
                    for t in conn.tools[:5]:
                        print(f"    - {t['name']}: {t.get('description', '')[:60]}")
                    if len(conn.tools) > 5:
                        print(f"    ... and {len(conn.tools) - 5} more")
        else:
            print("  No MCP servers connected. Configure in AURA.md under mcp_servers:")
    else:
        print(f"Unknown command: {cmd}")


def _handle_browse_command(agent, arg: str):
    """Handle /browse subcommands using the existing BrowserTool."""
    # Get or create the browser tool
    if 'browser' not in agent.tools:
        try:
            from aura.tools.browser import BrowserTool
            agent.tools['browser'] = BrowserTool(headless=False)
        except ImportError:
            print("Browser tool not available. Install playwright: pip install playwright && playwright install")
            return

    browser = agent.tools['browser']
    parts = arg.split(maxsplit=1)
    subcmd = parts[0].lower()
    subarg = parts[1] if len(parts) > 1 else ""

    if subcmd == "search":
        if not subarg:
            print("Usage: /browse search <query>")
            return
        query_url = f"https://www.google.com/search?q={subarg.replace(' ', '+')}"
        result = browser.open(query_url)
        if result.get("success"):
            print(f"  Searched: {subarg}")
            print(f"  Title: {result.get('title', 'N/A')}")
            links = browser.get_links()
            if links.get("success"):
                print(f"  Top results:")
                count = 0
                for link in links.get("links", []):
                    href = link.get("href", "")
                    text = link.get("text", "").strip()
                    if text and "google" not in href and len(text) > 5:
                        print(f"    - {text[:80]}")
                        print(f"      {href}")
                        count += 1
                        if count >= 5:
                            break
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")

    elif subcmd == "text":
        result = browser.get_text()
        if result.get("success"):
            print(f"  Page: {result.get('title', 'N/A')}")
            print(f"  URL: {result.get('url', 'N/A')}")
            text = result.get("text", "")
            print(f"  Text ({result.get('length', 0)} chars):\n")
            # Print first 2000 chars
            print(text[:2000])
            if len(text) > 2000:
                print(f"\n  ... ({len(text) - 2000} more chars)")
        else:
            print(f"  Error: {result.get('error', 'No page loaded')}")

    elif subcmd == "screenshot":
        result = browser.screenshot(subarg if subarg else None)
        if result.get("success"):
            print(f"  Screenshot saved: {result.get('path', 'N/A')}")
        else:
            print(f"  Error: {result.get('error', 'Screenshot failed')}")

    elif subcmd == "click":
        if not subarg:
            print("Usage: /browse click <css-selector>")
            return
        result = browser.click(subarg)
        if result.get("success"):
            print(f"  Clicked: {subarg}")
            print(f"  Now at: {result.get('url', 'N/A')}")
        else:
            print(f"  Error: {result.get('error', 'Click failed')}")

    elif subcmd == "links":
        result = browser.get_links()
        if result.get("success"):
            print(f"  URL: {result.get('url', 'N/A')}")
            print(f"  Links ({result.get('count', 0)}):")
            for link in result.get("links", [])[:20]:
                text = link.get("text", "").strip()
                href = link.get("href", "")
                if text:
                    print(f"    [{text[:60]}] {href}")
        else:
            print(f"  Error: {result.get('error', 'No page loaded')}")

    else:
        # Treat as URL
        result = browser.open(arg)
        if result.get("success"):
            print(f"  Title: {result.get('title', 'N/A')}")
            print(f"  URL: {result.get('url', 'N/A')}")
            print(f"  Status: {result.get('status', 'N/A')}")
        else:
            print(f"  Error: {result.get('error', 'Navigation failed')}")


def _handle_grep_command(agent, arg: str):
    """Handle /grep <pattern> [path] — search code content."""
    if not arg:
        print("Usage: /grep <pattern> [path]")
        print("  /grep 'def my_func'")
        print("  /grep 'import os' ./src")
        print("  /grep 'TODO' --type py")
        return

    tool = agent.tools.get("code_search")
    if not tool:
        from aura.tools.code_search import CodeSearchTool
        tool = CodeSearchTool()
        agent.tools["code_search"] = tool

    parts = arg.split()
    pattern = parts[0]
    path = "."

    # Parse flags
    file_type = None
    case_insensitive = False
    context = 0
    i = 1
    while i < len(parts):
        if parts[i] == "--type" and i + 1 < len(parts):
            file_type = parts[i + 1]
            i += 2
        elif parts[i] == "-i":
            case_insensitive = True
            i += 1
        elif parts[i] == "-C" and i + 1 < len(parts):
            context = int(parts[i + 1])
            i += 2
        else:
            path = parts[i]
            i += 1

    result = tool.grep(
        pattern=pattern, path=path, file_type=file_type,
        case_insensitive=case_insensitive, context_lines=context,
    )

    if not result.get("success"):
        print(f"  Error: {result.get('error')}")
        return

    matches = result.get("matches", [])
    total = result.get("total_matches", 0)
    print(f"\n  {total} matches in {result.get('files_searched', 0)} files:\n")
    for m in matches[:50]:
        print(f"  {m['file']}:{m['line']}\t{m['text']}")
        for ctx in m.get("before", []):
            print(f"    {ctx}")
        for ctx in m.get("after", []):
            print(f"    {ctx}")
    if total > 50:
        print(f"\n  ... and {total - 50} more matches")


def _handle_search_command(agent, arg: str):
    """Handle /search and /find — file pattern search and definitions."""
    if not arg:
        print("Usage: /search <glob-pattern>  or  /find def <name>")
        print("  /search '*.py'")
        print("  /find def MyClass")
        print("  /search structure")
        return

    tool = agent.tools.get("code_search")
    if not tool:
        from aura.tools.code_search import CodeSearchTool
        tool = CodeSearchTool()
        agent.tools["code_search"] = tool

    parts = arg.split(maxsplit=1)
    subcmd = parts[0].lower()

    if subcmd == "def" or subcmd == "definition":
        name = parts[1] if len(parts) > 1 else ""
        if not name:
            print("Usage: /find def <name>")
            return
        result = tool.find_definition(name=name)
        if result.get("success"):
            defs = result.get("definitions", [])
            print(f"\n  Found {len(defs)} definition(s) of '{name}':\n")
            for d in defs:
                print(f"  {d['file']}:{d['line']} ({d['kind']})")
                print(f"    {d['text']}")
        else:
            print(f"  Error: {result.get('error')}")

    elif subcmd == "ref" or subcmd == "references":
        name = parts[1] if len(parts) > 1 else ""
        if not name:
            print("Usage: /find ref <name>")
            return
        result = tool.find_references(name=name)
        if result.get("success"):
            refs = result.get("references", [])
            print(f"\n  Found {len(refs)} reference(s) to '{name}':\n")
            for r in refs[:30]:
                print(f"  {r['file']}:{r['line']}\t{r['text']}")
        else:
            print(f"  Error: {result.get('error')}")

    elif subcmd == "structure" or subcmd == "tree":
        path = parts[1] if len(parts) > 1 else "."
        result = tool.project_structure(path=path)
        if result.get("success"):
            print(f"\n{result['tree']}")
            s = result.get("stats", {})
            print(f"\n  {s.get('files', 0)} files, {s.get('dirs', 0)} dirs")
        else:
            print(f"  Error: {result.get('error')}")

    else:
        # Treat as glob pattern
        result = tool.glob(pattern=arg)
        if result.get("success"):
            files = result.get("files", [])
            print(f"\n  Found {result.get('total', 0)} files:\n")
            for f in files[:50]:
                size = f.get("size", 0)
                size_str = f"{size // 1024}KB" if size > 1024 else f"{size}B"
                print(f"  {f['path']}  ({size_str})")
            if result.get("truncated"):
                print(f"\n  ... truncated ({result['total']} total)")
        else:
            print(f"  Error: {result.get('error')}")


def _handle_edit_command(agent, arg: str):
    """Handle /edit <path> — read file with line numbers for editing context."""
    if not arg:
        print("Usage: /edit <file-path> [line-offset]")
        print("  /edit src/main.py")
        print("  /edit src/main.py 100")
        return

    tool = agent.tools.get("code_edit")
    if not tool:
        from aura.tools.code_edit import CodeEditTool
        tool = CodeEditTool()
        agent.tools["code_edit"] = tool

    parts = arg.split()
    path = parts[0]
    offset = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 100

    result = tool.read_file(path=path, offset=offset, limit=limit)
    if result.get("success"):
        print(f"\n  {result['showing']}  ({result['path']})\n")
        print(result["content"])
    else:
        print(f"  Error: {result.get('error')}")


def _handle_project_command(agent, arg: str):
    """Handle /project — detect project type, init AURA.md, show context."""
    tool = agent.tools.get("code_search")
    if not tool:
        from aura.tools.code_search import CodeSearchTool
        tool = CodeSearchTool()
        agent.tools["code_search"] = tool

    parts = arg.split(maxsplit=1) if arg else ["info"]
    subcmd = parts[0].lower()

    if subcmd == "init":
        from aura.tools.project_context import init_project
        path = parts[1] if len(parts) > 1 else "."
        print(init_project(path))

    elif subcmd == "detect" or subcmd == "info":
        path = parts[1] if len(parts) > 1 else "."
        result = tool.detect_project_type(path=path)
        if result.get("success"):
            print(f"\n  Project Type:     {result.get('project_type', 'unknown')}")
            print(f"  Language:         {result.get('language', 'N/A')}")
            print(f"  Stack:            {', '.join(result.get('stack', [])) or 'N/A'}")
            print(f"  Frameworks:       {', '.join(result.get('frameworks', [])) or 'N/A'}")
            print(f"  Package Manager:  {result.get('package_manager', 'N/A')}")
            print(f"  Key Files:        {', '.join(result.get('key_files', [])) or 'N/A'}")
        else:
            print(f"  Error: {result.get('error')}")

    elif subcmd == "context":
        from aura.tools.project_context import load_project_context
        path = parts[1] if len(parts) > 1 else None
        ctx = load_project_context(path)
        if ctx:
            print(f"\n{ctx}")
        else:
            print("  No AURA.md found. Create one with: /project init")

    elif subcmd == "index":
        path = parts[1] if len(parts) > 1 else "."
        from aura.tools.codebase_index import CodebaseIndex
        idx = CodebaseIndex(path)
        def on_progress(current, total, fpath):
            if current % 20 == 0 or current == total:
                print(f"  [{current}/{total}] {fpath}")
        print("  Indexing codebase...")
        result = idx.index(progress_callback=on_progress)
        print(f"\n  Done: {result['indexed']} files indexed, {result['total_chunks']} chunks, "
              f"{result['skipped']} unchanged, {result['elapsed']}s")
        idx.close()

    elif subcmd == "search":
        query = parts[1] if len(parts) > 1 else ""
        if not query:
            print("Usage: /project search <query>")
            return
        path = "."
        from aura.tools.codebase_index import CodebaseIndex
        idx = CodebaseIndex(path)
        # Auto-index if empty
        if idx.stats()["total_chunks"] == 0:
            print("  No index found, indexing first...")
            idx.index()
        results = idx.search(query, top_k=10)
        if results:
            print(f"\n  Results for '{query}':\n")
            for r in results:
                score_pct = f"{r['score']:.0%}"
                print(f"  [{score_pct}] {r['file_path']}:{r['line_start']} ({r['kind']}) {r['name']}")
                snippet = (r.get('content') or '')[:100].replace('\n', ' ')
                print(f"        {snippet}")
        else:
            print("  No results found.")
        idx.close()

    else:
        print("Usage: /project [info|detect|init|context|index|search] [path|query]")


def _handle_shell_command(agent, arg: str):
    """Handle /shell, /bash, /run — execute shell commands."""
    if not arg:
        print("Usage: /shell <command>")
        print("  /shell git status")
        print("  /run npm test")
        print("  /bash ls -la")
        return

    tool = agent.tools.get("shell_executor")
    if not tool:
        from aura.tools.shell_executor import ShellExecutorTool
        tool = ShellExecutorTool()
        agent.tools["shell_executor"] = tool

    # Stream output in real-time
    def on_line(line):
        print(f"  {line}")

    result = tool.run_streaming(command=arg, on_output=on_line)

    if not result.get("success"):
        error = result.get("error", "")
        if error:
            print(f"\n  Error: {error}")
    print(f"\n  [exit {result.get('exit_code', '?')}] ({result.get('elapsed', '?')}s)")


def _handle_evolve_command(agent, arg: str):
    """Handle /evolve — run GEPA skill evolution."""
    print("\n  [GEPA] Starting skill evolution...")
    try:
        from aura.evolution.runner import run_evolution

        parts = arg.split() if arg else []
        skill_ids = None
        dry_run = "--dry-run" in parts
        iterations = 10

        for i, p in enumerate(parts):
            if p == "--skill" and i + 1 < len(parts):
                skill_ids = [parts[i + 1]]
            if p == "--iterations" and i + 1 < len(parts):
                try:
                    iterations = int(parts[i + 1])
                except ValueError:
                    pass

        result = run_evolution(
            skill_ids=skill_ids,
            config_overrides={"max_iterations": iterations},
            dry_run=dry_run,
        )

        if result.get("error"):
            print(f"  [GEPA] Error: {result['error']}")
        elif result.get("dry_run"):
            print(f"  [GEPA] Would evolve: {result['skills']}")
        else:
            print(f"  [GEPA] Done! Improvement: +{result.get('improvement', 0):.3f}")
            print(f"  Score: {result.get('seed_score', 0):.3f} -> {result.get('best_score', 0):.3f}")
            print(f"  Skills updated: {result.get('skills_updated', 0)}")
            print(f"  Iterations: {result.get('iterations', 0)}, Evals: {result.get('total_evals', 0)}")
            print(f"  Time: {result.get('duration_seconds', 0):.1f}s")
            print(f"  Run saved to: {result.get('run_dir', 'N/A')}")

    except ImportError as e:
        print(f"  [GEPA] Import error: {e}")
    except Exception as e:
        print(f"  [GEPA] Failed: {e}")
    print()


def _handle_agent_command(agent, arg: str):
    """Handle /agent subcommands using the multi-agent orchestrator."""
    # Initialize orchestrator if needed
    if not hasattr(agent, 'orchestrator') or agent.orchestrator is None:
        try:
            from aura.multi_agent.orchestrator import MultiAgentOrchestrator

            def llm_func(system_prompt, user_message):
                return agent.brain.think(user_message, system_prompt=system_prompt, use_history=False)

            agent.orchestrator = MultiAgentOrchestrator(
                tool_registry=agent.tools,
                llm_func=llm_func
            )
        except Exception as e:
            print(f"Multi-agent system not available: {e}")
            return

    specialists = list(agent.orchestrator.specialists.keys())

    if not arg:
        print("\n  Available specialists:")
        for name in specialists:
            spec = agent.orchestrator.specialists[name]
            desc = getattr(spec, 'description', name.capitalize())
            print(f"    {name:12s} - {desc}")
        print(f"\n  Usage: /agent <specialist> <task>")
        print(f"         /agent parallel <task>")
        return

    parts = arg.split(maxsplit=1)
    specialist = parts[0].lower()
    task = parts[1] if len(parts) > 1 else ""

    if not task:
        print(f"Usage: /agent {specialist} <task>")
        return

    if specialist == "parallel":
        # Run all specialists in parallel
        print(f"  Running all specialists in parallel on: {task[:60]}...")
        from aura.multi_agent.protocol import AgentMessage, CollaborationMode
        message = AgentMessage(content=task, sender="user")
        results = agent.orchestrator._execute_parallel(specialists, message)
        for result in results:
            status = "OK" if result.success else "FAIL"
            print(f"\n  [{result.agent.upper()}] ({status}):")
            print(f"  {result.response[:500]}")
    elif specialist in specialists:
        # Run specific specialist
        print(f"  [{specialist.upper()}] Working on: {task[:60]}...")
        from aura.multi_agent.protocol import AgentMessage
        message = AgentMessage(content=task, sender="user")
        result = agent.orchestrator._execute_single(specialist, message)
        if result.success:
            print(f"\n  [{specialist.upper()}]:")
            print(f"  {result.response}")
        else:
            print(f"  Error: {result.response}")
    else:
        print(f"  Unknown specialist: {specialist}")
        print(f"  Available: {', '.join(specialists)}, parallel")


def print_result(result, is_fastpath: bool = False):
    """Print the agent run result using rich."""
    from aura.cli.display import console, show_response

    response = result.get("response", "")
    if response:
        show_response(response)
    else:
        mode = "Fast-path" if is_fastpath else f"{result.get('iterations', '?')} iterations"
        console.print(f"[dim]Completed ({mode})[/dim]")


if __name__ == "__main__":
    main()

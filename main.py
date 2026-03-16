#!/usr/bin/env python3
"""Main entry point for the Apprentice Agent."""

import os
os.environ["TQDM_DISABLE"] = "1"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import sys

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

    # Handle subcommands that don't need the full agent
    if args.command:
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

    # Non-interactive mode: run prompt, print response, exit
    if args.prompt:
        prompt = args.prompt
        # Read stdin if piped
        if not sys.stdin.isatty():
            try:
                stdin_text = sys.stdin.read()[:50000]
                if stdin_text.strip():
                    prompt = f"{stdin_text}\n\n{prompt}"
            except Exception:
                pass
        result = agent.run(prompt)
        response = result.get("response", "")
        if response:
            print(response)
        sys.exit(0 if result.get("success", True) else 1)

    if args.voice:
        run_voice_mode(agent, enable_barge_in=not args.no_barge_in)
    elif args.goal:
        # One-shot agentic mode: aura "fix the login bug"
        prompt = " ".join(args.goal) if isinstance(args.goal, list) else args.goal
        run_agentic_oneshot(agent, prompt, args)
    else:
        # Default: interactive chat mode (just type 'aura' to start)
        run_chat_mode(agent, speak=args.speak, trust=args.trust, model=args.model)


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


def run_chat_mode(agent, speak: bool = False, trust: bool = False, model: str = None):
    """Interactive CLI — agentic loop with status bar, model picker, tool calling."""
    from aura.cli.display import (
        console, show_banner, show_response,
        show_error, show_info, show_status_bar, show_help,
        show_welcome_info, show_tool_call,
    )
    from aura.cli.input import create_session, get_input
    from aura.cli.model_picker import pick_model, update_model_roles_from_config
    from aura.core.agentic_loop import AgenticLoop
    from aura.core.session import AgenticSession
    from aura.core.permissions import PermissionManager
    from aura.core.context import gather_context, get_aura_md_config

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
        session=session,
        aura_config=aura_config,
        router=chat_router,
    )
    # Store on agent so /clear and /trust can access it
    agent._agentic_loop = agentic
    agent._agentic_permissions = permissions
    agent._agentic_session = session

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
    show_status_bar(
        model=_current_model, project_type=_project_type,
        session_title=_session_title, message_count=_msg_count,
    )

    if speak:
        show_info("Voice output enabled")

    # Initialize model picker roles from config
    update_model_roles_from_config()

    session = create_session()

    while True:
        user_input = get_input(session)

        if user_input is None:
            console.print("\n[dim]Goodbye.[/dim]\n")
            break

        # Handle Alt+M model picker
        if user_input == "__MODEL_PICK__":
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
            show_status_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
            )
            continue

        if not user_input:
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
            show_status_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
            )
            continue

        # Run agentic loop directly (no threading — permission prompts need
        # to be visible on the main thread, and think_with_tools has its own
        # 120s timeout so it won't hang forever)
        show_info("Thinking...")
        try:
            def _on_tool(name, args, _result):
                desc = args.get("path") or args.get("pattern") or args.get("query") or ""
                if not desc and "command" in args:
                    desc = args["command"][:60]
                show_tool_call(name, str(desc))

            result = agentic.run(
                user_input,
                on_tool_call=_on_tool,
            )
        except KeyboardInterrupt:
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

        # Update status bar
        _msg_count += 1
        _current_model = agent.brain._model_override or "auto"
        cost_usd = 0.0
        try:
            stats = agent.brain.get_session_stats()
            cost_usd = stats.get("cost_usd", 0.0)
        except Exception:
            pass
        show_status_bar(
            model=_current_model, project_type=_project_type,
            session_title=_session_title, message_count=_msg_count,
            cost_usd=cost_usd,
        )

        if speak and response_text:
            try:
                agent._speak(response_text)
            except Exception:
                pass


def handle_command(agent, command: str, speak: bool = False):
    """Handle special commands in chat mode."""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/quit" or cmd == "/exit":
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
                    show_info("Model override cleared. Using auto-selection.")
                else:
                    agent.brain.set_model_override(choice)
                    show_info(f"Model locked to: {choice}")
        elif arg.lower() == "auto":
            agent.brain.set_model_override(None)
            print("Model override cleared. Using auto-selection.")
        else:
            agent.brain.set_model_override(arg)
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
            print("\nCreating execution plan...")
            plan = agent.create_plan(arg)
            print(f"\n  Task: {plan.get('task', arg)}")
            print(f"  Complexity: {plan.get('complexity', 'unknown')}")
            print(f"  Steps:")
            for i, step in enumerate(plan.get('steps', []), 1):
                print(f"    {i}. {step}")
            if plan.get('tools'):
                print(f"  Tools needed: {', '.join(plan['tools'])}")
            print()
            try:
                confirm = input("  Execute this plan? (yes/no): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                confirm = "no"
            if confirm in ("yes", "y"):
                print("\n  Executing plan...")
                result = agent.run(arg)
                print_result(result, is_fastpath=result.get("fast_path", False))
            else:
                print("  Plan cancelled.")
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
    elif cmd == "/hook":
        _handle_hook_command(agent, arg)
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
            print("\n  Agentic Sessions:\n")
            if agentic_sessions:
                for i, s in enumerate(agentic_sessions[:10], 1):
                    title = s.get("title", "Untitled")[:50]
                    msgs = s.get("message_count", 0)
                    model = s.get("model", "?")
                    print(f"    {i}. {title} ({msgs} msgs) [{model}]")
            else:
                print("    (none)")
            if brain_conversations:
                print("\n  Legacy Sessions:\n")
                for i, c in enumerate(brain_conversations[:5], 1):
                    title = c.get("title", "Untitled")[:50]
                    msgs = c.get("message_count", 0)
                    print(f"    {i}. {title} ({msgs} msgs)")
            print(f"\n  Usage: /sessions new | /sessions delete <id>")
    elif cmd == "/trust":
        if hasattr(agent, '_agentic_permissions'):
            agent._agentic_permissions.set_trust_mode(True)
        else:
            from aura.core.permissions import PermissionManager
            agent._agentic_permissions = PermissionManager()
            agent._agentic_permissions.set_trust_mode(True)
        print("  Trust mode enabled — all tool calls auto-approved.")
    elif cmd == "/context":
        if hasattr(agent, '_agentic_loop') and agent._agentic_loop.context_mgr:
            mgr = agent._agentic_loop.context_mgr
            report = mgr.usage_report(agent._agentic_loop._conversation_history)
            print(f"\n  Context Window:")
            print(f"    Model: {report['model'] or 'auto'}")
            print(f"    Used: ~{report['used_tokens']:,} / {report['budget']:,} tokens ({report['pct_used']}%)")
            print(f"    Max context: {report['max_tokens']:,}")
            print(f"    Compactions: {report['compactions']}")
            print()
        else:
            print("  Context tracking not available.")
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


def _handle_hook_command(agent, arg: str):
    """Handle /hook subcommands."""
    if not hasattr(agent, 'hooks') or agent.hooks is None:
        try:
            from aura.hooks import HooksManager
            agent.hooks = HooksManager(tools=agent.tools)
            agent.hooks.start_background(interval=15)
        except Exception as e:
            print(f"Hooks system not available: {e}")
            return

    if not arg:
        print("Usage: /hook list | /hook add <event> <condition> <action> <args> | /hook remove <id>")
        return

    parts = arg.split(maxsplit=1)
    subcmd = parts[0].lower()
    subarg = parts[1] if len(parts) > 1 else ""

    if subcmd == "list":
        hooks = agent.hooks.list_hooks()
        if not hooks:
            print("  No hooks registered.")
        else:
            print(f"\n  Registered hooks ({len(hooks)}):")
            for h in hooks:
                print(f"    [{h['id']}] {h['event']}:{h['condition']} -> {h['action']} {h.get('action_args', '')}")
    elif subcmd == "add":
        # Parse: /hook add schedule 09:00 notify "Good morning"
        add_parts = subarg.split(maxsplit=3)
        if len(add_parts) < 3:
            print("Usage: /hook add <event> <condition> <action> [args]")
            print("  Events:  schedule, file_modified, system_alert, clipboard_changed")
            print("  Actions: notify, speak, run_tool, log")
            print("  Example: /hook add schedule 09:00 notify Good morning!")
            return
        event = add_parts[0]
        condition = add_parts[1]
        action = add_parts[2]
        action_args = add_parts[3] if len(add_parts) > 3 else ""
        hook_id = agent.hooks.register(event, condition, action, action_args)
        print(f"  Hook registered with ID: {hook_id}")
    elif subcmd == "remove":
        if not subarg:
            print("Usage: /hook remove <id>")
            return
        success = agent.hooks.unregister(subarg)
        if success:
            print(f"  Hook {subarg} removed.")
        else:
            print(f"  Hook {subarg} not found.")
    else:
        print(f"  Unknown hook command: {subcmd}")


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

#!/usr/bin/env python3
"""Main entry point for the Apprentice Agent."""
from __future__ import annotations

# Stub _wmi BEFORE any other import — Python 3.12's platform.py imports _wmi
# at module level, and WMI can hang on Windows. This stub lets platform.machine()
# fall back to the PROCESSOR_ARCHITECTURE env var instead.
import sys as _sys, types as _types
if '_wmi' not in _sys.modules:
    _wmi_stub = _types.ModuleType('_wmi')
    _wmi_stub.exec_query = lambda *a, **k: (_ for _ in ()).throw(OSError("_wmi stubbed"))
    _sys.modules['_wmi'] = _wmi_stub

import argparse
import os
import sys
from typing import Any, NoReturn, Optional


def _get_console():
    """Lazy accessor for the shared Rich console (deferred to avoid import cost at startup)."""
    try:
        from aura.cli.display import console
        return console
    except ImportError:
        from rich.console import Console
        return Console()


def _suppress_warnings() -> None:
    """Silence noisy third-party warnings and progress bars at startup."""
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["SENTENCE_TRANSFORMERS_NO_PROGRESS_BAR"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    try:
        from tqdm import tqdm as _orig_tqdm
        from functools import partialmethod
        _orig_tqdm.__init__ = partialmethod(_orig_tqdm.__init__, disable=True)
    except ImportError:
        pass

    import warnings
    import logging
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="comtypes")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pycaw")
    warnings.filterwarnings("ignore", message="urllib3.*charset_normalizer")
    warnings.filterwarnings("ignore", message="Revert to STA COM")

    # Suppress torchao/triton warnings
    warnings.filterwarnings("ignore", module="torchao")
    os.environ["TORCHAO_DISABLE_TRITON"] = "1"

    # Suppress noisy aura internal log messages
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
    logging.getLogger("aura.auth").setLevel(logging.ERROR)
    logging.getLogger("aura.tools.web_search").setLevel(logging.ERROR)
    logging.getLogger("aura.tools.custom_loader").setLevel(logging.ERROR)
    logging.getLogger("aura.memory.store").setLevel(logging.ERROR)
    logging.getLogger("aura.memory.retrieval").setLevel(logging.ERROR)
    logging.getLogger("torchao").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Suppress sentence_transformers stdout load report
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "")
    warnings.filterwarnings("ignore", module="sentence_transformers")
    warnings.filterwarnings("ignore", module="transformers")


_SUBCOMMANDS = {
    "init", "setup", "doctor", "config", "models", "commit", "cost",
    "mcp-serve", "acp-serve", "exec", "ide", "log", "status", "recall",
    "start", "stop", "why", "heatmap", "worktree",
}
_OPTS_WITH_VALUES = {
    "--max-iterations", "--dream-date", "-p", "--prompt", "--login", "--logout",
    "--tier", "--budget", "--preference", "--model", "--format",
    "--channels", "-ch",
}


def _argv_has_subcommand(argv: list[str]) -> bool:
    """Return True if the first positional token in argv is a known subcommand.

    Needed because argparse subparsers are greedy: if we always register them,
    `aura "fix the login bug"` fails with 'invalid choice: fix'. We pre-scan
    to decide whether to register subparsers at all.
    """
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            return False
        if tok.startswith("-"):
            if "=" in tok:
                i += 1
                continue
            if tok in _OPTS_WITH_VALUES:
                i += 2
                continue
            i += 1
            continue
        return tok in _SUBCOMMANDS
    return False


def _build_argument_parser() -> tuple[argparse.ArgumentParser, bool]:
    """Build and return the CLI argument parser plus the use_subparsers flag.

    Returns a (parser, use_subparsers) tuple so callers can check whether
    subparser-mode was activated before calling parser.parse_args().
    """
    from aura._version import __version__
    parser = argparse.ArgumentParser(
        description="AURA - Autonomous Universal Reasoning Agent\n\n"
        "Usage examples:\n"
        "  aura                     Interactive chat mode\n"
        "  aura \"fix the login bug\"  One-shot agentic execution\n"
        "  aura init                Create AURA.md for current project\n"
        "  aura doctor              Check Ollama, models, dependencies\n"
        "  aura commit --all        AI-generated commit message\n"
        "  aura --voice             Voice mode with speech input/output\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog="aura"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Only register subparsers if a known subcommand is present in argv.
    # Otherwise argparse would greedily consume the first positional of a
    # goal-mode invocation (e.g. `aura "fix the login bug"`) and crash.
    # Also register when -h/--help is requested so the help text is complete.
    _wants_help = any(a in ("-h", "--help") for a in sys.argv[1:])
    use_subparsers = _argv_has_subcommand(sys.argv[1:]) or _wants_help
    if use_subparsers:
        subparsers = parser.add_subparsers(dest="command")
        subparsers.add_parser("init", help="Create AURA.md in current project")
        subparsers.add_parser("setup", help="Interactive setup wizard for new projects")
        subparsers.add_parser("doctor", help="Check Ollama, models, dependencies")
        subparsers.add_parser("config", help="Show current configuration")
        subparsers.add_parser("models", help="List available models with routing roles")
        sub_commit = subparsers.add_parser("commit", help="Smart commit with AI-generated message")
        sub_commit.add_argument("--all", "-a", action="store_true", help="Stage all changes")
        sub_cost = subparsers.add_parser("cost", help="Show session cost breakdown")
        sub_cost.add_argument("--by-model", dest="by_model", action="store_true",
                              help="Show per-model cost breakdown")
        sub_cost.add_argument("--by-provider", dest="by_provider", action="store_true",
                              help="Show per-provider cost breakdown")
        sub_cost.add_argument("--session", default="",
                              help="Filter to a specific session ID")
        subparsers.add_parser("status", help="Show current Aura state (Ollama, routing, bandit, daemon)")
        subparsers.add_parser("start", help="Start the Aura background daemon")
        subparsers.add_parser("stop", help="Stop the Aura background daemon")
        sub_why = subparsers.add_parser("why", help="Intent-to-Code Ledger lookup")
        sub_why.add_argument("why_target", nargs="?", default="", help="Path or path:line")
        sub_why.add_argument("--limit", dest="why_limit", type=int, default=5)
        sub_hm = subparsers.add_parser("heatmap", help="Show cognitive heatmap (tokens per tool/file)")
        sub_hm.add_argument("--session", dest="heatmap_session", default=None, help="Session ID (default: latest)")
        sub_wt = subparsers.add_parser("worktree", help="Manage git worktrees for isolated feature work")
        sub_wt.add_argument("worktree_name", nargs="?", default="", help="Worktree name")
        sub_wt.add_argument("--branch", dest="worktree_branch", default=None, help="Branch name (default: worktree name)")
        sub_wt.add_argument("--remove", dest="worktree_remove", action="store_true", help="Remove a worktree")
        sub_wt.add_argument("--force", dest="worktree_force", action="store_true", help="Force removal")
        sub_wt.add_argument("--open", dest="worktree_open", action="store_true", help="Open a new terminal in the worktree")
        sub_wt.add_argument("--list", dest="worktree_list", action="store_true", help="List existing worktrees")
        sub_recall = subparsers.add_parser("recall", help="Query UnifiedMemory from the shell")
        sub_recall.add_argument("recall_query", nargs="*", default=[], help="Topic to recall")
        sub_recall.add_argument("--limit", dest="recall_limit", type=int, default=5, help="Max results (default 5)")
        subparsers.add_parser("mcp-serve", help="Run as MCP server (JSON-RPC over stdio)")
        subparsers.add_parser("acp-serve", help="Run as ACP server (agent-client protocol, stdio)")
        sub_exec = subparsers.add_parser("exec", help="Non-interactive agent execution")
        sub_exec.add_argument("exec_prompt", nargs="?", default=None, help="Prompt to execute")
        sub_exec.add_argument(
            "--timeout", dest="exec_timeout", type=int, default=0,
            help="Hard wall-clock timeout in seconds; exits 124 if exceeded (0 = no timeout)",
        )
        sub_exec.add_argument(
            "--quiet", action="store_true",
            help="Suppress progress output; print only the final response",
        )
        sub_exec.add_argument(
            "--output-failures", dest="output_failures", action="store_true",
            help="Emit per-tool failure JSON to stderr (for CI consumption)",
        )
        sub_ide = subparsers.add_parser("ide", help="IDE integration setup")
        sub_ide.add_argument("action", nargs="?", default="setup", choices=["setup"], help="Action (default: setup)")
        sub_log = subparsers.add_parser("log", help="Query interaction history")
        sub_log.add_argument("action", choices=["search", "export", "stats", "recent"], nargs="?", default="recent")
        sub_log.add_argument("query", nargs="*", default=[])
        sub_log.add_argument("--session", default="")
        sub_log.add_argument("--format", dest="log_format", choices=["markdown", "json"], default="markdown")
        sub_log.add_argument("--limit", type=int, default=20)

    # Positional prompt for one-shot agentic mode
    parser.add_argument(
        "goal",
        nargs="*",
        help="One-shot agentic prompt (e.g., aura 'fix the login bug')"
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
        "--fast",
        action="store_true",
        help="Skip the agentic loop: call agent.run() directly. No tool calls, "
             "no permissions, no session. Use for cheap scripted queries where "
             "you only want a raw answer."
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
        choices=["fast", "balanced", "max"],
        default=None,
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
    _sandbox_group = parser.add_mutually_exclusive_group()
    _sandbox_group.add_argument(
        "--sandboxed",
        action="store_true",
        help="Read-only sandbox: only read/search tools allowed; writes/shell/git "
             "mutations are blocked. Overrides --trust for safety."
    )
    _sandbox_group.add_argument(
        "--workspace-write",
        dest="workspace_write",
        action="store_true",
        help="Workspace-write sandbox: file edits auto-approved in cwd; shell, "
             "spawn_agent, git push/pull still require approval."
    )
    _sandbox_group.add_argument(
        "--unrestricted",
        action="store_true",
        help="Unrestricted sandbox (default): per-tool permissions as configured."
    )
    parser.add_argument(
        "--preference",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Routing preference tier: fast (speed), balanced (default), quality (best output)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Use a specific model (e.g., --model kimi-k2.5:cloud)"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format for non-interactive mode (default: text). "
             "JSON mode denies tool-permission prompts unless --trust is set; "
             "denied tools emit a 'permission_denied' event so scripted "
             "consumers can react."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose mode: expand all tool output sections"
    )
    parser.add_argument(
        "--channels", "-ch",
        nargs="+",
        default=[],
        help="Enable channel bridges (telegram, extension)"
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "debate", "chain"],
        default="chat",
        help="Reasoning mode for non-interactive runs. Default 'chat' is normal "
             "agentic. 'debate' runs three-model cross-examination on the prompt. "
             "'chain' executes a step chain parsed as 'step1 -> step2 -> step3'."
    )
    parser.add_argument(
        "--routing-trace",
        action="store_true",
        help="Log every ModelRouter decision (category, confidence, tier, model) "
             "to stderr so scripts can audit which model handled a prompt."
    )
    return parser, use_subparsers


def _handle_resume(agent: Any, args: argparse.Namespace) -> Optional[str]:
    """List sessions and let the user pick one to resume. Returns selected session id or None."""
    from aura.core.session import AgenticSession as _SessionCheck
    _ses = _SessionCheck()
    agentic_sessions = _ses.list_sessions()
    brain_conversations = agent.brain.list_conversations()

    _resume_console = _get_console()
    if args.resume == "last":
        if agentic_sessions:
            latest = agentic_sessions[0]
            agent._resume_session_id = latest["id"]
            _resume_console.print(f"  [green]Resuming:[/green] {latest.get('title', 'Untitled')} [dim]({latest.get('message_count', 0)} messages)[/dim]")
        elif brain_conversations:
            latest = brain_conversations[0]
            agent.brain.switch_conversation(latest["id"])
            _resume_console.print(f"  [green]Resumed (legacy):[/green] {latest.get('title', 'Untitled')}")
        else:
            _resume_console.print("[dim]No previous sessions found.[/dim]")
        return None

    all_sessions = []
    for s in agentic_sessions:
        s["_source"] = "agentic"
        all_sessions.append(s)
    for c in brain_conversations:
        c["_source"] = "brain"
        all_sessions.append(c)
    all_sessions.sort(key=lambda x: x.get("updated_at", 0), reverse=True)

    if not all_sessions:
        _resume_console.print("[dim]No previous sessions found.[/dim]")
        return None

    _resume_console.print("\n  [bold]Recent sessions:[/bold]\n")
    for i, s in enumerate(all_sessions[:10], 1):
        title = s.get("title", "Untitled")[:50]
        msgs = s.get("message_count", 0)
        src = s.get("_source", "?")
        _resume_console.print(f"    [cyan]{i}.[/cyan] {title} [dim]({msgs} msgs) [{src}][/dim]")
    _resume_console.print()
    try:
        choice = input("  Pick a session (number): ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(all_sessions[:10]):
            picked = all_sessions[idx]
            if picked["_source"] == "agentic":
                agent._resume_session_id = picked["id"]
            else:
                agent.brain.switch_conversation(picked["id"])
            _resume_console.print(f"  [green]Resuming:[/green] {picked.get('title', 'Untitled')}")
        else:
            _resume_console.print("  [yellow]Invalid choice, starting new session.[/yellow]")
    except (ValueError, EOFError, KeyboardInterrupt):
        _resume_console.print("  [dim]Starting new session.[/dim]")
    return None


def main() -> None:
    _suppress_warnings()
    parser, use_subparsers = _build_argument_parser()

    args = parser.parse_args()
    if not use_subparsers:
        args.command = None

    # Apply sandbox tier as early as possible so any subsequent code path
    # (subcommands, one-shot, chat loop) sees the clamp.
    if getattr(args, "sandboxed", False):
        from aura.core.permissions import SandboxTier, set_sandbox_tier
        set_sandbox_tier(SandboxTier.READ_ONLY)
    elif getattr(args, "workspace_write", False):
        from aura.core.permissions import SandboxTier, set_sandbox_tier
        set_sandbox_tier(SandboxTier.WORKSPACE_WRITE)
    # --unrestricted is the default; no-op.

    # Routing trace: emit every router decision to stderr
    if getattr(args, "routing_trace", False):
        from aura.core.router import enable_routing_trace
        enable_routing_trace()

    # Validate conflicting flags
    if args.voice and args.prompt:
        parser.error("--voice and --prompt cannot be used together")

    # Handle auth commands (no agent needed)
    if args.login:
        if args.login.lower() == "chatgpt":
            from aura.auth.chatgpt_oauth import login
            sys.exit(0 if login() else 1)
        else:
            _get_console().print(f"[red]Unknown provider: {args.login}. Available: chatgpt[/red]")
            sys.exit(1)

    if args.logout:
        if args.logout.lower() == "chatgpt":
            from aura.auth.chatgpt_oauth import logout
            logout()
            sys.exit(0)
        else:
            _get_console().print(f"[red]Unknown provider: {args.logout}. Available: chatgpt[/red]")
            sys.exit(1)

    # Handle MCP server (lightweight, no agent needed)
    if args.command == "mcp-serve":
        from aura.core.mcp_server import main as mcp_main
        mcp_main()
        sys.exit(0)

    if args.command == "acp-serve":
        from aura.acp.server import run_acp_server
        run_acp_server()
        sys.exit(0)

    # Handle log subcommand (lightweight, no agent needed)
    if args.command == "log":
        from aura.cli.activity_log import ActivityLog
        _log_console = _get_console()
        log = ActivityLog()
        if args.action == "search":
            query = " ".join(args.query)
            results = log.search(query, limit=args.limit)
            for r in results:
                _log_console.print(f"[cyan][{r['model']}][/cyan] {r['prompt'][:80]}")
                _log_console.print(f"  [dim]-> {r['response'][:120]}[/dim]")
        elif args.action == "stats":
            stats = log.get_stats()
            for k, v in stats.items():
                _log_console.print(f"  [bold]{k}[/bold]: {v}")
        elif args.action == "export":
            if not args.session:
                _log_console.print("[yellow]Usage: aura log export --session <session_id>[/yellow]")
                sys.exit(1)
            md = log.export_session(args.session, format=args.log_format)
            print(md)  # raw output for export (may be piped)
        else:  # recent
            for r in log.get_recent(args.limit):
                _log_console.print(f"  [dim]{r['prompt'][:80]}[/dim]")
        sys.exit(0)

    # Handle exec subcommand — convert to non-interactive --prompt path
    if args.command == "exec":
        exec_prompt = getattr(args, "exec_prompt", None) or args.prompt
        if not exec_prompt:
            _get_console().print("[yellow]Usage: aura exec 'your prompt here'[/yellow]")
            sys.exit(1)
        args.prompt = exec_prompt  # normalize for the shared --prompt path below
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
        _dream_console = _get_console()
        date_label = args.dream_date or "today"
        _dream_console.print(f"[cyan][Dream] Analyzing metacognition logs for {date_label}...[/cyan]")
        result = run_dream_mode(args.dream_date)
        if result.get("success"):
            insights = result.get("insights", [])
            _dream_console.print(
                f"[green][Dream] Analyzed {result.get('logs_analyzed', 0)} log entries, "
                f"generated {len(insights)} insights, "
                f"stored {len(result.get('stored_ids', []))} in memory.[/green]"
            )
            for i, ins in enumerate(insights, 1):
                _dream_console.print(f"  [dim]{i}. {ins}[/dim]")
            cons = result.get("consolidation") or {}
            if cons:
                _dream_console.print(
                    f"[cyan][Dream] Consolidation: merged={cons.get('merged', 0)}, "
                    f"pruned={cons.get('pruned', 0)}[/cyan]"
                )
            sys.exit(0)
        err = result.get("error", "Unknown error")
        _dream_console.print(f"[dim][Dream] {err}[/dim]")
        # "No logs found" is informational, not a failure.
        sys.exit(0 if err == "No logs found" else 1)

    # Fast-init path for non-interactive flows — skips proactive, neurodream,
    # visible thinking, soul, tool_rag, adaptive_planner, fast_path_handler,
    # thinker, reasoning_tree. Cuts cold start ~3-5s for --prompt and exec.
    _fast_init = bool(getattr(args, "prompt", None)) or args.command == "exec"

    try:
        agent = ApprenticeAgent(fast_init=_fast_init)
    except ConnectionError as e:
        _get_console().print(f"\n[red][AURA] Cannot connect to Ollama: {e}[/red]")
        _get_console().print("[dim]Start it with: ollama serve[/dim]")
        sys.exit(1)
    except (FileNotFoundError, PermissionError) as e:
        _get_console().print(f"\n[red][AURA] Config/filesystem error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        _get_console().print(f"\n[red][AURA] Failed to initialize agent: {e}[/red]")
        _get_console().print("[dim]Run 'aura doctor' to diagnose the issue.[/dim]")
        sys.exit(1)
    agent.max_iterations = args.max_iterations
    agent.use_fastpath = not args.no_fastpath

    # Ensure agent cleanup runs on exit (KG flush, skill library save, etc.)
    import atexit
    atexit.register(agent.shutdown)

    # Handle session resume — try agentic sessions first, fall back to brain conversations
    if args.resume:
        _handle_resume(agent, args)

    # Read piped stdin if available (for composability)
    from aura.cli.pipe_mode import PipeOutput, read_piped_input, EXIT_SUCCESS, EXIT_ERROR

    if not args.prompt and not sys.stdin.isatty():
        piped = read_piped_input()
        if piped:
            args.prompt = piped

    # If stdout is piped and the user didn't pick an output format, assume
    # the caller is a script that wants structured JSONL output.
    if not sys.stdout.isatty() and args.format == "text":
        args.format = "json"

    # Initialize channel bridge if --channels specified (available to every path)
    bridge = None
    if args.channels:
        try:
            from aura.channels.channel_bridge import ChannelBridge
            bridge = ChannelBridge()

            for ch_name in args.channels:
                if ch_name == 'telegram':
                    from aura.channels.telegram_channel import TelegramChannel
                    bridge.add_channel(TelegramChannel())
                elif ch_name == 'extension':
                    from aura.channels.extension_channel import ExtensionChannel
                    bridge.add_channel(ExtensionChannel())
                elif ch_name == 'slack':
                    from aura.channels.slack_channel import SlackChannel
                    bridge.add_channel(SlackChannel())
                else:
                    _get_console().print(f"[red]Unknown channel: {ch_name}[/red]")
                    sys.exit(1)

            bridge.start()
        except Exception as e:
            _get_console().print(f"[red][AURA] Channel bridge failed to start: {e}[/red]")
            bridge = None

    # Unified non-interactive dispatch: -p/--prompt, `exec` subcommand (which
    # normalizes into args.prompt above), positional goal, and piped stdin all
    # feed into the same dispatcher. Default is the rich agentic loop; --fast
    # is the escape hatch for scripts that only want a raw one-shot answer.
    noninteractive_prompt = args.prompt
    if not noninteractive_prompt and args.goal:
        noninteractive_prompt = " ".join(args.goal)

    if noninteractive_prompt and not args.voice:
        if args.fast:
            pipe = PipeOutput(format=args.format)
            result = agent.run(noninteractive_prompt)
            response = result.get("response", "")
            model_used = result.get("model", "")
            if response:
                pipe.result({"response": response, "model": model_used})
            sys.exit(EXIT_SUCCESS if result.get("success", True) else EXIT_ERROR)
        else:
            from aura.cli.oneshot import run_agentic_oneshot
            run_agentic_oneshot(agent, noninteractive_prompt, args, bridge=bridge)
            # run_agentic_oneshot calls sys.exit(); unreachable below.

    if args.voice:
        from aura.cli.voice_mode import run_voice_mode
        run_voice_mode(agent, enable_barge_in=not args.no_barge_in, bridge=bridge)
    else:
        # Surface the latest dream report (< 48h old) on chat startup so
        # users see what the nightly consolidation produced without running
        # /memory or /dream manually.
        try:
            from aura.dream import load_latest_dream_report
            _report = load_latest_dream_report()
            if _report and _report.get("insights"):
                from aura.cli.display import console as _console
                _console.print(
                    f"\n[dim cyan]\u25ce last dream ({_report.get('date', '?')}): "
                    f"{_report.get('logs_analyzed', 0)} logs analyzed, "
                    f"{len(_report['insights'])} insights[/dim cyan]"
                )
                for i, _ins in enumerate(_report["insights"][:3], 1):
                    _console.print(f"[dim]    {i}. {str(_ins)[:100]}[/dim]")
                _console.print()
        except Exception:
            pass
        from aura.cli.chat_loop import run_chat_mode
        run_chat_mode(agent, speak=args.speak, trust=args.trust, model=args.model, verbose=args.verbose, tier=args.tier, bridge=bridge, preference=args.preference)


if __name__ == "__main__":
    main()

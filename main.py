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
from typing import Any, NoReturn


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
    "mcp-serve", "exec", "ide", "log",
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


def main() -> None:
    _suppress_warnings()
    from aura._version import __version__
    parser = argparse.ArgumentParser(
        description="AURA - Autonomous Universal Reasoning Agent",
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
        subparsers.add_parser("cost", help="Show session cost breakdown")
        subparsers.add_parser("mcp-serve", help="Run as MCP server (JSON-RPC over stdio)")
        sub_exec = subparsers.add_parser("exec", help="Non-interactive agent execution")
        sub_exec.add_argument("exec_prompt", nargs="?", default=None, help="Prompt to execute")
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
        help="Output format for non-interactive mode (default: text)"
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

    args = parser.parse_args()
    if not use_subparsers:
        args.command = None

    # Validate conflicting flags
    if args.voice and args.prompt:
        parser.error("--voice and --prompt cannot be used together")

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

    # Handle log subcommand (lightweight, no agent needed)
    if args.command == "log":
        from aura.cli.activity_log import ActivityLog
        log = ActivityLog()
        if args.action == "search":
            query = " ".join(args.query)
            results = log.search(query, limit=args.limit)
            for r in results:
                print(f"[{r['model']}] {r['prompt'][:80]}")
                print(f"  -> {r['response'][:120]}")
        elif args.action == "stats":
            stats = log.get_stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")
        elif args.action == "export":
            if not args.session:
                print("Usage: aura log export --session <session_id>")
                sys.exit(1)
            md = log.export_session(args.session, format=args.log_format)
            print(md)
        else:  # recent
            for r in log.get_recent(args.limit):
                print(f"  {r['prompt'][:80]}")
        sys.exit(0)

    # Handle exec subcommand — convert to non-interactive --prompt path
    if args.command == "exec":
        exec_prompt = getattr(args, "exec_prompt", None) or args.prompt
        if not exec_prompt:
            print("Usage: aura exec 'your prompt here'")
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
        date_label = args.dream_date or "today"
        print(f"[Dream] Analyzing metacognition logs for {date_label}...")
        result = run_dream_mode(args.dream_date)
        if result.get("success"):
            insights = result.get("insights", [])
            print(
                f"[Dream] Analyzed {result.get('logs_analyzed', 0)} log entries, "
                f"generated {len(insights)} insights, "
                f"stored {len(result.get('stored_ids', []))} in memory."
            )
            for i, ins in enumerate(insights, 1):
                print(f"  {i}. {ins}")
            cons = result.get("consolidation") or {}
            if cons:
                print(
                    f"[Dream] Consolidation: merged={cons.get('merged', 0)}, "
                    f"pruned={cons.get('pruned', 0)}"
                )
            sys.exit(0)
        err = result.get("error", "Unknown error")
        print(f"[Dream] {err}")
        # "No logs found" is informational, not a failure.
        sys.exit(0 if err == "No logs found" else 1)

    try:
        agent = ApprenticeAgent()
    except ConnectionError as e:
        print(f"\n[AURA] Cannot connect to Ollama: {e}")
        print("Start it with: ollama serve")
        sys.exit(1)
    except (FileNotFoundError, PermissionError) as e:
        print(f"\n[AURA] Config/filesystem error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[AURA] Failed to initialize agent: {e}")
        print("Run 'aura doctor' to diagnose the issue.")
        sys.exit(1)
    agent.max_iterations = args.max_iterations
    agent.use_fastpath = not args.no_fastpath

    # Ensure agent cleanup runs on exit (KG flush, skill library save, etc.)
    import atexit
    atexit.register(agent.shutdown)

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
        else:
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
    from aura.cli.pipe_mode import PipeOutput, read_piped_input, EXIT_SUCCESS, EXIT_ERROR

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

    # Initialize channel bridge if --channels specified
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
                else:
                    print(f"Unknown channel: {ch_name}")
                    sys.exit(1)

            bridge.start()
        except Exception as e:
            print(f"[AURA] Channel bridge failed to start: {e}")
            bridge = None

    if args.voice:
        from aura.cli.voice_mode import run_voice_mode
        run_voice_mode(agent, enable_barge_in=not args.no_barge_in, bridge=bridge)
    elif args.goal:
        from aura.cli.oneshot import run_agentic_oneshot
        prompt = " ".join(args.goal)
        run_agentic_oneshot(agent, prompt, args, bridge=bridge)
    else:
        from aura.cli.chat_loop import run_chat_mode
        run_chat_mode(agent, speak=args.speak, trust=args.trust, model=args.model, verbose=args.verbose, tier=args.tier, bridge=bridge, preference=args.preference)


if __name__ == "__main__":
    main()

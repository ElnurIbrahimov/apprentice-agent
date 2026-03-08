#!/usr/bin/env python3
"""Main entry point for the Apprentice Agent."""

import os
os.environ["TQDM_DISABLE"] = "1"

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="AURA - Autonomous Universal Reasoning Agent",
        prog="aura"
    )
    parser.add_argument(
        "goal",
        nargs="?",
        help="The goal for the agent to achieve"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start in interactive chat mode"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations for the agent loop (default: 10)"
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

    args = parser.parse_args()

    # Heavy imports deferred until after argparse (so --help is instant)
    from aura import ApprenticeAgent
    from aura.config import Config
    from aura.dream import run_dream_mode

    # Handle dream mode first (doesn't need agent)
    if args.dream:
        result = run_dream_mode(args.dream_date)
        sys.exit(0 if result.get("success") else 1)

    agent = ApprenticeAgent()
    agent.max_iterations = args.max_iterations
    agent.use_fastpath = not args.no_fastpath

    # Handle session resume
    if args.resume:
        conversations = agent.brain.list_conversations()
        if not conversations:
            print("No previous sessions found.")
        elif args.resume == "last":
            latest = conversations[0]  # Already sorted by updated_at desc
            agent.brain.switch_conversation(latest["id"])
            print(f"  Resumed: {latest.get('title', 'Untitled')} ({latest.get('message_count', 0)} messages)")
        else:
            # Show picker
            print("\n  Recent sessions:\n")
            for i, conv in enumerate(conversations[:10], 1):
                active = " *" if conv.get("is_active") else ""
                title = conv.get("title", "Untitled")[:50]
                msgs = conv.get("message_count", 0)
                print(f"    {i}. {title} ({msgs} msgs){active}")
            print()
            try:
                choice = input("  Pick a session (number): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(conversations[:10]):
                    picked = conversations[idx]
                    agent.brain.switch_conversation(picked["id"])
                    print(f"  Resumed: {picked.get('title', 'Untitled')}")
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
        from aura.cli.display import show_banner
        show_banner()
        result = agent.run(args.goal)
        print_result(result, is_fastpath=result.get("fast_path", False))
    else:
        # Default: interactive chat mode (just type 'aura' to start)
        run_chat_mode(agent, speak=args.speak)


def run_voice_mode(agent, enable_barge_in: bool = True):
    """Run the agent in voice conversation mode."""
    from aura.tools.voice import VoiceConversation  # lazy: avoids ~7s sesame_tts load
    conversation = VoiceConversation(agent, whisper_model="base", enable_barge_in=enable_barge_in)
    conversation.start()


def run_chat_mode(agent, speak: bool = False):
    """Interactive CLI — full agent loop with status bar, model picker, streaming."""
    import io
    import sys
    import threading
    from aura.cli.display import (
        console, show_banner, show_thinking, show_response,
        show_error, show_info, show_status_bar, show_help,
    )
    from aura.cli.input import create_session, get_input
    from aura.cli.model_picker import pick_model, update_model_roles_from_config

    show_banner()

    # Detect project type for status bar
    _project_type = ""
    try:
        from aura.tools.project_context import detect_and_load_context
        ctx = detect_and_load_context(".")
        _project_type = ctx.get("project_type", "") if isinstance(ctx, dict) else ""
    except Exception:
        pass

    # Status bar state
    _current_model = agent.brain._model_override or "auto"
    _session_title = ""
    _msg_count = len(agent.brain.conversation_history) if hasattr(agent.brain, 'conversation_history') else 0
    show_status_bar(
        model=_current_model, project_type=_project_type,
        session_title=_session_title, message_count=_msg_count,
    )

    if speak:
        show_info("Voice output enabled")

    # Register CLI permission callback for destructive actions
    def _cli_confirm(tool_name: str, action: str) -> bool:
        if tool_name == "code_edit_preview":
            print(f"\n  Proposed edit:\n")
            for line in action.split("\n")[:40]:
                if line.startswith("+") and not line.startswith("+++"):
                    print(f"  \033[32m{line}\033[0m")
                elif line.startswith("-") and not line.startswith("---"):
                    print(f"  \033[31m{line}\033[0m")
                else:
                    print(f"  {line}")
            if action.count("\n") > 40:
                print(f"  ... ({action.count(chr(10)) - 40} more lines)")
        else:
            print(f"\n  \u26a0 Permission required:")
            print(f"    Tool: {tool_name}")
            print(f"    Action: {action[:200]}")
        try:
            response = input("    Allow? (y/n/always): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if response == "always":
            for word in action.lower().split()[:3]:
                if len(word) > 3:
                    agent._approved_patterns.add(word)
            return True
        return response in ("y", "yes")

    agent.set_cli_confirm_callback(_cli_confirm)

    # Initialize model picker roles from config
    update_model_roles_from_config()

    session = create_session()

    while True:
        user_input = get_input(session)

        if user_input is None:
            console.print("\n[dim]Goodbye.[/dim]\n")
            break

        # Handle Ctrl+M model picker
        if user_input == "__MODEL_PICK__":
            _current_model = agent.brain._model_override or "auto"
            choice = pick_model(console, _current_model)
            if choice:
                if choice == "auto":
                    agent.brain.set_model_override(None)
                    _current_model = "auto"
                    show_info("Model set to auto-routing")
                else:
                    agent.brain.set_model_override(choice)
                    _current_model = choice
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
            import socket, json
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                s.connect(("127.0.0.1", 19733))
                s.send((json.dumps({"type": "activity"}) + "\n").encode())
        except Exception:
            pass

        # Handle ? for help
        if user_input.strip() == "?":
            show_help()
            continue

        if user_input.startswith("/"):
            handle_command(agent, user_input, speak=speak)
            _msg_count = len(agent.brain.conversation_history) if hasattr(agent.brain, 'conversation_history') else 0
            _current_model = agent.brain._model_override or "auto"
            show_status_bar(
                model=_current_model, project_type=_project_type,
                session_title=_session_title, message_count=_msg_count,
            )
            continue

        # Run agent in thread, capture its verbose stdout, show spinner while working
        result_holder = {}
        captured_output = io.StringIO()

        def _run():
            old_stdout = sys.stdout
            sys.stdout = captured_output
            try:
                result_holder["result"] = agent.run(user_input)
            except Exception as exc:
                result_holder["error"] = str(exc)
            finally:
                sys.stdout = old_stdout

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        with show_thinking():
            thread.join()

        if "error" in result_holder:
            show_error(result_holder["error"])
            continue

        result = result_holder["result"]
        response_text = result.get("response", "")

        show_response(response_text)

        # Update status bar after response
        _msg_count = len(agent.brain.conversation_history) if hasattr(agent.brain, 'conversation_history') else 0
        _current_model = agent.brain._model_override or "auto"
        show_status_bar(
            model=_current_model, project_type=_project_type,
            session_title=_session_title, message_count=_msg_count,
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
                print(f"  - {m['content'][:100]}...")
        else:
            print("Usage: /recall <query>")
    elif cmd == "/clear":
        agent.brain.clear_history()
        print("Conversation history cleared.")
    elif cmd == "/speak" or cmd == "/say":
        if arg:
            agent._speak(arg)
            print(f"[Spoke: {arg}]")
        else:
            print("Usage: /speak <text to speak>")
    elif cmd == "/model":
        if not arg:
            # Show current model + all available
            models = Config.get_all_models()
            override = agent.brain._model_override
            print("\n  Model Configuration:")
            print(f"    Override:  {override or '(auto)'}")
            print(f"    fast:      {models.get('fast', 'N/A')}")
            print(f"    reason:    {models.get('reason', 'N/A')}")
            print(f"    code:      {models.get('code', 'N/A')}")
            print(f"    vision:    {models.get('vision', 'N/A')}")
            if models.get('reason_cloud'):
                print(f"    cloud:     {models.get('reason_cloud', 'N/A')}")
            print("\n  Usage: /model <name> | /model auto")
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
    elif cmd == "/hook":
        _handle_hook_command(agent, arg)
    elif cmd == "/sessions":
        conversations = agent.brain.list_conversations()
        if not conversations:
            print("  No sessions found.")
            return
        parts_arg = arg.split(maxsplit=1) if arg else []
        subcmd = parts_arg[0].lower() if parts_arg else "list"
        if subcmd == "switch" and len(parts_arg) > 1:
            target = parts_arg[1]
            # Try matching by index number
            try:
                idx = int(target) - 1
                if 0 <= idx < len(conversations):
                    conv = conversations[idx]
                    agent.brain.switch_conversation(conv["id"])
                    print(f"  Switched to: {conv.get('title', 'Untitled')}")
                    return
            except ValueError:
                pass
            # Try matching by ID
            for conv in conversations:
                if conv["id"] == target:
                    agent.brain.switch_conversation(conv["id"])
                    print(f"  Switched to: {conv.get('title', 'Untitled')}")
                    return
            print(f"  Session not found: {target}")
        elif subcmd == "new":
            agent.brain.new_conversation(parts_arg[1] if len(parts_arg) > 1 else None)
            print("  Started new session.")
        else:
            print("\n  Sessions:\n")
            for i, conv in enumerate(conversations[:15], 1):
                active = " *" if conv.get("is_active") else ""
                title = conv.get("title", "Untitled")[:50]
                msgs = conv.get("message_count", 0)
                print(f"    {i}. {title} ({msgs} msgs){active}")
            print(f"\n  Usage: /sessions switch <number> | /sessions new [title]")
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

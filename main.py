#!/usr/bin/env python3
"""Main entry point for the Apprentice Agent."""

import os
os.environ["TQDM_DISABLE"] = "1"

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

    if args.voice:
        run_voice_mode(agent, enable_barge_in=not args.no_barge_in)
    elif args.goal:
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
    """Interactive CLI — every input goes through the full agent loop with all tools."""
    import io
    import sys
    import threading
    from aura.cli.display import console, show_banner, show_thinking, show_response, show_error, show_info
    from aura.cli.input import create_session, get_input

    show_banner()
    if speak:
        show_info("Voice output enabled")

    session = create_session()

    while True:
        user_input = get_input(session)

        if user_input is None:
            console.print("\n[dim]Goodbye.[/dim]\n")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            handle_command(agent, user_input, speak=speak)
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
    elif cmd == "/agent":
        _handle_agent_command(agent, arg)
    elif cmd == "/hook":
        _handle_hook_command(agent, arg)
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
    """Print the agent run result."""
    print("\n" + "=" * 60)
    if is_fastpath:
        print("FAST-PATH RESPONSE COMPLETE")
    else:
        print("AGENT RUN COMPLETE")
    print("=" * 60)
    print(f"Goal: {result['goal']}")
    print(f"Completed: {result['completed']}")
    if is_fastpath:
        print(f"Mode: Fast-path (no tool execution)")
    else:
        print(f"Iterations: {result['iterations']}")
    if result.get("final_evaluation"):
        print(f"Final evaluation: {result['final_evaluation'].get('progress', 'N/A')}")


if __name__ == "__main__":
    main()

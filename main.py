#!/usr/bin/env python3
"""Main entry point for the Apprentice Agent."""

import argparse
import sys

from apprentice_agent import ApprenticeAgent
from apprentice_agent.dream import run_dream_mode


def main():
    parser = argparse.ArgumentParser(
        description="Apprentice Agent - An AI agent with memory and reasoning"
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

    args = parser.parse_args()

    # Handle dream mode first (doesn't need agent)
    if args.dream:
        result = run_dream_mode(args.dream_date)
        sys.exit(0 if result.get("success") else 1)

    agent = ApprenticeAgent()
    agent.max_iterations = args.max_iterations
    agent.use_fastpath = not args.no_fastpath

    if args.chat:
        run_chat_mode(agent)
    elif args.goal:
        result = agent.run(args.goal)
        print_result(result, is_fastpath=result.get("fast_path", False))
    else:
        parser.print_help()


def run_chat_mode(agent: ApprenticeAgent):
    """Run the agent in interactive chat mode."""
    print("Apprentice Agent - Interactive Mode")
    print("Commands: /goal <text>, /recall <query>, /clear, /quit")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            handle_command(agent, user_input)
        else:
            response = agent.chat(user_input)
            print(f"\nAgent: {response}")


def handle_command(agent: ApprenticeAgent, command: str):
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
    else:
        print(f"Unknown command: {cmd}")


def print_result(result: dict, is_fastpath: bool = False):
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

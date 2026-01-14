#!/usr/bin/env python3
"""Main entry point for the Apprentice Agent."""

import argparse
import sys

from apprentice_agent import ApprenticeAgent


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

    args = parser.parse_args()

    agent = ApprenticeAgent()
    agent.max_iterations = args.max_iterations

    if args.chat:
        run_chat_mode(agent)
    elif args.goal:
        result = agent.run(args.goal)
        print_result(result)
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


def print_result(result: dict):
    """Print the agent run result."""
    print("\n" + "=" * 60)
    print("AGENT RUN COMPLETE")
    print("=" * 60)
    print(f"Goal: {result['goal']}")
    print(f"Completed: {result['completed']}")
    print(f"Iterations: {result['iterations']}")
    if result.get("final_evaluation"):
        print(f"Final evaluation: {result['final_evaluation'].get('progress', 'N/A')}")


if __name__ == "__main__":
    main()

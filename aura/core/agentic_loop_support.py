"""Shared prompt, history, and memory helpers for the agentic loop."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

console = None
MAX_TOOL_OUTPUT_CHARS = 15000


def _ensure_console():
    """Lazily resolve the Rich console without forcing a CLI import at module load."""

    global console
    if console is None:
        try:
            from aura.cli.display import console as display_console

            console = display_console
        except ImportError:
            from rich.console import Console

            console = Console()


AGENTIC_SYSTEM_PROMPT = """You are Aura, an AI coding agent. Act, don't talk. First response must be a tool call.

BEHAVIOR: Never ask permission, present options, or narrate plans. Just execute.

HOW TO CODE WELL:
1. Read before writing — ALWAYS read_file before editing.
2. Search, don't guess — grep for definitions/usages, glob to find files by name.
3. Plan multi-file changes before starting edits.
4. Minimal edits — surgical edit_file, don't rewrite working code.
5. Test after changes — run tests via shell. If they fail, read the error and fix.
6. One thing at a time — finish one logical change before the next.
7. Check errors — if a command fails, read output and fix the root cause.
8. When asked about a specific file, ALWAYS read it with read_file — never answer from memory.

TOOLS:
- read_file: ALWAYS read before editing. Read related files for context.
- grep: Find where things are defined or used.
- glob: Find files by name pattern.
- edit_file: Surgical string-match edits. Prefer over write_file for existing files.
- write_file: New files only.
- shell: Run commands, build, test. Always check output.
- search_web/fetch_url: Look up docs when needed.

RULES:
- Never modify files outside the project without permission.
- Use exact string matches from file content when editing.
- Past session memories are below when available.

{context}

{memories}
"""


def _extract_action_summary(msg: dict) -> str | None:
    """Extract a concise action description from a conversation message."""

    role = msg.get("role", "")
    content = (msg.get("content", "") or "").strip()

    if role == "user":
        if not content or len(content) < 5:
            return None
        if content.startswith("[Auto-test result]"):
            result_text = content[len("[Auto-test result]") :].strip()
            if "PASS" in result_text.upper() or "OK" in result_text.upper():
                return "- Tests were run and passed"
            return f"- Tests were run and failed: {result_text[:80]}"
        first_line = content.split("\n", 1)[0]
        return f"- User: {first_line[:120]}"

    if role == "assistant":
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tool_summaries = []
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    func = tool_call.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", {})
                else:
                    func = getattr(tool_call, "function", None)
                    name = getattr(func, "name", "?") if func else "?"
                    args = getattr(func, "arguments", {}) if func else {}

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                if not isinstance(args, dict):
                    args = {}

                if name in ("read_file", "read"):
                    tool_summaries.append(
                        f"read {args.get('path', args.get('file_path', '?'))}",
                    )
                elif name == "edit_file":
                    tool_summaries.append(f"edited {args.get('path', '?')}")
                elif name == "write_file":
                    tool_summaries.append(f"wrote {args.get('path', '?')}")
                elif name == "run_command":
                    cmd = args.get("command", "?")
                    tool_summaries.append(f"ran `{cmd[:60]}`")
                elif name == "grep":
                    tool_summaries.append(f"searched for '{args.get('pattern', '?')}'")
                elif name == "glob":
                    tool_summaries.append(
                        f"found files matching '{args.get('pattern', '?')}'",
                    )
                elif name == "git":
                    subcommand = args.get("subcommand", args.get("command", "?"))
                    tool_summaries.append(f"git {subcommand}")
                elif name == "web_search":
                    tool_summaries.append(
                        f"searched web for '{args.get('query', '?')[:50]}'",
                    )
                else:
                    tool_summaries.append(name)
            return f"- Agent {', '.join(tool_summaries)}"

        if content:
            first_line = content.split("\n", 1)[0]
            return f"- Agent responded: {first_line[:100]}"
        return None

    if role == "tool":
        return None

    return None


def _compact_history(history: list[dict]) -> list[dict]:
    """Summarize the oldest 2/3 of messages into a single summary message."""

    if len(history) < 6:
        return history

    keep_count = max(4, len(history) // 3)
    old_msgs = history[:-keep_count]
    recent_msgs = history[-keep_count:]

    summary_lines = []
    for msg in old_msgs:
        line = _extract_action_summary(msg)
        if line:
            summary_lines.append(line)

    summary_text = (
        "(earlier conversation with no notable actions)"
        if not summary_lines
        else "\n".join(summary_lines)
    )

    n_compressed = len(old_msgs)
    summary_msg = {
        "role": "user",
        "content": (
            "[Previous conversation summary]\n"
            f"{summary_text}\n"
            f"[End summary — {n_compressed} messages compressed]"
        ),
    }

    try:
        _ensure_console()
        console.print(
            "  [dim italic]Context compacted: "
            f"{n_compressed} messages summarized into conversation summary[/]",
        )
    except Exception as exc:
        logger.debug("[AgenticLoop] Compaction console print failed: %s", exc)

    return [summary_msg, *recent_msgs]


def _truncate(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Truncate tool output to prevent context explosion."""

    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n... ({len(text) - max_chars} chars truncated) ...\n\n" + text[-half:]


def _recall_memories(prompt: str, max_results: int = 5) -> str:
    """Query UnifiedMemory for relevant context."""

    try:
        from aura.memory.unified_memory import get_unified_memory

        unified_memory = get_unified_memory()
        results = unified_memory.query(prompt, k=max_results, min_score=0.2)
        if not results:
            return ""

        lines = ["## Relevant Memories"]
        for result in results:
            content = result.content[:300] if hasattr(result, "content") else str(result)[:300]
            source = getattr(result, "source", "memory")
            score = getattr(result, "score", 0)
            lines.append(f"- [{source}, relevance={score:.2f}] {content}")
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("[AgenticLoop] Memory recall failed (non-fatal): %s", exc)
        return ""


def _store_interaction(prompt: str, response: str) -> None:
    """Store the interaction in memory for future recall."""

    try:
        from aura.memory.unified_memory import get_unified_memory

        unified_memory = get_unified_memory()
        if len(response) > 50 and len(prompt) > 10:
            unified_memory.store(
                content=f"User asked: {prompt[:200]}\nAura responded: {response[:500]}",
                source="agentic_conversation",
                importance=0.4,
                tags=["agentic", "conversation"],
            )
    except Exception as exc:
        logger.debug("[AgenticLoop] Memory store failed (non-fatal): %s", exc)

"""Core agentic loop for Aura Dev CLI.

The LLM autonomously calls tools via Ollama's structured tool calling API,
loops until the task is complete (content-only response) or limits are hit.

Features wired in:
  - Diff preview on edit_file (shows colored diff before applying)
  - Auto-test after edits (runs project tests, feeds failures back to LLM)
  - Memory recall (injects relevant memories into system prompt)
  - Typewriter response display (streams final response character by character)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.text import Text

from .tool_schemas import AGENTIC_TOOLS
from .permissions import PermissionManager
from .session import AgenticSession
from .token_manager import ContextWindowManager

logger = logging.getLogger(__name__)

console = Console()

MAX_ITERATIONS = 50
MAX_TOOL_OUTPUT_CHARS = 15000


AGENTIC_SYSTEM_PROMPT = """You are Aura, an AI coding agent with persistent memory. You help users with software engineering tasks by reading files, writing code, running commands, and iterating until the task is complete.

You REMEMBER past conversations — relevant memories are provided below when available. Use them to maintain context across sessions.

## Rules
- Read files before modifying them — understand existing code first
- Show your reasoning before making changes
- After editing code, tests will run automatically — if they fail, fix the issue
- Ask for clarification if the task is ambiguous
- Never modify files outside the project directory without explicit permission
- Use grep/glob to find relevant files instead of guessing paths
- When editing, use exact string matches from the file content

## Available tools
You have tools for: reading files, searching code (grep/glob), editing files, writing new files, running shell commands, git operations, web search, and viewing project structure.

{context}

{memories}
"""


def _truncate(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Truncate tool output to prevent context explosion."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n... ({len(text) - max_chars} chars truncated) ...\n\n" + text[-half:]


def _recall_memories(prompt: str, max_results: int = 5) -> str:
    """Query UnifiedMemory for relevant context. Returns formatted string or empty."""
    try:
        from aura.memory.unified_memory import get_unified_memory
        um = get_unified_memory()
        results = um.query(prompt, k=max_results, min_score=0.2)
        if not results:
            return ""

        lines = ["## Relevant Memories"]
        for r in results:
            content = r.content[:300] if hasattr(r, 'content') else str(r)[:300]
            source = getattr(r, 'source', 'memory')
            score = getattr(r, 'score', 0)
            lines.append(f"- [{source}, relevance={score:.2f}] {content}")
        return "\n".join(lines)
    except Exception as e:
        logger.debug(f"[AgenticLoop] Memory recall failed (non-fatal): {e}")
        return ""


def _store_interaction(prompt: str, response: str) -> None:
    """Store the interaction in memory for future recall."""
    try:
        from aura.memory.unified_memory import get_unified_memory
        um = get_unified_memory()
        # Only store substantial interactions (not greetings)
        if len(response) > 50 and len(prompt) > 10:
            um.store(
                content=f"User asked: {prompt[:200]}\nAura responded: {response[:500]}",
                source="agentic_conversation",
                importance=0.4,
                tags=["agentic", "conversation"],
            )
    except Exception as e:
        logger.debug(f"[AgenticLoop] Memory store failed (non-fatal): {e}")


def _typewriter_print(text: str, delay: float = 0.008) -> None:
    """Print text with a typewriter effect for streaming feel."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        if char in ('.', '!', '?', '\n'):
            time.sleep(delay * 3)
        elif char == ' ':
            time.sleep(delay * 0.5)
        else:
            time.sleep(delay)
    sys.stdout.write('\n')
    sys.stdout.flush()


class ToolExecutor:
    """Executes tool calls by dispatching to existing Aura tool classes."""

    def __init__(self, project_root: str, sub_agent_mgr=None):
        self.project_root = project_root
        self.sub_agent_mgr = sub_agent_mgr
        self._fs = None
        self._code_edit = None
        self._search = None
        self._shell = None
        self._git = None
        self._brave = None
        self._tavily = None

    @property
    def fs(self):
        if self._fs is None:
            from aura.tools.filesystem import FileSystemTool
            self._fs = FileSystemTool()
        return self._fs

    @property
    def code_edit(self):
        if self._code_edit is None:
            from aura.tools.code_edit import CodeEditTool
            self._code_edit = CodeEditTool()
        return self._code_edit

    @property
    def search(self):
        if self._search is None:
            from aura.tools.code_search import CodeSearchTool
            self._search = CodeSearchTool()
        return self._search

    @property
    def shell(self):
        if self._shell is None:
            from aura.tools.shell_executor import ShellExecutorTool
            self._shell = ShellExecutorTool()
        return self._shell

    @property
    def git(self):
        if self._git is None:
            from aura.tools.git_tool import GitTool
            self._git = GitTool()
        return self._git

    def _resolve_path(self, path: str) -> str:
        """Resolve relative paths against project root."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.project_root, path)

    def execute(self, tool_name: str, args: dict) -> str:
        """Execute a tool call, return result as string for the LLM."""
        try:
            result = self._dispatch(tool_name, args)
            if isinstance(result, dict):
                return _truncate(json.dumps(result, indent=2, default=str))
            return _truncate(str(result))
        except Exception as e:
            logger.error(f"[ToolExecutor] {tool_name} failed: {e}")
            return json.dumps({"error": str(e)})

    def _dispatch(self, tool_name: str, args: dict) -> dict:
        if tool_name == "read_file":
            return self._read_file(args)
        elif tool_name == "grep":
            return self.search.grep(
                pattern=args["pattern"],
                path=self._resolve_path(args.get("path", ".")),
                file_type=args.get("file_type"),
                case_insensitive=args.get("case_insensitive", False),
                context_lines=2,
                max_results=50,
            )
        elif tool_name == "glob":
            return self.search.glob(
                pattern=args["pattern"],
                path=self._resolve_path(args.get("path", ".")),
            )
        elif tool_name == "list_dir":
            path = self._resolve_path(args.get("path", "."))
            return self.fs.list_directory(path)
        elif tool_name == "edit_file":
            return self._edit_file(args)
        elif tool_name == "write_file":
            return self._write_file(args)
        elif tool_name == "shell":
            return self.shell.run(
                command=args["command"],
                cwd=args.get("cwd", self.project_root),
                timeout=min(args.get("timeout", 60), 300),
            )
        elif tool_name == "git":
            return self._git_dispatch(args)
        elif tool_name == "search_web":
            return self._web_search(args)
        elif tool_name == "project_structure":
            return self.search.project_structure(
                path=self._resolve_path(args.get("path", ".")),
                max_depth=args.get("max_depth", 3),
            )
        elif tool_name == "spawn_agent":
            if self.sub_agent_mgr is None:
                return {"error": "Sub-agents not available"}
            return self.sub_agent_mgr.spawn(
                task=args.get("task", ""),
                role=args.get("role", "reader"),
            )
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _read_file(self, args: dict) -> dict:
        path = self._resolve_path(args["path"])
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}
        except PermissionError:
            return {"error": f"Permission denied: {path}"}

        offset = args.get("offset", 0)
        limit = args.get("limit", 0)
        if limit > 0:
            selected = lines[offset:offset + limit]
        else:
            selected = lines[offset:]

        # Add line numbers
        numbered = []
        for i, line in enumerate(selected, start=offset + 1):
            numbered.append(f"{i:>5}\t{line.rstrip()}")

        return {
            "success": True,
            "path": path,
            "total_lines": len(lines),
            "showing": f"{offset + 1}-{offset + len(selected)}",
            "content": "\n".join(numbered),
        }

    def _edit_file(self, args: dict) -> dict:
        """Edit file with diff preview. Uses CodeEditTool for fuzzy matching."""
        path = self._resolve_path(args["path"])
        # Use CodeEditTool which has fuzzy matching + backup + diff
        result = self.code_edit.edit(
            path=path,
            old_string=args["old_string"],
            new_string=args["new_string"],
        )

        # Show diff in console if edit succeeded
        if result.get("success") and result.get("diff"):
            from .diff_display import show_diff
            try:
                show_diff(path, args["old_string"], args["new_string"])
            except Exception:
                pass

        return result

    def _write_file(self, args: dict) -> dict:
        path = self._resolve_path(args["path"])
        return self.fs.write_file(path=path, content=args["content"], overwrite=True)

    def _git_dispatch(self, args: dict) -> dict:
        action = args.get("action", "status")
        repo = self.project_root

        if action == "status":
            return self.git.status(repo)
        elif action == "diff":
            return self.git.diff(repo, file=args.get("file"))
        elif action == "log":
            return self.git.log(repo, count=args.get("count", 5))
        elif action == "branch":
            return self.git.branch(repo)
        elif action == "add":
            return self.git.add(repo, files=args.get("files", "."))
        elif action == "commit":
            return self.git.commit(repo, message=args.get("message", ""))
        elif action == "push":
            return self.git.push(repo)
        elif action == "pull":
            return self.git.pull(repo)
        else:
            return {"error": f"Unknown git action: {action}"}

    def _web_search(self, args: dict) -> dict:
        query = args["query"]
        max_results = args.get("max_results", 8)

        # Try Tavily first, fall back to Brave
        try:
            if self._tavily is None:
                from aura.tools.tavily_tool import TavilyTool
                self._tavily = TavilyTool()
            return self._tavily.search(query=query, max_results=max_results)
        except Exception:
            pass

        try:
            if self._brave is None:
                from aura.tools.brave_search import BraveSearchTool
                self._brave = BraveSearchTool()
            return self._brave.run(query=query, count=max_results)
        except Exception as e:
            return {"error": f"Web search unavailable: {e}"}


class AgenticLoop:
    """Core autonomous loop: LLM calls tools until task is complete."""

    def __init__(
        self,
        brain,
        project_root: str = ".",
        permissions: Optional[PermissionManager] = None,
        model_override: str = None,
        max_iterations: int = MAX_ITERATIONS,
        budget_usd: Optional[float] = None,
        context: str = "",
        session: Optional[AgenticSession] = None,
    ):
        self.brain = brain
        self.project_root = os.path.abspath(project_root)
        self.permissions = permissions or PermissionManager()
        self.model_override = model_override
        self.max_iterations = max_iterations
        self.budget_usd = budget_usd
        self.context = context

        from .sub_agent import SubAgentManager
        self._sub_agent_mgr = SubAgentManager(self)
        self.executor = ToolExecutor(self.project_root, sub_agent_mgr=self._sub_agent_mgr)
        self.iteration = 0
        self.tool_calls_total = 0
        self._edits_this_turn = 0  # Track edits for auto-test
        self._is_sub_agent = False  # Prevent sub-agent-ception

        # Context window management
        effective_model = model_override or ""
        self.context_mgr = ContextWindowManager(effective_model)

        # Session persistence
        self.session = session

        # Persistent conversation history for interactive mode
        self._conversation_history: list[dict] = []

    def _build_system_prompt(self, prompt: str) -> str:
        """Build system prompt with context and relevant memories."""
        memories = _recall_memories(prompt)
        return AGENTIC_SYSTEM_PROMPT.format(
            context=self.context or "(No project context loaded)",
            memories=memories or "(No relevant memories found)",
        )

    def run(self, prompt: str, on_tool_call=None, on_response=None) -> dict:
        """Run the agentic loop until completion.

        Args:
            prompt: User's task/prompt
            on_tool_call: Callback(tool_name, args, result) for UI updates
            on_response: Callback(text, iteration) for streaming text

        Returns:
            {success, response, iterations, tool_calls, model}
        """
        # Reset per-turn counters
        self.iteration = 0
        self.tool_calls_total = 0
        self._edits_this_turn = 0

        system_prompt = self._build_system_prompt(prompt)

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Include prior conversation turns (keep last 40 messages to avoid context explosion)
        if self._conversation_history:
            history = self._conversation_history[-40:]
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        # Track in session
        if self.session:
            self.session.append({"role": "user", "content": prompt})

        final_response = ""
        model_used = ""

        while self.iteration < self.max_iterations:
            self.iteration += 1

            # Budget check
            if self.budget_usd is not None:
                stats = self.brain.get_session_stats()
                if stats["cost_usd"] >= self.budget_usd:
                    final_response = f"Budget limit reached (${self.budget_usd:.2f}). Stopping."
                    break

            # Context window management — compact if approaching limit
            messages = self.context_mgr.check_and_compact(messages, self.brain)

            # Call LLM with tools (sub-agents may have restricted tool sets)
            active_tools = getattr(self, '_sub_agent_tools', None) or AGENTIC_TOOLS
            result = self.brain.think_with_tools(
                messages=messages,
                tools=active_tools,
                model_override=self.model_override,
            )

            if "error" in result:
                final_response = f"Error: {result['error']}"
                break

            msg = result["message"]
            model_used = result.get("model", "")

            # Extract tool calls and content.
            # Ollama returns Pydantic objects — msg.tool_calls is None or list of ToolCall,
            # msg.content is None or str. Use `or` to handle None defaults.
            tool_calls = None
            content = ""
            if isinstance(msg, dict):
                tool_calls = msg.get("tool_calls")
                content = msg.get("content", "") or ""
            else:
                tool_calls = getattr(msg, "tool_calls", None)
                content = getattr(msg, "content", "") or ""

            if not tool_calls:
                # No tool calls — LLM is done
                final_response = content or "(No response)"
                if on_response:
                    on_response(final_response, self.iteration)
                break

            # Append assistant message with tool_calls to history
            assistant_msg = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
            messages.append(assistant_msg)
            if self.session:
                self.session.append(assistant_msg)

            # Show intermediate thinking if present
            if content and on_response:
                on_response(content, self.iteration)

            # Execute each tool call
            for tc in tool_calls:
                # Handle both Pydantic ToolCall objects and plain dicts
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    args = func.get("arguments", {})
                else:
                    func = getattr(tc, "function", None)
                    tool_name = getattr(func, "name", "") if func else ""
                    args = getattr(func, "arguments", {}) if func else {}

                try:
                    if isinstance(args, str):
                        args = json.loads(args)
                    if args is None:
                        args = {}
                except (json.JSONDecodeError, TypeError):
                    args = {}

                self.tool_calls_total += 1

                # Permission check
                if not self.permissions.check(tool_name, args):
                    tool_result = json.dumps({"error": "Permission denied by user"})
                    self._show_tool_status(tool_name, args, denied=True)
                else:
                    self._show_tool_status(tool_name, args)
                    tool_result = self.executor.execute(tool_name, args)

                    # Track edits for auto-test
                    if tool_name in ("edit_file", "write_file"):
                        self._edits_this_turn += 1

                if on_tool_call:
                    on_tool_call(tool_name, args, tool_result)

                # Append tool result to messages
                tool_msg = {"role": "tool", "content": tool_result}
                messages.append(tool_msg)
                if self.session:
                    self.session.append(tool_msg)

            # Auto-test: after processing all tool calls in this iteration,
            # if any edits were made, run tests and feed results back to LLM
            if self._edits_this_turn > 0:
                test_result = self._run_auto_test()
                if test_result:
                    messages.append({
                        "role": "tool",
                        "content": test_result,
                    })

        else:
            # Max iterations reached
            final_response = f"Reached maximum iterations ({self.max_iterations}). Last response:\n{final_response}"

        # Save user message and final assistant response to conversation history
        self._conversation_history.append({"role": "user", "content": prompt})
        if final_response:
            self._conversation_history.append({"role": "assistant", "content": final_response})
            if self.session:
                self.session.append({"role": "assistant", "content": final_response})

        # Update session stats and save
        if self.session:
            self.session.update_stats(
                iterations=self.iteration,
                tool_calls=self.tool_calls_total,
            )
            self.session.save()

        # Store in persistent memory (background, non-blocking)
        try:
            _store_interaction(prompt, final_response)
        except Exception:
            pass

        return {
            "success": not final_response.startswith("Error:"),
            "response": final_response,
            "iterations": self.iteration,
            "tool_calls": self.tool_calls_total,
            "model": model_used,
        }

    def _run_auto_test(self) -> Optional[str]:
        """Run project tests after edits. Returns test output for LLM or None."""
        try:
            from aura.tools.auto_verify import auto_verify
            console.print("  [dim cyan]auto[/dim cyan] running tests...", highlight=False)
            result = auto_verify(self.project_root, self.executor.shell)

            if result.get("skipped"):
                return None  # No test runner — don't inject anything

            if result.get("success"):
                console.print("  [green]tests passed[/green]", highlight=False)
                return None  # Tests passed — no need to tell LLM

            # Tests failed — feed output back to LLM so it can fix
            output = result.get("output", "Tests failed")
            cmd = result.get("test_command", "tests")
            console.print(f"  [red]tests failed[/red] ({cmd})", highlight=False)
            return json.dumps({
                "auto_test_result": "FAILED",
                "test_command": cmd,
                "output": output[:3000],
                "instruction": "Tests failed after your edit. Please read the error output and fix the issue.",
            })
        except Exception as e:
            logger.debug(f"[AgenticLoop] Auto-test error (non-fatal): {e}")
            return None

    def get_session(self) -> Optional[AgenticSession]:
        """Get the current session object."""
        return self.session

    def load_session(self, session_id: str) -> bool:
        """Load a session and restore conversation history."""
        if not self.session:
            return False
        messages = self.session.load(session_id)
        if not messages:
            return False
        # Restore conversation history from session (skip system messages)
        self._conversation_history = [m for m in messages if m.get("role") != "system"]
        return True

    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()

    def _show_tool_status(self, tool_name: str, args: dict, denied: bool = False):
        """Show compact tool call status in the console."""
        if denied:
            console.print(f"  [red]DENIED[/red] {tool_name}", highlight=False)
            return

        # Compact description
        desc = tool_name
        if tool_name == "read_file":
            desc = f"read {args.get('path', '?')}"
        elif tool_name == "grep":
            desc = f"grep '{args.get('pattern', '?')}'"
        elif tool_name == "glob":
            desc = f"glob '{args.get('pattern', '?')}'"
        elif tool_name == "list_dir":
            desc = f"ls {args.get('path', '.')}"
        elif tool_name == "edit_file":
            desc = f"edit {args.get('path', '?')}"
        elif tool_name == "write_file":
            desc = f"write {args.get('path', '?')}"
        elif tool_name == "shell":
            cmd = args.get("command", "?")
            if len(cmd) > 60:
                cmd = cmd[:57] + "..."
            desc = f"$ {cmd}"
        elif tool_name == "git":
            desc = f"git {args.get('action', '?')}"
        elif tool_name == "search_web":
            desc = f"search '{args.get('query', '?')[:40]}'"
        elif tool_name == "project_structure":
            desc = "project structure"
        elif tool_name == "spawn_agent":
            role = args.get("role", "reader")
            task_desc = args.get("task", "?")
            if len(task_desc) > 50:
                task_desc = task_desc[:47] + "..."
            desc = f"spawn {role}: {task_desc}"

        console.print(f"  [dim cyan]tool[/dim cyan] {desc}", highlight=False)


def run_agentic(
    brain,
    prompt: str,
    project_root: str = ".",
    permissions: Optional[PermissionManager] = None,
    model_override: str = None,
    max_iterations: int = MAX_ITERATIONS,
    budget_usd: Optional[float] = None,
    context: str = "",
    trust_mode: bool = False,
) -> dict:
    """Convenience function to run a single agentic task."""
    if permissions is None:
        permissions = PermissionManager()
    if trust_mode:
        permissions.set_trust_mode(True)

    loop = AgenticLoop(
        brain=brain,
        project_root=project_root,
        permissions=permissions,
        model_override=model_override,
        max_iterations=max_iterations,
        budget_usd=budget_usd,
        context=context,
    )

    def on_response(text, iteration):
        console.print(f"\n{text}\n")

    return loop.run(prompt, on_response=on_response)

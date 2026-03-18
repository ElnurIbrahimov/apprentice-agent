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
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
_TOOL_POOL = ThreadPoolExecutor(max_workers=4)
import atexit as _atexit
_atexit.register(_TOOL_POOL.shutdown, wait=True)


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

    def __init__(self, project_root: str, sub_agent_mgr=None, permissions=None, mcp_client=None):
        self.project_root = project_root
        self.sub_agent_mgr = sub_agent_mgr
        self.permissions = permissions
        self._mcp_client = mcp_client
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

    # Aliases for when the LLM uses agent-style names instead of dev-tool names
    _TOOL_ALIASES = {
        "filesystem": "list_dir",
        "code_search": "grep",
        "code_edit": "edit_file",
        "shell_executor": "shell",
        "web_search": "search_web",
        "tavily_search": "search_web",
        "brave_search": "search_web",
        "ls": "list_dir",
        "find": "glob",
        "cat": "read_file",
    }

    def _dispatch(self, tool_name: str, args: dict) -> dict:
        # Resolve aliases first
        tool_name = self._TOOL_ALIASES.get(tool_name, tool_name)

        # If args has 'action' but no structured params, try to parse it
        if "action" in args and len(args) == 1:
            args = self._parse_action_to_args(tool_name, args["action"])

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
            return self._list_dir(path)
        elif tool_name == "edit_file":
            return self._edit_file(args)
        elif tool_name == "write_file":
            return self._write_file(args)
        elif tool_name == "shell":
            return self.shell.run_sandboxed(
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
        elif tool_name.startswith("mcp_"):
            # Route to MCP client
            if self._mcp_client:
                return {"result": self._mcp_client.call_tool(tool_name, args)}
            return {"error": f"MCP client not available for: {tool_name}"}
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def _parse_action_to_args(self, tool_name: str, action: str) -> dict:
        """Convert a freeform 'action' string into structured args for a tool.

        When the LLM uses a generic {action: "..."} schema (from ToolRAG-generated schemas),
        we need to extract the actual parameters the tool expects.
        """
        action = action.strip()

        if tool_name == "read_file":
            # Strip common prefixes: "read README.md" → "README.md"
            for prefix in ("read ", "cat ", "open "):
                if action.lower().startswith(prefix):
                    action = action[len(prefix):].strip()
                    break
            return {"path": action}
        elif tool_name == "list_dir":
            # Strip common prefixes: "list ." → ".", "ls src/" → "src/"
            for prefix in ("list ", "ls ", "dir ", "list_directory "):
                if action.lower().startswith(prefix):
                    action = action[len(prefix):].strip()
                    break
            return {"path": action or "."}
        elif tool_name == "grep":
            parts = action.split(maxsplit=1)
            if len(parts) >= 2:
                return {"pattern": parts[0], "path": parts[1]}
            return {"pattern": action, "path": "."}
        elif tool_name == "glob":
            return {"pattern": action or "*"}
        elif tool_name == "shell":
            return {"command": action}
        elif tool_name == "search_web":
            return {"query": action}
        elif tool_name == "write_file":
            # Try to parse JSON
            try:
                import json as _json
                parsed = _json.loads(action)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {"error": f"Could not parse write_file arguments from: {action[:100]}"}
        elif tool_name == "edit_file":
            try:
                import json as _json
                parsed = _json.loads(action)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return {"error": "Could not parse edit_file arguments"}
        elif tool_name == "git":
            return {"action": action}
        elif tool_name == "project_structure":
            return {"path": action or "."}

        # Default: keep as action
        return {"action": action}

    def _list_dir(self, path: str) -> dict:
        """List directory contents — direct filesystem access, no sandbox."""
        try:
            entries = []
            for entry in sorted(os.listdir(path)):
                full = os.path.join(path, entry)
                if os.path.isdir(full):
                    entries.append(f"  {entry}/")
                else:
                    try:
                        size = os.path.getsize(full)
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024 * 1024:
                            size_str = f"{size // 1024}KB"
                        else:
                            size_str = f"{size // (1024 * 1024)}MB"
                        entries.append(f"  {entry} ({size_str})")
                    except OSError:
                        entries.append(f"  {entry}")
            return {
                "success": True,
                "path": path,
                "count": len(entries),
                "entries": "\n".join(entries),
            }
        except FileNotFoundError:
            return {"error": f"Directory not found: {path}"}
        except PermissionError:
            return {"error": f"Permission denied: {path}"}

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
        """Edit file with diff preview and approval. Uses CodeEditTool for fuzzy matching."""
        path = self._resolve_path(args["path"])

        # Checkpoint: snapshot file before editing
        _cp = getattr(self, '_checkpoint_mgr', None)
        if _cp:
            try:
                _cp.snapshot(path, label=f"before edit: {Path(path).name}")
            except Exception:
                pass  # Don't fail the edit if checkpoint fails

        # Step 1: dry-run to preview
        preview = self.code_edit.edit(
            path=path,
            old_string=args["old_string"],
            new_string=args["new_string"],
            dry_run=True,
        )
        if not preview.get("success"):
            return preview

        # Step 2: show diff and get approval (skip prompt in trust mode)
        from .diff_display import show_diff_and_confirm
        if not self.permissions.trust_mode:
            approved = show_diff_and_confirm(path, args["old_string"], args["new_string"], trust_mode=False)
            if not approved:
                return {"success": False, "error": "Edit rejected by user"}
        else:
            # Trust mode: show diff briefly, auto-approve
            from .diff_display import show_diff
            try:
                show_diff(path, args["old_string"], args["new_string"])
            except Exception as e:
                logger.debug(f"[AgenticLoop] non-critical: {e}")
        # Step 3: apply for real
        result = self.code_edit.edit(
            path=path,
            old_string=args["old_string"],
            new_string=args["new_string"],
        )

        # Broadcast to Artifacts live-preview if file is previewable
        if result.get("success"):
            self._maybe_broadcast_artifact(path)

        return result

    def _write_file(self, args: dict) -> dict:
        path = self._resolve_path(args["path"])
        # Checkpoint: snapshot existing file before overwriting
        _cp = getattr(self, '_checkpoint_mgr', None)
        if _cp and os.path.exists(path):
            try:
                _cp.snapshot(path, label=f"before write: {Path(path).name}")
            except Exception:
                pass
        result = self.fs.write_file(path=path, content=args["content"], overwrite=True)

        # Broadcast to Artifacts live-preview if file is previewable
        if result.get("success"):
            self._maybe_broadcast_artifact(path, content=args.get("content"))

        return result

    def _maybe_broadcast_artifact(self, path: str, content: str = None) -> None:
        """Broadcast file content to the Artifacts live-preview panel if previewable."""
        try:
            from api.routes.artifacts import is_previewable, broadcast_artifact
            if not is_previewable(path):
                return
            # Read file content if not provided (edit_file case)
            if content is None:
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception:
                    return
            if content:
                filename = Path(path).name
                broadcast_artifact(filename, content)
                logger.debug(f"[AgenticLoop] Artifact broadcast: {filename} ({len(content)} chars)")
        except ImportError:
            pass  # API routes not available (CLI-only mode)
        except Exception as e:
            logger.debug(f"[AgenticLoop] Artifact broadcast failed (non-fatal): {e}")

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
        # Note: Tavily returns {"error": ...} dict on failure (doesn't raise), so check result
        try:
            if self._tavily is None:
                from aura.tools.tavily_tool import TavilyTool
                self._tavily = TavilyTool()
            result = self._tavily.search(query=query, max_results=max_results)
            if "error" not in result:
                return result
            logger.debug(f"[AgenticLoop] Tavily error: {result.get('error')}, trying Brave")
        except Exception as e:
            logger.debug(f"[AgenticLoop] Tavily exception: {e}")
        try:
            if self._brave is None:
                from aura.tools.brave_search import BraveSearchTool
                self._brave = BraveSearchTool()
            result = self._brave.run(query=query, count=max_results)
            if isinstance(result, dict) and "error" not in result:
                return result
            if isinstance(result, dict):
                logger.debug(f"[AgenticLoop] Brave error: {result.get('error')}, trying SearXNG")
        except Exception as e:
            logger.debug(f"[AgenticLoop] Brave exception: {e}")
        # Final fallback: SearXNG
        try:
            from aura.tools.web_search import WebSearchTool
            ws = WebSearchTool()
            return ws.search(query=query, num_results=max_results)
        except Exception as e:
            return {"error": f"All search providers failed: {e}"}


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
        aura_config: dict = None,
        router=None,
    ):
        self.brain = brain
        self.project_root = os.path.abspath(project_root)
        self.permissions = permissions or PermissionManager()
        self.model_override = model_override
        self.max_iterations = max_iterations
        self.budget_usd = budget_usd
        self.context = context
        self._router = router  # Per-step model routing

        from .sub_agent import SubAgentManager
        self._sub_agent_mgr = SubAgentManager(self)
        self.executor = ToolExecutor(self.project_root, sub_agent_mgr=self._sub_agent_mgr, permissions=self.permissions)
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

        # Incremental tracking for _pick_step_model (avoids O(n) rescan each iteration)
        self._has_edits = False
        self._has_test_failure = False
        self._last_tools_were_reads = True

        # MCP client for external tool servers
        from .mcp_client import MCPClientManager
        self._mcp_client = MCPClientManager()
        if aura_config:
            try:
                self._mcp_client.load_from_config(aura_config)
            except Exception as e:
                logger.debug(f"[AgenticLoop] MCP client init failed (non-fatal): {e}")
        self.executor._mcp_client = self._mcp_client

    def _get_active_tools(self) -> list:
        """Get all active tools including MCP tools if connected."""
        base_tools = getattr(self, '_sub_agent_tools', None) or AGENTIC_TOOLS
        if hasattr(self, '_mcp_client') and self._mcp_client.connections:
            mcp_tools = self._mcp_client.list_all_tools()
            # Convert MCP tools to Ollama tool schema format
            ollama_mcp = []
            for t in mcp_tools:
                schema = {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
                    },
                }
                ollama_mcp.append(schema)
            return list(base_tools) + ollama_mcp
        return base_tools

    def _build_system_prompt(self, prompt: str) -> str:
        """Build system prompt with context and relevant memories. Hot-reloads AURA.md if changed."""
        memories = _recall_memories(prompt)

        # Hot-reload AURA.md if it changed on disk
        aura_md_path = os.path.join(self.project_root, "AURA.md")
        try:
            current_mtime = os.path.getmtime(aura_md_path)
            if not hasattr(self, '_aura_md_mtime') or current_mtime != self._aura_md_mtime:
                from .context import gather_context
                self.context = gather_context(self.project_root)
                self._aura_md_mtime = current_mtime
        except OSError:
            pass

        return AGENTIC_SYSTEM_PROMPT.format(
            context=self.context or "(No project context loaded)",
            memories=memories or "(No relevant memories found)",
        )

    def run(self, prompt: str, on_tool_call=None, on_response=None, steering_queue=None) -> dict:
        """Run the agentic loop until completion.

        Args:
            prompt: User's task/prompt
            on_tool_call: Callback(tool_name, args, result) for UI updates
            on_response: Callback(text, iteration) for streaming text
            steering_queue: Optional SteeringQueue for mid-turn user messages

        Returns:
            {success, response, iterations, tool_calls, model}
        """
        # Reset per-turn counters
        self.iteration = 0
        self.tool_calls_total = 0
        self._edits_this_turn = 0
        self._has_edits = False
        self._has_test_failure = False
        self._last_tools_were_reads = True

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
            self._edits_this_turn = 0  # Reset per iteration to avoid redundant auto-test

            # Budget check (before any injection or model call)
            if self.budget_usd is not None:
                stats = self.brain.get_session_stats()
                if stats["cost_usd"] >= self.budget_usd:
                    final_response = f"Budget limit reached (${self.budget_usd:.2f}). Stopping."
                    break

            # Mid-turn steering: inject queued user messages (after budget check,
            # and only after the first iteration so the original prompt runs clean)
            if steering_queue and self.iteration > 1:
                injection = steering_queue.format_injection()
                if injection:
                    messages.append({"role": "user", "content": injection})

            # Context window management — compact if approaching limit
            messages = self.context_mgr.check_and_compact(messages, self.brain)

            # Call LLM with tools (sub-agents may have restricted tool sets)
            active_tools = self._get_active_tools()

            # Per-step model routing: pick best model for THIS iteration
            # Check brain's live model override (user may change via /model mid-session)
            brain_override = getattr(self.brain, '_model_override', None)
            step_model = brain_override or self.model_override
            if self._router and not step_model:
                step_model = self._pick_step_model(messages)

            # Note: ChatGPT models can't do tool calling — brain.react_step()
            # auto-falls back to default Ollama model for tool steps.

            # Try streaming first, fall back to blocking
            accumulated = ""
            tool_calls = None
            content = ""
            stream_error = None

            try:
                # Show spinner
                model_tag = f" [{step_model.split(':')[0]}]" if step_model else ""
                sys.stdout.write(f"  \033[90m● thinking{model_tag}...\033[0m")
                sys.stdout.flush()

                for chunk_type, data in self.brain.think_with_tools_stream(
                    messages=messages, tools=active_tools,
                    model_override=step_model,
                ):
                    if chunk_type == "content":
                        if not accumulated:
                            # Clear the "thinking..." spinner but keep cursor on same line
                            sys.stdout.write("\r\033[K")
                            sys.stdout.write(f"  \033[90m● generating...\033[0m")
                            sys.stdout.flush()
                        accumulated += data
                    elif chunk_type == "tool_calls":
                        if not accumulated:
                            sys.stdout.write("\r\033[K")
                        tool_calls = data
                    elif chunk_type == "done":
                        model_used = data.get("model", "")
                        # Use cleaned content from adapter (strips <tool_call> tags)
                        if data.get("content"):
                            content = data["content"]
                    elif chunk_type == "error":
                        stream_error = data.get("error", "Unknown stream error")
                        break

                if accumulated:
                    sys.stdout.write("\r\033[K")  # Clear "generating..." status
                    sys.stdout.flush()
                elif not tool_calls:
                    sys.stdout.write("\r\033[K")  # Clear spinner if no output
                    sys.stdout.flush()

            except Exception as e:
                stream_error = str(e)

            # Fallback to non-streaming if streaming failed
            if stream_error:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                logger.debug(f"[AgenticLoop] Streaming failed ({stream_error}), falling back to blocking call")
                result = self.brain.think_with_tools(
                    messages=messages, tools=active_tools,
                    model_override=step_model,
                )
                if "error" in result:
                    final_response = f"Error: {result['error']}"
                    break

                msg = result["message"]
                model_used = result.get("model", "")
                if isinstance(msg, dict):
                    tool_calls = msg.get("tool_calls")
                    content = msg.get("content", "") or ""
                else:
                    tool_calls = getattr(msg, "tool_calls", None)
                    content = getattr(msg, "content", "") or ""
            else:
                content = content or accumulated  # prefer cleaned content from "done" event

            # Strip any lingering tool XML from content
            content = re.sub(r'</?tool_call>|</?tool_result[^>]*>', '', content).strip()
            content = re.sub(r'\n{3,}', '\n\n', content)

            if not tool_calls:
                # No tool calls — LLM is done
                final_response = content or "(No response)"
                if on_response and not accumulated:
                    # Only call on_response if we didn't already stream it
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

            # Parse all tool calls first
            parsed_calls = []
            for tc in tool_calls:
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
                except (json.JSONDecodeError, TypeError) as _parse_err:
                    logger.warning(f"[AgenticLoop] Failed to parse tool args for {tool_name}: {str(args)[:200]}")
                    args = {}
                parsed_calls.append((tool_name, args))

            # Permission checks + status display on main thread
            approved = []
            for tool_name, args in parsed_calls:
                self.tool_calls_total += 1
                if not self.permissions.check(tool_name, args):
                    approved.append((tool_name, args, json.dumps({"error": "Permission denied by user"})))
                    self._show_tool_status(tool_name, args, denied=True)
                else:
                    self._show_tool_status(tool_name, args)
                    approved.append((tool_name, args, None))  # None = needs execution

            # Execute: parallel if 2+ calls, direct if 1
            needs_exec = [(i, t, a) for i, (t, a, r) in enumerate(approved) if r is None]
            if len(needs_exec) == 1:
                idx, t, a = needs_exec[0]
                result = self.executor.execute(t, a)
                approved[idx] = (t, a, result)
            elif len(needs_exec) > 1:
                futures = {}
                for idx, t, a in needs_exec:
                    fut = _TOOL_POOL.submit(self.executor.execute, t, a)
                    futures[idx] = fut
                for idx, fut in futures.items():
                    try:
                        result = fut.result(timeout=300)
                    except Exception as e:
                        result = json.dumps({"error": f"Tool execution failed: {e}"})
                    t, a, _ = approved[idx]
                    approved[idx] = (t, a, result)

            # Collect results in original order
            for tool_name, args, tool_result in approved:
                if tool_name in ("edit_file", "write_file") and not self._tool_result_has_error(tool_result):
                    self._edits_this_turn += 1
                    self._has_edits = True
                    self._last_tools_were_reads = False
                elif tool_name in ("read_file", "grep", "glob", "list_dir", "project_structure"):
                    self._last_tools_were_reads = True
                else:
                    self._last_tools_were_reads = False
                if on_tool_call:
                    on_tool_call(tool_name, args, tool_result)
                tool_msg = {"role": "tool", "content": tool_result}
                messages.append(tool_msg)
                if self.session:
                    self.session.append(tool_msg)

            # Auto-test: after processing all tool calls in this iteration,
            # if any edits were made, run tests and feed results back to LLM
            if self._edits_this_turn > 0:
                test_result = self._run_auto_test()
                if test_result:
                    self._has_test_failure = True
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
        except Exception as e:
            logger.debug(f"[AgenticLoop] non-critical: {e}")
        return {
            "success": not final_response.startswith("Error:"),
            "response": final_response,
            "iterations": self.iteration,
            "tool_calls": self.tool_calls_total,
            "model": model_used,
        }

    @staticmethod
    def _tool_result_has_error(tool_result: str) -> bool:
        """Check if a tool result indicates an error.

        Tries structured JSON parsing first, falls back to string matching.
        """
        try:
            parsed = json.loads(tool_result)
            if isinstance(parsed, dict) and "error" in parsed:
                return True
            return False
        except (json.JSONDecodeError, TypeError, ValueError):
            # Fallback: string matching for non-JSON results
            return '"error"' in tool_result

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

    def _pick_step_model(self, messages: list[dict]) -> str:
        """Pick the best model for this iteration using phase-based routing.

        Uses incrementally-tracked instance variables (_has_edits, _has_test_failure,
        _last_tools_were_reads) updated when messages are appended in the main loop,
        instead of rescanning the full message history each iteration (O(1) vs O(n)).

        Phases:
          1. understand — first 1-2 iterations, reading/searching
          2. code       — once edit_file/write_file is called, stay here
          3. fix        — after test failure, stay on code model (not reasoning)

        Switching happens at most 1-2 times per task to avoid:
          - Model thrashing (different models interpret conversation differently)
          - Breaking coding flow by inserting a reasoning model mid-implementation
        """
        from .router import classify_task

        # Phase 1: First iteration — classify from user prompt
        if self.iteration == 1:
            user_prompt = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_prompt = msg.get("content", "")
            category = classify_task(user_prompt) if user_prompt else "orchestrator"

        # Phase 3: Test failure — stay on code model to fix
        elif self._has_test_failure and self._has_edits:
            category = "code_gen"

        # Phase 2: Coding phase — once edits start, stay on code_gen
        elif self._has_edits:
            category = "code_gen"

        # Still exploring/reading — use orchestrator (reliable, low hallucination)
        elif self._last_tools_were_reads:
            category = "orchestrator"

        else:
            category = "orchestrator"

        model = self._router.select(category)
        logger.debug(f"[AgenticLoop] Step {self.iteration} phase={category} -> {model}")
        return model

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
        elif tool_name.startswith("mcp_"):
            desc = f"mcp {tool_name[4:]}"

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
    aura_config: dict = None,
    router=None,
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
        aura_config=aura_config,
        router=router,
    )

    def on_response(text, iteration):
        console.print(f"\n{text}\n")

    return loop.run(prompt, on_response=on_response)

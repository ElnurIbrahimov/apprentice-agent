"""Core agentic loop for Aura Dev CLI.

The LLM autonomously calls tools via Ollama's structured tool calling API,
loops until the task is complete (content-only response) or limits are hit.

Features wired in:
  - Diff preview on edit_file (shows colored diff before applying)
  - Auto-test after edits (runs project tests, feeds failures back to LLM)
  - Memory recall (injects relevant memories into system prompt)
"""

import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from rich.text import Text

from .tool_schemas import AGENTIC_TOOLS
from .permissions import PermissionManager
from .session import AgenticSession
from .token_manager import ContextWindowManager

logger = logging.getLogger(__name__)

# Lazy import — set on first use to decouple core from CLI
console = None
def _ensure_console():
    global console
    if console is None:
        try:
            from aura.cli.display import console as _c
            console = _c
        except ImportError:
            from rich.console import Console
            console = Console()

MAX_ITERATIONS = 50
MAX_TOOL_OUTPUT_CHARS = 15000
from aura.pools import tool_pool as _tool_pool_fn

def _get_tool_pool():
    """Lazy accessor — pool created on first parallel tool call, not at import."""
    return _tool_pool_fn()


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
    """Extract a concise action description from a conversation message.

    Returns a short summary line like:
      - User asked to fix the auth bug in login.py
      - Agent read login.py, found the issue on line 42
      - Agent edited login.py to fix the token refresh logic
    Returns None for messages that don't contain useful info (empty, tool noise).
    """
    role = msg.get("role", "")
    content = msg.get("content", "") or ""
    content = content.strip()

    if role == "user":
        if not content or len(content) < 5:
            return None
        # Strip [Auto-test result] prefix for cleaner summaries
        if content.startswith("[Auto-test result]"):
            result_text = content[len("[Auto-test result]"):].strip()
            if "PASS" in result_text.upper() or "OK" in result_text.upper():
                return "- Tests were run and passed"
            return f"- Tests were run and failed: {result_text[:80]}"
        # Summarize user request
        first_line = content.split("\n", 1)[0]
        return f"- User: {first_line[:120]}"

    if role == "assistant":
        # Check for tool_calls first — they describe what the agent DID
        tc = msg.get("tool_calls")
        if tc:
            tool_summaries = []
            for t in tc:
                if isinstance(t, dict):
                    func = t.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", {})
                else:
                    func = getattr(t, "function", None)
                    name = getattr(func, "name", "?") if func else "?"
                    args = getattr(func, "arguments", {}) if func else {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                # Build human-readable tool description
                if name in ("read_file", "read"):
                    tool_summaries.append(f"read {args.get('path', args.get('file_path', '?'))}")
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
                    tool_summaries.append(f"found files matching '{args.get('pattern', '?')}'")
                elif name == "git":
                    sub = args.get("subcommand", args.get("command", "?"))
                    tool_summaries.append(f"git {sub}")
                elif name == "web_search":
                    tool_summaries.append(f"searched web for '{args.get('query', '?')[:50]}'")
                else:
                    tool_summaries.append(name)
            return f"- Agent {', '.join(tool_summaries)}"
        # Plain text response
        if content:
            first_line = content.split("\n", 1)[0]
            return f"- Agent responded: {first_line[:100]}"
        return None

    if role == "tool":
        # Skip tool results — they're bulky and the tool_call already captures the action
        return None

    return None


def _compact_history(history: list[dict]) -> list[dict]:
    """Summarize the oldest 2/3 of messages into a single summary message.

    Instead of just dropping old messages (losing all context), this builds
    a compact programmatic summary of what happened, so the LLM retains
    awareness of earlier conversation.
    """
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

    if not summary_lines:
        summary_text = "(earlier conversation with no notable actions)"
    else:
        summary_text = "\n".join(summary_lines)

    n_compressed = len(old_msgs)
    summary_msg = {
        "role": "user",
        "content": (
            f"[Previous conversation summary]\n"
            f"{summary_text}\n"
            f"[End summary — {n_compressed} messages compressed]"
        ),
    }

    try:
        _ensure_console()
        console.print(
            f"  [dim italic]Context compacted: {n_compressed} messages summarized into conversation summary[/]"
        )
    except Exception as e:
        logger.debug(f"[AgenticLoop] Compaction console print failed: {e}")

    return [summary_msg] + recent_msgs


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
            # Disable sandbox since ToolExecutor._resolve_path handles path containment
            self._fs = FileSystemTool(sandbox_enabled=False)
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
        """Resolve relative paths against project root with containment check."""
        if os.path.isabs(path):
            resolved = os.path.realpath(path)
        else:
            resolved = os.path.realpath(os.path.join(self.project_root, path))

        # LLM often prefixes paths with the project folder name (e.g., "Aura/main.py"
        # when project root is already D:/Aura). If resolved doesn't exist, try stripping.
        if not os.path.exists(resolved) and not os.path.isabs(path):
            project_name = os.path.basename(self.project_root)
            if path.startswith(project_name + "/") or path.startswith(project_name + "\\"):
                stripped = path[len(project_name) + 1:]
                alt = os.path.realpath(os.path.join(self.project_root, stripped))
                if os.path.exists(alt):
                    resolved = alt

        # Allow project root and specific safe subdirs under home
        allowed_roots = [os.path.realpath(self.project_root)]
        home = os.path.realpath(os.path.expanduser("~"))
        if home:
            # Only allow Aura's own data directories, not the entire home
            for subdir in [".aura", "Desktop"]:
                candidate = os.path.join(home, subdir)
                if os.path.isdir(candidate):
                    allowed_roots.append(os.path.realpath(candidate))
        # Block sensitive directories even if they're under an allowed root
        _SENSITIVE_DIRS = {".ssh", ".gnupg", ".aws", ".azure", ".kube",
                           ".docker", ".config/gcloud", "AppData/Roaming/1Password"}
        for sensitive in _SENSITIVE_DIRS:
            sensitive_path = os.path.realpath(os.path.join(home, sensitive))
            if resolved.startswith(sensitive_path + os.sep) or resolved == sensitive_path:
                raise PermissionError(f"Access to sensitive directory blocked: {path}")
        for root in allowed_roots:
            if resolved.startswith(root + os.sep) or resolved == root:
                return resolved
        raise PermissionError(f"Path outside allowed directories: {path}")

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
            if "error" in args:
                return args  # Surface the parse error directly to the LLM

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
            cwd = self._resolve_path(args.get("cwd", "."))
            result = self.shell.run_sandboxed(
                command=args["command"],
                cwd=cwd,
                timeout=min(args.get("timeout", 60), 300),
            )
            return self._enrich_shell_error(result, args.get("command", ""))
        elif tool_name == "git":
            return self._git_dispatch(args)
        elif tool_name == "search_web":
            return self._web_search(args)
        elif tool_name == "project_structure":
            return self.search.project_structure(
                path=self._resolve_path(args.get("path", ".")),
                max_depth=args.get("max_depth", 3),
            )
        elif tool_name == "fetch_url":
            return self._fetch_url(args)
        elif tool_name == "create_directory":
            path = self._resolve_path(args.get("path", ""))
            os.makedirs(path, exist_ok=True)
            return {"success": True, "path": path}
        elif tool_name == "move_file":
            src = self._resolve_path(args.get("source", ""))
            dst = self._resolve_path(args.get("destination", ""))
            import shutil
            shutil.move(src, dst)
            return {"success": True, "source": src, "destination": dst}
        elif tool_name == "multi_edit":
            return self.code_edit.multi_edit(
                path=self._resolve_path(args["path"]),
                edits=args.get("edits", []),
            )
        elif tool_name == "run_tests":
            return self._run_tests(args)
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
            except Exception as e:
                logger.debug(f"[AgenticLoop] write_file JSON parse failed: {e}")
            return {"error": f"Could not parse write_file arguments from: {action[:100]}"}
        elif tool_name == "edit_file":
            try:
                import json as _json
                parsed = _json.loads(action)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as e:
                logger.debug(f"[AgenticLoop] edit_file JSON parse failed: {e}")
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
            return self._read_file_not_found(path, args.get("path", ""))
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

    def _read_file_not_found(self, resolved_path: str, original_path: str) -> dict:
        """Enrich a file-not-found error with suggestions from glob search."""
        basename = os.path.basename(original_path or resolved_path)
        if not basename:
            return {"error": f"File not found: {resolved_path}"}

        suggestions = []
        try:
            # Search for files with the same name anywhere in project
            result = self.search.glob(
                pattern=f"**/{basename}",
                path=self.project_root,
                max_results=5,
            )
            if result.get("success") and result.get("files"):
                suggestions = [f["path"] for f in result["files"][:5]]
        except Exception as e:
            logger.debug(f"[AgenticLoop] File suggestion glob failed: {e}")

        if not suggestions:
            # Try partial match — strip extension and search broader
            stem = os.path.splitext(basename)[0]
            if len(stem) >= 3:
                try:
                    result = self.search.glob(
                        pattern=f"**/{stem}*",
                        path=self.project_root,
                        max_results=5,
                    )
                    if result.get("success") and result.get("files"):
                        suggestions = [f["path"] for f in result["files"][:5]]
                except Exception as e:
                    logger.debug(f"[AgenticLoop] Partial file suggestion failed: {e}")

        error_msg = f"File not found: {resolved_path}"
        if suggestions:
            return {"error": error_msg, "did_you_mean": suggestions}
        return {"error": error_msg}

    # Common command alternatives for "command not found" enrichment
    _COMMAND_ALTERNATIVES = {
        "python": ["python3", "py"],
        "python3": ["python", "py"],
        "py": ["python", "python3"],
        "pip": ["pip3", "python -m pip"],
        "pip3": ["pip", "python3 -m pip"],
        "npx": ["npm exec", "pnpm exec"],
        "npm": ["pnpm", "yarn"],
        "pnpm": ["npm", "yarn"],
        "yarn": ["npm", "pnpm"],
        "node": ["nodejs"],
        "nodejs": ["node"],
        "make": ["cmake", "nmake"],
        "gcc": ["cc", "clang"],
        "g++": ["c++", "clang++"],
        "curl": ["wget", "Invoke-WebRequest"],
        "wget": ["curl"],
        "cat": ["type", "Get-Content"],
        "ls": ["dir", "Get-ChildItem"],
        "rm": ["del", "Remove-Item"],
        "cp": ["copy", "Copy-Item"],
        "mv": ["move", "Move-Item"],
        "grep": ["findstr", "Select-String"],
    }

    def _enrich_shell_error(self, result: dict, command: str) -> dict:
        """Enrich shell errors with actionable suggestions."""
        if result.get("success", False):
            return result

        stderr = result.get("stderr", "") or result.get("error", "") or ""
        # Detect "command not found" patterns (bash, PowerShell, cmd)
        not_found_patterns = [
            "command not found",
            "not recognized as an internal or external command",
            "is not recognized as",
            "No such file or directory",
            "The term",  # PowerShell: "The term 'X' is not recognized"
        ]

        is_cmd_not_found = any(p.lower() in stderr.lower() for p in not_found_patterns)
        if not is_cmd_not_found:
            return result

        # Extract the failed command name (first token)
        cmd_name = command.strip().split()[0] if command.strip() else ""
        # Strip path prefixes if present
        cmd_name = os.path.basename(cmd_name)

        alternatives = self._COMMAND_ALTERNATIVES.get(cmd_name, [])
        if alternatives:
            result["suggestion"] = f"'{cmd_name}' not found. Try: {', '.join(alternatives)}"

        return result

    def _edit_file(self, args: dict) -> dict:
        """Edit file with diff preview and approval. Uses CodeEditTool for fuzzy matching."""
        path = self._resolve_path(args["path"])

        # Checkpoint: snapshot file before editing
        _cp = getattr(self, '_checkpoint_mgr', None)
        if _cp:
            try:
                _cp.snapshot(path, label=f"before edit: {Path(path).name}")
            except Exception as e:
                logger.debug(f"[AgenticLoop] Edit checkpoint failed (non-fatal): {e}")

        # Step 1: dry-run to preview
        preview = self.code_edit.edit(
            path=path,
            old_string=args["old_string"],
            new_string=args["new_string"],
            dry_run=True,
        )
        if not preview.get("success"):
            return preview

        # Step 2: show diff — auto-approve if edit_file is AUTO in permissions (like Claude Code)
        # The PermissionManager already approved this tool at the outer gate (line 988).
        # No second prompt needed — just show the diff for visibility.
        from .diff_display import show_diff
        try:
            show_diff(path, args["old_string"], args["new_string"])
        except Exception as e:
            logger.debug(f"[AgenticLoop] non-critical: {e}")

        # H2: Interactive accept/reject for careful mode
        if not getattr(self, '_trust_all_edits', False):
            perm_mode = getattr(self.permissions, 'mode', 'auto_edit') if self.permissions else 'auto_edit'
            if perm_mode == 'careful':
                try:
                    import sys
                    sys.stderr.write("  Apply this edit? [y/n/all]: ")
                    sys.stderr.flush()
                    choice = input().strip().lower()
                    if choice == "all":
                        self._trust_all_edits = True
                    elif choice not in ("y", "yes", ""):
                        return {"success": False, "skipped": True, "message": "Edit rejected by user"}
                except (EOFError, KeyboardInterrupt):
                    return {"success": False, "skipped": True, "message": "Edit cancelled"}

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
            except Exception as e:
                logger.debug(f"[AgenticLoop] Write checkpoint failed (non-fatal): {e}")
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
                except Exception as e:
                    logger.debug(f"[AgenticLoop] Artifact file read failed: {e}")
                    return
            if content:
                try:
                    filename = os.path.relpath(path, self.project_root)
                except Exception as e:
                    logger.debug(f"[AgenticLoop] relpath failed, using basename: {e}")
                    filename = Path(path).name
                filename = filename.replace("\\", "/")
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

        # Shared fallback chain: Tavily → Brave → Firecrawl
        from aura.tools.search_fallback import web_search_with_fallback
        return web_search_with_fallback(query=query, max_results=max_results)

    def _fetch_url(self, args: dict) -> dict:
        """Fetch a URL and return stripped text content."""
        url = args.get("url", "")
        if not url:
            return {"error": "No URL provided"}
        # SSRF protection: block internal/private network requests
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return {"error": f"Only http/https URLs allowed, got: {parsed.scheme}"}
            host = (parsed.hostname or "").lower()
            if not host:
                return {"error": "No hostname in URL"}
            _blocked = {"localhost", "127.0.0.1", "::1", "0.0.0.0",
                        "169.254.169.254", "metadata.google.internal"}
            if (host in _blocked or host.startswith("10.")
                    or host.startswith("192.168.")
                    or host.startswith("172.16.") or host.startswith("172.17.")
                    or host.startswith("172.18.") or host.startswith("172.19.")
                    or host.startswith("172.2") or host.startswith("172.3")
                    or host.endswith(".internal") or host.endswith(".local")):
                return {"error": f"Requests to internal/private addresses blocked: {host}"}
        except Exception as e:
            return {"error": f"URL validation failed: {e}"}
        try:
            import requests
            with requests.get(url, timeout=15, headers={"User-Agent": "Aura-Dev-Agent/1.0"}, stream=False) as resp:
                resp.raise_for_status()
                content = resp.text
        except Exception as e:
            return {"error": f"Failed to fetch {url}: {e}"}
        # Strip HTML tags
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            # Remove script/style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        except ImportError:
            # Fallback: simple regex strip
            text = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        # Truncate to 15000 chars
        if len(text) > 15000:
            text = text[:15000] + "\n... [truncated]"
        return {"success": True, "url": url, "length": len(text), "content": text}

    def _run_tests(self, args: dict) -> dict:
        """Run project tests, auto-detecting the test framework."""
        target = args.get("target", "")
        cwd = self.project_root

        # Try auto_verify if available
        try:
            from aura.tools.auto_verify import AutoVerifyTool
            av = AutoVerifyTool()
            result = av.run(project_root=cwd, target=target)
            if result and "error" not in result:
                return result
        except (ImportError, Exception) as e:
            logger.debug(f"[ToolExecutor] auto_verify unavailable: {e}")

        # Detect test framework and run
        cmd = None
        if os.path.exists(os.path.join(cwd, "pytest.ini")) or os.path.exists(os.path.join(cwd, "pyproject.toml")) or os.path.exists(os.path.join(cwd, "setup.py")):
            cmd = f"python -m pytest {target} -x -q --tb=short" if target else "python -m pytest -x -q --tb=short"
        elif os.path.exists(os.path.join(cwd, "package.json")):
            # Check for vitest or jest
            try:
                with open(os.path.join(cwd, "package.json"), "r") as f:
                    import json as _json
                    pkg = _json.load(f)
                deps = {**pkg.get("devDependencies", {}), **pkg.get("dependencies", {})}
                if "vitest" in deps:
                    cmd = f"npx vitest run {target}" if target else "npx vitest run"
                else:
                    cmd = f"npx jest {target}" if target else "npx jest"
            except Exception as e:
                logger.debug(f"[AgenticLoop] package.json test detection failed: {e}")
                cmd = f"npm test -- {target}" if target else "npm test"
        elif os.path.exists(os.path.join(cwd, "Cargo.toml")):
            cmd = f"cargo test {target}" if target else "cargo test"
        elif os.path.exists(os.path.join(cwd, "go.mod")):
            cmd = f"go test {target}" if target else "go test ./..."

        if not cmd:
            return {"error": "Could not detect test framework. Use shell tool to run tests manually."}

        return self.shell.run_sandboxed(command=cmd, cwd=cwd, timeout=120)


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

        # Action mode detected by intent classifier (set per run())
        self._current_action_mode = None

        # Budget warning flags (C1) and per-turn cost tracking (C2)
        self._budget_warned_50 = False
        self._budget_warned_80 = False
        self.last_turn_cost = 0.0

        # Hot files: recently read/edited file paths and content snapshots
        # Persists across iterations within a task so the LLM remembers what it was working on
        self._hot_files: list[str] = []  # recently touched file paths (max 10)
        self._hot_file_contents: dict[str, str] = {}  # path -> first 200 lines snapshot

        # Cancellation event for mid-loop abort (Ctrl+C)
        self._cancel_event = threading.Event()

        # Completion verification: one extra LLM call to check task was actually done
        self._verify_completion = True

        # Adaptive planner (enhancement — non-fatal if unavailable)
        self._planner = None
        try:
            from .adaptive_planner import AdaptivePlanner
            self._planner = AdaptivePlanner(brain=brain)
        except Exception as e:
            logger.debug(f"[AgenticLoop] AdaptivePlanner init failed (non-fatal): {e}")

        # MCP client for external tool servers
        from .mcp_client import MCPClientManager
        self._mcp_client = MCPClientManager()
        if aura_config:
            try:
                self._mcp_client.load_from_config(aura_config)
            except Exception as e:
                logger.debug(f"[AgenticLoop] MCP client init failed (non-fatal): {e}")
        self.executor._mcp_client = self._mcp_client

        # H1: Cost estimation toggle
        self._show_cost_estimates_enabled = False
        if aura_config:
            self._show_cost_estimates_enabled = aura_config.get("cost_estimate", False)

        # H2: Diff preview accept/reject toggle
        self._trust_all_edits = False

        # H5: Response cache
        self._response_cache = None
        try:
            from .response_cache import ResponseCache
            self._response_cache = ResponseCache()
        except Exception:
            pass

    def __del__(self):
        """Clean up MCP connections to prevent process leaks."""
        try:
            if hasattr(self, '_mcp_client') and self._mcp_client:
                self._mcp_client.disconnect_all()
        except Exception as e:
            # Can't use logger in __del__ reliably, but at least bind the exception
            pass  # MCP cleanup during GC — best-effort

    def cancel(self):
        """Signal the loop to stop after the current LLM/tool call finishes."""
        self._cancel_event.set()

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

        system_prompt = AGENTIC_SYSTEM_PROMPT.format(
            context=self.context or "(No project context loaded)",
            memories=memories or "(No relevant memories found)",
        )

        # Inject design system for frontend tasks
        try:
            from aura.prompts.design_system import DESIGN_SYSTEM_PROMPT, DESIGN_SYSTEM_MODES
            if self._current_action_mode in DESIGN_SYSTEM_MODES:
                system_prompt += "\n\n" + DESIGN_SYSTEM_PROMPT
        except ImportError:
            pass

        # Inject semantic codebase context (same as brain.py path)
        try:
            from aura.tools.codebase_index import CodebaseIndex
            from pathlib import Path as _P
            _idx_db = _P("data/codebase_index/index.db")
            _idx_legacy = _P(self.project_root) / ".aura" / "index.db"
            if (_idx_db.exists() or _idx_legacy.exists()) and len(system_prompt) < 22000:
                idx = CodebaseIndex(self.project_root)
                if idx.stats()["total_chunks"] > 0:
                    relevant = idx.search(prompt, top_k=5)
                    if relevant:
                        chunks = "\n".join(
                            f"[{r['file']}:{r.get('start_line','')}] {r['content'][:500]}"
                            for r in relevant
                        )
                        system_prompt += f"\n\n## Relevant codebase context\n{chunks}"
        except Exception as e:
            logger.debug(f"[AgenticLoop] Codebase context injection failed: {e}")

        # Pre-load files mentioned in the prompt
        try:
            import re as _re_files
            file_patterns = _re_files.findall(
                r'(?:^|\s)([a-zA-Z_][\w/\\.-]*\.(?:py|js|ts|jsx|tsx|rs|go|java|c|cpp|h|rb|php|swift|kt|sql|yaml|yml|json|toml|md))\b',
                prompt
            )
            if file_patterns:
                file_contents = []
                for fp in file_patterns[:3]:  # max 3 files
                    resolved = self.executor._resolve_path(fp)
                    if os.path.isfile(resolved) and os.path.getsize(resolved) < 50000:
                        with open(resolved, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read(8000)
                        file_contents.append(f"### {fp}\n```\n{content}\n```")
                if file_contents and len(system_prompt) < 22000:
                    system_prompt += "\n\n## Pre-loaded files from prompt\n" + "\n".join(file_contents)
        except Exception as e:
            logger.debug(f"[AgenticLoop] File pre-load failed: {e}")

        # Adaptive plan context: inject current plan so LLM knows what to do next
        try:
            if self._planner and self._planner.current_plan:
                plan_ctx = self._planner.current_plan.to_prompt_context()
                if plan_ctx:
                    system_prompt += "\n\n" + plan_ctx
        except Exception as e:
            logger.debug(f"[AgenticLoop] Plan context injection failed: {e}")

        # Hot files: inject recently touched file paths so LLM remembers
        # what it was working on across iterations and after context compaction
        if self._hot_files:
            hot_lines = ["## Recently touched files"]
            for fp in self._hot_files:
                name = Path(fp).name
                rel = os.path.relpath(fp, self.project_root)
                has_snapshot = "(content cached)" if fp in self._hot_file_contents else ""
                hot_lines.append(f"- {rel} {has_snapshot}")
            system_prompt += "\n\n" + "\n".join(hot_lines)

        return system_prompt

    def _track_hot_file(self, tool_name: str, args: dict, tool_result: str) -> None:
        """Track recently touched files for context injection.

        Called after each tool execution for read_file, edit_file, write_file.
        Keeps _hot_files deduplicated, most-recent-first, max 10 entries.
        For read_file, also snapshots the first 200 lines into _hot_file_contents.
        """
        resolved_name = self.executor._TOOL_ALIASES.get(tool_name, tool_name)
        if resolved_name not in ("read_file", "edit_file", "write_file"):
            return

        path = args.get("path")
        if not path:
            return

        # Resolve to absolute path for consistency
        try:
            path = self.executor._resolve_path(path)
        except (PermissionError, Exception):
            return

        # Deduplicate: remove if already present, then prepend (most recent first)
        if path in self._hot_files:
            self._hot_files.remove(path)
        self._hot_files.insert(0, path)

        # Cap at 10
        if len(self._hot_files) > 10:
            evicted = self._hot_files.pop()
            self._hot_file_contents.pop(evicted, None)

        # Snapshot first 200 lines on read_file (or edit/write if we can)
        if resolved_name == "read_file":
            try:
                # Extract content from the tool result if available
                parsed = json.loads(tool_result) if isinstance(tool_result, str) else tool_result
                if isinstance(parsed, dict) and parsed.get("content"):
                    # Take first 200 lines from the numbered content
                    lines = parsed["content"].split("\n")[:200]
                    self._hot_file_contents[path] = "\n".join(lines)
                    return
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            # Fallback: read from disk
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= 200:
                            break
                        lines.append(line.rstrip())
                    self._hot_file_contents[path] = "\n".join(lines)
            except Exception as e:
                logger.debug(f"[AgenticLoop] Hot file read failed for {path}: {e}")

    def _estimate_cost(self, prompt: str) -> dict:
        """Estimate cost based on task category and historical data."""
        try:
            from .router import classify_task
            category, confidence = classify_task(prompt)

            # Historical averages per category (fallback defaults)
            _AVG_ITERS = {"code_gen": 5, "small_edit": 2, "reasoning": 1,
                          "frontend": 4, "tool_dispatch": 3, "orchestrator": 3}
            _COST_PER_ITER = 0.005  # ~$0.005 per iteration average

            est_iters = _AVG_ITERS.get(category, 3)
            # Check router outcome stats for better estimate
            if hasattr(self, '_router') and self._router:
                stats = getattr(self._router, '_outcome_stats', {})
                for key, data in stats.items():
                    if key[0] == category and data.get("count", 0) >= 3:
                        est_iters = data["total_iters"] / data["count"]
                        break

            est_cost = est_iters * _COST_PER_ITER
            return {"estimated_cost": est_cost, "estimated_iterations": int(est_iters), "category": category}
        except Exception:
            return {"estimated_cost": 0, "estimated_iterations": 0, "category": "unknown"}

    def _inject_smart_context(self, prompt: str, system_prompt: str) -> str:
        """Embed prompt and find most relevant project files to inject."""
        try:
            import ollama
            import numpy as np

            resp = ollama.embed(model="nomic-embed-text:latest", input=prompt)
            if not resp or "embeddings" not in resp or not resp["embeddings"]:
                return system_prompt
            prompt_vec = np.array(resp["embeddings"][0])

            # Score candidate files
            import glob as _glob
            candidates = []
            for ext in ("*.py", "*.js", "*.ts", "*.go", "*.rs", "*.java"):
                candidates.extend(_glob.glob(os.path.join(self.project_root, "**", ext), recursive=True))
            candidates = [f for f in candidates[:50] if os.path.getsize(f) < 50000]

            # Remove files already in hot_files
            hot = set(getattr(self, '_hot_files', {}).keys())
            candidates = [f for f in candidates if f not in hot][:30]

            if not candidates:
                return system_prompt

            scored = []
            for fpath in candidates:
                rel = os.path.relpath(fpath, self.project_root)
                try:
                    file_resp = ollama.embed(model="nomic-embed-text:latest", input=rel)
                    if file_resp and file_resp.get("embeddings"):
                        file_vec = np.array(file_resp["embeddings"][0])
                        sim = float(np.dot(prompt_vec, file_vec) / (np.linalg.norm(prompt_vec) * np.linalg.norm(file_vec) + 1e-8))
                        scored.append((fpath, sim))
                except Exception:
                    continue

            scored.sort(key=lambda x: -x[1])
            top = scored[:3]

            if not top or top[0][1] < 0.3:
                return system_prompt

            parts = []
            for fpath, score in top:
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read(8000)
                    rel = os.path.relpath(fpath, self.project_root)
                    parts.append(f"### {rel} (relevance={score:.2f})\n```\n{content}\n```")
                except Exception:
                    continue

            if parts and len(system_prompt) < 25000:
                system_prompt += "\n\n## Auto-detected relevant files\n" + "\n".join(parts)

            return system_prompt
        except (ImportError, Exception):
            return system_prompt

    def plan_first(self, prompt: str) -> dict:
        """Generate a plan without executing anything. Returns plan dict.

        Uses the LLM to create a step-by-step plan for the given prompt.
        The plan can be displayed to the user for approval before execution.

        Returns:
            {"plan_text": str, "plan": ExecutionPlan, "prompt": str}
        """
        from aura.core.planner import (
            PLAN_GENERATION_PROMPT, parse_plan_from_llm,
        )

        plan_prompt = PLAN_GENERATION_PROMPT.format(task=prompt)
        system_prompt = self._build_system_prompt(prompt)

        try:
            response = self.brain.think(
                plan_prompt,
                system_prompt=system_prompt,
                use_history=False,
            )
            if isinstance(response, dict):
                response = response.get("response", response.get("content", str(response)))
        except (ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            return {"plan_text": "", "plan": None, "prompt": prompt, "error": str(e)}

        plan = parse_plan_from_llm(response)
        return {"plan_text": response, "plan": plan, "prompt": prompt}

    def run(self, prompt: str, on_tool_call=None, on_response=None, steering_queue=None, on_chunk=None, on_tool_start=None) -> dict:
        """Run the agentic loop until completion.

        Args:
            prompt: User's task/prompt
            on_tool_call: Callback(tool_name, args, result) for UI updates
            on_response: Callback(text, iteration) for streaming text
            steering_queue: Optional SteeringQueue for mid-turn user messages
            on_chunk: Callback(text) for live token streaming
            on_tool_start: Callback(tool_name, args) fired before tool execution

        Returns:
            {success, response, iterations, tool_calls, model}
        """
        # Ensure console is initialized for display output
        _ensure_console()

        # Reset per-turn counters
        self.iteration = 0
        self.tool_calls_total = 0
        self._edits_this_turn = 0
        self._has_edits = False
        self._has_test_failure = False
        self._last_tools_were_reads = True
        self._loop_error = False
        self._verification_done = False  # Only verify once per run

        # Capture baseline cost for per-turn cost tracking (C2)
        _prev_cost = 0.0
        try:
            _prev_cost = self.brain.get_session_stats().get("cost_usd", 0.0)
        except Exception:
            pass

        # Reset cancellation for this run
        self._cancel_event.clear()

        # Wire in loop guard to prevent infinite tool cycles
        from aura.reliability.loop_guard import get_guard
        _session_id = self.session.session_id if self.session else "default"
        guard = get_guard(_session_id)
        guard.reset()

        # ── Intent classification & model routing (same as web UI) ──
        self._current_action_mode = None
        _original_model_override = self.model_override  # preserve user-set override
        try:
            from api.services.agent_service import detect_action_mode, get_model_for_action
            action_mode = detect_action_mode(prompt)
            if action_mode:
                self._current_action_mode = action_mode
                logger.info(f"[AgenticLoop] Detected action mode: {action_mode}")
                # Apply model routing only if user hasn't set a manual override
                if not self.model_override:
                    routed_model = get_model_for_action(action_mode)
                    if routed_model:
                        self.model_override = routed_model
                        logger.info(f"[AgenticLoop] Model routed to: {routed_model}")
        except Exception as e:
            logger.debug(f"[AgenticLoop] Intent classification failed (non-fatal): {e}")

        # ── Adaptive planning: classify task and generate plan if complex ──
        try:
            if self._planner:
                self._planner.reset()
                is_complex = self._planner.classify(prompt)
                if is_complex:
                    plan = self._planner.generate_plan(prompt)
                    if plan:
                        logger.info(f"[AgenticLoop] Plan generated: {len(plan.steps)} steps")
                        try:
                            _ensure_console()
                            console.print(
                                f"  [dim cyan]plan[/dim cyan] {len(plan.steps)} steps generated",
                                highlight=False,
                            )
                        except Exception as e:
                            logger.debug(f"[AgenticLoop] Plan console print failed: {e}")
        except Exception as e:
            logger.debug(f"[AgenticLoop] Adaptive planning failed (non-fatal): {e}")

        system_prompt = self._build_system_prompt(prompt)

        # H4: Smart context injection via embeddings
        system_prompt = self._inject_smart_context(prompt, system_prompt)

        # H5: Inject learned corrections
        try:
            from .correction_tracker import CorrectionTracker
            _ct = CorrectionTracker()
            _corrections = _ct.to_system_prompt_fragment(prompt)
            if _corrections:
                system_prompt += "\n\n" + _corrections
        except Exception:
            pass

        # H1: Cost estimation before execution
        if self._show_cost_estimates_enabled and self.budget_usd:
            estimate = self._estimate_cost(prompt)
            if estimate["estimated_cost"] > 0.005:
                sys.stderr.write(f"\n  Estimated: ~${estimate['estimated_cost']:.3f}, ~{estimate['estimated_iterations']} iterations ({estimate['category']})\n")
                sys.stderr.flush()

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Include prior conversation turns (keep last 40 messages to avoid context explosion)
        if self._conversation_history:
            history = self._conversation_history[-40:]
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        final_response = ""
        model_used = ""

        # ── Visual feedback loop for frontend mode ──
        if self._current_action_mode == "frontend" and not getattr(self.brain, '_model_override', None):
            try:
                from aura.tools.visual_feedback import get_visual_feedback
                vfl = get_visual_feedback(brain=self.brain)
                if vfl:
                    logger.info("[AgenticLoop] Frontend mode: trying visual feedback loop")
                    result = vfl.generate_with_feedback(prompt)
                    if result and result.get("code"):
                        final_response = result["code"]
                        model_used = result.get("model_used", "")
                        # Store and return early — skip agentic loop
                        self._conversation_history.append({"role": "user", "content": prompt})
                        self._conversation_history.append({"role": "assistant", "content": final_response})
                        if self.session:
                            self.session.append({"role": "user", "content": prompt})
                            self.session.append({"role": "assistant", "content": final_response})
                            self.session.update_stats(iterations=1, tool_calls=0)
                            self.session.save()
                        try:
                            _store_interaction(prompt, final_response)
                        except Exception as e:
                            logger.debug(f"[AgenticLoop] Store interaction failed: {e}")
                        self._current_action_mode = None
                        return {
                            "success": True,
                            "response": final_response,
                            "iterations": 1,
                            "tool_calls": 0,
                            "model": model_used,
                        }
            except Exception as e:
                logger.warning(f"[AgenticLoop] Visual feedback failed, falling back to normal: {e}")

        while self.iteration < self.max_iterations:
            # ── Cancellation check: top of iteration ──
            if self._cancel_event.is_set():
                final_response = "Cancelled by user."
                break

            self.iteration += 1
            self._edits_this_turn = 0  # Reset per iteration to avoid redundant auto-test

            # Adaptive planner: tick and check for replan
            try:
                if self._planner and self._planner.current_plan:
                    self._planner.tick()
                    if self._planner.should_replan():
                        progress = self._planner.current_plan.progress_summary
                        self._planner.replan(results_so_far=progress)
                        logger.info("[AgenticLoop] Re-planned based on progress")
                        # Rebuild system prompt with updated plan
                        system_prompt = self._build_system_prompt(prompt)
                        messages[0] = {"role": "system", "content": system_prompt}
            except Exception as e:
                logger.debug(f"[AgenticLoop] Planner tick/replan failed (non-fatal): {e}")

            # Budget check with intermediate warnings (C1)
            if self.budget_usd is not None:
                stats = self.brain.get_session_stats()
                cost = stats.get("cost_usd", 0.0)
                pct = cost / self.budget_usd if self.budget_usd > 0 else 0

                if cost >= self.budget_usd:
                    final_response = f"Budget limit reached (${self.budget_usd:.2f}). Stopping."
                    self._loop_error = True
                    break

                if not self._budget_warned_80 and pct >= 0.80:
                    self._budget_warned_80 = True
                    sys.stderr.write(f"\n  \033[33m\u25b3 Budget 80% used (${cost:.3f}/${self.budget_usd:.2f})\033[0m\n")
                    sys.stderr.flush()
                elif not self._budget_warned_50 and pct >= 0.50:
                    self._budget_warned_50 = True
                    sys.stderr.write(f"\n  \033[33m\u25b3 Budget 50% used (${cost:.3f}/${self.budget_usd:.2f})\033[0m\n")
                    sys.stderr.flush()

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
                # Show spinner (suppressed when on_chunk handles display)
                if not on_chunk:
                    model_tag = f" [{step_model.split(':')[0]}]" if step_model else ""
                    sys.stdout.write(f"  \033[90m● thinking{model_tag}...\033[0m")
                    sys.stdout.flush()

                for chunk_type, data in self.brain.think_with_tools_stream(
                    messages=messages, tools=active_tools,
                    model_override=step_model,
                ):
                    if chunk_type == "content":
                        if not on_chunk and not accumulated:
                            # Clear the "thinking..." spinner but keep cursor on same line
                            sys.stdout.write("\r\033[K")
                            sys.stdout.write(f"  \033[90m● generating...\033[0m")
                            sys.stdout.flush()
                        accumulated += data
                        if on_chunk:
                            on_chunk(data)
                    elif chunk_type == "tool_calls":
                        if not on_chunk and not accumulated:
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

                if not on_chunk:
                    if accumulated:
                        sys.stdout.write("\r\033[K")  # Clear "generating..." status
                        sys.stdout.flush()
                    elif not tool_calls:
                        sys.stdout.write("\r\033[K")  # Clear spinner if no output
                        sys.stdout.flush()

            except ConnectionError as e:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                _model_label = step_model or "default model"
                final_response = (
                    f"Connection failed to {_model_label}.\n"
                    f"  - Is Ollama running? Try: ollama serve\n"
                    f"  - Check your network connection."
                )
                self._loop_error = True
                break
            except TimeoutError as e:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                _model_label = step_model or "default model"
                final_response = (
                    f"Request timed out for {_model_label}.\n"
                    f"  - The model may be overloaded or too large.\n"
                    f"  - Try a smaller model with: /model <name>"
                )
                self._loop_error = True
                break
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
                    self._loop_error = True
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
                if not content:
                    # Empty response — retry with a nudge instead of showing "(No response)"
                    self._empty_response_count = getattr(self, '_empty_response_count', 0) + 1
                    if self._empty_response_count > 3:
                        logger.error(f"[AgenticLoop] {self._empty_response_count} consecutive empty responses — aborting loop")
                        final_response = "The model failed to generate a response after multiple attempts. Please try again with a clearer prompt."
                        break
                    logger.warning(f"[AgenticLoop] Empty response #{self._empty_response_count} from model on iteration {self.iteration}, nudging")
                    messages.append({"role": "assistant", "content": ""})
                    messages.append({"role": "user", "content": "Continue. Execute the task using tools."})
                    continue

                # Detect "thinking without acting" — model said it would use tools
                # but didn't actually call any. Nudge it to execute.
                _thinking_phrases = (
                    "let me search", "let me look", "let me find", "i'll search",
                    "i will search", "let me do a", "let me check", "let me research",
                    "i'll look up", "let me query", "searching for",
                )
                _content_lower = content.lower().strip()
                _is_thinking_without_acting = (
                    self.iteration <= 2
                    and len(content) < 500
                    and any(_content_lower.startswith(p) or f"\n{p}" in _content_lower for p in _thinking_phrases)
                    and self.tool_calls_total == 0
                )
                if _is_thinking_without_acting:
                    _nudge_count = getattr(self, '_thinking_nudge_count', 0) + 1
                    self._thinking_nudge_count = _nudge_count
                    if _nudge_count <= 2:
                        logger.warning(f"[AgenticLoop] Model is thinking without acting (nudge #{_nudge_count}): '{content[:80]}...'")
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": "Don't just describe what you'll do — actually use the available tools now. Call web_search or the appropriate tool to execute."})
                        continue

                self._empty_response_count = 0  # Reset on successful response

                # ── Completion verification ──
                # Only verify when: verification enabled, 2+ iterations ran,
                # tools were actually used (not simple chat), and not already verified.
                if (
                    self._verify_completion
                    and not self._verification_done
                    and self.iteration >= 2
                    and self.tool_calls_total > 0
                ):
                    self._verification_done = True
                    incomplete_reason = self._verify_task_completion(prompt, content)
                    if incomplete_reason:
                        # Re-enter the loop with the missing context
                        logger.info(f"[AgenticLoop] Verification found incomplete work, continuing")
                        messages.append({"role": "assistant", "content": content})
                        messages.append({
                            "role": "user",
                            "content": (
                                f"[Verification check] You said you were done, but verification found: "
                                f"{incomplete_reason}\n\n"
                                f"Please complete the remaining work."
                            ),
                        })
                        continue

                final_response = content
                if on_response and not accumulated:
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
                    # Return error instead of calling tool with empty/wrong args
                    error_result = json.dumps({
                        "error": f"Malformed arguments for {tool_name}: could not parse JSON. "
                                 f"Please provide valid JSON arguments. Raw: {str(args)[:200]}"
                    })
                    messages.append({"role": "tool", "content": error_result})
                    if self.session:
                        self.session.append({"role": "tool", "content": error_result})
                    continue
                parsed_calls.append((tool_name, args))

            # Permission checks + status display on main thread
            approved = []
            for tool_name, args in parsed_calls:
                self.tool_calls_total += 1
                # Resolve aliases BEFORE permission check (LLM may send "find", "Glob", "cat", etc.)
                resolved_name = self.executor._TOOL_ALIASES.get(tool_name, tool_name).lower()
                if not self.permissions.check(resolved_name, args):
                    approved.append((tool_name, args, json.dumps({"error": "Permission denied by user"})))
                    # Only show tool status here when there's no external on_tool_call callback
                    # (the callback in chat_loop already handles display, so this avoids double display)
                    if not on_tool_call:
                        self._show_tool_status(tool_name, args, denied=True)
                else:
                    if not on_tool_call:
                        self._show_tool_status(tool_name, args)
                    approved.append((tool_name, args, None))  # None = needs execution

            # ── Cancellation check: before tool execution ──
            if self._cancel_event.is_set():
                final_response = f"Cancelled after {self.iteration} iterations."
                break

            # Fire on_tool_start before execution so UI can show tool immediately
            needs_exec = [(i, t, a) for i, (t, a, r) in enumerate(approved) if r is None]
            if on_tool_start:
                for _idx, _t, _a in needs_exec:
                    on_tool_start(_t, _a)

            # Execute: parallel if 2+ calls, direct if 1
            if len(needs_exec) == 1:
                idx, t, a = needs_exec[0]
                result = self.executor.execute(t, a)
                approved[idx] = (t, a, result)
            elif len(needs_exec) > 1:
                futures = {}
                _pool = _get_tool_pool()
                for idx, t, a in needs_exec:
                    fut = _pool.submit(self.executor.execute, t, a)
                    futures[idx] = fut
                for idx, fut in futures.items():
                    try:
                        result = fut.result(timeout=300)
                    except Exception as e:
                        result = json.dumps({"error": f"Tool execution failed: {e}"})
                    t, a, _ = approved[idx]
                    approved[idx] = (t, a, result)

            # Collect results in original order
            _guard_tripped = False
            for tool_name, args, tool_result in approved:
                if tool_name in ("edit_file", "write_file") and not self._tool_result_has_error(tool_result):
                    self._edits_this_turn += 1
                    self._has_edits = True
                    self._last_tools_were_reads = False
                elif tool_name in ("read_file", "grep", "glob", "list_dir", "project_structure"):
                    self._last_tools_were_reads = True
                else:
                    self._last_tools_were_reads = False

                # Track hot files for context injection
                self._track_hot_file(tool_name, args, tool_result)

                # Adaptive planner: advance step on successful tool completions
                try:
                    if (self._planner and self._planner.current_plan
                            and not self._tool_result_has_error(tool_result)):
                        # Advance on substantive actions (edits, writes, shell commands)
                        if tool_name in ("edit_file", "write_file", "shell", "run_tests"):
                            result_snippet = tool_result[:100] if tool_result else ""
                            self._planner.advance_step(result=result_snippet)
                except Exception as e:
                    logger.debug(f"[AgenticLoop] Planner advance failed: {e}")

                if on_tool_call:
                    on_tool_call(tool_name, args, tool_result)
                tool_msg = {"role": "tool", "content": tool_result}
                messages.append(tool_msg)
                if self.session:
                    self.session.append(tool_msg)

                # Loop guard: detect infinite tool cycles
                guard_result = guard.record(tool_name, str(args))
                if guard_result and guard_result.triggered:
                    final_response = guard_result.fallback_message
                    self._loop_error = True
                    _guard_tripped = True
                    break

                # ── Cancellation check: after each tool result ──
                if self._cancel_event.is_set():
                    final_response = f"Cancelled after {self.iteration} iterations."
                    self._loop_error = True
                    _guard_tripped = True  # reuse flag to break outer loop
                    break

            if _guard_tripped:
                break

            # Auto-test: after processing all tool calls in this iteration,
            # if any edits were made, run tests and feed results back to LLM
            if self._edits_this_turn > 0:
                test_result = self._run_auto_test()
                if test_result:
                    self._has_test_failure = True
                    messages.append({
                        "role": "user",
                        "content": f"[Auto-test result] {test_result}",
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

        # Keep history bounded — summarize old messages instead of dropping them
        if len(self._conversation_history) > 100:
            self._conversation_history = _compact_history(self._conversation_history)

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
        # Determine success via explicit flag rather than string prefix matching.
        # The loop sets _loop_error when it hits a real failure (LLM error, budget, guard trip).
        # Clean up action mode state and restore original model override
        self._current_action_mode = None
        self.model_override = _original_model_override

        # Compute per-turn cost (C2)
        try:
            _end_cost = self.brain.get_session_stats().get("cost_usd", 0.0)
            self.last_turn_cost = _end_cost - _prev_cost
        except Exception:
            self.last_turn_cost = 0.0

        hit_error = getattr(self, '_loop_error', False)
        return {
            "success": not hit_error,
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

    def _verify_task_completion(self, original_task: str, agent_response: str) -> Optional[str]:
        """Quick LLM check: did the agent actually complete the task?

        Uses the fast model to minimize cost. Returns None if complete,
        or a string describing what's missing if incomplete.
        """
        try:
            from aura.config import Config
            fast_model = Config.get_model("fast")

            verify_prompt = (
                f"Task the agent was given:\n{original_task[:1000]}\n\n"
                f"Agent's final response:\n{agent_response[:2000]}\n\n"
                "Did the agent complete ALL parts of the task? "
                "If anything is incomplete or was skipped, respond with INCOMPLETE: [what's missing]. "
                "If everything is done, respond with COMPLETE."
            )

            _ensure_console()
            console.print("  [dim cyan]verify[/dim cyan] checking completion...", highlight=False)

            result = self.brain.think(
                verify_prompt,
                system_prompt="You are a task completion verifier. Be brief and precise.",
                use_history=False,
                model_override=fast_model,
            )

            # brain.think returns str or dict
            if isinstance(result, dict):
                result = result.get("response", result.get("content", str(result)))
            result = str(result).strip()

            if result.upper().startswith("INCOMPLETE"):
                # Extract the reason after "INCOMPLETE:" prefix
                reason = result.split(":", 1)[1].strip() if ":" in result else result
                console.print(f"  [yellow]incomplete[/yellow] {reason[:120]}", highlight=False)
                logger.info(f"[AgenticLoop] Verification: INCOMPLETE — {reason[:200]}")
                return reason
            else:
                console.print("  [green]verified[/green] task complete", highlight=False)
                logger.info("[AgenticLoop] Verification: COMPLETE")
                return None

        except Exception as e:
            # Verification failure must not break the loop
            logger.debug(f"[AgenticLoop] Completion verification failed (non-fatal): {e}")
            return None

    def _run_auto_test(self) -> Optional[str]:
        """Run project tests after edits. Returns test output for LLM or None."""
        try:
            from aura.tools.auto_verify import auto_verify
            _ensure_console()
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

        # Phase 1: First iteration — classify from the latest user prompt
        if self.iteration == 1:
            user_prompt = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_prompt = msg.get("content", "")
                    break
            category, _conf = classify_task(user_prompt) if user_prompt else ("orchestrator", 1.0)

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
        """Clear conversation history and hot files (new session)."""
        self._conversation_history.clear()
        self._hot_files.clear()
        self._hot_file_contents.clear()

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

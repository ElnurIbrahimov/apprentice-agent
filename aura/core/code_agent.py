"""Code Agent Mode — LLM writes Python code as actions instead of JSON tool calls.

Roadmap 5.1: For complex multi-step tasks, the LLM writes Python code that calls
AURA's tools as functions. smolagents proved this uses ~30% fewer steps.

The code agent:
  - Prompts the LLM to produce Thought + ```python code blocks
  - Exposes AURA tools as simple Python functions (search_memory, web_search, etc.)
  - Executes the code in a restricted sandbox with timeout
  - Captures stdout, stderr, return values, and exceptions
  - Returns structured results back to the agent loop
"""

import ast
import os
import re
import sys
import json
import logging
import subprocess
import tempfile
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default timeout for code execution (seconds)
CODE_EXEC_TIMEOUT = 30

# Keywords that suggest a task is complex enough for code agent mode
COMPLEX_TASK_KEYWORDS = {
    "for each", "for every", "batch", "all files", "process all",
    "compare", "merge", "aggregate", "transform", "pipeline",
    "step by step", "multi-step", "first then", "after that",
    "combine", "cross-reference", "correlate",
    "parse", "extract and", "filter and", "sort and",
    "calculate", "compute across", "sum of", "average of",
    "generate report", "create summary from",
    "refactor", "migrate", "convert all",
}

# Environment variables safe to pass to the subprocess sandbox.
# Everything else is stripped to prevent leaking secrets/tokens.
_SAFE_ENV_KEYS = {"PATH", "TEMP", "TMP", "HOME", "SYSTEMROOT", "COMSPEC", "USERPROFILE"}

# Maximum bytes of stdout captured from subprocess (100 KB)
_MAX_STDOUT_BYTES = 100 * 1024


def _run_code_in_subprocess(code: str, timeout: int) -> dict:
    """Execute Python code in a subprocess with enforced timeout.

    The subprocess runs with a sanitized environment and in a temporary
    working directory.  On timeout the process is killed by the OS —
    unlike a thread, this actually stops execution.

    The executed code's stdout is captured.  If the last line of stdout
    is valid JSON, it is parsed into ``result["parsed_result"]``.

    Returns:
        dict with keys: success, stdout, stderr, parsed_result, error, timed_out
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "parsed_result": None,
        "error": "",
        "timed_out": False,
    }

    # Build sanitized environment
    safe_env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}

    tmp_dir = tempfile.gettempdir()
    tmp_file = None
    proc = None

    try:
        # Write code to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=tmp_dir, delete=False
        ) as f:
            f.write(code)
            tmp_file = f.name

        proc = subprocess.Popen(
            [sys.executable, tmp_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp_dir,
            env=safe_env,
            text=True,
        )

        try:
            stdout_data, stderr_data = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()  # Drain pipes to avoid zombie
            result["timed_out"] = True
            result["error"] = f"Code execution timed out after {timeout}s"
            logger.warning("[CodeAgent] Subprocess timed out after %ds", timeout)
            return result

        # Cap stdout to avoid memory issues
        stdout_data = (stdout_data or "")[:_MAX_STDOUT_BYTES]
        stderr_data = (stderr_data or "")[:_MAX_STDOUT_BYTES]

        result["stdout"] = stdout_data
        result["stderr"] = stderr_data

        if proc.returncode == 0:
            result["success"] = True
        else:
            result["error"] = stderr_data.strip() if stderr_data.strip() else f"Process exited with code {proc.returncode}"

        # Try to parse the last non-empty stdout line as JSON result
        if stdout_data.strip():
            last_line = stdout_data.strip().rsplit("\n", 1)[-1].strip()
            try:
                result["parsed_result"] = json.loads(last_line)
            except (json.JSONDecodeError, ValueError):
                pass  # Not JSON — that's fine

    except Exception as e:
        result["error"] = f"Subprocess launch failed: {e}"
        logger.error("[CodeAgent] Subprocess error: %s", e)
    finally:
        if tmp_file:
            try:
                os.unlink(tmp_file)
            except (OSError, FileNotFoundError):
                pass

    return result


# System prompt instructing the LLM to write Python code as actions
CODE_AGENT_SYSTEM_PROMPT = """You are Aura in CODE AGENT MODE. Instead of calling tools via JSON, you write Python code to accomplish tasks.

## How It Works
1. Think about what you need to do (write your reasoning as plain text).
2. Write a Python code block that uses the available tool functions.
3. The code will be executed and you'll see the results.
4. Repeat until the task is done, then give your final answer as plain text (no code block).

## Available Functions
These are pre-loaded in your execution environment:

```
search_memory(query: str, k: int = 5) -> list[dict]
    Search AURA's memory for relevant information.

query_kg(query: str, k: int = 5) -> list[dict]
    Query the knowledge graph for entities and relationships.

web_search(query: str) -> list[dict]
    Search the web and return results.

read_file(path: str) -> str
    Read a file and return its contents.

write_file(path: str, content: str) -> dict
    Write content to a file. Returns {"success": bool, "path": str}.

list_dir(path: str = ".") -> list[str]
    List files in a directory.

grep(pattern: str, path: str = ".") -> list[dict]
    Search file contents with regex. Returns matches with file/line info.

edit_file(path: str, old_text: str, new_text: str) -> dict
    Replace old_text with new_text in a file.

execute_code(code: str) -> dict
    Execute Python code in a sandboxed environment.
```

## Rules
- Put ALL code in a single ```python ... ``` block per step.
- Use `result = function(...)` to capture return values, then `print(result)` to see them.
- You can use standard Python: loops, conditionals, string ops, json, re, math, etc.
- Do NOT import os, subprocess, sys, or any dangerous modules — they are blocked.
- When done, respond with your final answer as plain text (NO code block).

## Format
Thought: <your reasoning>

```python
# your code here
result = web_search("query")
print(result)
```
"""


def _extract_code_block(text: str) -> Optional[str]:
    """Extract Python code from a ```python ... ``` block in LLM output."""
    # Match ```python ... ``` (greedy, handles multiple lines)
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: match ``` ... ``` without language tag
    pattern2 = r"```\s*\n(.*?)```"
    match2 = re.search(pattern2, text, re.DOTALL)
    if match2:
        code = match2.group(1).strip()
        # Basic heuristic: looks like Python?
        if any(kw in code for kw in ("print(", "def ", "for ", "import ", "result =", "= ")):
            return code

    return None


def _extract_thought(text: str) -> str:
    """Extract the thought/reasoning portion before the code block."""
    # Everything before the first ``` block
    idx = text.find("```")
    if idx > 0:
        thought = text[:idx].strip()
        # Strip "Thought:" prefix if present
        if thought.lower().startswith("thought:"):
            thought = thought[len("thought:"):].strip()
        return thought
    return text.strip()


class CodeAgentMode:
    """Executes LLM-generated Python code with AURA tools exposed as functions.

    Usage in the agent loop:
        code_agent = CodeAgentMode(agent)
        result = code_agent.execute_step(messages)
        # result = {"thought": str, "code": str, "output": str, "error": str, "done": bool}
    """

    def __init__(self, agent, timeout: int = CODE_EXEC_TIMEOUT):
        """
        Args:
            agent: ApprenticeAgent instance (for accessing tools, brain, memory).
            timeout: Max seconds per code execution.
        """
        self.agent = agent
        self.timeout = timeout

    def build_tool_namespace(self) -> dict:
        """Build a dict of Python functions the LLM can call in generated code.

        Maps AURA tools to simple function signatures. Each function wraps
        the actual tool execution with error handling.
        """
        agent = self.agent
        ns = {}

        # --- search_memory ---
        def search_memory(query: str, k: int = 5) -> list:
            try:
                from aura.memory.unified_memory import get_unified_memory
                um = get_unified_memory()
                results = um.query(query, k=k, min_score=0.1)
                return [{"content": r.content[:500], "source": r.source, "score": getattr(r, "score", 0)}
                        for r in results]
            except Exception as e:
                return [{"error": str(e)}]
        ns["search_memory"] = search_memory

        # --- query_kg ---
        def query_kg(query: str, k: int = 5) -> list:
            if agent.kg_bridge is not None:
                try:
                    ctx = agent.kg_bridge.get_context_for_query(query, k)
                    return [{"context": ctx}] if ctx else []
                except Exception as e:
                    return [{"error": str(e)}]
            return [{"error": "Knowledge graph not available"}]
        ns["query_kg"] = query_kg

        # --- web_search ---
        def web_search(query: str) -> list:
            for tool_name in ("tavily_search", "brave_search", "web_search"):
                tool = agent.tools.get(tool_name)
                if tool and hasattr(tool, "execute"):
                    try:
                        result = tool.execute(f"search {query}")
                        if isinstance(result, dict):
                            return [result]
                        return result if isinstance(result, list) else [{"result": str(result)[:2000]}]
                    except Exception:
                        continue
            return [{"error": "No search tool available"}]
        ns["web_search"] = web_search

        # --- read_file ---
        def read_file(path: str) -> str:
            tool = agent.tools.get("filesystem")
            if tool:
                try:
                    result = tool.read_file(path)
                    if isinstance(result, dict):
                        return result.get("content", result.get("error", str(result)))
                    return str(result)
                except Exception as e:
                    return f"Error: {e}"
            # Fallback: try ToolExecutor
            if hasattr(agent, "_tool_executor") and agent._tool_executor:
                r = agent._tool_executor.execute("read_file", {"path": path})
                return r[:8000] if isinstance(r, str) else str(r)[:8000]
            return "Error: filesystem tool not available"
        ns["read_file"] = read_file

        # --- write_file ---
        def write_file(path: str, content: str) -> dict:
            tool = agent.tools.get("filesystem")
            if tool:
                try:
                    result = tool.write_file(path, content)
                    return result if isinstance(result, dict) else {"success": True, "path": path}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            return {"success": False, "error": "filesystem tool not available"}
        ns["write_file"] = write_file

        # --- list_dir ---
        def list_dir(path: str = ".") -> list:
            tool = agent.tools.get("filesystem")
            if tool and hasattr(tool, "list_directory"):
                try:
                    result = tool.list_directory(path)
                    if isinstance(result, dict):
                        return result.get("entries", result.get("files", [str(result)]))
                    return result if isinstance(result, list) else [str(result)]
                except Exception as e:
                    return [f"Error: {e}"]
            return ["Error: filesystem tool not available"]
        ns["list_dir"] = list_dir

        # --- grep ---
        def grep(pattern: str, path: str = ".") -> list:
            tool = agent.tools.get("code_search")
            if tool and hasattr(tool, "execute"):
                try:
                    result = tool.execute(f"grep {pattern} {path}")
                    if isinstance(result, dict):
                        return result.get("matches", [result])
                    return result if isinstance(result, list) else [{"result": str(result)[:3000]}]
                except Exception as e:
                    return [{"error": str(e)}]
            return [{"error": "code_search tool not available"}]
        ns["grep"] = grep

        # --- shell removed: LLM-generated code must not have direct shell access ---

        # --- edit_file ---
        def edit_file(path: str, old_text: str, new_text: str) -> dict:
            tool = agent.tools.get("code_edit")
            if tool and hasattr(tool, "execute"):
                try:
                    action = json.dumps({"path": path, "old_text": old_text, "new_text": new_text})
                    result = tool.execute(action)
                    return result if isinstance(result, dict) else {"success": True}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            # Fallback to ToolExecutor
            if hasattr(agent, "_tool_executor") and agent._tool_executor:
                r = agent._tool_executor.execute("edit_file", {
                    "path": path, "old_string": old_text, "new_string": new_text
                })
                try:
                    return json.loads(r) if isinstance(r, str) else {"result": str(r)}
                except (json.JSONDecodeError, ValueError):
                    return {"result": str(r)[:2000]}
            return {"success": False, "error": "code_edit tool not available"}
        ns["edit_file"] = edit_file

        # --- execute_code ---
        def execute_code(code: str) -> dict:
            tool = agent.tools.get("code_executor")
            if tool and hasattr(tool, "execute"):
                try:
                    return tool.execute(code)
                except Exception as e:
                    return {"success": False, "error": str(e)}
            return {"success": False, "error": "code_executor not available"}
        ns["execute_code"] = execute_code

        return ns

    def execute_code_safely(self, code: str, namespace: dict) -> dict:
        """Execute Python code in a subprocess with enforced timeout.

        Pre-validates the code via AST checks and import allow-listing, then
        runs it in an isolated subprocess that the OS can actually kill on
        timeout (unlike the previous thread-based approach).

        Tool namespace functions (search_memory, web_search, etc.) are NOT
        available inside the subprocess — it is for pure computation only.
        Tool calls go through the normal ReAct loop.

        If E2B_API_KEY is set, prefers the E2B cloud sandbox over a local
        subprocess.

        Args:
            code: Python code string from LLM.
            namespace: Tool functions namespace (used only for key names in
                       the error message if the LLM tries to call tools).

        Returns:
            {"stdout": str, "stderr": str, "return_value": Any, "error": str, "timed_out": bool}
        """
        result = {
            "stdout": "",
            "stderr": "",
            "return_value": None,
            "error": "",
            "timed_out": False,
        }

        # ------------------------------------------------------------------
        # Pre-filter 1: AST validation — reject syntax errors and dangerous
        # constructs before spawning anything.
        # ------------------------------------------------------------------
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            result["error"] = f"Syntax error: {e}"
            return result

        # Blocked builtins / attributes (same set as before)
        _blocked_calls = {
            "eval", "exec", "compile", "__import__", "open",
            "input", "breakpoint", "exit", "quit",
            "globals", "locals", "vars",
        }
        _blocked_attrs = {
            "__class__", "__bases__", "__subclasses__", "__mro__",
            "__code__", "__globals__", "__builtins__", "__dict__",
            "__import__", "__loader__", "__spec__",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fname = ""
                if isinstance(node.func, ast.Name):
                    fname = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    fname = node.func.attr
                if fname in _blocked_calls:
                    result["error"] = f"Blocked function call: {fname}()"
                    return result

            if isinstance(node, ast.Attribute):
                if node.attr in _blocked_attrs:
                    result["error"] = f"Blocked attribute access: {node.attr}"
                    return result

        # ------------------------------------------------------------------
        # Pre-filter 2: Import allow-list — reject disallowed imports before
        # spawning the subprocess.
        # ------------------------------------------------------------------
        allowed_modules = {
            "json", "re", "math", "datetime", "collections",
            "itertools", "functools", "string", "textwrap",
            "statistics", "decimal", "fractions",
            "urllib.parse",
            "csv",
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split(".")[0]
                    if base not in allowed_modules and alias.name not in allowed_modules:
                        result["error"] = (
                            f"Import of '{alias.name}' is not allowed in code agent mode. "
                            f"Allowed: {', '.join(sorted(allowed_modules))}"
                        )
                        return result
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base = node.module.split(".")[0]
                    if base not in allowed_modules and node.module not in allowed_modules:
                        result["error"] = (
                            f"Import of '{node.module}' is not allowed in code agent mode. "
                            f"Allowed: {', '.join(sorted(allowed_modules))}"
                        )
                        return result

        # ------------------------------------------------------------------
        # Tier 1: E2B cloud sandbox if available
        # ------------------------------------------------------------------
        if os.getenv("E2B_API_KEY"):
            e2b_result = self._try_e2b(code)
            if e2b_result is not None:
                return e2b_result

        # ------------------------------------------------------------------
        # Tier 2: Local subprocess sandbox
        # ------------------------------------------------------------------
        sub_result = _run_code_in_subprocess(code, self.timeout)

        # Map subprocess result to the expected return format
        result["stdout"] = sub_result["stdout"][:8000]
        result["stderr"] = sub_result["stderr"][:4000]
        result["error"] = sub_result["error"]
        result["timed_out"] = sub_result["timed_out"]

        # If the subprocess produced a JSON result on the last line, use it
        if sub_result["parsed_result"] is not None:
            try:
                result["return_value"] = str(sub_result["parsed_result"])[:4000]
            except Exception:
                pass

        return result

    def _try_e2b(self, code: str) -> Optional[dict]:
        """Attempt execution via E2B cloud sandbox. Returns None if unavailable."""
        try:
            from e2b_code_interpreter import Sandbox  # type: ignore
        except ImportError:
            return None

        api_key = os.getenv("E2B_API_KEY", "")
        if not api_key:
            return None

        result = {
            "stdout": "",
            "stderr": "",
            "return_value": None,
            "error": "",
            "timed_out": False,
        }
        try:
            with Sandbox(api_key=api_key) as sbx:
                execution = sbx.run_code(code)
                output = "\n".join(str(r) for r in execution.results) if execution.results else ""
                error_msg = execution.error.value if execution.error else ""
                result["stdout"] = output[:8000]
                result["stderr"] = error_msg[:4000]
                if execution.error:
                    result["error"] = error_msg[:4000]
            logger.debug("[CodeAgent] E2B execution completed")
        except Exception as e:
            result["error"] = f"E2B sandbox error: {e}"
            logger.warning("[CodeAgent] E2B failed, would fall back to subprocess: %s", e)
            return None  # Fall through to subprocess

        return result

    def format_execution_result(self, exec_result: dict) -> str:
        """Format execution result as a string for the LLM conversation."""
        parts = []

        if exec_result.get("timed_out"):
            parts.append(f"TIMEOUT: Execution exceeded {self.timeout}s limit.")

        if exec_result.get("stdout"):
            parts.append(f"Output:\n{exec_result['stdout']}")

        if exec_result.get("stderr"):
            parts.append(f"Stderr:\n{exec_result['stderr']}")

        if exec_result.get("error"):
            parts.append(f"Error:\n{exec_result['error']}")

        if exec_result.get("return_value") and not exec_result.get("stdout"):
            parts.append(f"Return value:\n{exec_result['return_value']}")

        if not parts:
            parts.append("Code executed successfully (no output).")

        return "\n\n".join(parts)


def should_use_code_agent(goal: str) -> bool:
    """Decide if a task should use code agent mode vs standard tool mode.

    Code agent mode is better for:
    - Multi-step tasks that need looping/iteration
    - Data processing and transformation
    - Tasks that combine multiple tool results
    - Complex file operations across many files

    Returns True if code agent mode is recommended.
    """
    goal_lower = goal.lower()

    # Check for explicit code agent trigger
    if goal_lower.startswith("/code ") or "use code mode" in goal_lower:
        return True

    # Count how many complexity keywords match
    matches = sum(1 for kw in COMPLEX_TASK_KEYWORDS if kw in goal_lower)

    # Need at least 2 keyword matches for auto-activation
    if matches >= 2:
        return True

    # Long goals with multiple sentences often need multi-step reasoning
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', goal) if s.strip()]
    if len(sentences) >= 3 and any(kw in goal_lower for kw in COMPLEX_TASK_KEYWORDS):
        return True

    return False

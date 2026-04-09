"""Sub-agent spawning for the agentic loop.

Allows the LLM to spawn parallel sub-agents for tasks like
researching docs, reading multiple files, or independent analysis.
"""

import concurrent.futures
import json
import logging
import threading

logger = logging.getLogger(__name__)

from aura.pools import tool_pool as _tool_pool_fn

_active_coder = threading.Lock()  # Ensures only 1 coder sub-agent at a time

# Read-only tool subset for restricted roles
READ_ONLY_TOOLS = frozenset({
    "read_file", "grep", "glob", "list_dir",
    "project_structure", "search_web",
})


class BrainProxy:
    """Lightweight wrapper that isolates mutable state for sub-agents.

    Delegates LLM calls to the shared brain but maintains its own
    model_override, cost counters, and conversation buffer to avoid
    race conditions when multiple sub-agents run in parallel.
    """

    def __init__(self, parent_brain):
        self._parent = parent_brain
        self._model_override = parent_brain._model_override
        self._sub_input_tokens = 0
        self._sub_output_tokens = 0

    def think(self, *args, **kwargs):
        result = self._parent.think(*args, **kwargs)
        return result

    def think_with_tools(self, *args, **kwargs):
        return self._parent.think_with_tools(*args, **kwargs)

    def think_with_tools_stream(self, *args, **kwargs):
        return self._parent.think_with_tools_stream(*args, **kwargs)

    def get_session_stats(self):
        return self._parent.get_session_stats()

    def set_model_override(self, model):
        self._model_override = model

    def recall_relevant(self, *args, **kwargs):
        if hasattr(self._parent, 'recall_relevant'):
            return self._parent.recall_relevant(*args, **kwargs)
        return []

    def __getattr__(self, name):
        # Fall through to parent for any other attribute reads
        return getattr(self._parent, name)


class SubAgentManager:
    """Manages sub-agent lifecycle."""

    def __init__(self, parent_loop):
        self.parent = parent_loop
        self._active_count = 0
        self._lock = threading.Lock()

    def spawn(self, task: str, role: str = "reader") -> str:
        """Spawn sub-agent, wait for result, return formatted string.

        Roles:
          reader     — read-only tools only
          researcher — read-only + search_web
          coder      — full tools (max 1 concurrent, blocks if another coder active)
        """
        if self.parent._is_sub_agent:
            return json.dumps({"error": "Sub-agents cannot spawn other sub-agents"})

        if role not in ("reader", "researcher", "coder"):
            role = "reader"

        with self._lock:
            self._active_count += 1

        try:
            child_loop = self._build_child_loop(task, role)
            result = self._run_child(child_loop, task, role)
            return result
        except Exception as e:
            logger.error(f"[SubAgent] Spawn failed: {e}")
            return json.dumps({"error": f"Sub-agent failed: {e}"})
        finally:
            with self._lock:
                self._active_count -= 1

    def _build_child_loop(self, task: str, role: str):
        """Create child AgenticLoop with restricted tools and permissions."""
        from .agentic_loop import AgenticLoop
        from .permissions import PermissionManager
        from .tool_schemas import AGENTIC_TOOLS

        # Build restricted tool list based on role
        if role == "coder":
            tools = [t for t in AGENTIC_TOOLS if t["function"]["name"] != "spawn_agent"]
        else:
            allowed = set(READ_ONLY_TOOLS)
            if role == "researcher":
                allowed.add("search_web")
            tools = [t for t in AGENTIC_TOOLS if t["function"]["name"] in allowed]

        # Build permissions
        permissions = PermissionManager()
        if role == "coder":
            # Coder gets auto-approve for everything (inherits parent trust)
            if self.parent.permissions.trust_mode:
                permissions.set_trust_mode(True)
        else:
            # Read-only roles: auto-approve all (they only have safe tools)
            permissions.set_trust_mode(True)

        # Create child loop with isolated brain proxy
        child = AgenticLoop(
            brain=BrainProxy(self.parent.brain),
            project_root=self.parent.project_root,
            permissions=permissions,
            model_override=self.parent.model_override,
            max_iterations=15,
            budget_usd=None,
            context=self.parent.context,
        )
        child._is_sub_agent = True
        child._sub_agent_tools = tools

        return child

    def _run_child(self, loop, task: str, role: str) -> str:
        """Run child in thread pool with 120s timeout. Return result text."""
        def _execute():
            if role == "coder":
                with _active_coder:
                    return loop.run(task)
            else:
                return loop.run(task)

        try:
            future = _tool_pool_fn().submit(_execute)
            result = future.result(timeout=120)

            response = result.get("response", "")
            iterations = result.get("iterations", 0)
            tool_calls = result.get("tool_calls", 0)

            return json.dumps({
                "sub_agent_result": response,
                "role": role,
                "iterations": iterations,
                "tool_calls": tool_calls,
            })
        except concurrent.futures.TimeoutError:
            logger.warning("[SubAgent] Timed out after 120s")
            return json.dumps({"error": "Sub-agent timed out after 120 seconds"})
        except Exception as e:
            return json.dumps({"error": f"Sub-agent execution error: {e}"})

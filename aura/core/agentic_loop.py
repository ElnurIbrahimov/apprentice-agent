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
import sys
import threading
import uuid
from typing import Any, Optional

from . import agentic_loop_support as _loop_support
from .agentic_loop_events import LoopEventEmitter
from .agentic_loop_model_step import ModelStepController
from .agentic_loop_outcomes import LoopOutcome
from .agentic_loop_support import (
    AGENTIC_SYSTEM_PROMPT,
    _compact_history,
    _ensure_console,
    _recall_memories,
    _store_interaction,
)
from .agentic_loop_tool_calls import ToolCallCoordinator
from .permissions import PermissionManager
from .session import AgenticSession
from .token_manager import ContextWindowManager
from .tool_executor import ToolExecutor
from .tool_schemas import AGENTIC_TOOLS

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 50


class AgenticLoop:
    """Core autonomous loop: LLM calls tools until task is complete."""

    def __init__(
        self,
        brain,
        project_root: str = ".",
        permissions: Optional[PermissionManager] = None,
        model_override: str | None = None,
        max_iterations: int = MAX_ITERATIONS,
        budget_usd: Optional[float] = None,
        context: str = "",
        session: Optional[AgenticSession] = None,
        aura_config: dict | None = None,
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
        self.executor = ToolExecutor(self.project_root, sub_agent_mgr=self._sub_agent_mgr, permissions=self.permissions, brain=self.brain)
        self.iteration = 0
        self.tool_calls_total = 0
        self._edits_this_turn = 0  # Track edits for auto-test
        self._is_sub_agent = False  # Prevent sub-agent-ception
        self._current_run_id = ""

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

        # Reflexion state: count consecutive iterations that contained any
        # failed tool call so we can trigger a reflection pass when the
        # agent is repeatedly stuck. Reset to 0 on any clean iteration.
        self._tool_failure_streak = 0
        self._reflexion_fired_this_run = False
        self._tool_call_coordinator = ToolCallCoordinator(self)
        self._model_step_controller = ModelStepController(self)

        # Completion verification: one extra LLM call to check task was actually done
        self._verify_completion = True

        # Adaptive planner (enhancement — non-fatal if unavailable)
        self._planner = None
        try:
            from .adaptive_planner import AdaptivePlanner
            self._planner = AdaptivePlanner(brain=brain)
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.debug(f"[AgenticLoop] AdaptivePlanner init failed (non-fatal): {e}")

        # MCP client for external tool servers
        from .mcp_client import MCPClientManager
        self._mcp_client = MCPClientManager()
        # Clones set _owns_mcp=False so they don't tear down the parent's
        # connections in __del__. See clone_for_background().
        self._owns_mcp = True
        if aura_config:
            try:
                self._mcp_client.load_from_config(aura_config)
            except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                logger.debug(f"[AgenticLoop] MCP client init failed (non-fatal): {e}")
        self.executor._mcp_client = self._mcp_client

        # H1: Cost estimation toggle
        self._show_cost_estimates_enabled = False
        if aura_config:
            self._show_cost_estimates_enabled = aura_config.get("cost_estimate", False)
        # Store for later consumers (auto-verify, lint/typecheck)
        self._aura_config = aura_config or {}

        # H2: Diff preview accept/reject toggle
        self._trust_all_edits = False

        # Response cache is now owned by brain.py (module-level singleton)
        # so multiple sessions share hits. See aura.brain._get_response_cache.

    def __del__(self):
        """Clean up MCP connections to prevent process leaks."""
        try:
            if (
                getattr(self, "_owns_mcp", True)
                and hasattr(self, "_mcp_client")
                and self._mcp_client
            ):
                self._mcp_client.disconnect_all()
        except (OSError, ConnectionError, TimeoutError, ValueError):
            # Can't use logger in __del__ reliably, but at least bind the exception
            pass  # MCP cleanup during GC — best-effort

    def clone_for_background(self, permissions: PermissionManager) -> "AgenticLoop":
        """Create a sibling loop for background '&' tasks.

        Shares the expensive read-mostly state (brain, MCP client, project
        context, router, planner) so each background task doesn't re-spawn
        MCP servers or rebuild the context index. Gets its own permissions,
        executor, conversation history, and iteration counters so it can't
        pollute the foreground session's state or escalate privileges.

        The clone is marked ``_owns_mcp=False`` so its ``__del__`` won't
        disconnect MCP servers that the parent is still using.
        """
        from .sub_agent import SubAgentManager
        from .tool_executor import ToolExecutor

        clone = AgenticLoop.__new__(AgenticLoop)

        # Shared state (read-only or thread-safe).
        clone.brain = self.brain
        clone.project_root = self.project_root
        clone.context = self.context
        clone._router = self._router
        clone._mcp_client = self._mcp_client
        clone._owns_mcp = False
        clone._planner = getattr(self, "_planner", None)
        clone.context_mgr = self.context_mgr

        # Fresh mutable state.
        clone.permissions = permissions
        clone.model_override = self.model_override
        clone.max_iterations = 10  # bg tasks stay short
        clone.budget_usd = None
        clone.session = None
        clone.aura_config = getattr(self, "aura_config", None)
        clone._conversation_history = []
        clone.iteration = 0
        clone.tool_calls_total = 0
        clone._edits_this_turn = 0
        clone._is_sub_agent = False
        clone._current_run_id = ""
        clone._has_edits = False
        clone._has_test_failure = False
        clone._last_tools_were_reads = True
        clone._current_action_mode = None
        clone._budget_warned_50 = False
        clone._budget_warned_80 = False
        clone.last_turn_cost = 0.0
        clone._hot_files = []
        clone._hot_file_contents = {}
        clone._cancel_event = threading.Event()
        clone._tool_failure_streak = 0
        clone._reflexion_fired_this_run = False
        clone._verify_completion = False  # skip verification overhead for bg
        clone._show_cost_estimates_enabled = False
        clone._trust_all_edits = False

        # Fresh executor because permissions are bound at construction.
        clone._sub_agent_mgr = SubAgentManager(clone)
        clone.executor = ToolExecutor(
            clone.project_root,
            sub_agent_mgr=clone._sub_agent_mgr,
            permissions=permissions,
            brain=clone.brain,
        )
        clone.executor._mcp_client = clone._mcp_client
        clone._tool_call_coordinator = type(self._tool_call_coordinator)(clone)
        clone._model_step_controller = type(self._model_step_controller)(clone)
        return clone

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
            memories=memories.formatted or "(No relevant memories found)",
        )

        # Inject design system for frontend tasks
        try:
            from aura.prompts.design_system import DESIGN_SYSTEM_MODES, DESIGN_SYSTEM_PROMPT
            if self._current_action_mode in DESIGN_SYSTEM_MODES:
                system_prompt += "\n\n" + DESIGN_SYSTEM_PROMPT
        except ImportError:
            pass

        # Inject semantic codebase context (same as brain.py path)
        try:
            from pathlib import Path as _P

            from aura.tools.codebase_index import CodebaseIndex
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
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
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
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.debug(f"[AgenticLoop] File pre-load failed: {e}")

        # Adaptive plan context: inject current plan so LLM knows what to do next
        try:
            if self._planner and self._planner.current_plan:
                plan_ctx = self._planner.current_plan.to_prompt_context()
                if plan_ctx:
                    system_prompt += "\n\n" + plan_ctx
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.debug(f"[AgenticLoop] Plan context injection failed: {e}")

        # Hot files: inject recently touched file paths so LLM remembers
        # what it was working on across iterations and after context compaction
        if self._hot_files:
            hot_lines = ["## Recently touched files"]
            for fp in self._hot_files:
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
            except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                logger.debug(f"[AgenticLoop] Hot file read failed for {path}: {e}")

    def _estimate_cost(self, prompt: str) -> dict:
        """Estimate cost based on task category and historical data."""
        try:
            from .router import classify_task
            category, _confidence = classify_task(prompt)

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
        except (OSError, ConnectionError, TimeoutError, ValueError):
            return {"estimated_cost": 0, "estimated_iterations": 0, "category": "unknown"}

    # Keywords that suggest a code-task prompt (as opposed to conversation)
    _CODE_TASK_KEYWORDS = frozenset({
        "fix", "edit", "write", "implement", "refactor", "debug", "test",
        "add", "create", "delete", "remove", "update", "change", "modify",
        "function", "class", "method", "file", "import", "error", "bug",
        "code", "script", "module", "variable", "loop", "api", "endpoint",
    })
    # Obvious non-code conversational openers — never trigger smart context
    _NON_CODE_OPENERS = frozenset({
        "hi", "hello", "hey", "yo", "sup", "thanks", "thank", "ok", "okay",
        "cool", "nice", "great", "lol", "lmao", "gg", "good", "bad", "wow",
    })

    def _inject_smart_context(self, prompt: str, system_prompt: str) -> str:
        """Embed prompt and find the most relevant project files to inject.

        Skipped for short conversational prompts to avoid 30+ blocking embed
        calls that add 5-20s of latency per message with no benefit.
        """
        words = prompt.lower().split()

        # Hard-skip conversational openers ("hey aura what's up") even when
        # they happen to contain a code keyword like "update".
        if words and words[0].rstrip("?!.,") in self._NON_CODE_OPENERS:
            return system_prompt

        is_code_task = len(words) >= 5 and any(w in self._CODE_TASK_KEYWORDS for w in words)
        if not is_code_task:
            return system_prompt

        # Also skip if system prompt is already very large (> 30KB) to avoid
        # bloating context beyond what the model can handle effectively.
        if len(system_prompt) > 30000:
            return system_prompt

        try:
            import numpy as np
            import ollama

            resp = ollama.embed(model="nomic-embed-text:latest", input=prompt)
            if not resp or "embeddings" not in resp or not resp["embeddings"]:
                return system_prompt
            prompt_vec = np.array(resp["embeddings"][0])

            # Score candidate files — cap at 10 (not 30) to keep latency bounded.
            import glob as _glob
            candidates = []
            for ext in ("*.py", "*.js", "*.ts", "*.go", "*.rs", "*.java"):
                candidates.extend(_glob.glob(os.path.join(self.project_root, "**", ext), recursive=True))
            candidates = [f for f in candidates[:30] if os.path.getsize(f) < 50000]

            # Remove files already in hot_files
            hot = set(getattr(self, '_hot_files', []))
            candidates = [f for f in candidates if f not in hot][:10]

            if not candidates:
                return system_prompt

            # Batch all candidate paths into one embed call (vs N sequential HTTP round-trips)
            rels = [os.path.relpath(fpath, self.project_root) for fpath in candidates]
            scored = []
            try:
                batch_resp = ollama.embed(model="nomic-embed-text:latest", input=rels)
                batch_vecs = batch_resp.get("embeddings") if batch_resp else None
            except (OSError, ConnectionError, TimeoutError, ValueError):
                batch_vecs = None

            if batch_vecs and len(batch_vecs) == len(candidates):
                prompt_norm = np.linalg.norm(prompt_vec) + 1e-8
                for fpath, vec in zip(candidates, batch_vecs, strict=False):
                    try:
                        file_vec = np.array(vec)
                        sim = float(np.dot(prompt_vec, file_vec) / (prompt_norm * (np.linalg.norm(file_vec) + 1e-8)))
                        scored.append((fpath, sim))
                    except (OSError, ConnectionError, TimeoutError, ValueError):
                        continue
            else:
                # Fallback: serial per-path (original behavior) if batch response was short/missing
                for fpath, rel in zip(candidates, rels, strict=False):
                    try:
                        file_resp = ollama.embed(model="nomic-embed-text:latest", input=rel)
                        if file_resp and file_resp.get("embeddings"):
                            file_vec = np.array(file_resp["embeddings"][0])
                            sim = float(np.dot(prompt_vec, file_vec) / (np.linalg.norm(prompt_vec) * np.linalg.norm(file_vec) + 1e-8))
                            scored.append((fpath, sim))
                    except (OSError, ConnectionError, TimeoutError, ValueError):
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
                except (OSError, ConnectionError, TimeoutError, ValueError):
                    continue

            if parts and len(system_prompt) < 30000:
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
            PLAN_GENERATION_PROMPT,
            parse_plan_from_llm,
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

    def run(
        self,
        prompt: str,
        on_tool_call=None,
        on_response=None,
        steering_queue=None,
        on_chunk=None,
        on_tool_start=None,
        on_event=None,
        images: Optional[list] = None,
    ) -> dict:
        """Run the agentic loop until completion.

        Args:
            prompt: User's task/prompt
            on_tool_call: Callback(tool_name, args, result) for UI updates
            on_response: Callback(text, iteration) for streaming text
            steering_queue: Optional SteeringQueue for mid-turn user messages
            on_chunk: Callback(text) for live token streaming
            on_tool_start: Callback(tool_name, args) fired before tool execution
            on_event: Callback(LoopEvent) for structured loop events

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
        self._empty_response_count = 0
        self._thinking_nudge_count = 0
        self._current_run_id = f"run_{uuid.uuid4().hex[:8]}"
        self._current_run_prompt = prompt  # used by Intent-to-Code Ledger
        # Turn-scoped atomic rollback. Populated lazily on the first edit of
        # the turn so runs with no edits pay no cost. Reset at the top of every
        # run(); restored if verification fails and AURA.md allows it.
        self._turn_checkpoint_ids: list[str] = []
        self._turn_snapshotted_paths: set[str] = set()
        # Cognitive heatmap accumulators (Phase 9)
        if not hasattr(self, "_tokens_by_tool") or self._tokens_by_tool is None:
            self._tokens_by_tool = {}
        if not hasattr(self, "_tokens_by_file") or self._tokens_by_file is None:
            self._tokens_by_file = {}
        self._active_tool_for_heatmap = None
        self._active_file_for_heatmap = None
        # Let ToolExecutor read back-refs (for ledger, heatmap, etc.)
        try:
            self.executor._agentic_loop = self
        except (OSError, ConnectionError, TimeoutError, ValueError):
            pass
        # Let brain find the active loop so _record_tokens can update heatmap
        try:
            self.brain._active_agentic_loop = self
        except (OSError, ConnectionError, TimeoutError, ValueError):
            pass

        # Capture baseline cost for per-turn cost tracking (C2)
        _prev_cost = 0.0
        try:
            _prev_cost = self.brain.get_session_stats().get("cost_usd", 0.0)
        except (OSError, ConnectionError, TimeoutError, ValueError):
            logger.debug("Failed to read baseline cost for per-turn tracking", exc_info=True)

        # Reset cancellation for this run
        self._cancel_event.clear()

        # Reset Reflexion state so the one-shot per-run fire counter is fresh
        self._tool_failure_streak = 0
        self._reflexion_fired_this_run = False

        # Wire in loop guard to prevent infinite tool cycles
        from aura.reliability.loop_guard import get_guard
        _session_id = self.session.session_id if self.session else "default"
        guard = get_guard(_session_id)
        guard.reset()

        def _persist_loop_event(event) -> None:
            if self.session and event.type != "chunk":
                self.session.append_event(
                    {
                        "type": event.type,
                        "run_id": event.run_id,
                        "iteration": event.iteration,
                        "payload": event.payload,
                    }
                )

        event_emitter = LoopEventEmitter(
            self,
            on_emit=_persist_loop_event,
            on_event=on_event,
            on_tool_call=on_tool_call,
            on_response=on_response,
            on_chunk=on_chunk,
            on_tool_start=on_tool_start,
        )

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
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
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
                            _loop_support.console.print(
                                f"  [dim cyan]plan[/dim cyan] {len(plan.steps)} steps generated",
                                highlight=False,
                            )
                        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                            logger.debug(f"[AgenticLoop] Plan console print failed: {e}")
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
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
        except (OSError, ConnectionError, TimeoutError, ValueError):
            logger.debug("CorrectionTracker injection failed", exc_info=True)

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

        _user_msg = {"role": "user", "content": prompt}
        if images:
            _user_msg["images"] = list(images)
        messages.append(_user_msg)

        final_response = ""
        model_used = ""
        outcome: LoopOutcome | None = None

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
                        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                            logger.debug(f"[AgenticLoop] Store interaction failed: {e}")
                        self._current_action_mode = None
                        result = LoopOutcome.visual_feedback(final_response).to_result_dict(
                            iterations=1,
                            tool_calls=0,
                            model=model_used,
                        )
                        event_emitter.emit(
                            "run_finished",
                            status=result["status"],
                            success=result["success"],
                            response=final_response,
                            model=model_used,
                            tool_calls=0,
                        )
                        return result
            except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                logger.warning(f"[AgenticLoop] Visual feedback failed, falling back to normal: {e}")

        while self.iteration < self.max_iterations:
            # ── Cancellation check: top of iteration ──
            if self._cancel_event.is_set():
                outcome = LoopOutcome.cancelled("Cancelled by user.")
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
            except (OSError, ConnectionError, TimeoutError, ValueError) as e:
                logger.debug(f"[AgenticLoop] Planner tick/replan failed (non-fatal): {e}")

            # Budget check with intermediate warnings (C1)
            if self.budget_usd is not None:
                stats = self.brain.get_session_stats()
                cost = stats.get("cost_usd", 0.0)
                pct = cost / self.budget_usd if self.budget_usd > 0 else 0

                if cost >= self.budget_usd:
                    outcome = LoopOutcome.budget_limit(self.budget_usd)
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

            step_result = self._model_step_controller.request_step(
                messages=messages,
                active_tools=active_tools,
                step_model=step_model,
                event_emitter=event_emitter,
            )

            if step_result.status == "terminal":
                outcome = step_result.outcome
                break

            tool_calls = step_result.tool_calls
            content = step_result.content
            if step_result.model_used:
                model_used = step_result.model_used

            if step_result.status == "content_ready":
                content_result = self._model_step_controller.resolve_content_only(
                    prompt=prompt,
                    content=content,
                    delivery=step_result.delivery,
                )
                if content_result.status == "terminal":
                    outcome = content_result.outcome
                    if (
                        outcome
                        and outcome.status == "completed"
                        and event_emitter.listens_for("response")
                        and step_result.delivery == "blocking"
                    ):
                        event_emitter.emit(
                            "response",
                            text=outcome.response,
                            delivery=step_result.delivery,
                        )
                    break
                if content_result.extra_messages:
                    messages.extend(content_result.extra_messages)
                    continue
                outcome = LoopOutcome.completed(content_result.content or content)
                if event_emitter.listens_for("response"):
                    event_emitter.emit(
                        "response",
                        text=outcome.response,
                        delivery=step_result.delivery,
                    )
                break

            # Reset empty response counter — tool calls are productive activity
            self._empty_response_count = 0

            # Append assistant message with tool_calls to history
            self._tool_call_coordinator.append_assistant_tool_message(
                messages,
                content,
                tool_calls,
                event_emitter=event_emitter,
            )
            parsed_calls = self._tool_call_coordinator.parse_tool_calls(tool_calls, messages)
            approved = self._tool_call_coordinator.approve_and_execute(
                parsed_calls,
                event_emitter=event_emitter,
            )

            # ── Cancellation check: before tool execution ──
            if self._cancel_event.is_set():
                outcome = LoopOutcome.cancelled(
                    f"Cancelled after {self.iteration} iterations."
                )
                break

            batch_result = self._tool_call_coordinator.collect_results(
                approved,
                messages,
                guard,
                event_emitter=event_emitter,
            )

            # Reflexion hook: if this iteration's tool calls had errors, bump
            # the failure streak. Two consecutive failing iterations trigger
            # a one-shot diagnosis that gets injected as a system nudge so
            # the next iteration sees concrete next-action advice.
            self._maybe_reflect_on_failures(approved, messages, event_emitter=event_emitter)

            if batch_result.should_break:
                outcome = batch_result.outcome
                break

            # Verification stage: after processing all tool calls in this iteration,
            # if any edits were made, run typecheck/tests scoped to changed files
            # and feed failures back so the agent self-corrects on the next turn.
            # This replaces the legacy _run_auto_test callback.
            if self._edits_this_turn > 0:
                verification_msg = self._run_verification_stage(
                    event_emitter=event_emitter,
                )
                if verification_msg:
                    self._has_test_failure = True
                    messages.append({
                        "role": "user",
                        "content": verification_msg,
                    })
                else:
                    # Successful edit + green verification is forward progress.
                    try:
                        from aura.reliability.loop_guard import get_guard
                        sid = getattr(self.session, "session_id", "") if self.session else ""
                        if sid:
                            get_guard(sid).note_progress()
                    except (OSError, ConnectionError, TimeoutError, ValueError):
                        logger.debug("loop_guard note_progress failed", exc_info=True)

        else:
            # Max iterations reached
            outcome = LoopOutcome.max_iterations(
                self.max_iterations,
                outcome.response if outcome else final_response,
            )

        if outcome is None:
            outcome = LoopOutcome.completed(final_response)
        final_response = outcome.response

        # Save user message and final assistant response to conversation history and session
        _hist_user = {"role": "user", "content": prompt}
        if images:
            _hist_user["images"] = list(images)
        self._conversation_history.append(_hist_user)
        if self.session is not None:
            self.session.append(_hist_user)
        if final_response:
            self._conversation_history.append({"role": "assistant", "content": final_response})
            if self.session is not None:
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
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
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
        except (OSError, ConnectionError, TimeoutError, ValueError):
            self.last_turn_cost = 0.0

        result = outcome.to_result_dict(
            iterations=self.iteration,
            tool_calls=self.tool_calls_total,
            model=model_used,
        )
        event_emitter.emit(
            "run_finished",
            status=result["status"],
            success=result["success"],
            response=final_response,
            model=model_used,
            tool_calls=self.tool_calls_total,
        )
        return result

    def _maybe_reflect_on_failures(
        self,
        approved: list,
        messages: list[dict],
        event_emitter=None,
    ) -> None:
        """Trigger a Reflexion-style diagnosis after repeated tool failures.

        Counts errors in the just-finished iteration's `approved` tool list.
        If the streak hits 2 and Reflexion is enabled in Config, ask the
        brain to diagnose why and prepend the diagnosis as a system message
        so the next iteration sees it. Fires at most once per run to avoid
        cost spirals when the agent is stuck on a genuinely impossible task.
        """
        try:
            from aura.config import Config
            if not getattr(Config, "REFLEXION_ENABLED", False):
                return
        except (OSError, ConnectionError, TimeoutError, ValueError):
            return

        if self._reflexion_fired_this_run:
            return

        # Count failures in this iteration's approved list
        had_failure = False
        for _tool_name, _args, tool_result in approved:
            if self._tool_result_has_error(tool_result):
                had_failure = True
                break

        if had_failure:
            self._tool_failure_streak += 1
        else:
            self._tool_failure_streak = 0
            return

        if self._tool_failure_streak < 2:
            return

        # Build a compact reflection prompt — just the last 2-3 iterations
        # of tool calls + results, asking the brain to diagnose.
        recent = []
        for m in messages[-8:]:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            if not content:
                continue
            recent.append(f"{role}: {str(content)[:300]}")
        recent_str = "\n".join(recent[-6:])

        prompt = (
            "You are diagnosing why the agent's last two tool calls failed.\n\n"
            "Recent history:\n"
            f"{recent_str}\n\n"
            "Give a 2-3 sentence diagnosis of the root cause and a concrete "
            "next action the agent should try. Be specific about paths, "
            "commands, or arguments that should change."
        )

        try:
            reflection = self.brain.think(
                prompt,
                system_prompt="You are a concise debugging coach.",
                use_history=False,
            )
            if isinstance(reflection, dict):
                reflection = reflection.get("response") or reflection.get("content") or ""
            reflection = (reflection or "").strip()
            if not reflection:
                return
        except Exception as exc:
            logger.debug(f"[Reflexion] Diagnosis call failed: {exc}")
            return

        self._reflexion_fired_this_run = True

        # Inject as a system nudge the next iteration will see
        messages.append({
            "role": "system",
            "content": f"[Reflexion] {reflection[:500]}",
        })

        # Emit an event so the UI can surface it
        if event_emitter is not None and hasattr(event_emitter, "emit"):
            try:
                event_emitter.emit("reflexion", text=reflection[:500])
            except (OSError, ConnectionError, TimeoutError, ValueError):
                logger.debug("Failed to emit reflexion event", exc_info=True)

        logger.info(f"[Reflexion] fired after {self._tool_failure_streak} failures")

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
            # Fallback: heuristic for non-JSON results.  Don't fire on every
            # occurrence of the word "error" inside a code snippet or doc.
            return tool_result.strip().startswith("Error:") or tool_result.strip().startswith("Traceback")

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
            _loop_support.console.print("  [dim cyan]verify[/dim cyan] checking completion...", highlight=False)

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
                _loop_support.console.print(f"  [yellow]incomplete[/yellow] {reason[:120]}", highlight=False)
                logger.info(f"[AgenticLoop] Verification: INCOMPLETE — {reason[:200]}")
                return reason
            else:
                _loop_support.console.print("  [green]verified[/green] task complete", highlight=False)
                logger.info("[AgenticLoop] Verification: COMPLETE")
                return None

        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            # Verification failure must not break the loop
            logger.debug(f"[AgenticLoop] Completion verification failed (non-fatal): {e}")
            return None

    def _ensure_turn_checkpoint(self, paths: list[str]) -> None:
        """Snapshot *paths* into the turn-scoped checkpoint chain, skipping any
        already captured this turn. Called BEFORE each edit tool so the stored
        content reflects the pre-edit state. Cheap on repeated calls because
        already-snapshotted paths are tracked in `_turn_snapshotted_paths`.
        """
        cp_mgr = getattr(self, "_checkpoint_mgr", None)
        if cp_mgr is None or not paths:
            return
        new_paths = [p for p in paths if p and p not in self._turn_snapshotted_paths]
        if not new_paths:
            return
        try:
            cp_id = cp_mgr.snapshot_multi(new_paths, label=f"turn:{self._current_run_id}")
            self._turn_checkpoint_ids.append(cp_id)
            for p in new_paths:
                self._turn_snapshotted_paths.add(p)
        except (OSError, ConnectionError, TimeoutError, ValueError):
            logger.debug("_ensure_turn_checkpoint failed", exc_info=True)

    def _rollback_turn(self, event_emitter: Any = None) -> dict:
        """Restore every turn-scope checkpoint taken this run.

        Restores in REVERSE order so paths first touched (captured in the
        earliest checkpoint) win — that's the pre-turn truth. Returns a dict
        describing what happened so the caller can surface it to the agent
        and render a UI event.
        """
        cp_mgr = getattr(self, "_checkpoint_mgr", None)
        ids = list(self._turn_checkpoint_ids)
        result = {
            "attempted": len(ids),
            "restored": 0,
            "failed": 0,
            "paths": sorted(self._turn_snapshotted_paths),
            "partial": False,
        }
        if cp_mgr is None or not ids:
            return result
        # Reverse: later checkpoints first. A file that was first snapshotted
        # in the first checkpoint (the pre-turn state) will be restored last,
        # overwriting any later-checkpoint content with the original.
        for cp_id in reversed(ids):
            try:
                if cp_mgr.restore(cp_id):
                    result["restored"] += 1
                else:
                    result["failed"] += 1
            except (OSError, ConnectionError, TimeoutError, ValueError):
                logger.debug("_rollback_turn restore failed for %s", cp_id, exc_info=True)
                result["failed"] += 1
        result["partial"] = result["failed"] > 0
        # Clear turn state so subsequent runs start fresh. The individual
        # per-edit checkpoints remain in the CheckpointManager index, so
        # /rewind can still reach them if the user wants to inspect.
        self._turn_checkpoint_ids.clear()
        self._turn_snapshotted_paths.clear()
        if event_emitter is not None:
            try:
                event_emitter.emit(
                    "turn_rolled_back",
                    attempted=result["attempted"],
                    restored=result["restored"],
                    failed=result["failed"],
                    paths=result["paths"],
                    partial=result["partial"],
                )
            except (OSError, ConnectionError, TimeoutError, ValueError):
                logger.debug("turn_rolled_back emit failed", exc_info=True)
        return result

    def _clear_turn_checkpoint(self) -> None:
        """Discard the turn checkpoint bookkeeping without restoring. Called on
        successful verification so we don't accidentally roll back on a later
        unrelated failure."""
        self._turn_checkpoint_ids.clear()
        self._turn_snapshotted_paths.clear()

    def _run_verification_stage(self, event_emitter: Any = None) -> Optional[str]:
        """Run the VerificationStage for this turn's edits.

        Returns a conversation-injectable message on failure (so the next
        iteration sees the errors), or None on success/skipped.
        """
        try:
            # Lazy init — the first edit-bearing iteration builds the stage from
            # the loop's aura_config so later iterations reuse it.
            if getattr(self, "_verification_stage", None) is None:
                from aura.core.verification_stage import VerificationStage
                self._verification_stage = VerificationStage(
                    project_root=self.project_root,
                    aura_config=self._aura_config,
                    shell_tool=self.executor.shell if getattr(self, "executor", None) else None,
                )

            stage = self._verification_stage
            # Scope: files touched this turn. Fall back to _hot_files (recent
            # LRU) so multi-edit turns get a complete picture.
            changed = list(getattr(self, "_edited_files_this_turn", [])) or list(
                getattr(self, "_hot_files", [])
            )
            if not changed:
                return None

            sid = getattr(self.session, "session_id", "") if self.session else ""
            outcome = stage.run(changed, emitter=event_emitter, session_id=sid)

            # Surface a line in the console so the user sees it even if the
            # UI layer isn't rendering verification events yet.
            try:
                _ensure_console()
                dur = f"{outcome.duration_s:.1f}s"
                if outcome.success:
                    if outcome.mode != "none" and outcome.stages:
                        _loop_support.console.print(
                            f"  [green]verification[/green] passed · {outcome.mode} · {dur}",
                            highlight=False,
                        )
                else:
                    _loop_support.console.print(
                        f"  [red]verification[/red] failed · {outcome.mode} · {dur}",
                        highlight=False,
                    )
            except (OSError, ConnectionError, TimeoutError, ValueError):
                logger.debug("verification console print failed", exc_info=True)

            if outcome.success:
                # Discard the turn checkpoint — no rollback needed and we don't
                # want a later unrelated failure to trigger it.
                self._clear_turn_checkpoint()
                return None

            # Verification failed. If the config allows, roll back all edits
            # in this turn to pre-turn state so the agent re-plans from a
            # clean slate instead of trying to patch broken code.
            msg = outcome.to_conversation_message()
            rollback_enabled = True
            try:
                rollback_enabled = bool(
                    (self._aura_config or {})
                    .get("verification", {})
                    .get("rollback_on_failure", True)
                )
            except (OSError, ConnectionError, TimeoutError, ValueError):
                pass
            if rollback_enabled and self._turn_checkpoint_ids:
                rb = self._rollback_turn(event_emitter=event_emitter)
                suffix_lines = [
                    "",
                    f"[Rollback] Restored {rb['restored']}/{rb['attempted']} "
                    f"checkpoint(s). {len(rb['paths'])} file(s) reverted to "
                    f"pre-turn state.",
                ]
                if rb["partial"]:
                    suffix_lines.append(
                        "[Rollback] WARNING: partial restore — some files "
                        "may be in an inconsistent state. Inspect before "
                        "continuing."
                    )
                suffix_lines.append(
                    "[Rollback] Your previous edits are discarded. Re-plan "
                    "from current state before attempting the fix."
                )
                msg = msg + "\n" + "\n".join(suffix_lines)
            return msg
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
            logger.debug(f"[AgenticLoop] Verification stage error (non-fatal): {e}")
            return None

    def _run_auto_test(self) -> Optional[str]:
        """Run project tests + lint/typecheck after edits. Returns output for LLM or None."""
        try:
            from aura.tools.auto_verify import auto_verify, lint_and_typecheck
            _ensure_console()
            test_cmd = (self._aura_config or {}).get("test_cmd")
            lint_cmd = (self._aura_config or {}).get("lint_cmd")
            typecheck_cmd = (self._aura_config or {}).get("typecheck_cmd")

            _loop_support.console.print("  [dim cyan]auto[/dim cyan] running tests...", highlight=False)
            result = auto_verify(self.project_root, self.executor.shell, test_cmd_override=test_cmd)

            # Tests first — if they fail, return immediately
            if not result.get("skipped") and not result.get("success"):
                output = result.get("output", "Tests failed")
                cmd = result.get("test_command", "tests")
                _loop_support.console.print(f"  [red]tests failed[/red] ({cmd})", highlight=False)
                return json.dumps({
                    "auto_test_result": "FAILED",
                    "test_command": cmd,
                    "output": output[:3000],
                    "instruction": "Tests failed after your edit. Please read the error output and fix the issue.",
                })

            if result.get("skipped"):
                pass  # fall through to lint/typecheck
            else:
                _loop_support.console.print("  [green]tests passed[/green]", highlight=False)

            # Tests passed (or no runner) — run lint + typecheck
            _loop_support.console.print("  [dim cyan]auto[/dim cyan] lint + typecheck...", highlight=False)
            lr = lint_and_typecheck(
                self.project_root, self.executor.shell,
                lint_cmd_override=lint_cmd, typecheck_cmd_override=typecheck_cmd,
            )

            if lr.get("skipped") or lr.get("success"):
                if not lr.get("skipped"):
                    _loop_support.console.print("  [green]lint/typecheck passed[/green]", highlight=False)
                return None

            # Lint or typecheck failed
            fail_blocks = []
            for key in ("lint", "typecheck"):
                sub = lr.get(key)
                if sub and not sub.get("success"):
                    fail_blocks.append(f"[{key}] {sub.get('command')}\n{sub.get('output', '')[:2000]}")
            _loop_support.console.print("  [red]lint/typecheck failed[/red]", highlight=False)
            return json.dumps({
                "auto_test_result": "FAILED",
                "test_command": "lint/typecheck",
                "output": "\n\n".join(fail_blocks)[:3000],
                "instruction": "Lint or type errors after your edit. Please fix them.",
            })
        except (OSError, ConnectionError, TimeoutError, ValueError) as e:
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
            _loop_support.console.print(f"  [red]DENIED[/red] {tool_name}", highlight=False)
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

        _loop_support.console.print(f"  [dim cyan]tool[/dim cyan] {desc}", highlight=False)


def run_agentic(
    brain,
    prompt: str,
    project_root: str = ".",
    permissions: Optional[PermissionManager] = None,
    model_override: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
    budget_usd: Optional[float] = None,
    context: str = "",
    trust_mode: bool = False,
    aura_config: dict | None = None,
    router=None,
    resume_session_id: Optional[str] = None,
    on_response=None,
    on_chunk=None,
    on_tool_start=None,
    on_tool_call=None,
    on_event=None,
) -> dict:
    """Convenience function to run a single agentic task.

    Callers can pass any of the event callbacks (on_chunk, on_tool_start,
    on_tool_call, on_event, on_response) to subscribe to the loop's event
    stream. If on_response is None, the default prints the response to the
    shared console.

    When resume_session_id is provided, the loop loads that prior session's
    message history before executing the prompt — matching what ChatSession
    does for --resume in interactive mode.
    """
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

    if resume_session_id:
        try:
            loop.load_session(resume_session_id)
        except (OSError, ConnectionError, TimeoutError, ValueError):
            logger.debug("resume_session_load_failed", exc_info=True)

    if on_response is None:
        def on_response(text, iteration):
            _loop_support.console.print(f"\n{text}\n")

    return loop.run(
        prompt,
        on_response=on_response,
        on_chunk=on_chunk,
        on_tool_start=on_tool_start,
        on_tool_call=on_tool_call,
        on_event=on_event,
    )

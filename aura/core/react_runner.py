"""ReAct loop runner mixin — system prompt, tool schemas, tool execution, react step.

Extracted from agent.py (2026-04-06) to reduce class size.
All methods assume self has: brain, tools, memory, monologue, metacognition,
max_iterations, tool_rag, kg_bridge, adaptive_planner, _tool_executor,
_REACT_TOOL_MODEL, _REACT_CODE_MODEL, _REACT_REASON_MODEL.
"""

import json
import logging

logger = logging.getLogger(__name__)

# Constants (mirrored from agent.py to avoid circular imports)
MAX_RESULT = 8000   # Tool result truncation limit
TOOL_TIMEOUT = 30


class ReactMixin:
    """Mixin providing ReAct loop helpers: prompt building, tool execution, react step."""

    # Cloud models known to support tool calling well, in preference order
    _REACT_TOOL_MODEL = "glm-5:cloud"         # Fast, reliable tool calling
    _REACT_CODE_MODEL = "deepseek-v3.2:cloud"  # Better for code tasks
    _REACT_REASON_MODEL = "kimi-k2.5:cloud"    # Best for complex reasoning

    # =================================================================
    # ReAct Loop Helper Methods
    # =================================================================

    def _build_react_system_prompt(self, goal: str, ace_context: str = "") -> str:
        """Build system prompt for ReAct loop with memory, identity, emotion.

        Keeps total prompt compact for tool-calling mode where context is precious.
        Identity + emotion are capped at 1500 chars combined.
        """
        parts = []

        # Identity (truncate for tool-calling mode)
        from aura.identity import get_identity_prompt
        identity_text = get_identity_prompt()
        if len(identity_text) > 1200:
            identity_text = identity_text[:1200] + "..."
        parts.append(identity_text)

        # ALMA emotional tone (truncate to keep prompt lean)
        try:
            from aura.emotion.integration import get_emotional_tone_modifier
            tone = get_emotional_tone_modifier()
            if tone:
                parts.append(tone[:300])
        except (ImportError, AttributeError, KeyError, TypeError) as _tone_err:
            logger.debug(f"[Agent] Emotional tone modifier unavailable: {_tone_err}")

        # ACE context
        if ace_context:
            parts.append(ace_context)

        # UserProfile injection
        try:
            from aura.memory.user_profile import load_profile
            _profile = load_profile()
            _profile_str = _profile.to_system_prompt()
            if _profile_str:
                parts.append(_profile_str)
        except (ImportError, AttributeError, OSError, ValueError) as _profile_err:
            logger.debug(f"[Agent] User profile load failed: {_profile_err}")

        # Memory recall (parallel) — unified store + KG
        try:
            from aura.pools import bg_pool as _bg_pool_fn
            _AGENT_EXECUTOR = _bg_pool_fn()
            from aura.memory.unified_memory import get_unified_memory
            _ctx_futures = {}
            _umem = get_unified_memory()
            _ctx_futures["memory"] = _AGENT_EXECUTOR.submit(_umem.query, goal, 3)
            if self.kg_bridge is not None:
                _ctx_futures["kg"] = _AGENT_EXECUTOR.submit(self.kg_bridge.get_context_for_query, goal, 5)

            for _key, _fut in _ctx_futures.items():
                try:
                    _result = _fut.result(timeout=2.0)
                    if _key == "memory" and _result:
                        mem_text = "\n".join(r.content[:150] for r in _result[:3])
                        parts.append(f"[Relevant memories]\n{mem_text}")
                    elif _key == "kg" and _result:
                        parts.append(f"[Knowledge context]\n{str(_result)[:300]}")
                except (AttributeError, KeyError, TypeError, ValueError, TimeoutError, OSError) as _ctx_err:
                    logger.debug(f"[Agent] {_key} context retrieval failed: {_ctx_err}")
        except (AttributeError, TypeError, OSError) as _mem_err:
            logger.debug(f"[Agent] Memory/KG context setup failed: {_mem_err}")

        # ReAct instructions (plan-aware when a plan exists)
        react_instruction = (
            "You are an AI agent with access to tools. "
            "Use the provided tools to accomplish the user's goal. "
            "When done, respond with your final answer (no tool call). "
            "Be concise and direct.\n\n"
            "IMPORTANT: If the user asks about something you are not sure about, "
            "something recent, current events, news, real-time data (dates, prices, "
            "weather, scores, stock prices, exchange rates), or asks you to look "
            "something up or verify information — USE the web_search or tavily tool "
            "to search the internet FIRST. Do NOT guess or hallucinate. Always "
            "verify uncertain facts by searching."
        )
        if self.adaptive_planner and self.adaptive_planner.current_plan:
            react_instruction += (
                "\n\nYou have a plan for this task. Follow the plan steps in order. "
                "Reference the current step in your reasoning. "
                "If a step is blocked or unnecessary, skip it and move on."
            )
        parts.append(react_instruction)

        return "\n\n".join(parts)

    def _build_tool_schemas(self, goal: str) -> list:
        """Select relevant tool schemas for this goal.

        Always includes the 11 core dev tools (read_file, grep, list_dir, etc.)
        plus up to 4 additional agent-specific tools from Tool RAG.
        """
        try:
            from aura.core.tool_schemas import AGENTIC_TOOLS
            core_schemas = list(AGENTIC_TOOLS)
            core_names = {s["function"]["name"] for s in core_schemas}
        except ImportError:
            return []

        # Add a few agent-specific tools from RAG if relevant
        if self.tool_rag and self.tool_rag._schemas_loaded:
            extra = self.tool_rag.select_tools(goal, k=6)
            for schema in extra:
                name = schema["function"]["name"]
                if name not in core_names:
                    core_schemas.append(schema)
                    core_names.add(name)
                    if len(core_schemas) >= 15:  # Cap at 15 total tools
                        break

        return core_schemas

    def _execute_tool_call(self, tool_name: str, args: dict) -> str:
        """Execute a tool call and return result as string for the LLM.

        Priority:
        1. ToolExecutor (handles dev tools: read_file, grep, list_dir, shell, etc.)
           Also resolves aliases (filesystem→list_dir, code_search→grep, etc.)
        2. Agent tools (self.tools dict — screenshot, vision, search, etc.)
        3. Web search fallback chain (tavily → brave → web_search)
        """
        MAX_RESULT = 8000

        def _safe_truncate_json(obj, limit: int = MAX_RESULT) -> str:
            """Serialize and truncate without producing invalid JSON."""
            raw = json.dumps(obj, default=str)
            if len(raw) <= limit:
                return raw
            # Truncate the string representation and wrap as valid JSON
            return json.dumps({"result": raw[:limit - 60], "_truncated": True})

        # 1. Try ToolExecutor first — handles dev tools + aliases + sandbox bypass
        if self._tool_executor:
            result = self._tool_executor.execute(tool_name, args)
            # Check if executor returned an "Unknown tool" error — if so, fall through
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict) and parsed.get("error", "").startswith("Unknown tool"):
                        pass  # Fall through to agent tools
                    else:
                        return result[:MAX_RESULT] if len(result) <= MAX_RESULT else json.dumps({"result": result[:MAX_RESULT - 60], "_truncated": True})
                except (json.JSONDecodeError, ValueError):
                    return result[:MAX_RESULT]

        # 2. Agent tools dispatch (try loaded tools, then deferred registry)
        tool = self.tools.get(tool_name)
        if tool is None:
            # Try loading from deferred registry
            try:
                from aura.tools.deferred_registry import deferred_registry
                if deferred_registry and hasattr(deferred_registry, 'ensure_tool'):
                    tool = deferred_registry.ensure_tool(tool_name)
                    if tool:
                        self.tools[tool_name] = tool  # Cache for next call
            except Exception:
                pass
        if tool is not None:
            try:
                action = args.get("action", "") if isinstance(args, dict) else str(args)
                if not action:
                    # Build action string from structured args
                    action = " ".join(f"{v}" for v in args.values()) if isinstance(args, dict) else str(args)
                if hasattr(tool, 'execute'):
                    result = tool.execute(action)
                    return _safe_truncate_json(result)
            except Exception as e:  # Catch-all: unknown tool implementations may raise anything
                return json.dumps({"error": f"Tool '{tool_name}' failed: {e}"})

        # 3. Web search fallback chain (shared implementation)
        if tool_name in ("search_web", "web_search", "search"):
            query = args.get("query", args.get("action", str(args)))
            from aura.tools.search_fallback import web_search_with_fallback
            result = web_search_with_fallback(query=query, tool_registry=self.tools)
            return _safe_truncate_json(result)

        return json.dumps({"error": f"No handler for tool: {tool_name}"})

    def _pick_react_model(self, messages: list, iteration: int) -> str:
        """Pick model for this ReAct iteration.

        User's explicit model override always wins. Otherwise auto-route
        based on whether the conversation involves code edits.
        """
        # User explicitly selected a model — respect it
        if self.brain._model_override:
            return self.brain._model_override

        has_edits = False
        for msg in messages:
            for tc in msg.get("tool_calls", []) or []:
                fn_name = tc.get("function", {}).get("name", "")
                if fn_name in ("edit_file", "write_file"):
                    has_edits = True
                    break

        if has_edits:
            return self._REACT_CODE_MODEL
        return self._REACT_TOOL_MODEL

    def _store_episode(self, goal: str, response: str):
        """Store goal+response in memory after the loop ends. No LLM call."""
        if not response:
            return
        try:
            if self.memory:
                self.memory.store(
                    content=f"Goal: {goal[:200]}\nResult: {response[:300]}",
                    source="react_loop",
                    importance=0.5,
                )
        except (AttributeError, TypeError, OSError) as e:
            logger.debug(f"[AGENT] Episode storage failed: {e}")

    # =================================================================
    # ReAct Step — single-step agent method (Roadmap 1.1)
    # =================================================================

    def _react_step(
        self,
        messages: list,
        tool_schemas: list,
        iteration: int,
        state_hashes: set,
        consecutive_failures: int,
    ) -> dict:
        """Execute one ReAct step: LLM call + tool execution + deterministic evaluation.

        This replaces the old 4-phase OPAE loop (observe/plan/act/evaluate) with a
        single LLM call that combines thought + action, followed by deterministic
        evaluation of the tool result (no LLM call for eval).

        Args:
            messages: Conversation history (mutated in-place with new messages)
            tool_schemas: Tool schemas for Ollama tool calling
            iteration: Current iteration number
            state_hashes: Set of seen action hashes (mutated in-place)
            consecutive_failures: Current failure count

        Returns:
            {
                "status": "done" | "continue" | "error" | "incomplete",
                "response": str (final answer when done, error message when error),
                "tool_calls_count": int,
                "consecutive_failures": int,
            }
        """
        # Pick model for this step
        step_model = self._pick_react_model(messages, iteration)

        # === Single LLM call (thought + action combined) ===
        step = self.brain.react_step(messages, tool_schemas, model_override=step_model)

        # Handle LLM error with fallback chain
        if "error" in step:
            logger.error(f"[REACT] LLM error: {step['error']}")
            _fallback_models = ["glm-5:cloud", "deepseek-v3.2:cloud", "kimi-k2.5:cloud"]
            _fallback_models = [m for m in _fallback_models if m != step_model]
            for _fb_model in _fallback_models:
                logger.info(f"[REACT] Trying fallback model: {_fb_model}")
                step = self.brain.react_step(messages, tool_schemas, model_override=_fb_model)
                if "error" not in step:
                    break
            if "error" in step:
                return {"status": "error", "response": f"I encountered an error: {step['error']}",
                        "tool_calls_count": 0, "consecutive_failures": consecutive_failures}

        thought = step.get("thought", "")
        tool_calls = step.get("tool_calls", [])

        # Record thought in inner monologue
        if thought and self.monologue:
            self.monologue.think("reason", thought[:200])

        # === No tool calls: LLM is done, content is the final answer ===
        if step.get("done"):
            _stripped = (thought or "").strip()
            _looks_incomplete = (
                len(_stripped) < 20
                and _stripped
                and _stripped[-1] not in '.!?'
                and iteration < self.max_iterations - 1
            )
            if _looks_incomplete:
                # Nudge the LLM to provide a real answer or use a tool
                messages.append({"role": "assistant", "content": thought})
                messages.append({
                    "role": "user",
                    "content": "Please provide your complete answer, or use a tool if you need more information.",
                })
                return {"status": "incomplete", "response": "",
                        "tool_calls_count": 0, "consecutive_failures": consecutive_failures}

            return {"status": "done", "response": step.get("final_answer", thought),
                    "tool_calls_count": 0, "consecutive_failures": consecutive_failures}

        # === Has tool calls: serialize message, execute tools, evaluate deterministically ===

        # Build assistant message for conversation history
        msg_dict = {
            "role": "assistant",
            "content": thought,
            "tool_calls": [
                {"function": {"name": tc["name"], "arguments": tc["args"]}}
                for tc in tool_calls
            ],
        }
        messages.append(msg_dict)

        # Execute each tool call
        calls_this_step = 0
        for tc in tool_calls:
            tool_name = tc["name"]
            args = tc["args"]

            # Loop guard: state dedup using raw string keys (hash() can collide)
            try:
                state_key = f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}"
            except (TypeError, ValueError):
                state_key = f"{tool_name}:{args!s}"
            if state_key in state_hashes:
                messages.append({
                    "role": "tool",
                    "content": json.dumps({"error": "Duplicate action. Try a different approach."}),
                })
                continue
            state_hashes.add(state_key)

            # Execute tool
            tool_result = self._execute_tool_call(tool_name, args)
            calls_this_step += 1
            messages.append({"role": "tool", "content": tool_result})

            # Audit chain: log every tool call (OpenFang-inspired Merkle trail)
            try:
                from aura.security.audit_chain import get_audit_chain
                get_audit_chain().append(
                    action_type="tool_call",
                    action_data={"tool": tool_name, "args_preview": str(args)[:200]},
                    agent_id="main",
                    session_id=getattr(self, '_session_id', 'default'),
                )
            except (ImportError, AttributeError, TypeError, OSError) as _audit_err:
                logger.debug(f"[AuditChain] Append failed: {_audit_err}")

            # === Deterministic evaluation (no LLM call) ===
            eval_result = self._evaluate_tool_result(tool_result)

            if not eval_result["success"]:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({
                            "warning": f"Multiple failures ({consecutive_failures}). Try a different tool or approach.",
                        }),
                    })
                    # Don't reset to 0 — let the outer loop's >= 3 check work.
                    # The warning nudges the LLM, but persistent failures still abort.
            else:
                consecutive_failures = 0

            # Log deterministic eval to metacognition
            self.metacognition.log_evaluation(
                tool=tool_name,
                action=json.dumps(args, default=str)[:200],
                confidence=eval_result["confidence"],
                success=eval_result["success"],
                progress=eval_result["reason"],
                next_step="continue",
                result_summary=tool_result[:500],
                model_used=step.get("model", ""),
            )

        return {
            "status": "continue",
            "response": "",
            "tool_calls_count": calls_this_step,
            "consecutive_failures": consecutive_failures,
        }

    @staticmethod
    def _evaluate_tool_result(tool_result: str) -> dict:
        """Deterministic evaluation of a tool result — no LLM call.

        Tries to parse the result as JSON first and checks structured keys
        (``success``, ``error``). Falls back to string matching only when
        JSON parsing fails.

        Returns:
            {"success": bool, "confidence": int (0-100), "reason": str}
        """
        # --- Phase 1: Try structured JSON evaluation first ---
        try:
            parsed = json.loads(tool_result)
            if isinstance(parsed, dict):
                # Explicit "success" key is the strongest signal
                if "success" in parsed:
                    if parsed["success"]:
                        if "error" not in parsed:
                            return {"success": True, "confidence": 95, "reason": "Tool reported success"}
                        # success=True but also has error key — trust success flag
                        return {"success": True, "confidence": 80, "reason": "Tool reported success (with warning)"}
                    else:
                        reason = str(parsed.get("error", "Tool reported failure"))[:100]
                        return {"success": False, "confidence": 90, "reason": reason}

                # No "success" key, but has "error" key
                if "error" in parsed:
                    error_msg = str(parsed["error"])[:100]
                    return {"success": False, "confidence": 85, "reason": f"Error: {error_msg}"}

                # Structured dict with no success/error keys — assume OK
                if len(parsed) > 0:
                    return {"success": True, "confidence": 70, "reason": "Structured result (no error keys)"}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # --- Phase 2: Fallback to string matching for non-JSON output ---
        result_lower = tool_result.lower()

        has_traceback = 'traceback' in result_lower or 'exception' in result_lower
        has_not_found = 'not found' in result_lower or 'no such file' in result_lower
        has_permission = 'permission denied' in result_lower or 'access denied' in result_lower
        has_timeout = 'timeout' in result_lower or 'timed out' in result_lower

        if has_traceback:
            return {"success": False, "confidence": 90, "reason": "Exception traceback in output"}
        if has_permission:
            return {"success": False, "confidence": 95, "reason": "Permission denied"}
        if has_timeout:
            return {"success": False, "confidence": 90, "reason": "Operation timed out"}
        if has_not_found:
            return {"success": False, "confidence": 80, "reason": "Resource not found"}

        # No explicit indicators — assume success if we got non-empty output
        if len(tool_result.strip()) > 2:
            return {"success": True, "confidence": 70, "reason": "Non-empty result (no error indicators)"}

        # Empty or minimal output
        return {"success": False, "confidence": 50, "reason": "Empty or minimal output"}

    # =================================================================
    # Code Agent Mode — LLM writes Python code as actions (5.1)
    # =================================================================

    def _should_use_code_agent(self, goal: str) -> bool:
        """Decide whether to use code agent mode for this goal.

        Code agent mode is better for complex multi-step tasks where the LLM
        can write Python loops, conditionals, and compose tool calls in code.
        Standard tool mode remains the default for simple queries.

        Returns True if code agent mode should be used.
        """
        try:
            from aura.core.code_agent import CODE_AGENT_AVAILABLE, should_use_code_agent
        except ImportError:
            return False

        if not CODE_AGENT_AVAILABLE:
            return False

        # Check config flag (allow disabling globally)
        from aura.config import Config
        if not getattr(Config, 'CODE_AGENT_ENABLED', True):
            return False

        return should_use_code_agent(goal)

    def _react_step_code(
        self,
        messages: list,
        code_agent: "CodeAgentMode",
        tool_namespace: dict,
        iteration: int,
        consecutive_failures: int,
    ) -> dict:
        """Execute one ReAct code step: LLM writes Python code, we execute it.

        This is the code-agent-mode equivalent of _react_step(). Instead of
        Ollama structured tool calling, the LLM writes a Python code block
        that calls tool functions directly.

        Args:
            messages: Conversation history (mutated in-place)
            code_agent: CodeAgentMode instance
            tool_namespace: Dict of tool functions for the code sandbox
            iteration: Current iteration number
            consecutive_failures: Current failure count

        Returns:
            Same shape as _react_step():
            {"status": "done"|"continue"|"error", "response": str, ...}
        """
        step_model = self._pick_react_model(messages, iteration)

        # LLM call — produces thought + Python code block (no tools= param)
        step = self.brain.react_step_code(messages, model_override=step_model)

        if "error" in step:
            logger.error(f"[REACT-CODE] LLM error: {step['error']}")
            return {
                "status": "error",
                "response": f"Code agent error: {step['error']}",
                "tool_calls_count": 0,
                "consecutive_failures": consecutive_failures,
            }

        thought = step.get("thought", "")
        code = step.get("code")

        # Record thought
        if thought and self.monologue:
            self.monologue.think("reason", f"[code-mode] {thought[:200]}")

        # No code block = LLM is done, final answer
        if step.get("done"):
            final = step.get("final_answer", thought)
            if len(final.strip()) < 20 and iteration < self.max_iterations - 1:
                messages.append({"role": "assistant", "content": final})
                messages.append({
                    "role": "user",
                    "content": "Please provide your complete answer, or write more code if needed.",
                })
                return {
                    "status": "incomplete",
                    "response": "",
                    "tool_calls_count": 0,
                    "consecutive_failures": consecutive_failures,
                }
            return {
                "status": "done",
                "response": final,
                "tool_calls_count": 0,
                "consecutive_failures": consecutive_failures,
            }

        # Has code block — execute it
        messages.append({
            "role": "assistant",
            "content": step.get("final_answer", "") or f"Thought: {thought}\n\n```python\n{code}\n```",
        })

        logger.info(f"[REACT-CODE] Executing code block ({len(code)} chars)")
        # SECURITY: Validate LLM-generated code before execution
        from aura.security.tool_validator import validate_script_code
        is_valid, error_msg = validate_script_code(code, "react_code_block")
        if not is_valid:
            messages.append({"role": "user", "content": f"Code rejected: {error_msg}"})
            consecutive_failures += 1
            if consecutive_failures >= 3:
                return {
                    "status": "fallback_to_tools",
                    "response": "",
                    "tool_calls_count": 0,
                    "consecutive_failures": 0,
                }
            return {
                "status": "continue",
                "response": "",
                "tool_calls_count": 0,
                "consecutive_failures": consecutive_failures,
            }
        exec_result = code_agent.execute_code_safely(code, tool_namespace)
        formatted = code_agent.format_execution_result(exec_result)

        # Add execution result as a "user" message (simulates tool result)
        messages.append({"role": "user", "content": f"Code execution result:\n{formatted}"})

        # Deterministic eval of execution result
        has_error = bool(exec_result.get("error")) or exec_result.get("timed_out", False)
        if has_error:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                logger.warning("[REACT-CODE] 3 consecutive failures, falling back to standard tools")
                return {
                    "status": "fallback_to_tools",
                    "response": "",
                    "tool_calls_count": 0,
                    "consecutive_failures": 0,
                }
        else:
            consecutive_failures = 0

        # Log to metacognition
        self.metacognition.log_evaluation(
            tool="code_agent",
            action=code[:200],
            confidence=70 if not has_error else 30,
            success=not has_error,
            progress="Code executed" if not has_error else "Code failed",
            next_step="continue",
            result_summary=formatted[:500],
            model_used=step.get("model", ""),
        )

        return {
            "status": "continue",
            "response": "",
            "tool_calls_count": 1,
            "consecutive_failures": consecutive_failures,
        }

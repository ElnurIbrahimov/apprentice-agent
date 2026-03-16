"""Ollama API integration as the agent's reasoning engine."""

import os
import re
import json
import logging
import threading
import time
import shutil
import concurrent.futures
import atexit
import uuid
from enum import Enum
from typing import Optional, Callable, Any
import ollama

from .config import Config
from .identity import get_identity_prompt

# ChatGPT OAuth client (optional — uses ChatGPT Plus/Pro subscription)
try:
    from .auth.chatgpt_client import ChatGPTClient
    from .auth.chatgpt_oauth import is_authenticated as _chatgpt_authenticated
    CHATGPT_AVAILABLE = True
except ImportError:
    CHATGPT_AVAILABLE = False

logger = logging.getLogger(__name__)

# ALMA Emotional Intelligence System
try:
    from .emotion.integration import (
        get_emotional_tone_modifier,
        process_user_message,
        process_response_outcome,
        bridge_evoemo_detection,
        get_mood_emoji,
    )
    from .emotion.alma_engine import alma_engine, trigger_emotion
    ALMA_AVAILABLE = True
except ImportError:
    ALMA_AVAILABLE = False
    logger.warning("[BRAIN] ALMA emotional system not available")

# Default timeouts (in seconds)
LLM_TIMEOUT = 60  # 60 seconds for LLM calls
WARMUP_TIMEOUT = 10  # 10 seconds for warmup

# Neuromodulator bounds for safety (multipliers on default values)
NEURO_MIN_MULTIPLIER = 0.7   # Never reduce below 70% of default
NEURO_MAX_MULTIPLIER = 1.4   # Never increase above 140% of default


_wm_extraction_running = threading.Event()  # Rate limiter: skip if previous extraction still running
_wm_consecutive_failures = 0  # Circuit breaker: disable after repeated failures
_WM_CIRCUIT_BREAKER_THRESHOLD = 3  # Disable after N consecutive failures
_WM_CIRCUIT_BREAKER_RESET_AFTER = 300  # Re-enable after 5 minutes
_wm_circuit_broken_at: float = 0.0

def _run_world_model_extraction(conversation_id, messages):
    """Background thread target for world model extraction (ADV-02 Phase 2).

    Includes circuit breaker: after 3 consecutive failures, disables extraction
    for 5 minutes to prevent thread pool starvation.
    """
    global _wm_consecutive_failures, _wm_circuit_broken_at

    # Circuit breaker check
    if _wm_consecutive_failures >= _WM_CIRCUIT_BREAKER_THRESHOLD:
        if time.time() - _wm_circuit_broken_at < _WM_CIRCUIT_BREAKER_RESET_AFTER:
            logger.debug("[BRAIN] World model extraction circuit breaker OPEN — skipping")
            return
        else:
            logger.info("[BRAIN] World model extraction circuit breaker RESET — retrying")
            _wm_consecutive_failures = 0

    if _wm_extraction_running.is_set():
        logger.debug("[BRAIN] Skipping world model extraction — previous still running")
        return
    _wm_extraction_running.set()
    try:
        from aura.consciousness.world_model import get_world_model
        wm = get_world_model()
        wm.process_conversation(conversation_id, messages)
        _wm_consecutive_failures = 0  # Reset on success
    except Exception as e:
        _wm_consecutive_failures += 1
        if _wm_consecutive_failures >= _WM_CIRCUIT_BREAKER_THRESHOLD:
            _wm_circuit_broken_at = time.time()
            logger.warning(f"[BRAIN] World model extraction failed {_wm_consecutive_failures}x — circuit breaker OPEN for {_WM_CIRCUIT_BREAKER_RESET_AFTER}s")
        else:
            logger.debug(f"[BRAIN] World model extraction failed ({_wm_consecutive_failures}/{_WM_CIRCUIT_BREAKER_THRESHOLD}): {e}")

    # ADV-02 Phase 3: Quick proactive awareness analysis after extraction
    try:
        if getattr(Config, "PROACTIVE_AWARENESS_QUICK_AFTER_CHAT", True):
            from aura.consciousness.proactive_awareness import get_proactive_awareness_engine
            engine = get_proactive_awareness_engine()
            engine.run_quick_analysis()
    except Exception as e:
        logger.debug(f"[BRAIN] Proactive awareness quick analysis failed: {e}")
    finally:
        _wm_extraction_running.clear()


def _get_neuromodulator_levels() -> dict:
    """Get current neuromodulator levels from ALMA, with safe defaults.

    Returns dict with dopamine, serotonin, norepinephrine, oxytocin (all 0-1).
    Returns 0.5 for all if ALMA is unavailable.
    During sleep, applies NeuroDream oscillation-based neuromodulator offsets.
    """
    _DEFAULTS = {"dopamine": 0.5, "serotonin": 0.5, "norepinephrine": 0.5, "oxytocin": 0.5}
    try:
        from aura.emotion.alma_engine import alma_engine
        state = alma_engine.get_emotional_state()
        base = (state or {}).get("neuromodulators") or {}
        if not base:
            return _DEFAULTS
        # Apply sleep neuromodulator influence from NeuroDream
        try:
            from aura.tools.neurodream import get_neurodream
            nd = get_neurodream()
            if nd.current_phase.value != "awake":
                influence = nd.get_sleep_neuromodulator_influence()
                return {k: max(0.0, min(1.0, base[k] + influence.get(k, 0.0))) for k in base}
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        return base
    except Exception as e:
        logger.debug(f"[Brain] non-critical: {e}")
    return _DEFAULTS


def _neuro_scale(base_value: float, neuro_level: float, sensitivity: float = 0.5) -> float:
    """Scale a base value by a neuromodulator level.

    neuro_level=0.5 -> no change (returns base_value)
    neuro_level=1.0 -> increase by sensitivity * (NEURO_MAX_MULTIPLIER - 1)
    neuro_level=0.0 -> decrease by sensitivity * (1 - NEURO_MIN_MULTIPLIER)

    Safety: result is always clamped to [base * NEURO_MIN_MULTIPLIER, base * NEURO_MAX_MULTIPLIER]
    """
    # Map neuro_level 0-1 to multiplier centered at 1.0
    offset = (neuro_level - 0.5) * 2 * sensitivity  # -sensitivity to +sensitivity
    multiplier = 1.0 + offset

    # Clamp to safety bounds
    multiplier = max(NEURO_MIN_MULTIPLIER, min(NEURO_MAX_MULTIPLIER, multiplier))
    return base_value * multiplier

# Shared thread pool to prevent thread leaks (max 12 concurrent LLM calls)
_SHARED_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=12, thread_name_prefix="llm_worker")

# Dedicated background pool for long-running non-user tasks (world model, self-improvement)
# Keeps these from starving the shared chat pool
_BG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="aura-bg")

def _cleanup_executor():
    """Cleanup shared executor on exit."""
    _SHARED_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    _BG_EXECUTOR.shutdown(wait=False, cancel_futures=True)

atexit.register(_cleanup_executor)


def call_with_timeout(func: Callable, timeout: int = LLM_TIMEOUT, default: Any = None) -> Any:
    """Execute a function with timeout protection using shared thread pool.

    Args:
        func: Function to execute (should be a lambda or callable with no args)
        timeout: Timeout in seconds
        default: Value to return on timeout

    Returns:
        Function result or default on timeout
    """
    try:
        future = _SHARED_EXECUTOR.submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"LLM call timed out after {timeout}s")
            future.cancel()  # Try to cancel the pending task
            return default
        except concurrent.futures.CancelledError:
            logger.warning("LLM call was cancelled")
            return default
        except (ConnectionError, OSError) as e:
            logger.error(f"LLM connection error: {e}")
            return default
        except ValueError as e:
            logger.error(f"LLM value error (bad response?): {e}")
            return default
        except Exception as e:
            # Log unexpected errors with full context for debugging
            logger.exception(f"Unexpected LLM error: {type(e).__name__}: {e}")
            return default
    except RuntimeError as e:
        # Executor might be shut down, create a one-off
        logger.warning(f"Shared executor unavailable ({e}), using fallback")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.warning("Fallback LLM call timed out")
                return default
            except (ConnectionError, OSError, ValueError) as e:
                logger.error(f"Fallback LLM error: {e}")
                return default


class TaskType(Enum):
    """Types of tasks for model routing."""
    SIMPLE = "simple"       # Greetings, short answers, basic queries
    REASONING = "reasoning" # Planning, evaluation, complex decisions
    CODE = "code"           # Code generation, calculations
    VISION = "vision"       # Image analysis


class OllamaBrain:
    """Handles all interactions with Ollama API for reasoning and decision-making."""

    # Limit conversation history to prevent unbounded memory growth
    MAX_HISTORY_LENGTH = Config.HISTORY_LIMIT  # Keep last N messages (N/2 exchanges)

    # Auto-reset context after this many queries to prevent slowdown
    AUTO_RESET_INTERVAL = Config.AUTO_RESET_INTERVAL  # Reset every N queries

    # Ollama cloud configuration
    OLLAMA_CLOUD_HOST = "https://api.ollama.com"

    def __init__(self, warmup: bool = True):
        # Local Ollama client (for local models)
        self.client = ollama.Client(host=Config.OLLAMA_HOST)

        # Cloud Ollama client (for cloud models like deepseek-v3.1:671b-cloud)
        self._cloud_client = None
        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key:
            self._cloud_client = ollama.Client(
                host=self.OLLAMA_CLOUD_HOST,
                headers={"Authorization": f"Bearer {api_key}"}
            )
            logger.debug(f"[BRAIN] Ollama cloud client initialized")
        else:
            logger.debug(f"[BRAIN] Warning: OLLAMA_API_KEY not set, cloud models unavailable")

        # ChatGPT client (for chatgpt: prefixed models)
        self._chatgpt_client = None
        if CHATGPT_AVAILABLE and _chatgpt_authenticated():
            self._chatgpt_client = ChatGPTClient()
            logger.info("[BRAIN] ChatGPT OAuth client initialized")

        self.model = Config.MODEL_NAME
        self._max_history: int = Config.HISTORY_LIMIT
        self.conversation_history: list[dict] = []
        self._history_lock = threading.Lock()
        self._last_model_used: str = self.model  # Track for metacognition
        self._query_count: int = 0  # Track queries for auto-reset (resets every 15)
        self._total_query_count: int = 0  # Total queries (never resets)
        self._model_override: Optional[str] = None  # Manual model override (bypasses auto-selection)

        # Phase 4 — compaction notification flag
        self._compaction_pending: bool = False

        # Phase 4 — per-session token/cost tracking
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._session_cost_usd: float = 0.0
        self._token_lock = threading.Lock()
        # Rough USD cost per 1K tokens (input/output) for cloud models
        self._MODEL_COST_PER_1K: dict = {
            "gemini-3-flash-preview:cloud": (0.00015, 0.0006),
            "nemotron-3-nano:30b-cloud":    (0.0004, 0.0004),
            "kimi-k2.5:cloud":             (0.003,  0.003),
            "kimi-k2-thinking:cloud":      (0.01,   0.03),
            "qwen3.5:397b-cloud":          (0.004,  0.004),
            "cogito-2.1:671b-cloud":       (0.004,  0.004),
            "deepseek-v3.2:cloud":         (0.003,  0.003),
            "qwen3-coder:480b-cloud":      (0.004,  0.004),
            "devstral-2:123b-cloud":       (0.002,  0.002),
            "minimax-m2.5:cloud":          (0.004,  0.004),
        }
        # ChatGPT subscription models (cost is $0 — covered by subscription)
        if CHATGPT_AVAILABLE:
            from .auth.chatgpt_client import ALL_CHATGPT_MODELS
            for m in ALL_CHATGPT_MODELS:
                self._MODEL_COST_PER_1K[m] = (0.0, 0.0)
        self._DEFAULT_COST_PER_1K = (0.003, 0.003)  # fallback for unknown models

        # Setup persistent history storage (legacy single-conversation path)
        self._history_dir = Config.CHROMADB_PATH.parent / "conversation"
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self._history_dir / "history.json"

        # Multi-conversation support
        self._conversations_dir = Config.CHROMADB_PATH.parent / "conversations"
        self._conversations_dir.mkdir(parents=True, exist_ok=True)
        self._conversations_index_file = self._conversations_dir / "index.json"
        self._current_conversation_id: Optional[str] = None
        self._conversations_index_cache: dict | None = None

        # Migrate legacy history and initialize conversations
        self._migrate_legacy_history()
        self._load_history()

        # System prompt additions TTL cache (Fix 4: prevent 8+ module queries per rapid message)
        self._cached_system_additions: Optional[str] = None
        self._system_additions_ts: float = 0.0
        self._system_additions_lock = threading.RLock()

        # ALMA Emotional Intelligence
        self._alma_enabled = ALMA_AVAILABLE
        self._auto_emotional_tone = True  # Automatically add emotional tone to responses
        if self._alma_enabled:
            logger.info(f"[BRAIN] ALMA emotional system enabled {get_mood_emoji()}")

        # Screenshot path tracking (set by agent, read by brain for combined screenshot+vision tasks)
        self._last_screenshot_path: Optional[str] = None

        # Episodic memory auto-recall (lazy-init, best-effort)
        self._episodic_memory = None
        try:
            from aura_episodic_memory.memory_store import EpisodicMemoryStore
            from aura_episodic_memory.mcp_tools import QuickEpisodicMemory
            _store = EpisodicMemoryStore()
            self._episodic_memory = QuickEpisodicMemory(_store)
            logger.info("[BRAIN] Episodic memory auto-recall enabled")
        except Exception as _e:
            logger.debug(f"[BRAIN] Episodic memory not available: {_e}")

        if warmup:
            self._warmup_models()

    def _get_client_for_model(self, model: str) -> tuple:
        """Get the appropriate client (local, cloud, or ChatGPT) based on model name.

        Routing:
        - chatgpt:* models → ChatGPT OAuth client (Codex Responses API)
        - *-cloud / *:cloud models → Ollama cloud client
        - everything else → local Ollama client

        Returns:
            Tuple of (client, actual_model_name) - model name may be modified for fallback
        """
        # ChatGPT OAuth models (e.g., chatgpt:gpt-5.1-codex)
        if model.startswith("chatgpt:"):
            if self._chatgpt_client:
                logger.debug(f"[BRAIN] Using ChatGPT OAuth client for model: {model}")
                return self._chatgpt_client, model
            else:
                logger.warning(f"[BRAIN] ChatGPT not authenticated, cannot use {model}")
                # Fall through to default
                return self.client, Config.MODEL_FAST

        if model.endswith(("-cloud", ":cloud")):
            if self._cloud_client:
                logger.debug(f"[BRAIN] Using cloud client for model: {model}")
                return self._cloud_client, model
            else:
                # No api.ollama.com key — route through local Ollama bridge which
                # proxies cloud models (deepseek-v3.2:cloud, qwen3.5:397b-cloud, etc.)
                logger.debug(f"[BRAIN] No OLLAMA_API_KEY — routing {model} via local Ollama bridge")
                return self.client, model
        return self.client, model

    def _get_fallback_chain(self, model: str) -> list:
        """Return the fallback chain for the given model (Phase 4 — model fallback)."""
        chains = [
            Config.MODEL_FAST_CHAIN,
            Config.MODEL_REASON_CHAIN,
            Config.MODEL_CODE_CHAIN,
            Config.MODEL_VISION_CHAIN,
            Config.MODEL_THINK_CHAIN,
            Config.MODEL_LONGCTX_CHAIN,
        ]
        for chain in chains:
            if model in chain:
                return chain
        return []

    def _record_tokens(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Accumulate session token counts and estimated cost (Phase 4)."""
        in_rate, out_rate = self._MODEL_COST_PER_1K.get(model, self._DEFAULT_COST_PER_1K)
        cost = (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate
        with self._token_lock:
            self._session_input_tokens += input_tokens
            self._session_output_tokens += output_tokens
            self._session_cost_usd += cost

    def get_session_stats(self) -> dict:
        """Return per-session token usage and estimated cost (Phase 4)."""
        with self._token_lock:
            return {
                "input_tokens": self._session_input_tokens,
                "output_tokens": self._session_output_tokens,
                "total_tokens": self._session_input_tokens + self._session_output_tokens,
                "cost_usd": round(self._session_cost_usd, 6),
                "queries": self._total_query_count,
            }

    def think_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model_override: str = None,
        options: dict = None,
    ) -> dict:
        """Call Ollama with structured tool calling (tools= parameter).

        This is the core method for the agentic dev CLI loop. Unlike think(),
        it does NOT manage history — the agentic loop owns the message list.
        It does NOT use neuromodulators — agentic mode uses fixed low temperature.

        Args:
            messages: Full message list (system + user + assistant + tool results)
            tools: Ollama tool schemas (from tool_schemas.AGENTIC_TOOLS)
            model_override: Force a specific model instead of auto-routing
            options: Ollama options (temperature, num_predict, etc.)

        Returns:
            {message, model, input_tokens, output_tokens} or {error} on failure
        """
        # Use MODEL_CODE (devstral/minimax) — not MODEL_FAST (gemini) which
        # crashes on tool-result turns with "missing thought_signature" error
        model = model_override or self._model_override or Config.MODEL_CODE
        client, actual_model = self._get_client_for_model(model)

        # ChatGPT client doesn't support tools= parameter
        if actual_model.startswith("chatgpt:"):
            return {"error": "ChatGPT models don't support structured tool calling. Use an Ollama model."}

        llm_options = options or {"temperature": 0.2, "num_predict": 4096}

        try:
            response = call_with_timeout(
                lambda: client.chat(
                    model=actual_model,
                    messages=messages,
                    tools=tools,
                    options=llm_options,
                ),
                timeout=120,
                default=None,
            )
        except Exception as e:
            logger.error(f"[BRAIN] think_with_tools error: {e}")
            return {"error": str(e)}

        if response is None:
            # Try fallback chain
            chain = self._get_fallback_chain(actual_model)
            for fallback_model in chain:
                if fallback_model == actual_model:
                    continue
                try:
                    fb_client, fb_actual = self._get_client_for_model(fallback_model)
                    logger.info(f"[BRAIN] Tool-call fallback: {actual_model} -> {fb_actual}")
                    response = call_with_timeout(
                        lambda m=fb_actual, c=fb_client: c.chat(
                            model=m, messages=messages, tools=tools, options=llm_options,
                        ),
                        timeout=120,
                        default=None,
                    )
                    if response is not None:
                        actual_model = fb_actual
                        break
                except Exception:
                    continue

        if response is None:
            return {"error": "All models failed to respond"}

        # Track tokens (Pydantic .get() returns None not the default, so use `or 0`)
        input_tokens = response.get("prompt_eval_count", 0) or 0
        output_tokens = response.get("eval_count", 0) or 0
        self._record_tokens(actual_model, input_tokens, output_tokens)

        return {
            "message": response.get("message", {}),
            "model": actual_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    # -----------------------------------------------------------------
    # ReAct Step — single LLM call combining thought + action (1.1)
    # -----------------------------------------------------------------

    def react_step(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        model_override: str = None,
    ) -> dict:
        """Execute one ReAct step: single LLM call that produces thought + tool call(s).

        This is the core brain method for roadmap 1.1 (Collapse to ReAct Loop).
        It wraps think_with_tools and parses the response into a structured result
        that the agent loop can act on without additional LLM calls.

        Args:
            messages: Full conversation history (system + user + assistant + tool results)
            tool_schemas: Ollama tool schemas (from Tool RAG selection)
            model_override: Force a specific model

        Returns:
            {
                "thought": str,       # LLM's reasoning (content text)
                "tool_calls": list,    # Parsed tool calls: [{"name": str, "args": dict}, ...]
                "done": bool,          # True if LLM returned final answer (no tool calls)
                "final_answer": str,   # Final answer text (only when done=True)
                "model": str,          # Model used
                "input_tokens": int,
                "output_tokens": int,
            }
            or {"error": str} on failure.
        """
        result = self.think_with_tools(messages, tool_schemas, model_override=model_override)

        if "error" in result:
            return result

        raw_msg = result.get("message", {})

        # Handle both dict and Pydantic message objects
        if isinstance(raw_msg, dict):
            content = raw_msg.get("content", "") or ""
            raw_tool_calls = raw_msg.get("tool_calls")
        else:
            content = getattr(raw_msg, "content", "") or ""
            raw_tool_calls = getattr(raw_msg, "tool_calls", None)

        # Parse tool calls into a clean list
        parsed_calls = []
        if raw_tool_calls:
            for tc in raw_tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    raw_args = fn.get("arguments", {})
                else:
                    fn = getattr(tc, "function", None)
                    tool_name = getattr(fn, "name", "") if fn else ""
                    raw_args = getattr(fn, "arguments", {}) if fn else {}

                # Parse arguments
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except (json.JSONDecodeError, TypeError):
                        args = {"action": raw_args}
                elif isinstance(raw_args, dict):
                    args = raw_args
                elif raw_args is None:
                    args = {}
                else:
                    args = {"action": str(raw_args)}

                parsed_calls.append({"name": tool_name, "args": args})

        done = not bool(parsed_calls)

        return {
            "thought": content,
            "tool_calls": parsed_calls,
            "done": done,
            "final_answer": content if done else "",
            "model": result.get("model", ""),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "raw_message": raw_msg,  # Kept for conversation history serialization
        }

    # -----------------------------------------------------------------
    # ReAct Code Step — LLM writes Python code instead of tool calls (5.1)
    # -----------------------------------------------------------------

    def react_step_code(
        self,
        messages: list[dict],
        model_override: str = None,
    ) -> dict:
        """Execute one ReAct code step: LLM produces Thought + Python code block.

        Unlike react_step() which uses Ollama structured tool calling,
        this method prompts the LLM to write Python code as its action.
        No tools= parameter is passed — the LLM generates free-form text.

        Args:
            messages: Full conversation history (system + user + assistant + code results)
            model_override: Force a specific model

        Returns:
            {
                "thought": str,       # LLM's reasoning text
                "code": str | None,   # Python code block (None if final answer)
                "done": bool,         # True if no code block (final answer)
                "final_answer": str,  # Full text when done
                "model": str,
                "input_tokens": int,
                "output_tokens": int,
            }
            or {"error": str} on failure.
        """
        model = model_override or self._model_override or Config.MODEL_CODE
        client, actual_model = self._get_client_for_model(model)

        if actual_model.startswith("chatgpt:"):
            return {"error": "ChatGPT models don't support code agent mode via this path."}

        llm_options = {"temperature": 0.2, "num_predict": 4096}

        try:
            response = call_with_timeout(
                lambda: client.chat(
                    model=actual_model,
                    messages=messages,
                    options=llm_options,
                ),
                timeout=120,
                default=None,
            )
        except Exception as e:
            logger.error(f"[BRAIN] react_step_code error: {e}")
            return {"error": str(e)}

        if response is None:
            return {"error": "Model failed to respond (timeout)"}

        # Extract content from response
        raw_msg = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
        if isinstance(raw_msg, dict):
            content = raw_msg.get("content", "") or ""
        else:
            content = getattr(raw_msg, "content", "") or ""

        # Track tokens
        input_tokens = (response.get("prompt_eval_count", 0) if isinstance(response, dict)
                        else getattr(response, "prompt_eval_count", 0)) or 0
        output_tokens = (response.get("eval_count", 0) if isinstance(response, dict)
                         else getattr(response, "eval_count", 0)) or 0
        self._record_tokens(actual_model, input_tokens, output_tokens)

        # Parse: extract thought and code block
        from aura.core.code_agent import _extract_code_block, _extract_thought

        code = _extract_code_block(content)
        thought = _extract_thought(content)
        done = code is None

        return {
            "thought": thought,
            "code": code,
            "done": done,
            "final_answer": content if done else "",
            "model": actual_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def think_with_tools_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        model_override: str = None,
        options: dict = None,
    ):
        """Streaming version of think_with_tools(). Yields (chunk_type, data) tuples.

        chunk_type: "content" | "tool_calls" | "done" | "error"
        data: str for content, list for tool_calls, dict for done/error
        """
        model = model_override or self._model_override or Config.MODEL_CODE
        client, actual_model = self._get_client_for_model(model)

        if actual_model.startswith("chatgpt:"):
            yield ("error", {"error": "ChatGPT models don't support structured tool calling."})
            return

        llm_options = options or {"temperature": 0.2, "num_predict": 4096}

        try:
            stream = client.chat(
                model=actual_model,
                messages=messages,
                tools=tools,
                options=llm_options,
                stream=True,
            )

            accumulated_content = ""
            tool_calls = None
            input_tokens = 0
            output_tokens = 0

            for chunk in stream:
                # Ollama streaming returns ChatResponse objects or dicts
                if isinstance(chunk, dict):
                    msg = chunk.get("message", {})
                    done = chunk.get("done", False)
                    if done:
                        input_tokens = chunk.get("prompt_eval_count", 0) or 0
                        output_tokens = chunk.get("eval_count", 0) or 0
                else:
                    msg = getattr(chunk, "message", None)
                    done = getattr(chunk, "done", False)
                    if done:
                        input_tokens = getattr(chunk, "prompt_eval_count", 0) or 0
                        output_tokens = getattr(chunk, "eval_count", 0) or 0

                if msg is None:
                    if done:
                        break
                    continue

                # Content chunk
                c = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
                if c:
                    accumulated_content += c
                    yield ("content", c)

                # Tool calls (usually only in final chunk)
                tc = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                if tc:
                    tool_calls = tc

                if done:
                    break

            if tool_calls:
                yield ("tool_calls", tool_calls)

            self._record_tokens(actual_model, input_tokens, output_tokens)

            yield ("done", {
                "content": accumulated_content,
                "tool_calls": tool_calls,
                "model": actual_model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })

        except Exception as e:
            yield ("error", {"error": str(e)})

    def _warmup_models(self) -> None:
        """Warm up local Ollama models with a keep-alive ping. Skipped for cloud models."""
        models_to_warm = [
            m for m in [Config.MODEL_FAST, Config.MODEL_REASON, Config.MODEL_CODE]
            if not m.endswith(("-cloud", ":cloud"))
        ]
        if not models_to_warm:
            logger.info("[BRAIN] All models are cloud-hosted — skipping warmup")
            return
        for model in models_to_warm:
            try:
                call_with_timeout(
                    lambda m=model: self.client.generate(model=m, prompt="", keep_alive="30m"),
                    timeout=WARMUP_TIMEOUT,
                    default=None
                )
                logger.info(f"[BRAIN] Warmed up local model: {model}")
            except Exception as e:
                logger.warning(f"[BRAIN] Warmup failed for {model}: {e}")

    def _load_history(self) -> None:
        """Load conversation history from disk."""
        try:
            if self._history_file.exists():
                data = json.loads(self._history_file.read_text(encoding="utf-8"))
                self.conversation_history = data.get("history", [])
                self._query_count = data.get("query_count", 0)
                self._total_query_count = data.get("total_query_count", 0)
                logger.info(f"[BRAIN] Loaded {len(self.conversation_history)} messages from history (total queries: {self._total_query_count})")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"[BRAIN] Could not load history: {e}")
            self.conversation_history = []

    def _save_history(self) -> None:
        """Save conversation history to disk."""
        with self._history_lock:
            self._save_history_unlocked()

    def _save_history_unlocked(self) -> None:
        """Save conversation history to disk — caller MUST hold _history_lock."""
        data_str = json.dumps(
            {
                "history": self.conversation_history,
                "query_count": self._query_count,
                "total_query_count": self._total_query_count,
            },
            indent=2,
            ensure_ascii=False,
        )
        # Capture path so a concurrent conversation switch
        # cannot cause us to write to the wrong file in the background.
        path = self._history_file
        _BG_EXECUTOR.submit(lambda p=path, d=data_str: p.write_text(d, encoding="utf-8"))
        self._update_conversation_index_entry()

    def _save_history_snapshot(self, history: list, query_count: int, total_query_count: int) -> None:
        """Save a pre-copied history list to disk (called OUTSIDE _history_lock).

        Serializes JSON on the calling thread (fast), then writes to disk
        in the background pool to avoid blocking request threads on I/O.
        """
        try:
            data_str = json.dumps(
                {"history": history, "query_count": query_count, "total_query_count": total_query_count},
                indent=2, ensure_ascii=False,
            )
            path = self._history_file
            _BG_EXECUTOR.submit(lambda p=path, d=data_str: p.write_text(d, encoding="utf-8"))
            self._update_conversation_index_entry()
        except (IOError, RuntimeError) as e:
            logger.warning(f"[BRAIN] Could not save history snapshot: {e}")

    # =========================================================================
    # Multi-Conversation Management
    # =========================================================================

    def _migrate_legacy_history(self) -> None:
        """Migrate existing single-conversation history into multi-conversation system."""
        index = self._load_conversations_index()
        if index:
            # Already have conversations, check if we need to set current
            if not self._current_conversation_id:
                # Find most recently updated conversation
                sorted_convs = sorted(index, key=lambda c: c.get("updated_at", 0), reverse=True)
                if sorted_convs:
                    self._current_conversation_id = sorted_convs[0]["id"]
                    # Point history file to current conversation
                    conv_dir = self._conversations_dir / self._current_conversation_id
                    if conv_dir.exists():
                        self._history_file = conv_dir / "history.json"
            return

        # No conversations index yet — migrate legacy history if it exists
        legacy_file = self._history_dir / "history.json"
        if legacy_file.exists():
            try:
                data = json.loads(legacy_file.read_text(encoding="utf-8"))
                history = data.get("history", [])
                if history:
                    # Create first conversation from legacy data
                    conv_id = self._generate_conversation_id()
                    conv_dir = self._conversations_dir / conv_id
                    conv_dir.mkdir(parents=True, exist_ok=True)

                    # Copy history to new location
                    new_history_file = conv_dir / "history.json"
                    new_history_file.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8"
                    )

                    # Generate title from first user message
                    title = self._auto_title(history)

                    # Get preview from last message
                    preview = ""
                    if history:
                        last_msg = history[-1].get("content", "")
                        preview = last_msg[:100]

                    # Create index with migrated conversation
                    index_entry = {
                        "id": conv_id,
                        "title": title,
                        "created_at": int(time.time()),
                        "updated_at": int(time.time()),
                        "message_count": len(history),
                        "preview": preview,
                    }
                    self._save_conversations_index([index_entry])
                    self._current_conversation_id = conv_id
                    self._history_file = new_history_file
                    logger.info(f"[BRAIN] Migrated legacy history to conversation: {conv_id} ({title})")
                    return
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[BRAIN] Could not migrate legacy history: {e}")

        # No legacy history either — create a default conversation
        conv_id = self._create_conversation_dir("New Chat")
        self._current_conversation_id = conv_id
        conv_dir = self._conversations_dir / conv_id
        self._history_file = conv_dir / "history.json"

    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        return f"conv_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    def _auto_title(self, messages: list) -> str:
        """Generate a title from the first user message."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "").strip()
                # Strip file attachment context markers
                if "[FILE_ATTACHMENT_CONTEXT]" in content:
                    # Try to find the user request after the context
                    parts = content.split("User request:")
                    if len(parts) > 1:
                        content = parts[-1].strip()
                    else:
                        content = content.split("\n")[0].strip()
                # Truncate to 50 chars
                if len(content) > 50:
                    content = content[:47] + "..."
                return content or "New Chat"
        return "New Chat"

    def _create_conversation_dir(self, title: str) -> str:
        """Create a new conversation directory and add to index."""
        conv_id = self._generate_conversation_id()
        conv_dir = self._conversations_dir / conv_id
        conv_dir.mkdir(parents=True, exist_ok=True)

        # Write empty history
        empty_data = {"history": [], "query_count": 0, "total_query_count": 0}
        (conv_dir / "history.json").write_text(
            json.dumps(empty_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Add to index
        index = self._load_conversations_index()
        index.append({
            "id": conv_id,
            "title": title,
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "message_count": 0,
            "preview": "",
        })
        self._save_conversations_index(index)
        return conv_id

    def _load_conversations_index(self) -> list:
        """Load the conversations index."""
        if self._conversations_index_cache is not None:
            return self._conversations_index_cache
        try:
            if self._conversations_index_file.exists():
                data = json.loads(self._conversations_index_file.read_text(encoding="utf-8"))
                self._conversations_index_cache = data
                return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"[BRAIN] Could not load conversations index: {e}")
        self._conversations_index_cache = []
        return []

    def _invalidate_conversation_cache(self) -> None:
        """Invalidate the in-memory conversations index cache.

        Must be called by any method that mutates conversations (create, delete,
        rename, switch) so the next _load_conversations_index() re-reads from disk.
        """
        self._conversations_index_cache = None

    def _save_conversations_index(self, index: list) -> None:
        """Save the conversations index."""
        try:
            self._conversations_index_file.write_text(
                json.dumps(index, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            self._conversations_index_cache = index
        except IOError as e:
            logger.warning(f"[BRAIN] Could not save conversations index: {e}")

    def _update_conversation_index_entry(self) -> None:
        """Update the current conversation's index entry with latest metadata."""
        if not self._current_conversation_id:
            return
        index = self._load_conversations_index()
        for entry in index:
            if entry["id"] == self._current_conversation_id:
                entry["updated_at"] = int(time.time())
                entry["message_count"] = len(self.conversation_history)
                if self.conversation_history:
                    last_msg = self.conversation_history[-1].get("content", "")
                    entry["preview"] = last_msg[:100]
                # Update title if still "New Chat" and we have messages
                if entry["title"] == "New Chat" and self.conversation_history:
                    entry["title"] = self._auto_title(self.conversation_history)
                break
        self._save_conversations_index(index)

    def create_conversation(self, title: Optional[str] = None) -> str:
        """Create a new conversation.

        Args:
            title: Optional title, defaults to "New Chat"

        Returns:
            The new conversation's ID
        """
        # Save current conversation first
        self._save_history()
        self._update_conversation_index_entry()
        self._invalidate_conversation_cache()

        effective_title = title or "New Chat"
        conv_id = self._create_conversation_dir(effective_title)

        # Switch to the new conversation
        self._current_conversation_id = conv_id
        conv_dir = self._conversations_dir / conv_id
        self._history_file = conv_dir / "history.json"
        self.conversation_history = []
        self._query_count = 0
        logger.info(f"[BRAIN] Created new conversation: {conv_id} ({effective_title})")
        return conv_id

    def list_conversations(self) -> list:
        """List all conversations.

        Returns:
            List of conversation summaries sorted by updated_at descending
        """
        index = self._load_conversations_index()
        # Sort by updated_at descending (most recent first)
        index.sort(key=lambda c: c.get("updated_at", 0), reverse=True)
        # Add is_active flag
        for entry in index:
            entry["is_active"] = entry["id"] == self._current_conversation_id
        return index

    def switch_conversation(self, conversation_id: str) -> bool:
        """Switch to a different conversation.

        Args:
            conversation_id: ID of conversation to switch to

        Returns:
            True if switched successfully
        """
        if conversation_id == self._current_conversation_id:
            return True

        conv_dir = self._conversations_dir / conversation_id
        if not conv_dir.exists():
            logger.warning(f"[BRAIN] Conversation not found: {conversation_id}")
            return False

        with self._history_lock:
            # Save current conversation (unlocked — we already hold the lock)
            self._save_history_unlocked()
            self._update_conversation_index_entry()
            self._invalidate_conversation_cache()

            # Load new conversation
            self._current_conversation_id = conversation_id
            self._history_file = conv_dir / "history.json"
            self._load_history()
        logger.info(f"[BRAIN] Switched to conversation: {conversation_id} ({len(self.conversation_history)} messages)")
        return True

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: ID of conversation to delete

        Returns:
            True if deleted successfully
        """
        conv_dir = self._conversations_dir / conversation_id
        if not conv_dir.exists():
            return False

        # Remove directory
        try:
            shutil.rmtree(conv_dir)
        except OSError as e:
            logger.error(f"[BRAIN] Failed to delete conversation dir: {e}")
            return False

        # Invalidate cache before re-reading index
        self._invalidate_conversation_cache()

        # Remove from index
        index = self._load_conversations_index()
        index = [c for c in index if c["id"] != conversation_id]
        self._save_conversations_index(index)

        # If we deleted the active conversation, switch to another or create new
        if conversation_id == self._current_conversation_id:
            if index:
                self.switch_conversation(index[0]["id"])
            else:
                conv_id = self._create_conversation_dir("New Chat")
                self._current_conversation_id = conv_id
                self._history_file = self._conversations_dir / conv_id / "history.json"
                self.conversation_history = []
                self._query_count = 0

        logger.info(f"[BRAIN] Deleted conversation: {conversation_id}")
        return True

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        """Rename a conversation.

        Args:
            conversation_id: ID of conversation to rename
            title: New title

        Returns:
            True if renamed successfully
        """
        self._invalidate_conversation_cache()
        index = self._load_conversations_index()
        for entry in index:
            if entry["id"] == conversation_id:
                entry["title"] = title
                self._save_conversations_index(index)
                logger.info(f"[BRAIN] Renamed conversation {conversation_id}: {title}")
                return True
        return False

    def get_current_conversation_id(self) -> Optional[str]:
        """Get the current conversation ID."""
        return self._current_conversation_id

    def get_conversation_messages(self, conversation_id: str) -> list:
        """Get messages for a specific conversation without switching.

        Args:
            conversation_id: ID of conversation

        Returns:
            List of messages
        """
        if conversation_id == self._current_conversation_id:
            return list(self.conversation_history)

        conv_dir = self._conversations_dir / conversation_id
        history_file = conv_dir / "history.json"
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text(encoding="utf-8"))
                return data.get("history", [])
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def save_conversation_to_memory(self, conversation_id: Optional[str] = None) -> dict:
        """Save a conversation's content to AURA's long-term memory (A-MEM).

        Args:
            conversation_id: ID of conversation to save, or None for current

        Returns:
            Dict with success status and details
        """
        conv_id = conversation_id or self._current_conversation_id
        if not conv_id:
            return {"success": False, "error": "No active conversation"}

        messages = self.get_conversation_messages(conv_id)
        if not messages:
            return {"success": False, "error": "Conversation is empty"}

        # Get conversation title
        index = self._load_conversations_index()
        title = "Unknown"
        for entry in index:
            if entry["id"] == conv_id:
                title = entry["title"]
                break

        # Build a summary of the conversation
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        assistant_messages = [m["content"] for m in messages if m.get("role") == "assistant"]

        # Create a condensed version for memory
        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Skip file attachment context markers for cleaner memory
            if "[FILE_ATTACHMENT_CONTEXT]" in content:
                parts = content.split("User request:")
                content = parts[-1].strip() if len(parts) > 1 else content[:200]
            conversation_text += f"{role.upper()}: {content[:300]}\n"

        # Truncate if too long (keep under 2000 chars for memory)
        if len(conversation_text) > 2000:
            conversation_text = conversation_text[:1900] + "\n...(truncated)"

        memory_content = f"Conversation: {title}\n\n{conversation_text}"

        # Try to save to A-MEM
        primary_error: Optional[str] = None
        try:
            from aura.tools.amem import get_amem
            amem = get_amem()
            note = amem.add(
                content=memory_content,
                tags=["conversation", "chat_history", title.lower().replace(" ", "_")[:30]],
                category="conversation",
                source="conversation_save",
                importance=0.6,
            )
            logger.info(f"[BRAIN] Saved conversation {conv_id} to A-MEM (note: {note.id})")
            return {
                "success": True,
                "note_id": note.id,
                "message_count": len(messages),
                "title": title,
            }
        except Exception as e:
            primary_error = str(e)
            logger.error(f"[BRAIN] Failed to save conversation to memory: {e}")

        # Fallback: try hybrid memory
        try:
            from aura.tools.hybrid_amem import get_hybrid_memory
            hybrid = get_hybrid_memory()
            result = hybrid.remember(
                content=memory_content,
                memory_type="episodic",
                tags=["conversation", "chat_history"],
                importance=0.6,
                source="conversation_save",
            )
            logger.info(f"[BRAIN] Saved conversation {conv_id} to hybrid memory")
            return {
                "success": True,
                "note_id": result.get("note_id"),
                "message_count": len(messages),
                "title": title,
            }
        except Exception as e2:
            logger.error(f"[BRAIN] Hybrid memory fallback also failed: {e2}")
            return {"success": False, "error": f"Memory save failed: {primary_error}; {e2}"}

    def clear_history(self):
        """Clear conversation history to free memory."""
        with self._history_lock:
            self.conversation_history.clear()
            self._query_count = 0
        self._save_history()
        logger.info("[BRAIN] Conversation history cleared")

    def reset_context(self, model: Optional[str] = None):
        """Reset Ollama's context for a model to prevent slowdown.

        Args:
            model: Model to reset, or None for current model
        """
        target_model = model or self._last_model_used
        try:
            # Unload and reload the model to clear its context
            self.client.generate(
                model=target_model,
                prompt="",
                keep_alive="0"  # Unload immediately
            )
            logger.info(f"[BRAIN] Reset context for {target_model}")
        except Exception as e:
            logger.debug(f"[BRAIN] Context reset failed (ok if model not loaded): {e}")

    def full_reset(self):
        """Full reset: clear history and reset Ollama context."""
        self.clear_history()
        self.reset_context()
        logger.info("[BRAIN] Full reset completed")

    def _quick_generate(self, prompt: str, timeout: int = 30) -> str:
        """Use MODEL_FAST for cheap/fast generation (summarization, planning).

        No history, no system prompt injection — just prompt -> response.
        Wrapped in call_with_timeout to prevent thread pool starvation.

        Args:
            prompt: The prompt to send
            timeout: Max seconds to wait for response

        Returns:
            Generated response string
        """
        fast_model = Config.MODEL_FAST
        try:
            client, actual_model = self._get_client_for_model(fast_model)
            response = call_with_timeout(
                lambda: client.chat(
                    model=actual_model,
                    messages=[{"role": "user", "content": prompt}]
                ),
                timeout=timeout,
                default=None,
            )
            if response is None:
                logger.warning(f"[BRAIN] Quick generate timed out after {timeout}s")
                return ""
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"[BRAIN] Quick generate failed: {e}")
            return ""

    def compact_history(self, focus: str = None) -> str:
        """Compact conversation history synchronously."""
        return self._do_compact_history(focus)

    def _do_compact_history(self, focus: str = None) -> str:
        """Compact conversation history by summarizing older messages.

        Takes the oldest 2/3 of conversation_history, asks LLM to summarize
        them in 2-4 sentences, then replaces history with:
        [summary as system message] + recent 1/3.

        Args:
            focus: Optional topic to focus the summary on

        Returns:
            The summary text, or empty string if nothing to compact
        """
        with self._history_lock:
            history = list(self.conversation_history)
        if len(history) < 6:
            return ""

        # Split: oldest 2/3 to summarize, keep recent 1/3
        split_point = (len(history) * 2) // 3
        old_messages = [m for m in history[:split_point] if m.get("role") != "system"]
        recent_messages = history[split_point:]

        # Build summary prompt
        conversation_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content'][:300]}"
            for msg in old_messages
        )

        focus_instruction = ""
        if focus:
            focus_instruction = f" Focus especially on topics related to: {focus}."

        summary_prompt = (
            f"Summarize this conversation in 2-4 concise sentences. "
            f"Capture the key topics, decisions, and any important context.{focus_instruction}\n\n"
            f"{conversation_text}"
        )

        summary = self._quick_generate(summary_prompt)
        if not summary:
            return ""

        # Replace history: summary as system message + recent messages
        new_history = [
            {"role": "system", "content": f"[Conversation summary] {summary}"}
        ] + recent_messages
        with self._history_lock:
            self.conversation_history = new_history
        self._save_history()

        logger.info(f"[BRAIN] Compacted {len(old_messages)} messages into summary, kept {len(recent_messages)} recent")
        return summary

    def _check_auto_reset(self):
        """Check if auto-reset is needed and perform it.

        Instead of just resetting the counter, compacts history to preserve
        context. Falls back to simple reset if compaction fails.
        """
        with self._history_lock:
            self._query_count += 1
            self._total_query_count += 1  # Total count never resets
            needs_compact = self._query_count >= self.AUTO_RESET_INTERVAL
            if needs_compact:
                self._query_count = 0
        if not needs_compact:
            return
        logger.info(f"[BRAIN] Auto-compact triggered (total: {self._total_query_count})")
        # Submit compaction to background (it handles its own save on completion)
        try:
            self.compact_history()
        except Exception as e:
            logger.warning(f"[BRAIN] Auto-compact submission failed: {e}")

    def _get_cached_system_additions(self) -> str:
        """Return TTL-cached subsystem additions for the system prompt.

        Queries all consciousness/emotion/user-model modules and caches the
        combined result for 12 seconds. This prevents 8+ module round-trips on
        every rapid sequential message. Thread-safe via _system_additions_lock.
        """
        with self._system_additions_lock:
            if self._cached_system_additions is not None and (time.time() - self._system_additions_ts) < 12.0:
                return self._cached_system_additions

            additions = []

            # === LEARNED CONTEXT INJECTION (Phase 4D: Letta-style) ===
            try:
                from aura.tools.neurodream import get_neurodream
                nd = get_neurodream()
                learned_ctx = nd.get_learned_context_prompt()
                if learned_ctx:
                    additions.append(learned_ctx)
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
            # === CALENDAR CONTEXT INJECTION (Phase 5D) ===
            try:
                from aura.proactive.monitors.calendar_monitor import get_calendar_monitor
                cm = get_calendar_monitor()
                cal_ctx = cm.get_context_for_prompt()
                if cal_ctx:
                    additions.append(cal_ctx)
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
            # === SELF-MODEL INJECTION (Phase 6B: Metacognitive Self-Improvement) ===
            try:
                from aura.consciousness.metacognition import get_metacognitive_engine
                mc = get_metacognitive_engine()
                self_model_ctx = mc.get_self_model_prompt()
                if self_model_ctx:
                    additions.append(self_model_ctx)
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
            # === USER MODEL INJECTION (Phase 6C / ADV-04: Theory of Mind) ===
            try:
                if Config.MULTI_USER_ENABLED:
                    from aura.multi_user import get_multi_user_manager
                    manager = get_multi_user_manager()
                    user_model = manager.get_active_user_model()
                    if user_model:
                        user_model_ctx = user_model.get_context_for_prompt()
                        if user_model_ctx:
                            additions.append(user_model_ctx)
                else:
                    from aura.proactive.theory_of_mind import get_theory_of_mind
                    tom = get_theory_of_mind()
                    user_model_ctx = tom.get_context_for_prompt()
                    if user_model_ctx:
                        additions.append(user_model_ctx)
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
            # === MOTIVATION INJECTION (Phase 6E: Intrinsic Motivation) ===
            try:
                from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
                im = get_intrinsic_motivation()
                motivation_ctx = im.get_context_for_prompt()
                if motivation_ctx:
                    additions.append(motivation_ctx)
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
            # Global Workspace Theory injection removed
            # === WORLD STATE INJECTION (ADV-02: Persistent World Model) ===
            try:
                from aura.consciousness.world_model import get_world_model
                wm = get_world_model()
                world_ctx = wm.get_context_summary()
                if world_ctx:
                    additions.append(world_ctx)
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
            result = "\n\n".join(additions)
            # Cap module additions to 4K chars to prevent context overflow
            if len(result) > 4000:
                logger.warning(f"[BRAIN] System additions too large ({len(result)} chars), truncating to 4000")
                result = result[:4000]
            self._cached_system_additions = result
            self._system_additions_ts = time.time()
            return result

    @staticmethod
    def _classify_budget(query: str) -> int:
        q = query.lower().strip()
        # Only classify truly minimal responses (yes/no/ok) as small
        if q in ("yes", "no", "ok", "thanks", "thx", "bye", "k"):
            return Config.BUDGET_SMALL   # one-word replies
        if any(kw in q for kw in ("explain", "analyze", "compare", "research", "write", "implement")):
            return Config.BUDGET_LARGE   # complex
        return Config.BUDGET_MEDIUM       # default (greetings, questions, etc.)

    @staticmethod
    def _build_budget_instruction(budget: int) -> str:
        return f"\n\n[Response budget: ~{budget} tokens. Be appropriately concise.]"

    def _build_full_system_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str],
        tone_modifier: Optional[str],
    ) -> str:
        """Build the complete system prompt for a think/think_stream call.

        Shared by think() and think_stream() to avoid duplicate logic:
        identity → caller system_prompt → subsystem context (TTL-cached) →
        emotional tone → ALMA modulation → budget instruction.
        """
        identity_prompt = get_identity_prompt()
        full = f"{identity_prompt}\n\n{system_prompt}" if system_prompt else identity_prompt

        # === SUBSYSTEM CONTEXT INJECTION (cached, TTL=12s) ===
        # Side-effect calls (observe/record) still run every time; only context
        # retrieval is cached to avoid 8+ module round-trips on rapid messages.
        try:
            if Config.MULTI_USER_ENABLED:
                from aura.multi_user import get_multi_user_manager
                manager = get_multi_user_manager()
                user_model = manager.get_active_user_model()
                if user_model:
                    user_model.observe_message(prompt, role="user")
            else:
                from aura.proactive.theory_of_mind import get_theory_of_mind
                tom = get_theory_of_mind()
                tom.observe_message(prompt, role="user")
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        try:
            from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
            get_intrinsic_motivation().record_interaction()  # Satisfies social drive
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        sys_additions = self._get_cached_system_additions()
        if sys_additions:
            full = f"{full}\n\n{sys_additions}"

        # === PROJECT CONTEXT INJECTION (AURA.md + auto-detect fallback) ===
        try:
            _now = time.time()
            _cwd = os.getcwd()
            # 60-second cache to avoid re-detecting every query
            if (not hasattr(self, '_project_ctx_cache')
                    or self._project_ctx_cache is None
                    or _now - getattr(self, '_project_ctx_ts', 0) > 60
                    or getattr(self, '_project_ctx_cwd', '') != _cwd):
                from aura.tools.project_context import detect_and_load_context
                self._project_ctx_cache = detect_and_load_context(_cwd)
                self._project_ctx_ts = _now
                self._project_ctx_cwd = _cwd

            ctx = self._project_ctx_cache
            if ctx and ctx.get("has_aura_md"):
                full = f"{full}\n\n## Active Project Context\n{ctx['aura_md_content']}"
            elif ctx and ctx.get("project_type") and ctx["project_type"] != "unknown":
                parts = [f"**Type:** {ctx['project_type']}"]
                if ctx.get("stack"):
                    parts.append(f"**Stack:** {', '.join(ctx['stack'])}")
                if ctx.get("frameworks"):
                    parts.append(f"**Frameworks:** {', '.join(ctx['frameworks'])}")
                if ctx.get("key_files"):
                    parts.append(f"**Key Files:** {', '.join(ctx['key_files'][:10])}")
                full = f"{full}\n\n## Auto-Detected Project Context\n" + "\n".join(parts)
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        # === SEMANTIC CODEBASE CONTEXT ===
        # Skip expensive operations if prompt is already near 12K cap
        MAX_SYSTEM_PROMPT_CHARS = 12000
        try:
            from aura.tools.codebase_index import CodebaseIndex
            _cwd = os.getcwd()
            _idx_db = Path(_cwd) / ".aura" / "index.db"
            if _idx_db.exists() and len(full) < MAX_SYSTEM_PROMPT_CHARS - 1000:
                idx = CodebaseIndex(_cwd)
                try:
                    if idx.stats()["total_chunks"] > 0:
                        relevant = idx.search(prompt, top_k=3)
                        if relevant and relevant[0]["score"] > 0.3:
                            ctx_parts = []
                            for r in relevant:
                                if r["score"] > 0.3:
                                    ctx_parts.append(f"**{r['file_path']}:{r['line_start']}** ({r['kind']} `{r['name']}`):\n```\n{r['content'][:300]}\n```")
                            if ctx_parts:
                                full = f"{full}\n\n## Relevant Code\n" + "\n\n".join(ctx_parts)
                finally:
                    idx.close()
        except Exception:
            pass

        # Apply emotional style prompt — behavioral directives from ALMA
        # The style prompt already encodes verbosity/formality/enthusiasm as behavior
        if tone_modifier:
            full = f"{full}\n\n{tone_modifier}"
        elif self._alma_enabled and self._auto_emotional_tone:
            try:
                from aura.emotion.integration import get_emotional_style_prompt
                alma_style = get_emotional_style_prompt()
                if alma_style:
                    full = f"{full}\n\n{alma_style}"
            except Exception as e:
                logger.debug(f"[BRAIN] ALMA style prompt failed: {e}")
        # === EPISODIC MEMORY AUTO-RECALL ===
        # Surface relevant past context for non-trivial queries (best-effort, never blocks)
        # Skip if already near budget cap
        if len(prompt) > 25 and len(full) < MAX_SYSTEM_PROMPT_CHARS - 500:
            try:
                if hasattr(self, '_episodic_memory') and self._episodic_memory:
                    memories = self._episodic_memory.quick_recall(prompt, limit=3)
                    if memories:
                        memory_ctx = "\n\n## Relevant Past Context\n"
                        for m in memories:
                            ts = m.get("timestamp", "")[:10] if m.get("timestamp") else ""
                            memory_ctx += f"- [{ts}] {m.get('title', '')}: {m.get('summary', '')}\n"
                        full = f"{full}{memory_ctx}"
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
        budget = self._classify_budget(prompt)
        full = f"{full}{self._build_budget_instruction(budget)}"

        # Safety: cap system prompt to ~12K chars (~3K tokens) to leave room
        # for conversation history and response in the model's context window.
        if len(full) > MAX_SYSTEM_PROMPT_CHARS:
            logger.warning(f"[BRAIN] System prompt too large ({len(full)} chars), truncating to {MAX_SYSTEM_PROMPT_CHARS}")
            full = full[:MAX_SYSTEM_PROMPT_CHARS] + "\n\n[System context truncated for length]"

        return full

    def think(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_history: bool = True,
        task_type: Optional[TaskType] = None,
        tone_modifier: Optional[str] = None
    ) -> str:
        """Generate a response using Ollama for reasoning tasks.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            use_history: Whether to include conversation history
            task_type: Type of task for model routing (auto-detected if None)
            tone_modifier: Optional emotional tone modifier from EvoEmo/ALMA
        """
        # Check if auto-reset is needed to prevent slowdown
        self._check_auto_reset()

        # ALMA: Process user message for emotional triggers
        if self._alma_enabled and use_history:
            try:
                process_user_message(prompt)
            except Exception as e:
                logger.debug(f"[BRAIN] ALMA message processing failed: {e}")

        # Select model based on task type, then apply outcome-aware routing overlay
        model = self._select_model(prompt, task_type)
        model = self._routing_stats_override(model, task_type)
        self._last_model_used = model

        full_system_prompt = self._build_full_system_prompt(prompt, system_prompt, tone_modifier)

        # Auto-compact: cloud models have 128K-256K context, compact at ~60% (~150 msgs)
        if use_history and len(self.conversation_history) > 150:
            try:
                summary = self.compact_history()
                if summary:
                    logger.info(f"[BRAIN] Auto-compacted history → {len(self.conversation_history)} msgs remain")
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
        messages = []
        if full_system_prompt:
            messages.append({"role": "system", "content": full_system_prompt})
        if use_history:
            with self._history_lock:
                messages.extend(self.conversation_history[-self._max_history:])
        messages.append({"role": "user", "content": prompt})

        # Neuromodulator: Serotonin modulates patience (timeout)
        # High serotonin = more patience = longer timeout; low = impatient = shorter
        neuro = _get_neuromodulator_levels()
        adjusted_timeout = max(45, int(_neuro_scale(LLM_TIMEOUT, neuro["serotonin"], sensitivity=0.3)))
        _llm_start_ts = time.time()  # Track LLM latency for routing stats
        logger.debug(f"[BRAIN] Calling {model} with timeout={adjusted_timeout}s (serotonin={neuro['serotonin']:.2f})")

        # Get appropriate client (local or cloud) - may return fallback model
        client, actual_model = self._get_client_for_model(model)
        if actual_model != model:
            logger.info(f"[BRAIN] Model fallback: {model} -> {actual_model}")

        # Update last model used to reflect ACTUAL model, not requested
        self._last_model_used = actual_model

        # === PHASE 1: Record real thinking — LLM inference starting ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tm.record_real_thought("formulating", f"reasoning with {actual_model}...", intensity=0.7, source="brain")
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        # Neuromodulator: Dopamine modulates temperature (creativity/exploration)
        # High dopamine = slightly higher temp = more creative; low = more conservative
        base_temp = 0.7
        adjusted_temp = round(_neuro_scale(base_temp, neuro["dopamine"], sensitivity=0.25), 2)

        # Neuromodulator: Serotonin modulates num_predict (response thoroughness)
        # High serotonin = patience = longer responses allowed; low = terse
        base_num_predict = 1024
        adjusted_num_predict = int(_neuro_scale(base_num_predict, neuro["serotonin"], sensitivity=0.3))

        # Neuromodulator: Norepinephrine modulates top_p (focus vs exploration)
        # High norepinephrine = alert/focused = lower top_p (more deterministic)
        # Low norepinephrine = relaxed = higher top_p (more varied responses)
        base_top_p = 0.9
        adjusted_top_p = round(base_top_p - (neuro["norepinephrine"] - 0.5) * 0.15, 2)
        adjusted_top_p = max(0.7, min(0.95, adjusted_top_p))

        # Neuromodulator: Acetylcholine modulates repeat_penalty (attention precision)
        # High acetylcholine = focused attention = higher repeat penalty (less repetitive)
        base_repeat_penalty = 1.1
        ach = neuro.get("acetylcholine", 0.5)
        adjusted_repeat_penalty = round(_neuro_scale(base_repeat_penalty, ach, sensitivity=0.15), 2)

        # Budget-forced num_predict: hard cap based on query complexity
        # The budget classification gives: conversational=BUDGET_SMALL, default=BUDGET_MEDIUM, complex=BUDGET_LARGE
        # We use 2x multiplier for breathing room while still enforcing a ceiling
        budget_tokens = self._classify_budget(prompt)
        budget_num_predict = budget_tokens * 2
        # Take the lower of neuromodulator-adjusted and budget-forced cap
        effective_num_predict = max(512, min(adjusted_num_predict, budget_num_predict))

        llm_options = {
            "temperature": adjusted_temp,
            "num_predict": effective_num_predict,
            "top_p": adjusted_top_p,
            "repeat_penalty": adjusted_repeat_penalty,
        }

        logger.debug(
            f"[BRAIN] Neuro-modulated LLM: temp={adjusted_temp} "
            f"(DA={neuro['dopamine']:.2f}), "
            f"num_predict={adjusted_num_predict} "
            f"(5HT={neuro['serotonin']:.2f}), "
            f"top_p={adjusted_top_p} "
            f"(NE={neuro['norepinephrine']:.2f})"
        )

        # Record neuromodulator influence on thinking panel
        try:
            from api.routes.thinking import record_thought
            neuro_effects = []
            if abs(neuro["dopamine"] - 0.5) > 0.1:
                neuro_effects.append(f"DA={'high' if neuro['dopamine']>0.5 else 'low'}")
            if abs(neuro["serotonin"] - 0.5) > 0.1:
                neuro_effects.append(f"5HT={'high' if neuro['serotonin']>0.5 else 'low'}")
            if abs(neuro["norepinephrine"] - 0.5) > 0.1:
                neuro_effects.append(f"NE={'high' if neuro['norepinephrine']>0.5 else 'low'}")
            if neuro_effects:
                record_thought(
                    "observing",
                    f"neuromodulators influencing response: {', '.join(neuro_effects)}",
                    0.4, "emotion"
                )
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        # Call with timeout protection (serotonin-modulated)
        response = call_with_timeout(
            lambda: client.chat(model=actual_model, messages=messages, options=llm_options),
            timeout=adjusted_timeout,
            default=None
        )

        if response is None:
            # ===== Phase 4 Fix 4B: Try fallback models in chain =====
            chain = self._get_fallback_chain(actual_model)
            for fallback_model in chain:
                if fallback_model == actual_model:
                    continue
                try:
                    fb_client, fb_actual = self._get_client_for_model(fallback_model)
                    logger.info(f"[BRAIN] Fallback attempt: {actual_model} → {fb_actual}")
                    response = call_with_timeout(
                        lambda m=fb_actual, c=fb_client: c.chat(model=m, messages=messages, options=llm_options),
                        timeout=adjusted_timeout,
                        default=None,
                    )
                    if response is not None:
                        actual_model = fb_actual
                        self._last_model_used = actual_model
                        logger.info(f"[BRAIN] Fallback succeeded with: {actual_model}")
                        break
                except Exception:
                    continue

        if response is None:
            logger.warning(f"[BRAIN] All models in chain failed, returning error message")
            # Record failure to routing stats
            _BG_EXECUTOR.submit(
                self._record_routing_outcome, actual_model, task_type, False,
                (time.time() - _llm_start_ts) * 1000
            )
            return "I'm having trouble processing that right now. Please try again."

        assistant_message = response["message"]["content"]

        # Record routing success to stats store (background, non-blocking)
        _BG_EXECUTOR.submit(
            self._record_routing_outcome, actual_model, task_type, True,
            (time.time() - _llm_start_ts) * 1000
        )

        # ===== Phase 4 Fix 4D: Track tokens and cost =====
        _in_tok = response.get("prompt_eval_count", 0) or 0
        _out_tok = response.get("eval_count", 0) or 0
        if _in_tok or _out_tok:
            self._record_tokens(actual_model, _in_tok, _out_tok)

        # ===== Phase 4 Fix 4C: Compaction notice =====
        if self._compaction_pending:
            self._compaction_pending = False
            assistant_message = (
                "_[Context compacted — older messages summarized to preserve memory]_\n\n"
                + assistant_message
            )

        if use_history:
            with self._history_lock:
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                # Enforce history limit (mirrors think_stream behaviour)
                if len(self.conversation_history) > self._max_history:
                    self.conversation_history = self.conversation_history[-self._max_history:]
                recent = list(self.conversation_history[-6:])
                _history_snapshot = list(self.conversation_history)
                _qc = self._query_count
                _tqc = self._total_query_count
            # Disk I/O outside the lock to avoid serializing concurrent requests
            self._save_history_snapshot(_history_snapshot, _qc, _tqc)
        else:
            recent = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_message},
            ]

        # === SELF-IMPROVEMENT: Record interaction outcome ===
        try:
            from aura.consciousness.self_improvement import get_self_improvement_engine
            _BG_EXECUTOR.submit(
                get_self_improvement_engine().record_chat_outcome,
                prompt, assistant_message, actual_model
            )
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        self._trigger_world_model_extraction(list(recent), _BG_EXECUTOR)

        return assistant_message

    def think_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_history: bool = True,
        task_type: Optional[TaskType] = None,
        tone_modifier: Optional[str] = None,
        model_override: Optional[str] = None
    ):
        """Generate a streaming response using Ollama for reasoning tasks.

        This is the streaming version of think() that yields chunks as they arrive.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            use_history: Whether to include conversation history
            task_type: Type of task for model routing (auto-detected if None)
            tone_modifier: Optional emotional tone modifier from EvoEmo/ALMA
            model_override: Explicit model to use (bypasses all routing)

        Yields:
            str: Response chunks as they are generated
        """
        # Check if auto-reset is needed to prevent slowdown
        self._check_auto_reset()

        # ALMA: Process user message for emotional triggers
        if self._alma_enabled and use_history:
            try:
                process_user_message(prompt)
            except Exception as e:
                logger.debug(f"[BRAIN] ALMA message processing failed: {e}")

        # Use explicit model_override if provided (thread-safe, no shared state)
        if model_override:
            logger.info(f"[BRAIN] Using explicit model override: {model_override}")
            model = model_override
        else:
            # Select model based on task type, then apply outcome-aware routing overlay
            model = self._select_model(prompt, task_type)
            model = self._routing_stats_override(model, task_type)
        self._last_model_used = model

        full_system_prompt = self._build_full_system_prompt(prompt, system_prompt, tone_modifier)

        # Auto-compact: cloud models have 128K-256K context, compact at ~60% (~150 msgs)
        if use_history and len(self.conversation_history) > 150:
            try:
                summary = self.compact_history()
                if summary:
                    logger.info(f"[BRAIN] Auto-compacted history → {len(self.conversation_history)} msgs remain")
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
        messages = []
        if full_system_prompt:
            messages.append({"role": "system", "content": full_system_prompt})
        if use_history:
            with self._history_lock:
                messages.extend(self.conversation_history[-self._max_history:])
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"[BRAIN] Streaming call to {model}")

        # Get appropriate client (local or cloud) - may return fallback model
        client, actual_model = self._get_client_for_model(model)
        if actual_model != model:
            logger.info(f"[BRAIN] Model fallback: {model} -> {actual_model}")

        # Update last model used to reflect ACTUAL model, not requested
        self._last_model_used = actual_model

        # === PHASE 1: Record real thinking — streaming inference starting ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tm.record_real_thought("formulating", f"streaming response with {actual_model}...", intensity=0.7, source="brain")
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        # Neuromodulator: Dopamine modulates temperature (creativity/exploration)
        neuro = _get_neuromodulator_levels()
        base_temp = 0.7
        adjusted_temp = round(_neuro_scale(base_temp, neuro["dopamine"], sensitivity=0.25), 2)

        # Neuromodulator: Serotonin modulates num_predict (response thoroughness)
        base_num_predict = 1024
        adjusted_num_predict = int(_neuro_scale(base_num_predict, neuro["serotonin"], sensitivity=0.3))

        # Neuromodulator: Norepinephrine modulates top_p (focus vs exploration)
        base_top_p = 0.9
        adjusted_top_p = round(base_top_p - (neuro["norepinephrine"] - 0.5) * 0.15, 2)
        adjusted_top_p = max(0.7, min(0.95, adjusted_top_p))

        # Neuromodulator: Acetylcholine modulates repeat_penalty (attention precision)
        base_repeat_penalty = 1.1
        ach = neuro.get("acetylcholine", 0.5)
        adjusted_repeat_penalty = round(_neuro_scale(base_repeat_penalty, ach, sensitivity=0.15), 2)

        # Budget-forced num_predict: hard cap based on query complexity
        # The budget classification gives: conversational=BUDGET_SMALL, default=BUDGET_MEDIUM, complex=BUDGET_LARGE
        # We use 2x multiplier for breathing room while still enforcing a ceiling
        budget_tokens = self._classify_budget(prompt)
        budget_num_predict = budget_tokens * 2
        # Take the lower of neuromodulator-adjusted and budget-forced cap
        effective_num_predict = max(512, min(adjusted_num_predict, budget_num_predict))

        llm_options = {
            "temperature": adjusted_temp,
            "num_predict": effective_num_predict,
            "top_p": adjusted_top_p,
            "repeat_penalty": adjusted_repeat_penalty,
        }

        full_response = ""

        # ===== Phase 4 Fix 4C: Compaction notice on streaming path =====
        if self._compaction_pending:
            self._compaction_pending = False
            notice = "_[Context compacted — older messages summarized to preserve memory]_\n\n"
            yield notice
            full_response += notice

        # ===== Phase 4 Fix 4B+4D: Streaming with fallback chain + token tracking =====
        _stream_in_tok = 0
        _stream_out_tok = 0
        _models_to_try = [actual_model] + [
            m for m in self._get_fallback_chain(actual_model) if m != actual_model
        ]

        _STREAM_STALE_TIMEOUT = 90  # seconds without a chunk → abort

        for _try_model in _models_to_try:
            try:
                _try_client, _try_actual = self._get_client_for_model(_try_model)
                if _try_model != _models_to_try[0]:
                    logger.info(f"[BRAIN] Stream fallback: {actual_model} → {_try_actual}")
                stream = _try_client.chat(model=_try_actual, messages=messages, stream=True, options=llm_options)
                _last_chunk_time = time.time()
                _stream_timed_out = False
                for chunk in stream:
                    now = time.time()
                    if now - _last_chunk_time > _STREAM_STALE_TIMEOUT:
                        logger.warning(f"[BRAIN] Stream stale for {_STREAM_STALE_TIMEOUT}s, aborting")
                        _stream_timed_out = True
                        break
                    _last_chunk_time = now
                    if chunk and "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        full_response += content
                        yield content
                    # Extract token counts from final done chunk
                    if chunk.get("done"):
                        _stream_in_tok = chunk.get("prompt_eval_count", 0) or 0
                        _stream_out_tok = chunk.get("eval_count", 0) or 0
                if _stream_timed_out:
                    raise TimeoutError(f"Stream stale for {_STREAM_STALE_TIMEOUT}s")
                actual_model = _try_actual
                self._last_model_used = actual_model
                break
            except Exception as e:
                if _try_model == _models_to_try[-1]:
                    import traceback
                    _tb = traceback.format_exc()
                    logger.error(f"[BRAIN] All stream models failed: {e}\n{_tb}")
                    fallback = "I'm having trouble processing that right now. Please try again."
                    yield fallback
                    full_response += fallback
                else:
                    logger.warning(f"[BRAIN] Stream model {_try_model} failed, trying next: {e}")
                continue

        if _stream_in_tok or _stream_out_tok:
            self._record_tokens(actual_model, _stream_in_tok, _stream_out_tok)

        # Update history after streaming completes
        if use_history and full_response:
            with self._history_lock:
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": full_response})
                # Enforce history limit
                if len(self.conversation_history) > self._max_history:
                    self.conversation_history = self.conversation_history[-self._max_history:]
                recent = list(self.conversation_history[-6:])
                _history_snapshot = list(self.conversation_history)
                _qc = self._query_count
                _tqc = self._total_query_count
            # Disk I/O outside the lock to avoid serializing concurrent requests
            self._save_history_snapshot(_history_snapshot, _qc, _tqc)
        else:
            recent = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": full_response},
            ]

        # === SELF-IMPROVEMENT: Record interaction outcome ===
        try:
            from aura.consciousness.self_improvement import get_self_improvement_engine
            _BG_EXECUTOR.submit(
                get_self_improvement_engine().record_chat_outcome,
                prompt, full_response, actual_model
            )
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        self._trigger_world_model_extraction(list(recent), _BG_EXECUTOR)

    def _trigger_world_model_extraction(self, recent: list, executor=None) -> None:
        """Submit background world model extraction (deduplicates logic from think/think_stream)."""
        try:
            from aura.consciousness.world_model import get_world_model
            wm = get_world_model()
            if not wm.enabled:
                return
            conv_id = self.get_current_conversation_id()
            if executor is not None:
                try:
                    executor.submit(_run_world_model_extraction, conv_id, recent)
                    return
                except RuntimeError:
                    pass  # Executor shut down — fall through to daemon thread
            threading.Thread(
                target=_run_world_model_extraction,
                args=(conv_id, recent),
                daemon=True,
                name=f"wm-extract-{conv_id[:8]}",
            ).start()
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
    def _is_complex_query(self, prompt: str) -> bool:
        """Detect if a query is complex and needs cloud model.

        Complex queries include:
        - Research/analysis requests
        - Multi-step reasoning
        - Comparisons requiring deep knowledge
        - Long-form content generation
        """
        prompt_lower = prompt.lower()
        words = prompt.split()

        # Long prompts are likely complex
        if len(words) > 50:
            return True

        # Complex task indicators — must be explicit task requests, not conversational references
        # Bad: 'research', 'review', 'tell me about' — match casual questions like
        #      "what do you think about this research?" → wrongly triggers 397B model
        complex_patterns = [
            'write an essay', 'write a report', 'write a detailed',
            'comprehensive analysis', 'in-depth analysis', 'thorough analysis',
            'deep dive into', 'deep search', 'investigate in detail',
            'pros and cons of', 'advantages and disadvantages',
            'step by step guide', 'detailed explanation of',
            'compare and contrast',
        ]

        if any(pattern in prompt_lower for pattern in complex_patterns):
            return True

        return False

    def set_model_override(self, model: Optional[str]) -> None:
        """Set a manual model override that bypasses auto-selection.

        Args:
            model: Model name to force, or None to return to auto-selection
        """
        self._model_override = model
        if model:
            logger.info(f"[BRAIN] Model override set: {model}")
        else:
            logger.info("[BRAIN] Model override cleared, returning to auto-selection")

    def _get_domain_confidence(self, prompt: str) -> tuple:
        """Get domain and confidence score from metacognition for a prompt.

        Returns:
            (domain_name: str, confidence: float) tuple.
            Falls back to (None, 0.5) if metacognition unavailable.
        """
        try:
            from aura.consciousness.metacognition import get_metacognitive_engine
            engine = get_metacognitive_engine()
            domain = engine.get_domain_for_query(prompt)
            if domain is None:
                return (None, 0.5)
            caps = engine.assess_capabilities()
            cap = caps.get(domain.value)
            if cap and cap.confidence > 0.1:
                return (domain.value, cap.score)
            return (domain.value, 0.5)
        except Exception:
            return (None, 0.5)

    def _should_escalate_to_system2(self, prompt: str, task_type: Optional[TaskType] = None) -> tuple:
        """Decide whether to use System 2 (deliberative) over System 1 (fast).

        Implements Kahneman-inspired dual-process routing:
        - Direct System 2 triggers for known complex patterns
        - Confidence-based escalation via metacognition
        - Neuromodulator tie-breaking for mid-range confidence

        Returns:
            (use_system2: bool, domain: str, confidence: float, reason: str)
        """
        # Direct System 2 triggers
        if self._is_complex_query(prompt):
            return (True, None, 0.0, "complex_query_heuristic")
        if task_type == TaskType.REASONING:
            return (True, None, 0.0, "explicit_reasoning_task")

        # Confidence-based escalation
        domain, confidence = self._get_domain_confidence(prompt)

        if confidence < Config.S2_CONFIDENCE_THRESHOLD:
            return (True, domain, confidence, "low_confidence")
        if confidence > Config.S1_CONFIDENCE_THRESHOLD:
            return (False, domain, confidence, "high_confidence")

        # Mid-range confidence: use neuromodulator state as tie-breaker
        neuro = _get_neuromodulator_levels()
        if neuro["norepinephrine"] > 0.6:
            return (True, domain, confidence, "high_norepinephrine")
        if neuro["dopamine"] > 0.7:
            return (False, domain, confidence, "high_dopamine")

        return (False, domain, confidence, "default_fast")

    # ------------------------------------------------------------------
    # Outcome-aware routing helpers
    # ------------------------------------------------------------------

    def _routing_stats_override(self, model: str, task_type: Optional[TaskType] = None) -> str:
        """Apply outcome-aware routing stats overlay to heuristic model selection.

        Only activates when ENABLE_OUTCOME_AWARE_ROUTING=True and RoutingStats
        has ≥MIN_SAMPLES data for the selected chain + microtask category.
        Falls back to heuristic model unchanged when data is insufficient.
        """
        if not getattr(Config, "ENABLE_OUTCOME_AWARE_ROUTING", True):
            return model
        try:
            from aura.reliability.routing_stats import get_routing_stats, MicrotaskCategory
            _CAT_MAP = {
                TaskType.CODE:      MicrotaskCategory.CODE_EDIT,
                TaskType.VISION:    MicrotaskCategory.LONG_DOC_EXTRACTION,
                TaskType.REASONING: MicrotaskCategory.TOOL_SELECTION,
                TaskType.SIMPLE:    MicrotaskCategory.GENERAL,
            }
            category = _CAT_MAP.get(task_type, MicrotaskCategory.GENERAL)
            chain = self._get_fallback_chain(model) or [model]
            stats_model = get_routing_stats().select_model_for_task(category, chain)
            if stats_model and stats_model != model:
                logger.info(
                    "[BRAIN] RoutingStats override: %s → %s (cat=%s)",
                    model, stats_model, category,
                )
                return stats_model
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        return model

    def _record_routing_outcome(
        self, model: str, task_type: Optional[TaskType], success: bool, latency_ms: float
    ) -> None:
        """Record routing outcome to RoutingStatsStore (called in background)."""
        try:
            from aura.reliability.routing_stats import get_routing_stats, MicrotaskCategory
            _CAT_MAP = {
                TaskType.CODE:      MicrotaskCategory.CODE_EDIT,
                TaskType.VISION:    MicrotaskCategory.LONG_DOC_EXTRACTION,
                TaskType.REASONING: MicrotaskCategory.TOOL_SELECTION,
                TaskType.SIMPLE:    MicrotaskCategory.GENERAL,
            }
            category = _CAT_MAP.get(task_type, MicrotaskCategory.GENERAL)
            get_routing_stats().record(category, model, success=success, latency_ms=latency_ms)
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
    def _select_model(self, prompt: str, task_type: Optional[TaskType] = None) -> str:
        """Select the appropriate model based on task type and complexity.

        SYSTEM 1/SYSTEM 2 HYBRID ROUTING (Kahneman dual-process):
        - System 1 (fast): Simple queries, high confidence → MODEL_FAST
        - System 2 (deliberative): Complex queries, low confidence → MODEL_REASON
        - Specialized: Vision/Code tasks use dedicated model chains
        - Cloud: Complex queries that need cloud-scale models

        Args:
            prompt: The prompt to analyze
            task_type: Explicit task type, or None for auto-detection

        Returns:
            Model name to use
        """
        # Check for manual override first
        if self._model_override:
            logger.info(f"[BRAIN] Using manual model override: {self._model_override}")
            return self._model_override

        # Short conversational queries always use fast model — skip all escalation logic.
        # No 397B model needed for "what do you think?" or "how does this work?"
        words = prompt.split()
        # Code/dev keywords override the short-query fast-model forcing
        _CODE_KWS = {
            'code', 'bug', 'fix', 'debug', 'test', 'tests', 'function',
            'script', 'error', 'implement', 'refactor', 'compile', 'run',
            'deploy', 'build', 'import', 'class', 'method', 'api',
            'database', 'query', 'sql', 'python', 'javascript',
        }
        if len(words) <= 5 and any(kw in prompt.lower() for kw in _CODE_KWS):
            logger.info(f"[BRAIN] Short code query ({len(words)} words) → code model")
            return Config.get_model("code")
        # Trivial queries (≤5 words): always fast model, no escalation possible
        if len(words) <= 5:
            logger.info(f"[BRAIN] Trivial query ({len(words)} words) → fast model (forced)")
            return Config.MODEL_FAST
        # Short queries (6-15 words): fast model unless complex
        if len(words) <= 15 and not self._is_complex_query(prompt):
            logger.info(f"[BRAIN] Short query ({len(words)} words) → fast model")
            return Config.MODEL_FAST

        use_cloud = self._is_complex_query(prompt)
        prompt_lower = prompt.lower()

        # Specialized task routing (Vision/Code have dedicated models)
        if task_type == TaskType.VISION or any(kw in prompt_lower for kw in ['image', 'picture', 'screenshot', 'photo', 'analyze image']):
            return Config.get_model("vision")

        if task_type == TaskType.CODE:
            return Config.get_model("code")

        # Code detection from prompt keywords
        code_patterns = [
            'calculate', 'compute', 'factorial', 'fibonacci', 'prime',
            'print(', 'import ', 'def ', 'python',
            'code', 'script', 'function', 'algorithm',
            'debug', 'fix this', 'fix the', 'write a script', 'implement',
            'refactor', 'class ', 'method', 'variable', 'loop',
            'error', 'exception', 'traceback', 'bug', 'syntax'
        ]
        if any(pattern in prompt_lower for pattern in code_patterns):
            return Config.get_model("code")

        # Identity questions always use reasoning model
        identity_patterns = [
            'what is your name', 'who are you', 'your name', 'are you called',
            'what should i call you', 'introduce yourself', 'tell me about yourself',
            'what are you', 'are you an ai', 'are you a bot', 'what model are you'
        ]
        if any(pattern in prompt_lower for pattern in identity_patterns):
            return Config.get_model("reason")

        # System 1/System 2 decision for all other queries
        use_s2, domain, confidence, reason = self._should_escalate_to_system2(prompt, task_type)

        # Apply explicit thinking-mode override + cognitive load
        try:
            from aura.thinking_mode import get_thinking_mode_manager
            tmm = get_thinking_mode_manager()
            use_s2, reason = tmm.get_effective_decision(use_s2)
            # Track this query in cognitive load window
            was_complex = use_cloud or (task_type == TaskType.CODE)
            tmm.cognitive_load.record_query(confidence, was_complex, use_s2)
        except Exception:
            pass  # Graceful fallback if thinking_mode not available

        if use_s2:
            model = Config.get_model("reason")
            logger.info(f"[BRAIN] System 2 (deliberative): domain={domain}, confidence={confidence:.2f}, reason={reason}, model={model}")
            return model
        else:
            model = Config.MODEL_FAST
            logger.info(f"[BRAIN] System 1 (fast): domain={domain}, confidence={confidence:.2f}, reason={reason}, model={model}")
            return model

    def get_last_model_used(self) -> str:
        """Get the model used in the last think() call."""
        return self._last_model_used

    def observe(self, context: dict) -> str:
        """Process observations about the current state.

        DEPRECATED: Use react_step() instead. Kept for backward compatibility.
        """
        prompt = f"""Context:
{self._format_context(context)}

List 3-5 key observations. Be brief."""

        return self.think(prompt, system_prompt=self._observer_prompt())

    def plan(self, goal: str, observations: str, available_tools: list[str]) -> str:
        """Create a plan to achieve the goal based on observations.

        DEPRECATED: Use react_step() instead. Kept for backward compatibility.
        """
        # === PHASE 1: Record real thinking — planning ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tm.record_real_thought("analyzing", f"planning approach for: {goal[:60]}", intensity=0.7, source="brain")
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        tool_descriptions = self._get_tool_descriptions(available_tools)

        # Store goal for decide_action reference
        self._current_goal = goal

        # Detect screenshot+vision combo for multi-step handling
        goal_lower = goal.lower()
        self._current_goal_is_screenshot_and_vision = (
            any(kw in goal_lower for kw in ('screenshot', 'screen capture', 'capture screen'))
            and any(kw in goal_lower for kw in ('and describe', 'and analyze', 'and tell me', 'then describe'))
        )

        prompt = f"""Goal: {goal}

Observations: {observations[:500]}

Available tools:
{tool_descriptions}

IMPORTANT RULES:
- For web/internet/online info → use web_search (code_executor CANNOT access the internet)
- For local files/folders → use filesystem or code_search
- For running code → use code_executor
- For screenshots → use screenshot tool
- For image analysis → use vision tool

Create a short 1-3 step plan. Be specific about which tool to use for each step."""

        return self.think(prompt, system_prompt=self._planner_prompt())

    def decide_action(self, plan: str, available_tools: list[str]) -> dict:
        """Decide the next action to take based on the plan.

        DEPRECATED: Use react_step() instead. Kept for backward compatibility.
        """
        # === PHASE 1: Record real thinking — deciding action ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tools_short = ", ".join(available_tools[:4])
            tm.record_real_thought("connecting", f"selecting tool from: {tools_short}", intensity=0.6, source="brain")
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        tool_descriptions = self._get_tool_descriptions(available_tools)

        # Special case: screenshot+vision combo where screenshot is already done
        is_screenshot_and_vision = getattr(self, '_current_goal_is_screenshot_and_vision', False)
        screenshot_path = getattr(self, '_last_screenshot_path', None)
        extra_context = ""
        if is_screenshot_and_vision and screenshot_path:
            extra_context = f"\nNote: Screenshot was already captured at: {screenshot_path}. Use vision tool to analyze it now."

        prompt = f"""Plan: {plan[:500]}
{extra_context}
Available tools:
{tool_descriptions}

Pick ONE action. Reply ONLY in this format:

TOOL: <tool_name>
ACTION: <actual code, path, query, or command>
REASONING: <why>

RULES:
- For web/internet/online info → use web_search (code_executor CANNOT access the internet)
- For local files/folders → use filesystem (list, read, write, delete)
- For code search/grep → use code_search
- For editing code files → use code_edit
- For running Python code → use code_executor
- For screenshots → use screenshot
- For image analysis → use vision
- For shell commands → use shell_executor

Examples:

TOOL: filesystem
ACTION: list C:/Users/asus/Desktop/MyProject
REASONING: see project contents

TOOL: web_search
ACTION: Bitcoin price today
REASONING: search internet for price

TOOL: code_search
ACTION: grep "def main" *.py
REASONING: find main function"""

        response = self.think(prompt, system_prompt=self._actor_prompt())
        return self._parse_action_response(response)

    def evaluate(self, action: str, result: str, goal: str) -> dict:
        """Evaluate the result of an action.

        DEPRECATED: Use deterministic evaluation in agent._evaluate_tool_result() instead.
        Kept for backward compatibility.
        """
        # === PHASE 1: Record real thinking — evaluating result ===
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tm.record_real_thought("observing", f"evaluating result of: {action[:50]}", intensity=0.5, source="brain")
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")
        # Truncate result to avoid overwhelming the model
        result_truncated = result[:1000] if len(result) > 1000 else result

        # Check for multi-step tasks (screenshot + vision)
        goal_lower = goal.lower()
        is_screenshot_and_vision = getattr(self, '_current_goal_is_screenshot_and_vision', False)

        # If this is a combined task and we just did screenshot, continue to vision
        if is_screenshot_and_vision and 'screenshot' in action.lower() and 'success' in result_truncated.lower():
            extra_instruction = """
IMPORTANT: This is a 2-step task (screenshot + describe/analyze).
If only screenshot was done, say NEXT: continue (still need to analyze the image).
Only say NEXT: complete if BOTH screenshot AND vision/description are done."""
        else:
            extra_instruction = ""

        prompt = f"""Goal: {goal}
Action: {action}
Result: {result_truncated}
{extra_instruction}
Reply ONLY in this format:

SUCCESS: yes OR no
CONFIDENCE: 0-100 (how confident are you the goal is fully achieved)
PROGRESS: one sentence about progress
NEXT: continue OR complete OR retry

If the goal is achieved, say NEXT: complete"""

        response = self.think(prompt, system_prompt=self._evaluator_prompt())
        return self._parse_evaluation_response(response)

    def summarize_for_memory(self, episode: dict) -> str:
        """Create a memory-worthy summary of an episode.

        DEPRECATED: Episode storage in react loop uses no LLM call.
        Kept for backward compatibility.
        """
        prompt = f"""Summarize in 2-3 sentences:
Goal: {episode.get('goal', 'N/A')}
Actions: {episode.get('actions', [])}
Outcome: {episode.get('outcome', 'N/A')}"""

        return self.think(prompt, system_prompt=self._memory_prompt(), use_history=False)

    def unload_model(self, model: str = None) -> bool:
        """Unload a model from Ollama to free VRAM.

        Args:
            model: Model name to unload. If None, unloads the last used model.

        Returns:
            True if successful, False otherwise.
        """
        model_to_unload = model or self._last_model_used
        try:
            # Send empty generate with keep_alive=0 to unload (with timeout)
            result = call_with_timeout(
                lambda: self.client.generate(
                    model=model_to_unload,
                    prompt="",
                    keep_alive="0s"
                ),
                timeout=10,
                default=None
            )
            return result is not None
        except Exception:
            return False

    def unload_all_models(self) -> dict:
        """Unload all commonly used models to free VRAM.

        Returns:
            Dict with unload status for each model.
        """
        models = [Config.MODEL_FAST, Config.MODEL_REASON, Config.MODEL_CODE, Config.MODEL_VISION]
        results = {}
        for model in models:
            results[model] = self.unload_model(model)
        return results

    def _format_context(self, context: dict) -> str:
        """Format context dictionary for prompts."""
        return "\n".join(f"- {k}: {v}" for k, v in context.items())

    def summarize(self, content: str, goal: str) -> str:
        """Summarize content in relation to a goal."""
        prompt = f"""Goal: {goal}

Content to summarize:
{content[:2000]}

Write a clear, concise summary (3-5 sentences) of the key points relevant to the goal."""

        return self.think(prompt, system_prompt="You summarize information clearly and concisely.", use_history=False)

    def _get_tool_descriptions(self, available_tools: list[str]) -> str:
        """Get clear descriptions for available tools."""
        descriptions = {
            "filesystem": "filesystem - list or read LOCAL files on this computer. ACTION: 'list <path>' or 'read <path>'",
            "web_search": "web_search - search the INTERNET for information. ACTION: the search query",
            "code_executor": "code_executor - run Python code and get the output. Use for calculations, data processing. ACTION: the Python code",
            "screenshot": "screenshot - capture a screenshot of the screen. ACTION: 'capture' or 'capture region x y width height'",
            "vision": "vision - analyze images using AI vision model. ACTION: 'analyze <image_path>' or 'describe screen <path>' or 'read text <path>'",
            "pdf_reader": "pdf_reader - read, extract text, or search PDF files. ACTION: 'read <path>' or 'extract <path> pages 1-5' or 'search <path> query' or 'info <path>'",
            "clipboard": "clipboard - read, write, or analyze clipboard content. ACTION: 'read' or 'write <text>' or 'analyze'",
            "system_control": "system_control - get system info (CPU, RAM, GPU, disk), control volume/brightness, open apps, lock screen. ACTION: 'get_system_info' or 'get_volume' or 'set_volume <level>' or 'open_app <name>'",
            "notifications": "notifications - set reminders, schedule notifications, create conditional alerts. ACTION: 'add_reminder <msg> in <time>' or 'add_scheduled <msg> <time> <repeat>' or 'list' or 'remove <id>'",
            "tool_builder": "tool_builder - create, test, enable, disable, or list custom tools. ACTION: 'list' or 'test <name>' or 'enable <name>' or 'disable <name>' or 'rollback <name>'",
            "summarize": "summarize - summarize gathered information. ACTION: 'results'",
            "calendar": "calendar - manage events, appointments, schedules. ACTION: 'add <title> on <date> at <time>' or 'today' or 'upcoming' or 'list <date>' or 'remove <id>' or 'search <query>'",
            "shell_executor": "shell_executor - execute shell/terminal commands with persistent sessions. ACTION: the command to run (e.g. 'ls -la', 'git status', 'python script.py')",
            "screen_reader": "screen_reader - read text from screen via OCR, detect active window, monitor for changes. ACTION: 'read' or 'read_region x y w h' or 'active_window' or 'watch <keyword>'",
            "email": "email - read and send emails. ACTION: 'fetch' or 'fetch unread' or 'read <id>' or 'send to:<addr> subject:<subj> body:<text>' or 'search <query>' or 'setup'",
            "spaced_repetition": "spaced_repetition - flashcard learning with spaced repetition. ACTION: 'review' or 'add front:<q> back:<a>' or 'answer <id> <quality 0-5>' or 'due' or 'stats' or 'auto_generate <text>'",
            "task_manager": "task_manager - manage tasks, projects, kanban boards. ACTION: 'add <title>' or 'list' or 'board' or 'update <id> status:<status>' or 'projects' or 'overdue' or 'search <query>'",
            "api_tester": "api_tester - test HTTP APIs and REST endpoints. ACTION: 'GET <url>' or 'POST <url> body:<json>' or 'PUT <url> body:<json>' or 'DELETE <url>' or 'history' or 'inspect <id>'",
            "database": "database - query SQLite databases, inspect schemas, import/export CSV. ACTION: SQL query like 'SELECT * FROM table' or 'schema' or 'tables' or 'import <csv> <table>' or 'export <table>'",
            "audio_transcriber": "audio_transcriber - transcribe audio/video files to text using Whisper. ACTION: 'transcribe <path>' or 'translate <path>' or 'detect <path>' or 'list' or 'status'",
            "clipboard_history": "clipboard_history - clipboard history with search, pinning, categories. ACTION: 'capture' or 'list' or 'search <query>' or 'pin <id>' or 'restore <id>' or 'stats'",
            "research": "research - save, search, and organize research notes and findings. ACTION: 'save title:<title> content:<text> category:<cat>' or 'search <query>' or 'list' or 'list <category>' or 'read <filename>' or 'stats' or 'skills' or 'tag <tagname>'"
        }
        return "\n".join(descriptions.get(t, t) for t in available_tools)

    def _parse_action_response(self, response: str) -> dict:
        """Parse the action decision response with better extraction for local models."""
        result = {"tool": None, "action": None, "reasoning": None, "raw": response}

        # Try to find TOOL, ACTION, REASONING in the response
        for line in response.split("\n"):
            line = line.strip()
            if line.upper().startswith("TOOL:"):
                tool = line[5:].strip().lower()
                # Clean up common variations
                tool = tool.replace("**", "").replace("`", "").strip()
                if "code" in tool or "execute" in tool or "python" in tool or "run" in tool:
                    tool = "code_executor"
                elif "summar" in tool:
                    tool = "summarize"
                elif "web" in tool or "search" in tool:
                    tool = "web_search"
                elif "file" in tool or "fs" in tool:
                    tool = "filesystem"
                elif "screenshot" in tool or "screen" in tool or "capture" in tool:
                    tool = "screenshot"
                elif "vision" in tool or "llava" in tool or "image" in tool or "analyze" in tool:
                    tool = "vision"
                elif "pdf" in tool or "document" in tool:
                    tool = "pdf_reader"
                elif "clipboard" in tool or "copy" in tool or "paste" in tool:
                    tool = "clipboard"
                elif "system" in tool or "control" in tool:
                    tool = "system_control"
                elif "notif" in tool or "remind" in tool or "schedule" in tool or "alert" in tool:
                    tool = "notifications"
                elif "tool_builder" in tool or "builder" in tool or "create tool" in tool or "custom tool" in tool:
                    tool = "tool_builder"
                elif "calendar" in tool or "event" in tool or "schedule" in tool or "agenda" in tool:
                    tool = "calendar"
                elif "shell" in tool or "terminal" in tool or "command" in tool or "bash" in tool:
                    tool = "shell_executor"
                elif "screen_reader" in tool or "ocr" in tool or "monitor" in tool or "active window" in tool:
                    tool = "screen_reader"
                elif "email" in tool or "mail" in tool or "inbox" in tool or "send email" in tool:
                    tool = "email"
                elif "flashcard" in tool or "spaced" in tool or "repetition" in tool or "review card" in tool:
                    tool = "spaced_repetition"
                elif "task_manager" in tool or "task" in tool or "kanban" in tool or "todo" in tool:
                    tool = "task_manager"
                elif "api_tester" in tool or "api test" in tool or "http" in tool or "rest" in tool:
                    tool = "api_tester"
                elif "database" in tool or "sqlite" in tool or "sql" in tool or "db" in tool:
                    tool = "database"
                elif "audio" in tool or "transcrib" in tool or "whisper" in tool or "speech" in tool:
                    tool = "audio_transcriber"
                elif "clipboard_history" in tool or "clip hist" in tool:
                    tool = "clipboard_history"
                elif "research" in tool or "save research" in tool or "notes" in tool:
                    tool = "research"
                result["tool"] = tool
            elif line.upper().startswith("ACTION:"):
                action = line[7:].strip()
                # Clean up the action - remove common prefixes local models add
                action = self._clean_action(action)
                result["action"] = action
            elif line.upper().startswith("REASONING:"):
                result["reasoning"] = line[10:].strip()

        # Fallback: try to extract from less structured responses
        if not result["tool"]:
            response_lower = response.lower()
            # Check for code executor indicators
            if "code_executor" in response_lower or "python" in response_lower or "calculate" in response_lower or "factorial" in response_lower or "print(" in response:
                result["tool"] = "code_executor"
            # Check for filesystem indicators
            elif "filesystem" in response_lower or "list " in response_lower or "read " in response_lower or "directory" in response_lower:
                result["tool"] = "filesystem"
            elif "summarize" in response_lower or "summary" in response_lower:
                result["tool"] = "summarize"
            elif "web_search" in response_lower or "internet" in response_lower or "online" in response_lower:
                result["tool"] = "web_search"
            elif "screenshot" in response_lower or "capture screen" in response_lower:
                result["tool"] = "screenshot"
            elif "vision" in response_lower or "analyze image" in response_lower or "describe image" in response_lower:
                result["tool"] = "vision"
            elif "pdf_reader" in response_lower or "read pdf" in response_lower or "extract pdf" in response_lower:
                result["tool"] = "pdf_reader"
            elif "clipboard" in response_lower or "paste" in response_lower or "copy to" in response_lower:
                result["tool"] = "clipboard"
            elif "system_control" in response_lower or "system info" in response_lower or "cpu" in response_lower or "ram" in response_lower or "volume" in response_lower or "brightness" in response_lower:
                result["tool"] = "system_control"
            elif "notification" in response_lower or "reminder" in response_lower or "remind" in response_lower or "schedule" in response_lower or "alert" in response_lower:
                result["tool"] = "notifications"
            elif "tool_builder" in response_lower or "create tool" in response_lower or "custom tool" in response_lower or "list tools" in response_lower:
                result["tool"] = "tool_builder"
            elif "calendar" in response_lower or "event" in response_lower or "agenda" in response_lower or "appointment" in response_lower:
                result["tool"] = "calendar"
            elif "shell" in response_lower or "terminal" in response_lower or "command line" in response_lower or "run command" in response_lower:
                result["tool"] = "shell_executor"
            elif "screen reader" in response_lower or "ocr" in response_lower or "read screen" in response_lower or "active window" in response_lower:
                result["tool"] = "screen_reader"
            elif "email" in response_lower or "inbox" in response_lower or "send mail" in response_lower or "check mail" in response_lower:
                result["tool"] = "email"
            elif "flashcard" in response_lower or "spaced repetition" in response_lower or "review card" in response_lower:
                result["tool"] = "spaced_repetition"
            elif "task_manager" in response_lower or "kanban" in response_lower or "todo list" in response_lower or "project board" in response_lower:
                result["tool"] = "task_manager"
            elif "api_tester" in response_lower or "test api" in response_lower or "http request" in response_lower or "rest api" in response_lower:
                result["tool"] = "api_tester"
            elif "database" in response_lower or "sql query" in response_lower or "sqlite" in response_lower or "run query" in response_lower:
                result["tool"] = "database"
            elif "audio_transcriber" in response_lower or "transcribe" in response_lower or "speech to text" in response_lower or "whisper" in response_lower:
                result["tool"] = "audio_transcriber"
            elif "clipboard_history" in response_lower or "clipboard history" in response_lower or "clip history" in response_lower:
                result["tool"] = "clipboard_history"
            elif "save research" in response_lower or "research note" in response_lower or "save finding" in response_lower:
                result["tool"] = "research"

        if not result["action"] and result["tool"] == "web_search":
            # Try to extract a search query from the response
            result["action"] = self._extract_search_query(response)

        # If action looks like Python code, ensure tool is code_executor
        if result["tool"] != "code_executor" and result["action"]:
            if any(ind in result["action"] for ind in ['print(', 'import ', 'def ', 'for i in']):
                result["tool"] = "code_executor"

        return result

    def _generate_default_code(self) -> str:
        """Generate default Python code based on the current goal."""
        goal = getattr(self, '_current_goal', '').lower()

        # Prime number check
        if 'prime' in goal:
            # Extract number from goal
            numbers = re.findall(r'\d+', goal)
            if numbers:
                n = numbers[0]
                return f"n = {n}; is_prime = n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)); print(str(n) + ' is ' + ('' if is_prime else 'not ') + 'a prime number')"

        # Factorial
        if 'factorial' in goal:
            numbers = re.findall(r'\d+', goal)
            if numbers:
                n = numbers[0]
                return f"import math; print(f'factorial({n}) = {{math.factorial({n})}}')"

        # Fibonacci
        if 'fibonacci' in goal or 'fib' in goal:
            numbers = re.findall(r'\d+', goal)
            if numbers:
                n = numbers[0]
                return f"def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2); print(f'fibonacci({n}) = {{fib({n})}}')"

        # Default: just print hello
        return "print('Code executed successfully')"

    def _clean_action(self, action: str) -> str:
        """Clean up action string from verbose local model outputs."""
        # Remove common prefixes that local models add
        prefixes_to_remove = [
            "use web_search tool to search for",
            "use web_search to search for",
            "search for",
            "search the web for",
            "search:",
            "query:",
            "use filesystem to",
            "use the",
        ]

        action_lower = action.lower()
        for prefix in prefixes_to_remove:
            if action_lower.startswith(prefix):
                action = action[len(prefix):].strip()
                action_lower = action.lower()

        # Remove quotes if present
        action = action.strip('"\'')

        # Remove markdown formatting
        action = action.replace("**", "").replace("`", "")

        return action.strip()

    def _extract_search_query(self, response: str) -> str:
        """Extract a search query from a verbose response."""
        # Look for quoted text
        quoted = re.findall(r'["\']([^"\']+)["\']', response)
        if quoted:
            return quoted[0]

        # Look for text after common patterns
        patterns = [
            r'search (?:for |query[: ]+)?["\']?([^"\'\n]+)',
            r'query[: ]+([^\n]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip().strip('"\'')

        # Fallback: return a default query based on context
        return "latest news"

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from a verbose LLM response."""
        # Look for code blocks
        code_block = re.search(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        # Look for lines that look like Python code
        code_indicators = ['print(', 'import ', 'def ', 'for ', 'while ', 'if ', '=']
        for line in response.split('\n'):
            line = line.strip()
            if any(ind in line for ind in code_indicators):
                # This looks like code
                return line

        # Look for code after "ACTION:" anywhere in response
        action_match = re.search(r'ACTION:\s*(.+)', response, re.IGNORECASE)
        if action_match:
            return action_match.group(1).strip()

        return None

    def _parse_evaluation_response(self, response: str) -> dict:
        """Parse the evaluation response."""
        result = {"success": False, "confidence": 0, "progress": None, "next": None, "raw": response}

        response_lower = response.lower()

        for line in response.split("\n"):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            if line_lower.startswith("success:"):
                result["success"] = "yes" in line_lower or "true" in line_lower
            elif line_lower.startswith("confidence:"):
                # Extract numeric confidence value
                conf_str = line_stripped[11:].strip()
                # Extract first number found
                conf_match = re.search(r'\d+', conf_str)
                if conf_match:
                    result["confidence"] = min(100, max(0, int(conf_match.group())))
            elif line_lower.startswith("progress:"):
                result["progress"] = line_stripped[9:].strip()
            elif line_lower.startswith("next:"):
                next_val = line_stripped[5:].strip().lower()
                result["next"] = next_val
                # Check if goal is complete
                if "complete" in next_val or "done" in next_val or "achieved" in next_val:
                    result["success"] = True

        # Fallback detection
        if result["progress"] is None:
            if "success" in response_lower or "found" in response_lower:
                result["progress"] = "Made progress"

        return result

    def _default_system_prompt(self) -> str:
        return "You are a helpful AI assistant. Be concise and direct."

    def _observer_prompt(self) -> str:
        return "You analyze situations. List only key observations. Be very brief."

    def _planner_prompt(self) -> str:
        return """You create simple action plans. Be brief.
CRITICAL: For ANY calculation, math, Python, or code task -> use code_executor FIRST. Do NOT search the web for how to do it. Just write and execute the code directly.
For local files -> filesystem.
For internet info -> web_search."""

    def _actor_prompt(self) -> str:
        return """You select actions. Output ONLY: TOOL, ACTION, REASONING lines.
For code_executor: ACTION must be actual Python code with print() to show results.
Do NOT describe code - write the actual code!"""

    def _evaluator_prompt(self) -> str:
        return """You evaluate results. Follow the format exactly.
Say 'NEXT: complete' when the goal is achieved."""

    def _memory_prompt(self) -> str:
        return "Summarize in 2-3 short sentences. Focus on what happened and what was learned."

    # =========================================================================
    # ALMA Emotional Intelligence Methods
    # =========================================================================

    def set_emotional_tone(self, enabled: bool = True):
        """Enable or disable automatic emotional tone in responses.

        Args:
            enabled: Whether to automatically add emotional context to prompts
        """
        self._auto_emotional_tone = enabled
        logger.info(f"[BRAIN] Automatic emotional tone: {'enabled' if enabled else 'disabled'}")

    def trigger_emotional_response(self, emotion: str, intensity: float = 0.7, reason: str = "manual"):
        """Trigger an emotional response in AURA.

        Args:
            emotion: Name of emotion (joy, curious, excited, etc.)
            intensity: Strength of emotion (0.0 to 1.0)
            reason: Why this emotion was triggered
        """
        if self._alma_enabled:
            try:
                trigger_emotion(emotion, intensity, reason)
                logger.debug(f"[BRAIN] Triggered emotion: {emotion} ({intensity})")
            except Exception as e:
                logger.warning(f"[BRAIN] Failed to trigger emotion: {e}")

    def get_emotional_state(self) -> Optional[dict]:
        """Get AURA's current emotional state.

        Returns:
            Dictionary with emotional state info, or None if ALMA not available
        """
        if not self._alma_enabled:
            return None
        try:
            return alma_engine.get_emotional_state()
        except Exception as e:
            logger.warning(f"[BRAIN] Failed to get emotional state: {e}")
            return None

    def get_mood_emoji(self) -> str:
        """Get emoji representing AURA's current mood.

        Returns:
            Mood emoji string
        """
        if self._alma_enabled:
            try:
                return get_mood_emoji()
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
        return "🤖"

    def update_emotional_state(self, success: bool = True, user_satisfied: bool = True):
        """Update emotional state after an interaction.

        Args:
            success: Whether the response was successful
            user_satisfied: Whether the user seemed satisfied
        """
        if self._alma_enabled:
            try:
                process_response_outcome(success, user_satisfied)
            except Exception as e:
                logger.debug(f"[BRAIN] Failed to update emotional state: {e}")

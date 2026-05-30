"""Ollama API integration as the agent's reasoning engine."""

import atexit
import concurrent.futures
import json
import logging
import os
import threading
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Optional

import ollama

from aura.pools import bg_submit as _bg_submit
from aura.pools import llm_pool as _llm_pool_fn

from .brain_support import (
    LLM_TIMEOUT,
    _llm_retry,
    call_with_timeout,
)
from .brain_support import (
    get_neuromodulator_levels as _get_neuromodulator_levels,
)
from .brain_support import (
    is_rate_limit_error as _is_rate_limit_error,
)
from .brain_support import (
    neuro_scale as _neuro_scale,
)
from .brain_support import (
    resp_content as _resp_content,
)
from .brain_support import (
    resp_get as _resp_get,
)
from .brain_support import (
    run_world_model_extraction as _run_world_model_extraction,
)
from .brain_support import (
    user_facing_llm_error as _user_facing_llm_error,
)
from .config import Config
from .core.conversation_mixin import ConversationMixin
from .core.model_router_mixin import ModelRouterMixin
from .core.token_manager import estimate_messages_tokens, get_context_window
from .prompt_builder import SystemPromptBuilder, build_budget_instruction, classify_budget

# ChatGPT OAuth client (optional — uses ChatGPT Plus/Pro subscription)
try:
    from .auth.chatgpt_client import ChatGPTClient
    from .auth.chatgpt_oauth import is_authenticated as _chatgpt_authenticated
    CHATGPT_AVAILABLE = True
except ImportError:
    CHATGPT_AVAILABLE = False

logger = logging.getLogger(__name__)


# Module-level response cache — shared across all OllamaBrain instances so
# multiple sessions benefit from cache hits. Lazily created on first access.
_response_cache_singleton = None
_response_cache_lock = threading.Lock()


def _get_response_cache():
    """Return the process-wide ResponseCache, or None if init fails."""
    global _response_cache_singleton
    if _response_cache_singleton is not None:
        return _response_cache_singleton
    with _response_cache_lock:
        if _response_cache_singleton is None:
            try:
                from aura.core.response_cache import ResponseCache
                _response_cache_singleton = ResponseCache()
            except Exception:
                logger.debug("response_cache_init_failed", exc_info=True)
                return None
    return _response_cache_singleton

# ALMA Emotional Intelligence System
try:
    from .emotion.alma_engine import alma_engine, trigger_emotion
    from .emotion.integration import (
        get_mood_emoji,
        process_response_outcome,
        process_user_message,
    )
    ALMA_AVAILABLE = True
except ImportError:
    ALMA_AVAILABLE = False
    logger.debug("[BRAIN] ALMA emotional system not available")

WARMUP_TIMEOUT = 10  # 10 seconds for warmup
STREAM_STALE_TIMEOUT = 45  # seconds without a chunk before aborting stream

# Shared pools — centralized in aura.pools (2026-03-22)
# Use functions instead of cached references to avoid stale pool handles
# after pool re-initialization (e.g., in tests or daemon restarts).
# Don't cache pool references — call functions to get fresh handles
# atexit cleanup now handled by aura.pools

class TaskType(Enum):
    """Types of tasks for model routing."""
    SIMPLE = "simple"       # Greetings, short answers, basic queries
    REASONING = "reasoning" # Planning, evaluation, complex decisions
    CODE = "code"           # Code generation, calculations
    VISION = "vision"       # Image analysis


class OllamaBrain(ConversationMixin, ModelRouterMixin):
    """Handles all interactions with Ollama API for reasoning and decision-making."""

    # Limit conversation history to prevent unbounded memory growth
    MAX_HISTORY_LENGTH = Config.HISTORY_LIMIT  # Keep last N messages (N/2 exchanges)

    # Auto-reset context after this many queries to prevent slowdown
    AUTO_RESET_INTERVAL = Config.AUTO_RESET_INTERVAL  # Reset every N queries

    # Ollama cloud configuration
    OLLAMA_CLOUD_HOST = "https://api.ollama.com"

    def _refresh_local_ollama_status(self) -> None:
        """Check if local Ollama is reachable and has chat models.

        Called at init and periodically (every 120s) to recover from
        transient startup failures.
        """
        self._local_ollama_last_check = time.time()
        try:
            result = self.client.list()
            chat_models = [m.model for m in result.models
                          if not any(x in m.model.lower() for x in ("embed", "nomic", "ocr"))]
            if chat_models:
                if not self._local_ollama_ok:
                    logger.info(f"[BRAIN] Local Ollama has {len(chat_models)} chat models")
                self._local_ollama_ok = True
            else:
                self._local_ollama_ok = False
                logger.info("[BRAIN] Local Ollama has no chat models — using cloud for all requests")
        except Exception:
            self._local_ollama_ok = False
            logger.info("[BRAIN] Local Ollama not reachable — using cloud for all requests")

    def __init__(self, warmup: bool = True):
        # Local Ollama client (for local models)
        # Explicit timeout prevents infinite hangs on unresponsive models
        import httpx
        _ollama_timeout = httpx.Timeout(connect=3.0, read=60.0, write=10.0, pool=10.0)
        self.client = ollama.Client(host=Config.OLLAMA_HOST, timeout=_ollama_timeout)

        # Check if local Ollama has useful models (not just embedding/OCR models)
        self._local_ollama_ok = False
        self._local_ollama_last_check = 0.0
        # Run non-blocking — don't hang startup if Ollama is slow/unreachable
        threading.Thread(target=self._refresh_local_ollama_status, daemon=True).start()

        self._cloud_client = None  # Default to None — prevents AttributeError in routing
        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key:
            # Register with the credential pool so multi-key OLLAMA_API_KEY=k1,k2,k3
            # rotates on rate/billing failures.  Single-key setups pass through unchanged.
            try:
                from aura.providers.credential_pool import get_pool
                get_pool().register("ollama_cloud", "OLLAMA_API_KEY")
                pooled_key = get_pool().acquire("ollama_cloud") or api_key
            except Exception:
                pooled_key = api_key

            # Cloud models: 90s read timeout (cloud can be slower than local)
            _cloud_timeout = httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0)

            def _build_cloud_client(active_key: str) -> "ollama.Client":
                return ollama.Client(
                    host=self.OLLAMA_CLOUD_HOST,
                    headers={"Authorization": f"Bearer {active_key}"},
                    timeout=_cloud_timeout,
                )

            raw_cloud_client = _build_cloud_client(pooled_key)

            # Wrap the cloud client with classifier-driven retries so transient
            # 429/5xx/timeout failures on api.ollama.com recover transparently
            # instead of bubbling up as ResponseError.
            from aura.reliability.ollama_wrapper import ResilientOllamaClient
            self._cloud_client = ResilientOllamaClient(
                raw_cloud_client,
                provider_label="ollama_cloud",
                api_key=pooled_key,
                host=self.OLLAMA_CLOUD_HOST,
                timeout=_cloud_timeout,
                rebuild_client=_build_cloud_client,
            )
            logger.debug("[BRAIN] Ollama cloud client initialized (resilient)")
        else:
            logger.debug("[BRAIN] Warning: OLLAMA_API_KEY not set, cloud models unavailable")

        # ChatGPT client (for chatgpt: prefixed models)
        self._chatgpt_client = None
        if CHATGPT_AVAILABLE and _chatgpt_authenticated():
            self._chatgpt_client = ChatGPTClient()
            logger.info("[BRAIN] ChatGPT OAuth client initialized")

        self.model = Config.MODEL_NAME
        self._max_history: int = Config.HISTORY_LIMIT
        self.conversation_history: list[dict] = []
        self._history_lock = threading.RLock()
        self._last_model_used: str = self.model  # Track for metacognition
        # Warm-slot tracker — Ollama Pro sweet spot is 2 concurrent models warm.
        # LRU of most-recently-used models; dispatcher prefers these when scores tie.
        from collections import deque as _deque
        self._warm_models: _deque[str] = _deque(maxlen=2)
        self._warm_models_lock = threading.Lock()
        self._query_count: int = 0  # Track queries for auto-reset (resets every 15)
        self._total_query_count: int = 0  # Total queries (never resets)
        self._model_override: Optional[str] = None  # Manual model override (bypasses auto-selection)
        self._action_mode: Optional[str] = None  # Current action mode (set by agent_service per request)

        # Phase 4 — compaction notification flag + last-model tracker.
        # _state_lock protects the read-modify-write sequences in think/think_stream/
        # _auto_compact_if_needed, where a race could lose the notice or misreport
        # which model handled the most recent call.
        self._compaction_pending: bool = False
        self._state_lock = threading.Lock()

        # User inference priority: background tasks should yield when active.
        # Reference-counted so concurrent think()/think_stream() calls don't
        # prematurely clear the flag when one finishes before the other.
        self._user_inference_active = threading.Event()
        self._inference_refcount = 0
        self._inference_rc_lock = threading.Lock()
        self._user_inference_started_at: float = 0.0

        # Adaptive timeout based on recent LLM latencies (replaces serotonin-modulated timeout)
        self._recent_latencies: deque = deque(maxlen=100)
        self._latency_lock = threading.Lock()

        # Circuit breaker for think() — prevents repeated failures from cascading
        self._consecutive_think_failures: int = 0
        self._think_circuit_open_at: float = 0.0
        _THINK_CIRCUIT_BREAKER_COOLDOWN = 30  # seconds
        self._think_cb_cooldown: float = _THINK_CIRCUIT_BREAKER_COOLDOWN
        self._think_cb_threshold: int = 3  # consecutive failures before opening
        self._cb_lock = threading.Lock()  # Protects circuit breaker reads + writes

        # Phase 4 — per-session token/cost tracking
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._session_cost_usd: float = 0.0
        self._token_lock = threading.Lock()
        # Model costs derive from the catalog — single source of truth.
        from aura.models_catalog import MODELS as _CATALOG
        self._MODEL_COST_PER_1K: dict = {
            name: (p.cost_in_per_1k, p.cost_out_per_1k)
            for name, p in _CATALOG.items()
        }
        # ChatGPT subscription models (cost is $0 — covered by subscription)
        if CHATGPT_AVAILABLE:
            from .auth.chatgpt_client import ALL_CHATGPT_MODELS
            for m in ALL_CHATGPT_MODELS:
                self._MODEL_COST_PER_1K[m] = (0.0, 0.0)
        self._DEFAULT_COST_PER_1K = (0.003, 0.003)  # fallback for unknown models

        # Setup persistent history storage (legacy single-conversation path)
        # CHROMADB_PATH removed (2026-03-22); conversations live under data/
        _data_dir = Path(os.getenv("AURA_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
        self._history_dir = _data_dir / "conversation"
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._history_file = self._history_dir / "history.json"

        # Multi-conversation support
        self._conversations_dir = _data_dir / "conversations"
        self._conversations_dir.mkdir(parents=True, exist_ok=True)
        self._conversations_index_file = self._conversations_dir / "index.json"
        self._current_conversation_id: Optional[str] = None
        self._conversations_index_cache: dict | None = None
        self._conversations_index_lock = threading.Lock()

        # System prompt builder (owns project context + subsystem caches)
        self._prompt_builder = SystemPromptBuilder()

        # Migrate legacy history and initialize conversations
        self._migrate_legacy_history()
        self._load_history()

        # (system additions cache moved into SystemPromptBuilder)

        # ALMA Emotional Intelligence
        self._alma_enabled = ALMA_AVAILABLE
        self._auto_emotional_tone = True  # Automatically add emotional tone to responses
        if self._alma_enabled:
            logger.info(f"[BRAIN] ALMA emotional system enabled {get_mood_emoji()}")

        # Screenshot path tracking (set by agent, read by brain for combined screenshot+vision tasks)
        self._last_screenshot_path: Optional[str] = None

        # Circuit breaker state persistence — survives restarts so rapid-fail
        # loops don't hammer the LLM on reboot within the cooldown window.
        self._cb_state_file = _data_dir / "brain_state.json"
        self._load_cb_state()

        if warmup:
            self._warmup_models()

        # Register cleanup on process exit
        atexit.register(self.close)

    def _load_cb_state(self) -> None:
        """Restore circuit breaker state from disk if still within cooldown."""
        try:
            if not self._cb_state_file.exists():
                return
            data = json.loads(self._cb_state_file.read_text(encoding="utf-8"))
            failures = int(data.get("failures", 0))
            open_at = float(data.get("open_at", 0.0))
            if failures >= self._think_cb_threshold and (time.time() - open_at) < self._think_cb_cooldown:
                self._consecutive_think_failures = failures
                self._think_circuit_open_at = open_at
                logger.warning(
                    f"[BRAIN] Restored circuit breaker OPEN state from disk "
                    f"({failures} failures, {int(time.time() - open_at)}s ago)"
                )
            else:
                # Stale — clean it up
                self._cb_state_file.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"[BRAIN] Could not load circuit breaker state: {e}")

    def _save_cb_state(self) -> None:
        """Persist circuit breaker state atomically. Caller holds _cb_lock."""
        try:
            if self._consecutive_think_failures == 0:
                self._cb_state_file.unlink(missing_ok=True)
                return
            payload = {
                "failures": self._consecutive_think_failures,
                "open_at": self._think_circuit_open_at,
            }
            from aura.paths import atomic_write_json
            atomic_write_json(self._cb_state_file, payload, indent=0)
        except Exception as e:
            logger.debug(f"[BRAIN] Could not save circuit breaker state: {e}")

    def close(self) -> None:
        """Close HTTP clients and free connection pools."""
        try:
            if hasattr(self, 'client') and self.client is not None:
                self.client._client.close()
        except Exception as e:
            logger.debug(f"[BRAIN] Local client close failed: {e}")
        try:
            if hasattr(self, '_cloud_client') and self._cloud_client is not None:
                self._cloud_client._client.close()
        except Exception as e:
            logger.debug(f"[BRAIN] Cloud client close failed: {e}")
        try:
            if hasattr(self, '_chatgpt_client'):
                self._chatgpt_client = None
        except Exception as e:
            logger.debug(f"[BRAIN] ChatGPT client cleanup failed: {e}")
        logger.debug("[BRAIN] HTTP clients closed")


    # --- Model routing: see aura.core.model_router_mixin ---

    def _record_tokens(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Accumulate session token counts and estimated cost (Phase 4)."""
        in_rate, out_rate = self._MODEL_COST_PER_1K.get(model, self._DEFAULT_COST_PER_1K)
        cost = (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate
        with self._token_lock:
            self._session_input_tokens += input_tokens
            self._session_output_tokens += output_tokens
            self._session_cost_usd += cost
        # Cognitive heatmap: if an agentic loop is active and has set a tool
        # context, bump per-tool / per-file counters on it.
        try:
            loop = getattr(self, "_active_agentic_loop", None)
            if loop is not None:
                tool = getattr(loop, "_active_tool_for_heatmap", None)
                f = getattr(loop, "_active_file_for_heatmap", None)
                total = int(input_tokens) + int(output_tokens)
                if total > 0:
                    if tool:
                        loop._tokens_by_tool[tool] = loop._tokens_by_tool.get(tool, 0) + total
                    if f:
                        loop._tokens_by_file[f] = loop._tokens_by_file.get(f, 0) + total
        except Exception:
            logger.debug("cognitive_heatmap_tracking_failed", exc_info=True)

    def _consume_compaction_notice(self) -> bool:
        """Atomically test-and-clear the compaction-pending flag.

        Returns True exactly once per compaction event, even under concurrent
        think() / think_stream() calls, so a second call that races with the
        first sees False rather than emitting a duplicate notice.
        """
        with self._state_lock:
            if self._compaction_pending:
                self._compaction_pending = False
                return True
            return False

    def _touch_warm_slot(self, model: str) -> None:
        """Mark `model` as recently used. Keeps warm-set bounded to 2 (Ollama Pro sweet spot)."""
        if not model:
            return
        with self._warm_models_lock:
            try:
                if model in self._warm_models:
                    # Move to the right (most-recent end) by removing + re-appending
                    self._warm_models.remove(model)
            except ValueError:
                # Racy disappearance — harmless, fall through to append.
                pass
            self._warm_models.append(model)

    def get_warm_models(self) -> list[str]:
        """Return currently-warm models (order: oldest -> newest)."""
        with self._warm_models_lock:
            return list(self._warm_models)

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

    # -----------------------------------------------------------------
    # Shared helpers for tool-calling methods (think_with_tools,
    # think_with_tools_stream, react_step_code)
    # -----------------------------------------------------------------

    def _tool_call_with_fallback(
        self,
        client,
        actual_model: str,
        messages: list,
        llm_options: dict,
        tools: list | None = None,
        timeout: int = 120,
    ) -> tuple:
        """Execute an LLM tool call with fallback chain. Shared by tool-calling methods.

        Args:
            client: Ollama client
            actual_model: Model name
            messages: Message list
            llm_options: LLM options dict
            tools: Tool schemas (None for code-only calls)
            timeout: Timeout in seconds

        Returns:
            (response, actual_model) tuple. response is None if all models failed.
        """
        chat_kwargs = {"model": actual_model, "messages": messages, "options": llm_options}
        if tools is not None:
            chat_kwargs["tools"] = tools

        try:
            response = call_with_timeout(
                lambda: client.chat(**chat_kwargs),
                timeout=timeout,
                default=None,
            )
        except Exception as e:
            logger.error(f"[BRAIN] Tool call error ({actual_model}): {e}")
            return None, actual_model

        if response is None:
            chain = self._get_fallback_chain(actual_model)
            # Cumulative deadline: total fallback time capped at original timeout
            import time as _time
            _deadline = _time.monotonic() + timeout
            for fallback_model in chain:
                if fallback_model == actual_model:
                    continue
                remaining = _deadline - _time.monotonic()
                if remaining <= 5:  # Not enough time for another attempt
                    logger.info(f"[BRAIN] Fallback deadline reached, {len(chain)} models remaining")
                    break
                fb_timeout = min(int(remaining), 30)  # Cap each fallback at 30s
                try:
                    fb_client, fb_actual = self._get_client_for_model(fallback_model)
                    logger.info(f"[BRAIN] Tool-call fallback: {actual_model} -> {fb_actual} (timeout={fb_timeout}s)")
                    fb_kwargs = {"model": fb_actual, "messages": messages, "options": llm_options}
                    if tools is not None:
                        fb_kwargs["tools"] = tools
                    response = call_with_timeout(
                        lambda kw=fb_kwargs, c=fb_client: c.chat(**kw),
                        timeout=fb_timeout,
                        default=None,
                    )
                    if response is not None:
                        actual_model = fb_actual
                        break
                except Exception as e:
                    logger.warning(f"[BRAIN] Fallback model {fb_actual} failed: {e}")
                    continue

        return response, actual_model

    def _extract_tool_response_tokens(self, response, actual_model: str) -> tuple:
        """Extract and record token counts from a tool-call response.

        Returns:
            (input_tokens, output_tokens) tuple
        """
        input_tokens = _resp_get(response, "prompt_eval_count", 0) or 0
        output_tokens = _resp_get(response, "eval_count", 0) or 0
        self._record_tokens(actual_model, input_tokens, output_tokens)
        return input_tokens, output_tokens

    def think_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model_override: str | None = None,
        options: dict | None = None,
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
        client, actual_model, llm_options = self._resolve_tool_model(model_override, options)

        # ChatGPT: use prompt-based tool calling adapter
        if actual_model.startswith("chatgpt:"):
            return self._think_with_tools_chatgpt(messages, tools, actual_model, client)

        response, actual_model = self._tool_call_with_fallback(
            client, actual_model, messages, llm_options, tools=tools,
        )

        if response is None:
            return {"error": _user_facing_llm_error(model=actual_model)}

        input_tokens, output_tokens = self._extract_tool_response_tokens(response, actual_model)

        return {
            "message": _resp_get(response, "message", {}),
            "model": actual_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    # -----------------------------------------------------------------
    # ChatGPT prompt-based tool calling
    # -----------------------------------------------------------------

    def _think_with_tools_chatgpt(
        self,
        messages: list,
        tools: list,
        model: str,
        client,
    ) -> dict:
        """Tool calling for ChatGPT via prompt injection + XML parsing.

        Injects tool descriptions into the system prompt, sends to ChatGPT,
        parses <tool_call> blocks from the text response.
        Returns same format as think_with_tools() for transparent integration.
        """
        from aura.core.prompt_tool_adapter import parse_tool_calls

        # Build tool-augmented messages
        aug_messages = self._inject_tool_prompt(messages, tools)

        try:
            response = call_with_timeout(
                lambda: client.chat(model=model, messages=aug_messages),
                timeout=120,
                default=None,
            )
        except Exception as e:
            logger.error(f"[BRAIN] ChatGPT tool call failed: {e}")
            return {"error": str(e)}

        if response is None:
            return {"error": "ChatGPT request timed out"}

        text = _resp_content(response)
        content, tool_calls = parse_tool_calls(text)

        msg = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls

        in_tok = _resp_get(response, "prompt_eval_count", 0) or 0
        out_tok = _resp_get(response, "eval_count", 0) or 0
        self._record_tokens(model, in_tok, out_tok)

        return {
            "message": msg,
            "model": model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }

    def _think_with_tools_stream_chatgpt(
        self,
        messages: list,
        tools: list,
        model: str,
        client,
    ):
        """Streaming tool calling for ChatGPT.

        Streams text chunks, then parses tool calls from the accumulated text.
        Yields same tuple format as think_with_tools_stream().
        """
        from aura.core.prompt_tool_adapter import parse_tool_calls

        aug_messages = self._inject_tool_prompt(messages, tools)
        accumulated = ""
        display_buffer = ""
        in_tool_call = False  # Stop displaying once <tool_call> detected

        try:
            for chunk in client.chat(model=model, messages=aug_messages, stream=True):
                delta = _resp_content(chunk) if chunk else ""
                if not delta:
                    continue
                accumulated += delta

                if in_tool_call:
                    continue  # Silently accumulate, don't display

                # Check if this chunk or the buffer contains <tool_call>
                display_buffer += delta
                if "<tool_call>" in display_buffer or "<tool_call" in display_buffer:
                    # Flush everything before the tag, then stop displaying
                    before = display_buffer.split("<tool_call")[0].rstrip()
                    if before:
                        yield ("content", before)
                    in_tool_call = True
                    continue

                # No tag detected — safe to yield for display
                # But hold back last 12 chars in case tag spans chunks
                if len(display_buffer) > 12:
                    safe = display_buffer[:-12]
                    display_buffer = display_buffer[-12:]
                    yield ("content", safe)

            # Flush remaining buffer if no tool call was detected
            if not in_tool_call and display_buffer:
                yield ("content", display_buffer)

        except Exception as e:
            logger.error(f"[BRAIN] ChatGPT stream error: {e}")
            yield ("error", {"error": str(e)})
            return

        # Parse tool calls from full accumulated text
        content, tool_calls = parse_tool_calls(accumulated)

        if tool_calls:
            yield ("tool_calls", tool_calls)

        yield ("done", {
            "content": content,
            "tool_calls": tool_calls,
            "model": model,
        })

    def _inject_tool_prompt(self, messages: list, tools: list) -> list:
        """Inject tool descriptions into the system message."""
        if not tools:
            return messages

        from aura.core.prompt_tool_adapter import build_tool_prompt
        tool_text = build_tool_prompt(tools)

        aug = []
        injected = False
        for msg in messages:
            if msg.get("role") == "system" and not injected:
                aug.append({**msg, "content": msg.get("content", "") + "\n" + tool_text})
                injected = True
            else:
                aug.append(msg)

        # No system message found — prepend one
        if not injected:
            aug.insert(0, {"role": "system", "content": tool_text})

        return aug

    # -----------------------------------------------------------------
    # ReAct Step — single LLM call combining thought + action (1.1)
    # -----------------------------------------------------------------

    def react_step(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        model_override: str | None = None,
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
        model_override: str | None = None,
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
        client, actual_model, llm_options = self._resolve_tool_model(model_override)

        # Code agent uses same prompt-based approach for ChatGPT
        if actual_model.startswith("chatgpt:"):
            return self._think_with_tools_chatgpt(messages, [], actual_model, client)

        response, actual_model = self._tool_call_with_fallback(
            client, actual_model, messages, llm_options, tools=None,
        )

        if response is None:
            return {"error": _user_facing_llm_error(model=actual_model)}

        # Extract content from response
        content = _resp_content(response)

        # Track tokens
        input_tokens, output_tokens = self._extract_tool_response_tokens(response, actual_model)

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
        model_override: str | None = None,
        options: dict | None = None,
    ):
        """Streaming version of think_with_tools(). Yields (chunk_type, data) tuples.

        chunk_type: "content" | "tool_calls" | "done" | "error"
        data: str for content, list for tool_calls, dict for done/error
        """
        # Check circuit breaker (same as think() / think_stream())
        cb_response = self._check_think_circuit_breaker()
        if cb_response:
            yield ("error", {"error": cb_response})
            return

        client, actual_model, llm_options = self._resolve_tool_model(model_override, options)

        if actual_model.startswith("chatgpt:"):
            yield from self._think_with_tools_stream_chatgpt(messages, tools, actual_model, client)
            return

        try:
            _start_stream = _llm_retry(
                lambda: client.chat(
                    model=actual_model,
                    messages=messages,
                    tools=tools,
                    options=llm_options,
                    stream=True,
                )
            )
            stream = _start_stream()

            accumulated_content = ""
            tool_calls = None
            input_tokens = 0
            output_tokens = 0
            _last_chunk_time = time.time()

            for chunk in stream:
                now = time.time()
                if now - _last_chunk_time > STREAM_STALE_TIMEOUT:
                    logger.warning(f"[BRAIN] Tool stream stale for {STREAM_STALE_TIMEOUT}s, aborting")
                    yield ("error", {"error": f"Stream stale for {STREAM_STALE_TIMEOUT}s", "model": actual_model})
                    return
                _last_chunk_time = now
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

            # Reset circuit breaker on successful stream completion
            with self._cb_lock:
                if self._consecutive_think_failures:
                    self._consecutive_think_failures = 0
                    self._save_cb_state()

            yield ("done", {
                "content": accumulated_content,
                "tool_calls": tool_calls,
                "model": actual_model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })

        except Exception as e:
            error_msg = _user_facing_llm_error(e, model=actual_model)
            yield ("error", {"error": error_msg})

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
                raw_history = data.get("history", [])
                # Bound loaded history to prevent memory bloat from corrupted/huge files
                if len(raw_history) > self._max_history:
                    raw_history = raw_history[-self._max_history:]
                self.conversation_history = raw_history
                self._query_count = data.get("query_count", 0)
                self._total_query_count = data.get("total_query_count", 0)
                logger.info(f"[BRAIN] Loaded {len(self.conversation_history)} messages from history (total queries: {self._total_query_count})")
        except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
            logger.warning("[BRAIN] Could not load history: %s", e)
            self.conversation_history = []

    def _save_history(self) -> None:
        """Save conversation history to disk."""
        with self._history_lock:
            self._save_history_unlocked()

    def _save_history_unlocked(self) -> None:
        """Save conversation history to disk — caller MUST hold _history_lock."""
        # Snapshot metadata while holding the lock so the index update
        # is consistent with the serialized data, even if the history
        # list is mutated before the background write completes.
        snap_count = len(self.conversation_history)
        snap_last = (self.conversation_history[-1].get("content", "")
                     if self.conversation_history else None)
        snap_history = list(self.conversation_history)
        data_str = json.dumps(
            {
                "history": snap_history,
                "query_count": self._query_count,
                "total_query_count": self._total_query_count,
            },
            indent=2,
            ensure_ascii=False,
        )
        # Capture path so a concurrent conversation switch
        # cannot cause us to write to the wrong file in the background.
        path = self._history_file
        def _bg_write(p, d):
            try:
                p.write_text(d, encoding="utf-8")
            except OSError as _e:
                logger.warning(f"[BRAIN] Background history write failed: {_e}")
        _bg_submit(_bg_write, path, data_str)
        self._update_conversation_index_entry(
            snap_message_count=snap_count, snap_last_content=snap_last,
            snap_history=snap_history,
        )

    def _save_history_snapshot(self, history: list, query_count: int, total_query_count: int, history_path=None) -> None:
        """Save a pre-copied history list to disk (called OUTSIDE _history_lock).

        Serializes JSON on the calling thread (fast), then writes to disk
        in the background pool to avoid blocking request threads on I/O.

        Args:
            history_path: Path captured inside _history_lock by the caller.
                          Falls back to self._history_file if not provided.
        """
        try:
            data_str = json.dumps(
                {"history": history, "query_count": query_count, "total_query_count": total_query_count},
                indent=2, ensure_ascii=False,
            )
            path = history_path or self._history_file
            def _bg_write_snap(p, d):
                try:
                    p.write_text(d, encoding="utf-8")
                except OSError as _e:
                    logger.warning(f"[BRAIN] Background history write failed: {_e}")
            _bg_submit(_bg_write_snap, path, data_str)
            self._update_conversation_index_entry(
                snap_message_count=len(history),
                snap_last_content=history[-1].get("content", "") if history else None,
                snap_history=history,
            )
        except (IOError, RuntimeError) as e:
            logger.warning(f"[BRAIN] Could not save history snapshot: {e}")



    # --- Conversation management: see aura.core.conversation_mixin ---

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
        Skips if user inference is active to avoid contention.

        Args:
            prompt: The prompt to send
            timeout: Max seconds to wait for response

        Returns:
            Generated response string
        """
        try:
            from aura.pools import is_shutting_down
            if is_shutting_down():
                return ""
        except Exception:
            logger.debug("Shutdown check failed in _quick_generate", exc_info=True)

        # Yield to user inference — poll briefly in case it finishes soon.
        # Event.wait(timeout) returns immediately when already set, so we poll manually.
        if self._user_inference_active.is_set():
            deadline = time.monotonic() + 5.0
            while self._user_inference_active.is_set() and time.monotonic() < deadline:
                time.sleep(0.2)
            if self._user_inference_active.is_set():
                logger.debug("[BRAIN] _quick_generate skipped — user inference still active after 5s")
                return ""
        fast_model = Config.MODEL_FAST
        try:
            client, actual_model = self._get_client_for_model(fast_model)
            # Use bg_pool (not llm_pool) to avoid competing with user inference
            from aura.pools import bg_pool as _bg_pool_fn
            try:
                future = _bg_pool_fn().submit(
                    lambda: client.chat(
                        model=actual_model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                )
                response = future.result(timeout=timeout)
            except Exception:
                response = None
            if response is None:
                logger.warning(f"[BRAIN] Quick generate timed out after {timeout}s")
                return ""
            return _resp_content(response)
        except Exception as e:
            logger.error(f"[BRAIN] Quick generate failed: {e}")
            return ""

    def compact_history(self, focus: str | None = None) -> str:
        """Compact conversation history synchronously."""
        return self._do_compact_history(focus)

    def _do_compact_history(self, focus: str | None = None) -> str:
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
            snapshot_len = len(self.conversation_history)
        if len(history) < 6:
            return ""

        # Split: oldest 2/3 to summarize, keep recent 1/3
        split_point = (len(history) * 2) // 3
        old_messages = [m for m in history[:split_point] if m.get("role") != "system"]
        recent_messages = history[split_point:]

        # Build summary prompt
        conversation_text = "\n".join(
            f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')[:1000]}"
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

        # Replace history: summary as system message + recent messages.
        # Re-acquire lock and check if new messages arrived during the LLM call.
        new_history = [
            {"role": "system", "content": f"[Conversation summary] {summary}"},
            *recent_messages,
        ]
        with self._history_lock:
            current_len = len(self.conversation_history)
            if current_len > snapshot_len:
                # Messages were added while we were summarizing — append them
                new_messages = self.conversation_history[snapshot_len:]
                new_history.extend(new_messages)
                logger.info(
                    "[BRAIN] Preserved %d messages added during compaction",
                    len(new_messages),
                )
            elif current_len < snapshot_len:
                # History was truncated (e.g., by _update_history_and_cleanup)
                # while we were compacting. The current history is authoritative —
                # do NOT overwrite it with our stale new_history.
                logger.info(
                    "[BRAIN] History truncated during compaction (%d -> %d), "
                    "preserving current state — skipping compaction result",
                    snapshot_len, current_len,
                )
                self._compaction_pending = True
                return summary
            self.conversation_history = new_history
            self._compaction_pending = True
        self._save_history()

        logger.info(f"[BRAIN] Compacted {len(old_messages)} messages into summary, kept {len(recent_messages)} recent")
        return summary

    def _call_with_rate_limit_retry(self, func, timeout, max_retries=2, base_delay=1.0):
        """Submit func to the pool with timeout, retrying on 429 rate-limit errors.

        Unlike _retry_on_rate_limit, this sleeps on the *calling* thread between
        pool submissions — not inside the pool thread — preventing pool starvation.

        Returns None if all attempts fail or time out.
        """
        for attempt in range(max_retries + 1):
            # Bail immediately if the interpreter is shutting down to avoid
            # repeated RuntimeError noise from a dead pool.
            try:
                from aura.pools import is_shutting_down
                if is_shutting_down():
                    return None
            except Exception:
                logger.debug("Shutdown check failed in _run_with_retry", exc_info=True)
            try:
                future = _llm_pool_fn().submit(func)
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(f"LLM call timed out after {timeout}s")
                future.cancel()
                return None
            except concurrent.futures.CancelledError:
                return None
            except RuntimeError:
                # Pool shut down during interpreter teardown
                return None
            except Exception as e:
                if attempt < max_retries and _is_rate_limit_error(e):
                    delay = min(base_delay * (2 ** attempt), 3.0)
                    logger.warning(f"[BRAIN] Rate limited (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s")
                    time.sleep(delay)  # Sleep on CALLING thread, not pool thread
                else:
                    if isinstance(e, (ConnectionError, OSError)):
                        logger.error(f"LLM connection error: {e}")
                        return None
                    raise
        return None

    def _get_context_limit(self) -> int:
        """Return the context window size for the current model."""
        model = self._last_model_used or self.model or Config.MODEL_NAME
        return get_context_window(model)

    def _check_auto_reset(self):
        """Check if token-based auto-compaction is needed and perform it.

        Triggers background compaction when conversation history exceeds
        70% of the model's context window. This replaced the old count-based
        heuristic (query_count >= 15) which was a poor proxy for actual
        context usage.
        """
        with self._history_lock:
            self._query_count += 1
            self._total_query_count += 1  # Total count never resets
            history_snapshot = list(self.conversation_history)

        # Token-based compaction trigger: compact at 70% of context window
        history_tokens = estimate_messages_tokens(history_snapshot)
        context_limit = self._get_context_limit()
        threshold = int(context_limit * 0.7)

        if history_tokens <= threshold:
            return

        logger.info(
            "[BRAIN] Auto-compact triggered: %d tokens > %d (70%% of %d) (total queries: %d)",
            history_tokens, threshold, context_limit, self._total_query_count,
        )
        # Submit compaction to background executor (non-blocking)
        try:
            _bg_submit(self.compact_history)
        except Exception as e:
            logger.warning(f"[BRAIN] Auto-compact submission failed: {e}")

    @staticmethod
    def _classify_budget(query: str) -> int:
        return classify_budget(query)

    def _build_neuro_llm_options(self, prompt: str, neuro: dict) -> dict:
        """Build neuromodulator-adjusted LLM options. Shared by think() and think_stream()."""
        # Dopamine modulates temperature (creativity/exploration)
        base_temp = 0.7
        adjusted_temp = round(_neuro_scale(base_temp, neuro["dopamine"], sensitivity=0.25), 2)

        # Serotonin modulates num_predict (response thoroughness)
        base_num_predict = 1024
        adjusted_num_predict = int(_neuro_scale(base_num_predict, neuro["serotonin"], sensitivity=0.3))

        # Norepinephrine modulates top_p (focus vs exploration)
        base_top_p = 0.9
        adjusted_top_p = round(base_top_p - (neuro["norepinephrine"] - 0.5) * 0.15, 2)
        adjusted_top_p = max(0.7, min(0.95, adjusted_top_p))

        # Acetylcholine modulates repeat_penalty (attention precision)
        base_repeat_penalty = 1.1
        ach = neuro.get("acetylcholine", 0.5)
        adjusted_repeat_penalty = round(_neuro_scale(base_repeat_penalty, ach, sensitivity=0.15), 2)

        # Budget-forced num_predict cap
        budget_tokens = self._classify_budget(prompt)
        budget_num_predict = budget_tokens * 2
        effective_num_predict = max(512, min(adjusted_num_predict, budget_num_predict))

        logger.debug(
            f"[BRAIN] Neuro-modulated LLM: temp={adjusted_temp} "
            f"(DA={neuro['dopamine']:.2f}), "
            f"num_predict={effective_num_predict} "
            f"(5HT={neuro['serotonin']:.2f}), "
            f"top_p={adjusted_top_p} "
            f"(NE={neuro['norepinephrine']:.2f})"
        )

        return {
            "temperature": adjusted_temp,
            "num_predict": effective_num_predict,
            "top_p": adjusted_top_p,
            "repeat_penalty": adjusted_repeat_penalty,
        }

    def _get_adaptive_timeout(self) -> int:
        """Compute adaptive timeout from recent LLM latency history.

        Uses p95 of recent latencies * 1.5, clamped to [45, 120].
        Falls back to 60s if fewer than 10 samples.
        """
        with self._latency_lock:
            if len(self._recent_latencies) < 10:
                return LLM_TIMEOUT  # default 60s
            sorted_latencies = sorted(self._recent_latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
        return max(45, min(120, int(p95 * 1.5)))

    def _record_latency(self, elapsed: float) -> None:
        """Record a successful LLM call latency for adaptive timeout."""
        with self._latency_lock:
            self._recent_latencies.append(elapsed)

    def _check_think_circuit_breaker(self) -> Optional[str]:
        """Check if think() circuit breaker is open.

        Returns a degraded response string if the circuit is open and cooldown
        hasn't elapsed, or None if the circuit is closed (proceed normally).
        """
        with self._cb_lock:
            if self._consecutive_think_failures >= self._think_cb_threshold:
                elapsed = time.time() - self._think_circuit_open_at
                if elapsed < self._think_cb_cooldown:
                    remaining = int(self._think_cb_cooldown - elapsed)
                    logger.warning(
                        f"[BRAIN] Think circuit breaker OPEN — {self._consecutive_think_failures} "
                        f"consecutive failures, {remaining}s cooldown remaining"
                    )
                    return (
                        "I'm experiencing connectivity issues with the language model. "
                        f"Retrying in ~{remaining}s. Please try again shortly."
                    )
                # Cooldown elapsed — allow one attempt (half-open state)
                logger.info("[BRAIN] Think circuit breaker half-open — allowing retry")
        return None

    @staticmethod
    def _build_budget_instruction(budget: int) -> str:
        return build_budget_instruction(budget)

    def _build_full_system_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str],
        tone_modifier: Optional[str],
    ) -> str:
        """Build the complete system prompt — delegates to SystemPromptBuilder."""
        return self._prompt_builder.build(
            prompt,
            system_prompt,
            tone_modifier,
            action_mode=self._action_mode,
            alma_enabled=self._alma_enabled,
            auto_emotional_tone=self._auto_emotional_tone,
            skill_list_fn=getattr(self, 'skill_list_summaries', None),
        )

    # -----------------------------------------------------------------
    # Shared setup / teardown for think() and think_stream()
    # -----------------------------------------------------------------

    def _prepare_chat_think(
        self,
        prompt: str,
        system_prompt: Optional[str],
        use_history: bool,
        task_type: Optional[TaskType],
        tone_modifier: Optional[str],
        model_override: Optional[str],
    ) -> dict:
        """Shared pre-LLM setup for think() and think_stream().

        Performs: auto-reset, ALMA processing, model selection, system prompt
        building, auto-compaction, user_inference_active flag set.

        Returns a context dict with all values needed by the caller:
            model, full_system_prompt, task_type
        """
        self._check_auto_reset()

        # ALMA: Process user message for emotional triggers
        if self._alma_enabled and use_history:
            try:
                process_user_message(prompt)
            except Exception as e:
                logger.debug(f"[BRAIN] ALMA message processing failed: {e}")

        # Model selection
        if model_override:
            logger.info(f"[BRAIN] Using explicit model override: {model_override}")
            model = model_override
        else:
            model = self._select_model(prompt, task_type)
            model = self._routing_stats_override(model, task_type)
        self._last_model_used = model
        self._touch_warm_slot(model)

        full_system_prompt = self._build_full_system_prompt(prompt, system_prompt, tone_modifier)

        # Note: auto-compaction is handled by _check_auto_reset() above (via background executor).
        # A duplicate synchronous compaction was removed here to prevent double triggers.

        # Safety valve: clear stale flag from abandoned generators.
        # In CPython, generator finally blocks run on GC, but this guards
        # against delayed collection or reference cycles.
        if self._user_inference_active.is_set():
            elapsed = time.time() - getattr(self, '_user_inference_started_at', 0)
            if elapsed > 120:
                logger.warning("[BRAIN] Clearing stale _user_inference_active flag (%.0fs old)", elapsed)
                with self._inference_rc_lock:
                    self._inference_refcount = 0
                    self._user_inference_active.clear()

        return {
            "model": model,
            "full_system_prompt": full_system_prompt,
            "task_type": task_type,
        }

    def _build_chat_messages(
        self,
        prompt: str,
        full_system_prompt: str,
        use_history: bool,
    ) -> list:
        """Build the message list for think()/think_stream().

        Returns:
            List of message dicts ready for the LLM call.
        """
        messages = []
        if full_system_prompt:
            messages.append({"role": "system", "content": full_system_prompt})
        if use_history:
            with self._history_lock:
                # Deep copy history snapshot so concurrent think() calls
                # don't interleave messages during the LLM call window
                messages.extend([dict(m) for m in self.conversation_history[-self._max_history:]])
        messages.append({"role": "user", "content": prompt})
        return messages

    def _update_history_and_cleanup(
        self,
        prompt: str,
        assistant_message: str,
        actual_model: str,
        use_history: bool,
    ) -> list:
        """Shared post-LLM teardown for think() and think_stream().

        Updates conversation history, saves to disk, triggers self-improvement
        and world model extraction in background.

        Returns:
            recent messages list (for world model extraction).
        """
        with self._history_lock:
            if use_history:
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                if len(self.conversation_history) > self._max_history:
                    self.conversation_history = self.conversation_history[-self._max_history:]
                recent = list(self.conversation_history[-6:])
                _history_snapshot = list(self.conversation_history)
                _qc = self._query_count
                _tqc = self._total_query_count
                _hpath = self._history_file
            else:
                recent = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_message},
                ]
                _history_snapshot = None

        # Disk I/O outside the lock to avoid serializing concurrent requests
        if _history_snapshot is not None:
            self._save_history_snapshot(_history_snapshot, _qc, _tqc, history_path=_hpath)

        # Self-improvement: record interaction outcome (background)
        try:
            from aura.consciousness.self_improvement import get_self_improvement_engine
            _bg_submit(
                get_self_improvement_engine().record_chat_outcome,
                prompt, assistant_message, actual_model
            )
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")

        self._trigger_world_model_extraction(list(recent))
        return recent

    def think(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_history: bool = True,
        task_type: Optional[TaskType] = None,
        tone_modifier: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> str:
        """Generate a response using Ollama for reasoning tasks.

        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            use_history: Whether to include conversation history
            task_type: Type of task for model routing (auto-detected if None)
            tone_modifier: Optional emotional tone modifier from EvoEmo/ALMA
            model_override: Explicit model to use (bypasses all routing and stats)
        """
        try:
            from aura.pools import is_shutting_down
            if is_shutting_down():
                return _user_facing_llm_error(model="shutdown")
        except Exception:
            logger.debug("Shutdown check failed in think()", exc_info=True)

        # Circuit breaker: if too many consecutive failures, return degraded response
        cb_response = self._check_think_circuit_breaker()
        if cb_response is not None:
            return cb_response

        ctx = self._prepare_chat_think(
            prompt, system_prompt, use_history, task_type, tone_modifier, model_override,
        )
        model = ctx["model"]
        full_system_prompt = ctx["full_system_prompt"]

        self._user_inference_started_at = time.time()
        self._begin_user_inference()
        try:
            # Phase 1: Build messages (_build_chat_messages copies history under _history_lock)
            messages = self._build_chat_messages(prompt, full_system_prompt, use_history)
            client, actual_model = self._resolve_chat_client(model)
            neuro = _get_neuromodulator_levels()
            llm_options = self._build_neuro_llm_options(prompt, neuro)

            # Taint tracking: scan user messages for secrets and redact before LLM
            try:
                from aura.security.taint_tracker import get_tracker
                tracker = get_tracker()
                taint_matches, redacted_prompt = tracker.check_and_track(prompt, session_id="brain")
                if taint_matches:
                    logger.warning(
                        f"[BRAIN] Detected {len(taint_matches)} secret(s) in user message — "
                        f"redacting before LLM call"
                    )
                    # Replace the prompt in messages with the redacted version
                    for msg in reversed(messages):
                        if msg.get("role") == "user" and msg.get("content") == prompt:
                            msg["content"] = redacted_prompt
                            break
            except Exception as e:
                logger.warning(f"[BRAIN] Taint tracker failed — secrets may pass through undetected: {e}")

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

            # Response cache — skip the LLM entirely on a hit. Only safe when
            # history is OFF (cache keyed on prompt+system+model only) and no
            # neuromodulator temperature variability is in play.
            _cache = _get_response_cache()
            _cache_key_ok = (
                _cache is not None
                and not use_history
                and not tone_modifier
            )
            if _cache_key_ok:
                _cached = _cache.get(prompt, full_system_prompt or "", actual_model)
                if _cached is not None:
                    logger.info(f"[BRAIN] Response cache HIT for {actual_model}")
                    return _cached

            # Phase 2: LLM call (NO lock held — allows parallel think/fleet/debate)
            adjusted_timeout = self._get_adaptive_timeout()
            _llm_start_ts = time.time()
            logger.debug(f"[BRAIN] Calling {model} with adaptive timeout={adjusted_timeout}s")

            # Call with timeout protection (adaptive latency-based) + 429 retry.
            # Rate-limit retries sleep on the calling thread, not in the pool,
            # to prevent pool starvation.
            response = self._call_with_rate_limit_retry(
                lambda: client.chat(model=actual_model, messages=messages, options=llm_options),
                timeout=adjusted_timeout + 20,
            )

            if response is None:
                chain = self._get_fallback_chain(actual_model)
                for fallback_model in chain:
                    if fallback_model == actual_model:
                        continue
                    try:
                        fb_client, fb_actual = self._get_client_for_model(fallback_model)
                        logger.info(f"[BRAIN] Fallback attempt: {actual_model} -> {fb_actual}")
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
                    except Exception as e:
                        logger.warning(f"[BRAIN] Fallback model failed: {e}")
                        continue

            if response is None:
                logger.warning("[BRAIN] All models in chain failed, returning error message")
                with self._cb_lock:
                    self._consecutive_think_failures += 1
                    if self._consecutive_think_failures >= self._think_cb_threshold:
                        self._think_circuit_open_at = time.time()
                        logger.warning(
                            f"[BRAIN] Think circuit breaker OPENED after "
                            f"{self._consecutive_think_failures} consecutive failures — "
                            f"cooldown {self._think_cb_cooldown}s"
                        )
                    self._save_cb_state()
                _bg_submit(
                    self._record_routing_outcome, actual_model, task_type, False,
                    (time.time() - _llm_start_ts) * 1000
                )
                return _user_facing_llm_error(model=actual_model)

            assistant_message = _resp_content(response)

            # Record latency for adaptive timeout and reset circuit breaker
            _llm_elapsed = time.time() - _llm_start_ts
            self._record_latency(_llm_elapsed)
            with self._cb_lock:
                if self._consecutive_think_failures:
                    self._consecutive_think_failures = 0
                    self._save_cb_state()

            _bg_submit(
                self._record_routing_outcome, actual_model, task_type, True,
                _llm_elapsed * 1000
            )

            # Track tokens and cost
            _in_tok = _resp_get(response, "prompt_eval_count", 0) or 0
            _out_tok = _resp_get(response, "eval_count", 0) or 0
            if _in_tok or _out_tok:
                self._record_tokens(actual_model, _in_tok, _out_tok)

            # Compaction notice — atomic consume so a racing think_stream() won't
            # emit a duplicate and can't lose a fresh signal.
            if self._consume_compaction_notice():
                assistant_message = (
                    "_[Context compacted — older messages summarized to preserve memory]_\n\n"
                    + assistant_message
                )

            # Phase 3: Update history (uses _history_lock internally)
            self._update_history_and_cleanup(prompt, assistant_message, actual_model, use_history)

            # Response cache write — only for cache-eligible calls
            if _cache_key_ok and assistant_message and _cache is not None:
                try:
                    _cache.put(prompt, full_system_prompt or "", actual_model, assistant_message)
                except Exception:
                    logger.debug("response_cache_put_failed", exc_info=True)

            return assistant_message
        finally:
            self._end_user_inference()

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
        # Check circuit breaker (same as think() — prevents streaming when circuit is open)
        cb_response = self._check_think_circuit_breaker()
        if cb_response:
            yield cb_response
            return

        ctx = self._prepare_chat_think(
            prompt, system_prompt, use_history, task_type, tone_modifier, model_override,
        )
        model = ctx["model"]
        full_system_prompt = ctx["full_system_prompt"]

        self._user_inference_started_at = time.time()
        self._begin_user_inference()
        try:
            # Build messages (_build_chat_messages copies history under _history_lock)
            messages = self._build_chat_messages(prompt, full_system_prompt, use_history)
            logger.debug(f"[BRAIN] Streaming call to {model}")
            client, actual_model = self._resolve_chat_client(model)
            neuro = _get_neuromodulator_levels()
            llm_options = self._build_neuro_llm_options(prompt, neuro)

            full_response = ""
            _all_models_failed = False

            # Compaction notice on streaming path — atomic consume.
            if self._consume_compaction_notice():
                notice = "_[Context compacted — older messages summarized to preserve memory]_\n\n"
                yield notice
                full_response += notice

            # Streaming with fallback chain + token tracking
            _stream_in_tok = 0
            _stream_out_tok = 0
            _models_to_try = [actual_model] + [
                m for m in self._get_fallback_chain(actual_model) if m != actual_model
            ]

            _stale_timeout = STREAM_STALE_TIMEOUT

            for _try_model in _models_to_try:
                try:
                    _try_client, _try_actual = self._get_client_for_model(_try_model)
                    if _try_model != _models_to_try[0]:
                        logger.info(f"[BRAIN] Stream fallback: {actual_model} -> {_try_actual}")
                    _start_stream = _llm_retry(
                        lambda c=_try_client, m=_try_actual: c.chat(model=m, messages=messages, stream=True, options=llm_options)
                    )
                    stream = _start_stream()
                    _last_chunk_time = time.time()
                    _stream_timed_out = False
                    for chunk in stream:
                        now = time.time()
                        if now - _last_chunk_time > _stale_timeout:
                            logger.warning(f"[BRAIN] Stream stale for {_stale_timeout}s, aborting")
                            _stream_timed_out = True
                            break
                        _last_chunk_time = now
                        content = _resp_content(chunk) if chunk else ""
                        if content:
                            full_response += content
                            yield content
                        # Extract token counts from final done chunk
                        if _resp_get(chunk, "done", False):
                            _stream_in_tok = _resp_get(chunk, "prompt_eval_count", 0) or 0
                            _stream_out_tok = _resp_get(chunk, "eval_count", 0) or 0
                    if _stream_timed_out:
                        raise TimeoutError(f"Stream stale for {_stale_timeout}s")
                    actual_model = _try_actual
                    self._last_model_used = actual_model
                    break
                except Exception as e:
                    if _try_model == _models_to_try[-1]:
                        import traceback
                        _tb = traceback.format_exc()
                        logger.error("[BRAIN] All stream models failed: %s\n%s", e, _tb)
                        # Use _try_model (always defined) — _try_actual may be unassigned
                        # if _get_client_for_model raised before the tuple unpack
                        _err_model = locals().get("_try_actual", _try_model)
                        fallback = _user_facing_llm_error(e, model=_err_model)
                        yield fallback
                        full_response += fallback
                        _all_models_failed = True
                    else:
                        logger.warning(f"[BRAIN] Stream model {_try_model} failed, trying next: {e}")
                    continue

            if _stream_in_tok or _stream_out_tok:
                self._record_tokens(actual_model, _stream_in_tok, _stream_out_tok)

            # Reset circuit breaker on successful stream completion
            if not _all_models_failed:
                with self._cb_lock:
                    if self._consecutive_think_failures:
                        self._consecutive_think_failures = 0
                        self._save_cb_state()

            # Update history (skip if all models failed to avoid polluting history)
            if not _all_models_failed:
                self._update_history_and_cleanup(prompt, full_response, actual_model, use_history)
            else:
                # Still trigger self-improvement and world model extraction
                try:
                    from aura.consciousness.self_improvement import get_self_improvement_engine
                    _bg_submit(
                        get_self_improvement_engine().record_chat_outcome,
                        prompt, full_response, actual_model
                    )
                except Exception as e:
                    logger.debug(f"[Brain] non-critical: {e}")
                recent = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": full_response},
                ]
                self._trigger_world_model_extraction(list(recent))
        finally:
            self._end_user_inference()

    def _begin_user_inference(self):
        """Mark a user inference as active (reference-counted for concurrent calls)."""
        with self._inference_rc_lock:
            self._inference_refcount += 1
            self._user_inference_active.set()

    def _end_user_inference(self):
        """Mark a user inference as complete (only clears when all concurrent calls finish)."""
        with self._inference_rc_lock:
            self._inference_refcount = max(0, self._inference_refcount - 1)
            if self._inference_refcount == 0:
                self._user_inference_active.clear()

    def _trigger_world_model_extraction(self, recent: list, executor=None) -> None:
        """Submit background world model extraction (deduplicates logic from think/think_stream).

        Uses the shared bg_submit() path, which already has a semaphore-capped
        fallback for pool-shutdown; we no longer hand-roll an uncapped daemon
        thread here (that could spawn unboundedly during shutdown chaos).
        """
        try:
            from aura.consciousness.world_model import get_world_model
            wm = get_world_model()
            if not wm.enabled:
                return
            conv_id = self.get_current_conversation_id() or "unknown"
            if executor is not None:
                try:
                    executor.submit(_run_world_model_extraction, conv_id, recent, self._user_inference_active)
                    return
                except RuntimeError:
                    pass  # Fall through to the shared pool path.
            from aura.pools import bg_submit
            bg_submit(_run_world_model_extraction, conv_id, recent, self._user_inference_active)
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")

    def summarize(self, content: str, goal: str) -> str:
        """Summarize content in relation to a goal."""
        prompt = f"""Goal: {goal}

Content to summarize:
{content[:2000]}

Write a clear, concise summary (3-5 sentences) of the key points relevant to the goal."""

        return self.think(prompt, system_prompt="You summarize information clearly and concisely.", use_history=False)

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

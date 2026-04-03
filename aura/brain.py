"""Ollama API integration as the agent's reasoning engine."""

import os
import json
import logging
import threading
import time
import shutil
import concurrent.futures
import atexit
import uuid
from collections import deque
from pathlib import Path
from enum import Enum
from typing import Optional, Callable, Any
import ollama
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from .config import Config
from .identity import get_identity_prompt
from .core.token_manager import estimate_messages_tokens, get_context_window

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
        get_mood_emoji,
    )
    from .emotion.alma_engine import alma_engine, trigger_emotion
    ALMA_AVAILABLE = True
except ImportError:
    ALMA_AVAILABLE = False
    logger.warning("[BRAIN] ALMA emotional system not available")

# Default timeouts (in seconds)
LLM_TIMEOUT = 120  # 120 seconds for LLM calls (cloud models need more time)
WARMUP_TIMEOUT = 10  # 10 seconds for warmup

# Neuromodulator bounds for safety (multipliers on default values)
NEURO_MIN_MULTIPLIER = 0.7   # Never reduce below 70% of default
NEURO_MAX_MULTIPLIER = 1.4   # Never increase above 140% of default


_wm_extraction_lock = threading.Lock()  # Rate limiter: skip if previous extraction still running
_wm_consecutive_failures = 0  # Circuit breaker: disable after repeated failures
_WM_CIRCUIT_BREAKER_THRESHOLD = 3  # Disable after N consecutive failures
_WM_CIRCUIT_BREAKER_RESET_AFTER = 300  # Re-enable after 5 minutes
_wm_circuit_broken_at: float = 0.0
_wm_lock = threading.Lock()  # Protect circuit breaker globals


def _resp_content(response) -> str:
    """Extract content from an Ollama response (dict or Pydantic object)."""
    if response is None:
        return ""
    if isinstance(response, dict):
        msg = response.get("message", {})
        return msg.get("content", "") if isinstance(msg, dict) else ""
    # Pydantic ChatResponse object
    msg = getattr(response, "message", None)
    if msg is not None:
        return getattr(msg, "content", "") or ""
    return ""


def _resp_get(response, key, default=None):
    """Get a field from an Ollama response (dict or Pydantic)."""
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)

def _run_world_model_extraction(conversation_id, messages, user_inference_event=None):
    """Background thread target for world model extraction (ADV-02 Phase 2).

    Includes circuit breaker: after 3 consecutive failures, disables extraction
    for 5 minutes to prevent thread pool starvation.
    Skips if user inference is active to avoid contention.
    """
    global _wm_consecutive_failures, _wm_circuit_broken_at

    # Yield to user inference
    if user_inference_event and user_inference_event.is_set():
        logger.debug("[BRAIN] World model extraction skipped — user inference active")
        return

    # Circuit breaker check (protected by lock)
    with _wm_lock:
        if _wm_consecutive_failures >= _WM_CIRCUIT_BREAKER_THRESHOLD:
            if time.time() - _wm_circuit_broken_at < _WM_CIRCUIT_BREAKER_RESET_AFTER:
                logger.debug("[BRAIN] World model extraction circuit breaker OPEN — skipping")
                return
            else:
                logger.info("[BRAIN] World model extraction circuit breaker RESET — retrying")
                _wm_consecutive_failures = 0

    if not _wm_extraction_lock.acquire(blocking=False):
        logger.debug("[BRAIN] Skipping world model extraction — previous still running")
        return
    try:
        from aura.consciousness.world_model import get_world_model
        wm = get_world_model()
        wm.process_conversation(conversation_id, messages)
        with _wm_lock:
            _wm_consecutive_failures = 0
            _wm_circuit_broken_at = 0.0
    except Exception as e:
        with _wm_lock:
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
        _wm_extraction_lock.release()


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

# Shared pools — centralized in aura.pools (2026-03-22)
# Use functions instead of cached references to avoid stale pool handles
# after pool re-initialization (e.g., in tests or daemon restarts).
from aura.pools import llm_pool as _llm_pool_fn, bg_pool as _bg_pool_fn
# Don't cache pool references — call functions to get fresh handles
# atexit cleanup now handled by aura.pools


_BG_FALLBACK_SEM = threading.Semaphore(8)


def _bg_submit(fn, *args, **kwargs):
    """Submit work to the background pool, with fallback for shutdown."""
    try:
        _bg_pool_fn().submit(fn, *args, **kwargs)
    except RuntimeError:
        # Pool shut down — run in daemon thread as last resort, capped at 8
        if not _BG_FALLBACK_SEM.acquire(blocking=False):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "_bg_submit: fallback thread cap reached (8), dropping task %s", fn
            )
            return
        def _run_and_release():
            try:
                fn(*args, **kwargs)
            finally:
                _BG_FALLBACK_SEM.release()
        threading.Thread(target=_run_and_release, daemon=True).start()


# --- Retry decorator for transient network/connection errors ---
try:
    import httpx as _httpx
    _RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OSError, _httpx.TimeoutException)
except ImportError:
    _RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OSError)

# 2 attempts max (not 3) — fail fast, let model fallback chain handle it
_llm_retry = retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _ollama_health_check() -> tuple[bool, list[str]]:
    """Quick check if Ollama is reachable. Returns (is_ok, available_model_names)."""
    try:
        resp = requests.get(f"{Config.OLLAMA_HOST}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return True, models
        return False, []
    except Exception:
        return False, []


def _user_facing_llm_error(original_error: Exception | None = None, model: str | None = None) -> str:
    """Return a user-friendly error message after LLM call failure, with Ollama health context."""
    model_tag = f" ({model})" if model else ""
    ollama_ok, available_models = _ollama_health_check()

    if not ollama_ok:
        return (
            f"[LLM Error] Could not connect to Ollama{model_tag}.\n"
            f"  - Is Ollama running? Try: ollama serve\n"
            f"  - Is the model available? Try: ollama list\n"
            f"  - For cloud models, check OLLAMA_API_KEY in .env"
        )

    # Ollama is reachable — check if the requested model exists
    if model and available_models:
        # Normalize: strip :latest tag for comparison
        normalized = [m.split(":")[0] for m in available_models]
        model_base = model.split(":")[0]
        if model_base not in normalized and model not in available_models:
            return (
                f"[LLM Error] Model '{model}' not found on this Ollama instance.\n"
                f"  - Available models: {', '.join(available_models[:8])}\n"
                f"  - Pull it with: ollama pull {model}\n"
                f"  - For cloud models, check OLLAMA_API_KEY in .env"
            )

    if original_error:
        err_type = type(original_error).__name__
        if isinstance(original_error, (ConnectionError, OSError)):
            return (
                f"[LLM Error] Connection failed for {model or 'model'}: {err_type}\n"
                f"  - Ollama is reachable but the request failed.\n"
                f"  - The model may have crashed. Try: ollama run {model or '<model>'}"
            )
        if isinstance(original_error, TimeoutError) or "timeout" in str(original_error).lower():
            return (
                f"[LLM Error] Request timed out for {model or 'model'}.\n"
                f"  - The model may be too large for your hardware.\n"
                f"  - Try a smaller model or increase timeout."
            )
        return f"[LLM Error] {err_type} for {model or 'model'}: {original_error}"

    return (
        f"[LLM Error] No response from {model or 'the language model'}.\n"
        f"  - Try again or run: aura doctor"
    )


def call_with_timeout(func: Callable, timeout: int = LLM_TIMEOUT, default: Any = None) -> Any:
    """Execute a function with timeout protection using shared thread pool.

    Wraps the callable with tenacity retry (3 attempts, exponential backoff)
    for transient ConnectionError/TimeoutError/OSError before submitting.

    Args:
        func: Function to execute (should be a lambda or callable with no args)
        timeout: Timeout in seconds
        default: Value to return on timeout

    Returns:
        Function result or default on timeout
    """
    # No retry wrapping here — caller-level _retry_on_rate_limit handles retries.
    # Double retry (tenacity + caller) caused timeout multiplication.

    try:
        future = _llm_pool_fn().submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"LLM call timed out after {timeout}s")
            future.cancel()
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


def _is_rate_limit_error(e: Exception) -> bool:
    """Check if an exception is a rate limit (429) error."""
    msg = str(e).lower()
    return "429" in msg or "rate" in msg or "too many" in msg


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
        # Explicit timeout prevents infinite hangs on unresponsive models
        import httpx
        _ollama_timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        self.client = ollama.Client(host=Config.OLLAMA_HOST, timeout=_ollama_timeout)

        # Check if local Ollama has useful models (not just embedding/OCR models)
        self._local_ollama_ok = False
        try:
            result = self.client.list()
            chat_models = [m.model for m in result.models
                          if not any(x in m.model.lower() for x in ("embed", "nomic", "ocr"))]
            if chat_models:
                self._local_ollama_ok = True
                logger.info(f"[BRAIN] Local Ollama has {len(chat_models)} chat models")
            else:
                logger.info("[BRAIN] Local Ollama has no chat models — using cloud for all requests")
        except Exception:
            logger.info("[BRAIN] Local Ollama not reachable — using cloud for all requests")

        # Cloud Ollama client (for cloud models like kimi-k2.5:cloud)
        self._cloud_client = None
        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key:
            # Cloud models: 90s read timeout (cloud can be slower than local)
            _cloud_timeout = httpx.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0)
            self._cloud_client = ollama.Client(
                host=self.OLLAMA_CLOUD_HOST,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=_cloud_timeout,
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
        self._history_lock = threading.RLock()
        # Serialize concurrent think()/think_stream() calls to prevent
        # interleaved conversation history and token counter corruption
        self._think_lock = threading.Lock()
        self._last_model_used: str = self.model  # Track for metacognition
        self._query_count: int = 0  # Track queries for auto-reset (resets every 15)
        self._total_query_count: int = 0  # Total queries (never resets)
        self._model_override: Optional[str] = None  # Manual model override (bypasses auto-selection)
        self._action_mode: Optional[str] = None  # Current action mode (set by agent_service per request)

        # Phase 4 — compaction notification flag
        self._compaction_pending: bool = False

        # Fix 9B — user inference priority: background tasks should yield
        self._user_inference_active = threading.Event()

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
        # Rough USD cost per 1K tokens (input/output) for cloud models
        self._MODEL_COST_PER_1K: dict = {
            "nemotron-3-super:cloud":      (0.0004, 0.0004),
            "kimi-k2.5:cloud":             (0.003,  0.003),
            "qwen3.5:397b-cloud":          (0.004,  0.004),
            "qwen3.5:cloud":              (0.004,  0.004),
            "deepseek-v3.2:cloud":         (0.003,  0.003),
            "glm-5:cloud":                (0.003,  0.003),
            "qwen3-coder:480b-cloud":      (0.004,  0.004),
            "qwen3-coder-next:cloud":      (0.003,  0.003),
            "minimax-m2.7:cloud":          (0.004,  0.004),
            "minimax-m2.5:cloud":          (0.004,  0.004),
            "gpt-oss:120b-cloud":          (0.003,  0.003),
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

        # Project context cache (initialized here for thread safety)
        self._project_ctx_cache = None
        self._project_ctx_ts: float = 0.0
        self._project_ctx_cwd: str = ""

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

        # Episodic memory consolidated into UnifiedMemory (2026-03-22)
        self._episodic_memory = None

        if warmup:
            self._warmup_models()

        # Register cleanup on process exit
        import atexit
        atexit.register(self.close)

    def close(self) -> None:
        """Close HTTP clients and free connection pools."""
        try:
            if hasattr(self, 'client') and self.client is not None:
                self.client._client.close()
        except Exception:
            pass
        try:
            if hasattr(self, '_cloud_client') and self._cloud_client is not None:
                self._cloud_client._client.close()
        except Exception:
            pass
        try:
            if hasattr(self, '_chatgpt_client'):
                self._chatgpt_client = None
        except Exception:
            pass
        logger.debug("[BRAIN] HTTP clients closed")

    def _get_client_for_model(self, model: str) -> tuple:
        """Get the appropriate client (local, cloud, ChatGPT, or direct API) based on model name.

        Routing:
        - chatgpt:* models → ChatGPT OAuth client (Codex Responses API)
        - anthropic:*/openai:*/gemini:*/grok:*/perplexity:*/deepseek:*/minimax:*/qwen:*/kimi:*/glm:*
          → Direct API provider client
        - *-cloud / *:cloud models → Ollama cloud client (or local bridge)
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
                # Fall back to cloud client if available, not local Ollama
                if self._cloud_client:
                    return self._cloud_client, Config.MODEL_FAST
                return self.client, Config.MODEL_FAST

        # Direct API providers (anthropic:, openai:, gemini:, grok:, etc.)
        if ":" in model and not model.endswith(("-cloud", ":cloud", ":latest")):
            prefix = model.split(":")[0]
            try:
                from aura.providers import get_provider
                provider = get_provider(prefix)
                if provider and provider.is_configured():
                    logger.debug(f"[BRAIN] Using {provider.display_name} API for model: {model}")
                    return provider, model
                elif provider:
                    logger.warning(f"[BRAIN] {provider.display_name} API key not set, cannot use {model}")
                    # Fall back to fast model instead of silently routing
                    # the provider-prefixed name to local Ollama (which would fail).
                    if self._cloud_client:
                        return self._cloud_client, Config.MODEL_FAST
                    return self.client, Config.MODEL_FAST
            except Exception as e:
                logger.debug(f"[BRAIN] Provider lookup failed for {prefix}: {e}")
                # Same fallback on provider error
                if self._cloud_client:
                    return self._cloud_client, Config.MODEL_FAST
                return self.client, Config.MODEL_FAST

        if model.endswith(("-cloud", ":cloud")):
            # Cloud models: prefer the dedicated cloud client (api.ollama.com) if available,
            # fall back to local Ollama bridge (for setups where Ollama Pro runs locally).
            if self._cloud_client:
                logger.debug(f"[BRAIN] Using Ollama cloud API for model: {model}")
                return self._cloud_client, model
            logger.debug(f"[BRAIN] Using local Ollama bridge for cloud model: {model}")
            return self.client, model

        # For local models / "auto": check if local Ollama is actually reachable.
        # If not, route to cloud with a cloud-capable model instead of failing.
        if self._cloud_client and not self._local_ollama_ok:
            logger.info(f"[BRAIN] Local Ollama not available, routing '{model}' to cloud")
            cloud_model = model if model.endswith(":cloud") else Config.MODEL_FAST
            return self._cloud_client, cloud_model

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

    # -----------------------------------------------------------------
    # Shared helpers for tool-calling methods (think_with_tools,
    # think_with_tools_stream, react_step_code)
    # -----------------------------------------------------------------

    def _resolve_tool_model(
        self,
        model_override: str = None,
        options: dict = None,
    ) -> tuple:
        """Resolve model, client, and LLM options for tool-calling methods.

        Returns:
            (client, actual_model, llm_options) tuple
        """
        model = model_override or self._model_override or Config.MODEL_CODE
        client, actual_model = self._get_client_for_model(model)
        llm_options = options or {"temperature": 0.2, "num_predict": 4096}
        return client, actual_model, llm_options

    def _tool_call_with_fallback(
        self,
        client,
        actual_model: str,
        messages: list,
        llm_options: dict,
        tools: list = None,
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
        model_override: str = None,
        options: dict = None,
    ):
        """Streaming version of think_with_tools(). Yields (chunk_type, data) tuples.

        chunk_type: "content" | "tool_calls" | "done" | "error"
        data: str for content, list for tool_calls, dict for done/error
        """
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
            _STREAM_STALE_TIMEOUT = 45  # 45s between chunks (was 90s — too slow)
            _last_chunk_time = time.time()

            for chunk in stream:
                now = time.time()
                if now - _last_chunk_time > _STREAM_STALE_TIMEOUT:
                    logger.warning(f"[BRAIN] Tool stream stale for {_STREAM_STALE_TIMEOUT}s, aborting")
                    yield ("error", {"error": f"Stream stale for {_STREAM_STALE_TIMEOUT}s", "model": actual_model})
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
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"[BRAIN] Could not load history: {e}")
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
        """Load the conversations index (thread-safe)."""
        with self._conversations_index_lock:
            if self._conversations_index_cache is not None:
                return list(self._conversations_index_cache)
            try:
                if self._conversations_index_file.exists():
                    data = json.loads(self._conversations_index_file.read_text(encoding="utf-8"))
                    self._conversations_index_cache = data
                    return list(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[BRAIN] Could not load conversations index: {e}")
            self._conversations_index_cache = []
            return []

    def _invalidate_conversation_cache(self) -> None:
        """Invalidate the in-memory conversations index cache.

        Must be called by any method that mutates conversations (create, delete,
        rename, switch) so the next _load_conversations_index() re-reads from disk.
        """
        with self._conversations_index_lock:
            self._conversations_index_cache = None

    def _save_conversations_index(self, index: list) -> None:
        """Save the conversations index (thread-safe)."""
        with self._conversations_index_lock:
            try:
                self._conversations_index_file.write_text(
                    json.dumps(index, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                self._conversations_index_cache = index
            except IOError as e:
                logger.warning(f"[BRAIN] Could not save conversations index: {e}")

    def _update_conversation_index_entry(
        self,
        snap_message_count: int | None = None,
        snap_last_content: str | None = None,
        snap_history: list | None = None,
    ) -> None:
        """Update the current conversation's index entry with latest metadata.

        When called from _save_history_unlocked, snapshot values are passed so
        the index stays consistent with the serialised data even if the live
        history list changes between serialisation and this call.
        """
        if not self._current_conversation_id:
            return
        msg_count = snap_message_count if snap_message_count is not None else len(self.conversation_history)
        last_content = snap_last_content if snap_last_content is not None else (
            self.conversation_history[-1].get("content", "") if self.conversation_history else None
        )
        history_for_title = snap_history if snap_history is not None else self.conversation_history
        # Update the in-memory cache under the lock (fast, no I/O),
        # then flush to disk outside the lock via _BG_EXECUTOR.
        # Lock ordering invariant: _history_lock -> _conversations_index_lock
        with self._conversations_index_lock:
            index = self._conversations_index_cache
            if index is None:
                # Cold start: need disk read (rare — only if cache was invalidated)
                try:
                    if self._conversations_index_file.exists():
                        index = json.loads(
                            self._conversations_index_file.read_text(encoding="utf-8")
                        )
                    else:
                        index = []
                except Exception:
                    index = []
            for entry in index:
                if entry["id"] == self._current_conversation_id:
                    entry["updated_at"] = int(time.time())
                    entry["message_count"] = msg_count
                    if last_content is not None:
                        entry["preview"] = last_content[:100]
                    # Update title if still "New Chat" and we have messages
                    if entry["title"] == "New Chat" and msg_count > 0:
                        entry["title"] = self._auto_title(history_for_title)
                    break
            # Store a copy so concurrent mutations don't affect the bg write
            self._conversations_index_cache = list(index)

        # Disk write outside lock, in background — uses the snapshot (immutable string)
        _index_snapshot = json.dumps(index, indent=2, ensure_ascii=False)
        def _bg_write_index(path, data):
            try:
                path.write_text(data, encoding="utf-8")
            except IOError as e:
                logger.warning(f"[BRAIN] Could not save conversations index: {e}")
        _bg_submit(_bg_write_index, self._conversations_index_file, _index_snapshot)

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
        # Return copies with is_active flag to avoid mutating cached index
        result = []
        for entry in index:
            entry_copy = dict(entry)
            entry_copy["is_active"] = entry["id"] == self._current_conversation_id
            result.append(entry_copy)
        return result

    def _validate_conversation_path(self, conversation_id: str) -> Path | None:
        """Validate that a conversation_id resolves to a safe path inside _conversations_dir.

        Returns the resolved path if safe, or None if the ID is invalid or escapes.
        """
        try:
            conv_dir = (self._conversations_dir / conversation_id).resolve()
            conv_dir.relative_to(self._conversations_dir.resolve())
        except (ValueError, OSError):
            logger.warning(f"[BRAIN] Blocked path traversal attempt: {conversation_id!r}")
            return None
        return conv_dir

    def switch_conversation(self, conversation_id: str) -> bool:
        """Switch to a different conversation.

        Args:
            conversation_id: ID of conversation to switch to

        Returns:
            True if switched successfully
        """
        if conversation_id == self._current_conversation_id:
            return True

        conv_dir = self._validate_conversation_path(conversation_id)
        if conv_dir is None or not conv_dir.exists():
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
        conv_dir = self._validate_conversation_path(conversation_id)
        if conv_dir is None or not conv_dir.exists():
            return False

        # Safety: only delete directories that contain a history.json (our canary)
        if not (conv_dir / "history.json").exists():
            logger.warning(f"[BRAIN] Refusing to delete non-conversation dir: {conv_dir}")
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
                # Sort by most recently updated so we switch to the latest conversation
                index.sort(key=lambda c: c.get("updated_at", 0), reverse=True)
                target_id = index[0]["id"]
                with self._history_lock:
                    target_dir = self._conversations_dir / target_id
                    self._current_conversation_id = target_id
                    self._history_file = target_dir / "history.json"
                    self._load_history()
            else:
                conv_id = self._create_conversation_dir("New Chat")
                with self._history_lock:
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
        if self._validate_conversation_path(conversation_id) is None:
            return False
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

        conv_dir = self._validate_conversation_path(conversation_id)
        if conv_dir is None:
            return []
        history_file = conv_dir / "history.json"
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text(encoding="utf-8"))
                return data.get("history", [])
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[Brain] Failed to load conversation history from {history_file}: {e}")
        return []

    def save_conversation_to_memory(self, conversation_id: Optional[str] = None) -> dict:
        """Save a conversation's content to AURA's long-term memory (UnifiedMemory).

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

        # Save to UnifiedMemory (primary — A-MEM removed 2026-03-22)
        try:
            from aura.memory.unified_memory import get_unified_memory
            umem = get_unified_memory()
            result = umem.store(
                content=memory_content,
                source="conversation_save",
                importance=0.6,
                tags=["conversation", "chat_history"],
            )
            logger.info(f"[BRAIN] Saved conversation {conv_id} to UnifiedMemory")
            return {
                "success": True,
                "note_id": result.get("store", ""),
                "message_count": len(messages),
                "title": title,
            }
        except Exception as e:
            logger.error(f"[BRAIN] Failed to save conversation: {e}")
            return {"success": False, "error": f"Memory save failed: {e}"}

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
        # Yield to user inference
        if self._user_inference_active.is_set():
            logger.debug("[BRAIN] _quick_generate skipped — user inference active")
            return ""
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
            return _resp_content(response)
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
            snapshot_len = len(self.conversation_history)
        if len(history) < 6:
            return ""

        # Split: oldest 2/3 to summarize, keep recent 1/3
        split_point = (len(history) * 2) // 3
        old_messages = [m for m in history[:split_point] if m.get("role") != "system"]
        recent_messages = history[split_point:]

        # Build summary prompt
        conversation_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content'][:1000]}"
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
            {"role": "system", "content": f"[Conversation summary] {summary}"}
        ] + recent_messages
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
                # History was truncated (e.g., by _update_history_and_cleanup).
                # The messages after the truncation point are already in the
                # current history — grab everything that's still there beyond
                # what we already included in recent_messages.
                logger.info(
                    "[BRAIN] History truncated during compaction (%d -> %d), "
                    "preserving current tail",
                    snapshot_len, current_len,
                )
            self.conversation_history = new_history
        self._save_history()

        logger.info(f"[BRAIN] Compacted {len(old_messages)} messages into summary, kept {len(recent_messages)} recent")
        return summary

    def _retry_on_rate_limit(self, func, max_retries=3, base_delay=2.0):
        """Retry a function call with exponential backoff on rate limit (429) errors."""
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                if attempt < max_retries and _is_rate_limit_error(e):
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"[BRAIN] Rate limited (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    raise

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
            _PER_SOURCE_CAP = 1000  # max chars per subsystem
            _TOTAL_CAP = 2000       # max chars for all additions combined

            def _cap(text: str, source: str) -> str:
                """Cap a subsystem's contribution and log if it was oversized."""
                if text and len(text) > _PER_SOURCE_CAP:
                    logger.warning(
                        f"[BRAIN] Subsystem '{source}' returned {len(text)} chars, capping to {_PER_SOURCE_CAP}"
                    )
                    # Try to cut at a sentence boundary; fall back to hard cut
                    cut = text[:_PER_SOURCE_CAP].rfind('. ')
                    return text[:cut + 1] if cut > _PER_SOURCE_CAP // 2 else text[:_PER_SOURCE_CAP]
                return text

            # === LEARNED CONTEXT INJECTION (Phase 4D: Letta-style) ===
            try:
                from aura.tools.neurodream import get_neurodream
                nd = get_neurodream()
                learned_ctx = nd.get_learned_context_prompt()
                if learned_ctx:
                    additions.append(_cap(learned_ctx, "NeuroDream"))
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
            # === CALENDAR CONTEXT INJECTION (Phase 5D) ===
            try:
                from aura.proactive.monitors.calendar_monitor import get_calendar_monitor
                cm = get_calendar_monitor()
                cal_ctx = cm.get_context_for_prompt()
                if cal_ctx:
                    additions.append(_cap(cal_ctx, "CalendarMonitor"))
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
            # === SELF-MODEL INJECTION (Phase 6B: Metacognitive Self-Improvement) ===
            try:
                from aura.consciousness.metacognition import get_metacognitive_engine
                mc = get_metacognitive_engine()
                self_model_ctx = mc.get_self_model_prompt()
                if self_model_ctx:
                    additions.append(_cap(self_model_ctx, "MetacognitionEngine"))
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
                            additions.append(_cap(user_model_ctx, "TheoryOfMind/MultiUser"))
                else:
                    from aura.proactive.theory_of_mind import get_theory_of_mind
                    tom = get_theory_of_mind()
                    user_model_ctx = tom.get_context_for_prompt()
                    if user_model_ctx:
                        additions.append(_cap(user_model_ctx, "TheoryOfMind"))
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
            # === MOTIVATION INJECTION (Phase 6E: Intrinsic Motivation) ===
            try:
                from aura.consciousness.intrinsic_motivation import get_intrinsic_motivation
                im = get_intrinsic_motivation()
                motivation_ctx = im.get_context_for_prompt()
                if motivation_ctx:
                    additions.append(_cap(motivation_ctx, "IntrinsicMotivation"))
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
            # Global Workspace Theory injection removed
            # === WORLD STATE INJECTION (ADV-02: Persistent World Model) ===
            try:
                from aura.consciousness.world_model import get_world_model
                wm = get_world_model()
                world_ctx = wm.get_context_summary()
                if world_ctx:
                    additions.append(_cap(world_ctx, "WorldModel"))
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
            result = "\n\n".join(additions)
            # Cap total additions to prevent context overflow
            if len(result) > _TOTAL_CAP:
                logger.warning(f"[BRAIN] System additions total {len(result)} chars (from {len(additions)} sources), truncating to {_TOTAL_CAP}")
                result = result[:_TOTAL_CAP]
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

        # === WEB SEARCH INSTRUCTION ===
        web_search_instruction = (
            "IMPORTANT: If the user asks about something you are not sure about, "
            "something recent, current events, news, real-time data (dates, prices, "
            "weather, scores, stock prices, exchange rates), or asks you to look "
            "something up or verify information — you MUST use the web_search or "
            "tavily tool to search the internet FIRST. Do NOT guess or make up "
            "answers. Always verify uncertain facts by searching."
        )
        full = f"{full}\n\n{web_search_instruction}"

        # === DESIGN SYSTEM INJECTION (frontend/rapid/artifact modes) ===
        try:
            from aura.prompts.design_system import DESIGN_SYSTEM_PROMPT, DESIGN_SYSTEM_MODES
            if self._action_mode and self._action_mode in DESIGN_SYSTEM_MODES:
                full = f"{full}\n\n{DESIGN_SYSTEM_PROMPT}"
                logger.debug(f"[BRAIN] Design system prompt injected for mode: {self._action_mode}")
        except ImportError:
            logger.debug("[BRAIN] aura.prompts.design_system not available")

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
            if (self._project_ctx_cache is None
                    or _now - self._project_ctx_ts > 60
                    or self._project_ctx_cwd != _cwd):
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
        MAX_SYSTEM_PROMPT_CHARS = 25000
        try:
            from aura.tools.codebase_index import CodebaseIndex
            _cwd = os.getcwd()
            _idx_db = Path("data/codebase_index/index.db")
            # Also check legacy path
            _idx_db_legacy = Path(_cwd) / ".aura" / "index.db"
            if (_idx_db.exists() or _idx_db_legacy.exists()) and len(full) < MAX_SYSTEM_PROMPT_CHARS - 1000:
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
        except Exception as e:
            logger.debug(f"[BRAIN] Code context retrieval failed: {e}")

        # === PROGRESSIVE SKILL CATALOG ===
        # Show available skills so the LLM knows it can load full procedures on demand.
        # skill_list_summaries() comes from SkillManagerMixin — may not be present yet.
        if len(full) < MAX_SYSTEM_PROMPT_CHARS - 1500:
            try:
                if hasattr(self, 'skill_list_summaries') and callable(self.skill_list_summaries):
                    skill_summaries = self.skill_list_summaries()
                    if skill_summaries:
                        skill_lines = "\n".join(
                            f"- {s['name']}: {s.get('description', 'no description')}"
                            for s in skill_summaries
                        )
                        # Cap skill section to 500 chars to stay within budget
                        if len(skill_lines) > 500:
                            skill_lines = skill_lines[:500].rsplit("\n", 1)[0]
                        full = f"{full}\n\n[Available Skills - use load_skill tool to load full procedures]\n{skill_lines}"
            except Exception as e:
                logger.debug(f"[BRAIN] Skill catalog injection failed: {e}")

        # === DEFERRED TOOL LISTING ===
        # Show tools that exist but aren't loaded yet — LLM can activate via tool_search.
        if len(full) < MAX_SYSTEM_PROMPT_CHARS - 1000:
            try:
                from aura.tools.loader import get_deferred_tool_list
                deferred_tools = get_deferred_tool_list()
                if deferred_tools:
                    tool_lines = "\n".join(
                        f"- {t['name']}: {t.get('description', 'no description')}"
                        for t in deferred_tools
                    )
                    # Cap deferred tools section to 800 chars
                    if len(tool_lines) > 800:
                        tool_lines = tool_lines[:800].rsplit("\n", 1)[0]
                    full = f"{full}\n\n[Additional Tools - use tool_search to find and activate]\n{tool_lines}"
            except ImportError:
                logger.debug("[BRAIN] aura.tools.loader.get_deferred_tool_list not available yet")
            except Exception as e:
                logger.debug(f"[BRAIN] Deferred tool listing failed: {e}")

        # One-liner instruction for skill/tool discovery (always added if space permits)
        hint = "Use load_skill or tool_search when a task matches a listed skill or requires a specialized tool."
        if len(full) + len(hint) + 4 < MAX_SYSTEM_PROMPT_CHARS:
            full = f"{full}\n\n{hint}"

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
                from aura.memory.unified_memory import get_unified_memory
                um_results = get_unified_memory().query(prompt, k=3)
                if um_results:
                    memory_ctx = "\n\n## Relevant Past Context\n"
                    for m in um_results:
                        ts = m.metadata.get("created_at", "")[:10] if m.metadata.get("created_at") else ""
                        memory_ctx += f"- [{ts}] {m.content[:120]}\n"
                    full = f"{full}{memory_ctx}"
            except Exception as e:
                logger.debug(f"[Brain] non-critical: {e}")
        budget = self._classify_budget(prompt)
        full = f"{full}{self._build_budget_instruction(budget)}"

        # Safety: cap system prompt to ~12K chars (~3K tokens) to leave room
        # for conversation history and response in the model's context window.
        if len(full) > MAX_SYSTEM_PROMPT_CHARS:
            logger.warning(f"[BRAIN] System prompt too large ({len(full)} chars), truncating to {MAX_SYSTEM_PROMPT_CHARS}")
            cut = full[:MAX_SYSTEM_PROMPT_CHARS].rfind('\n\n')
            if cut > MAX_SYSTEM_PROMPT_CHARS // 2:
                full = full[:cut]
            else:
                full = full[:MAX_SYSTEM_PROMPT_CHARS]
            full += "\n\n[System context truncated for length]"

        return full

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
                self._user_inference_active.clear()

        self._user_inference_started_at = time.time()
        self._user_inference_active.set()

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
        """Build the message list for think()/think_stream() inside _think_lock.

        Returns:
            List of message dicts ready for the LLM call.
        """
        messages = []
        if full_system_prompt:
            messages.append({"role": "system", "content": full_system_prompt})
        if use_history:
            with self._history_lock:
                messages.extend(self.conversation_history[-self._max_history:])
        messages.append({"role": "user", "content": prompt})
        return messages

    def _resolve_chat_client(self, model: str) -> tuple:
        """Resolve client and actual model, record thinking panel event.

        Returns:
            (client, actual_model) tuple
        """
        client, actual_model = self._get_client_for_model(model)
        if actual_model != model:
            logger.info(f"[BRAIN] Model fallback: {model} -> {actual_model}")
        self._last_model_used = actual_model

        # Record real thinking — LLM inference starting
        try:
            from api.routes.thinking import get_manager as get_thinking_manager
            tm = get_thinking_manager()
            tm.record_real_thought("formulating", f"reasoning with {actual_model}...", intensity=0.7, source="brain")
        except Exception as e:
            logger.debug(f"[Brain] non-critical: {e}")

        return client, actual_model

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
        # Circuit breaker: if too many consecutive failures, return degraded response
        cb_response = self._check_think_circuit_breaker()
        if cb_response is not None:
            return cb_response

        ctx = self._prepare_chat_think(
            prompt, system_prompt, use_history, task_type, tone_modifier, model_override,
        )
        model = ctx["model"]
        full_system_prompt = ctx["full_system_prompt"]

        try:
            # Phase 1: Snapshot under lock (brief — only reads shared history)
            with self._think_lock:
                messages = self._build_chat_messages(prompt, full_system_prompt, use_history)
                client, actual_model = self._resolve_chat_client(model)
                neuro = _get_neuromodulator_levels()
                llm_options = self._build_neuro_llm_options(prompt, neuro)
            # _think_lock released — LLM call proceeds without blocking other callers

            # Taint tracking: scan user messages for secrets (OpenFang-inspired)
            try:
                from aura.security.taint_tracker import get_tracker
                tracker = get_tracker()
                taint_matches, _ = tracker.check_and_track(prompt, session_id="brain")
                if taint_matches:
                    logger.warning(
                        f"[BRAIN] Detected {len(taint_matches)} secret(s) in user message — "
                        f"taint level tracked for session"
                    )
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

            # Phase 2: LLM call (NO lock held — allows parallel think/fleet/debate)
            adjusted_timeout = self._get_adaptive_timeout()
            _llm_start_ts = time.time()
            logger.debug(f"[BRAIN] Calling {model} with adaptive timeout={adjusted_timeout}s")

            # Call with timeout protection (adaptive latency-based) + 429 retry
            response = call_with_timeout(
                lambda: self._retry_on_rate_limit(
                    lambda: client.chat(model=actual_model, messages=messages, options=llm_options)
                ),
                timeout=adjusted_timeout + 20,
                default=None
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
                logger.warning(f"[BRAIN] All models in chain failed, returning error message")
                with self._cb_lock:
                    self._consecutive_think_failures += 1
                    if self._consecutive_think_failures >= self._think_cb_threshold:
                        self._think_circuit_open_at = time.time()
                        logger.warning(
                            f"[BRAIN] Think circuit breaker OPENED after "
                            f"{self._consecutive_think_failures} consecutive failures — "
                            f"cooldown {self._think_cb_cooldown}s"
                        )
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
                self._consecutive_think_failures = 0

            _bg_submit(
                self._record_routing_outcome, actual_model, task_type, True,
                _llm_elapsed * 1000
            )

            # Track tokens and cost
            _in_tok = _resp_get(response, "prompt_eval_count", 0) or 0
            _out_tok = _resp_get(response, "eval_count", 0) or 0
            if _in_tok or _out_tok:
                self._record_tokens(actual_model, _in_tok, _out_tok)

            # Compaction notice
            if self._compaction_pending:
                self._compaction_pending = False
                assistant_message = (
                    "_[Context compacted — older messages summarized to preserve memory]_\n\n"
                    + assistant_message
                )

            # Phase 3: Update history under lock (brief — only writes shared history)
            with self._think_lock:
                self._update_history_and_cleanup(prompt, assistant_message, actual_model, use_history)

            return assistant_message
        finally:
            self._user_inference_active.clear()

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
        ctx = self._prepare_chat_think(
            prompt, system_prompt, use_history, task_type, tone_modifier, model_override,
        )
        model = ctx["model"]
        full_system_prompt = ctx["full_system_prompt"]

        try:
            # Acquire _think_lock briefly to snapshot history and prepare messages.
            # Cannot hold through the streaming generator, but this prevents a
            # concurrent think() from reading stale history while we're mid-stream.
            with self._think_lock:
                messages = self._build_chat_messages(prompt, full_system_prompt, use_history)
                logger.debug(f"[BRAIN] Streaming call to {model}")
                client, actual_model = self._resolve_chat_client(model)
                neuro = _get_neuromodulator_levels()
                llm_options = self._build_neuro_llm_options(prompt, neuro)
            # _think_lock released — streaming proceeds without blocking other callers

            full_response = ""
            _all_models_failed = False

            # Compaction notice on streaming path
            if self._compaction_pending:
                self._compaction_pending = False
                notice = "_[Context compacted — older messages summarized to preserve memory]_\n\n"
                yield notice
                full_response += notice

            # Streaming with fallback chain + token tracking
            _stream_in_tok = 0
            _stream_out_tok = 0
            _models_to_try = [actual_model] + [
                m for m in self._get_fallback_chain(actual_model) if m != actual_model
            ]

            _STREAM_STALE_TIMEOUT = 45  # seconds without a chunk -> abort (was 90)

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
                        if now - _last_chunk_time > _STREAM_STALE_TIMEOUT:
                            logger.warning(f"[BRAIN] Stream stale for {_STREAM_STALE_TIMEOUT}s, aborting")
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
                        raise TimeoutError(f"Stream stale for {_STREAM_STALE_TIMEOUT}s")
                    actual_model = _try_actual
                    self._last_model_used = actual_model
                    break
                except Exception as e:
                    if _try_model == _models_to_try[-1]:
                        import traceback
                        _tb = traceback.format_exc()
                        logger.error(f"[BRAIN] All stream models failed: {e}\n{_tb}")
                        fallback = _user_facing_llm_error(e, model=_try_actual)
                        yield fallback
                        full_response += fallback
                        _all_models_failed = True
                    else:
                        logger.warning(f"[BRAIN] Stream model {_try_model} failed, trying next: {e}")
                    continue

            if _stream_in_tok or _stream_out_tok:
                self._record_tokens(actual_model, _stream_in_tok, _stream_out_tok)

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
            self._user_inference_active.clear()

    def _trigger_world_model_extraction(self, recent: list, executor=None) -> None:
        """Submit background world model extraction (deduplicates logic from think/think_stream)."""
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
                    pass  # Executor shut down — fall through to daemon thread
            threading.Thread(
                target=_run_world_model_extraction,
                args=(conv_id, recent, self._user_inference_active),
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

    def set_action_mode(self, mode: Optional[str]) -> None:
        """Set the current action mode for context-aware prompt injection.

        Used by agent_service to pass the detected action mode so that
        _build_full_system_prompt can inject mode-specific prompts
        (e.g., design system for frontend/artifact modes).

        Args:
            mode: Action mode string or None to clear
        """
        self._action_mode = mode

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
        except Exception as e:
            logger.debug(f"[BRAIN] Capability check failed: {e}")
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

    def _routing_stats_override(self, model: str, task_type: Optional[TaskType] = None, user_selected: bool = False) -> str:
        """Apply outcome-aware routing stats overlay to heuristic model selection.

        Only activates when ENABLE_OUTCOME_AWARE_ROUTING=True and RoutingStats
        has ≥MIN_SAMPLES data for the selected chain + microtask category.
        Falls back to heuristic model unchanged when data is insufficient.

        Args:
            user_selected: If True, the model was explicitly chosen by the user
                           (via UI dropdown or CLI). NEVER override user selections.
        """
        if not getattr(Config, "ENABLE_OUTCOME_AWARE_ROUTING", True):
            return model
        # Don't override user's explicit model choice (parameter OR instance-level)
        if user_selected or self._model_override:
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
        except Exception as e:
            logger.debug(f"[BRAIN] Thinking mode not available: {e}")

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

    # (observe/plan/decide_action/evaluate removed — OPAE loop replaced by ReAct)

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

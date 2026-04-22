"""Shared helper functions for :mod:`aura.brain`."""

from __future__ import annotations

import concurrent.futures
import logging
import sys
import threading
import time
from typing import Any, Callable

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aura.config import Config
from aura.pools import llm_pool as _llm_pool_fn

logger = logging.getLogger(__name__)

try:
    from aura.emotion.alma_engine import alma_engine

    ALMA_AVAILABLE = True
except ImportError:
    ALMA_AVAILABLE = False
    alma_engine = None

LLM_TIMEOUT = 120
NEURO_MIN_MULTIPLIER = 0.7
NEURO_MAX_MULTIPLIER = 1.4
_NEURO_DEFAULTS = {"dopamine": 0.5, "serotonin": 0.5, "norepinephrine": 0.5, "oxytocin": 0.5}

_wm_extraction_lock = threading.Lock()
_wm_consecutive_failures = 0
_WM_CIRCUIT_BREAKER_THRESHOLD = 3
_WM_CIRCUIT_BREAKER_RESET_AFTER = 300
_wm_circuit_broken_at = 0.0
_wm_lock = threading.Lock()

try:
    import httpx as _httpx

    _RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OSError, _httpx.TimeoutException)
except ImportError:
    _RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OSError)

_llm_retry = retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def resp_content(response) -> str:
    """Extract content from an Ollama response."""
    if response is None:
        return ""
    if isinstance(response, dict):
        msg = response.get("message", {})
        return msg.get("content", "") if isinstance(msg, dict) else ""
    msg = getattr(response, "message", None)
    if msg is not None:
        return getattr(msg, "content", "") or ""
    return ""


def resp_get(response, key, default=None):
    """Get a field from an Ollama response."""
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def run_world_model_extraction(conversation_id, messages, user_inference_event=None) -> None:
    """Process world-model extraction with a simple circuit breaker."""
    global _wm_consecutive_failures, _wm_circuit_broken_at

    if user_inference_event and user_inference_event.is_set():
        logger.debug("[BRAIN] World model extraction skipped, user inference active")
        return

    with _wm_lock:
        if _wm_consecutive_failures >= _WM_CIRCUIT_BREAKER_THRESHOLD:
            if time.time() - _wm_circuit_broken_at < _WM_CIRCUIT_BREAKER_RESET_AFTER:
                logger.debug("[BRAIN] World model extraction circuit breaker OPEN")
                return
            logger.info("[BRAIN] World model extraction circuit breaker RESET")
            _wm_consecutive_failures = 0
            _wm_circuit_broken_at = 0.0

    if not _wm_extraction_lock.acquire(blocking=False):
        logger.debug("[BRAIN] Skipping world model extraction, previous run still active")
        return

    try:
        from aura.consciousness.world_model import get_world_model

        get_world_model().process_conversation(conversation_id, messages)
        with _wm_lock:
            _wm_consecutive_failures = 0
            _wm_circuit_broken_at = 0.0
    except Exception as exc:
        with _wm_lock:
            _wm_consecutive_failures += 1
            if _wm_consecutive_failures >= _WM_CIRCUIT_BREAKER_THRESHOLD:
                _wm_circuit_broken_at = time.time()
                logger.warning(
                    "[BRAIN] World model extraction failed %sx, circuit OPEN for %ss",
                    _wm_consecutive_failures,
                    _WM_CIRCUIT_BREAKER_RESET_AFTER,
                )
            else:
                logger.debug(
                    "[BRAIN] World model extraction failed (%s/%s): %s",
                    _wm_consecutive_failures,
                    _WM_CIRCUIT_BREAKER_THRESHOLD,
                    exc,
                )
    finally:
        _wm_extraction_lock.release()

    try:
        if getattr(Config, "PROACTIVE_AWARENESS_QUICK_AFTER_CHAT", True):
            from aura.consciousness.proactive_awareness import get_proactive_awareness_engine

            get_proactive_awareness_engine().run_quick_analysis()
    except Exception as exc:
        logger.debug("[BRAIN] Proactive awareness quick analysis failed: %s", exc)


def get_neuromodulator_levels() -> dict:
    """Get current neuromodulator levels from ALMA, with safe defaults."""
    if not ALMA_AVAILABLE:
        return _NEURO_DEFAULTS.copy()
    try:
        state = alma_engine.get_emotional_state()
        base = (state or {}).get("neuromodulators") or {}
        if not base:
            return _NEURO_DEFAULTS.copy()
        try:
            from aura.tools.neurodream import get_neurodream

            neurodream = get_neurodream()
            if neurodream.current_phase.value != "awake":
                influence = neurodream.get_sleep_neuromodulator_influence()
                return {
                    key: max(0.0, min(1.0, base[key] + influence.get(key, 0.0)))
                    for key in base
                }
        except Exception as exc:
            logger.debug("[BRAIN] NeuroDream influence unavailable: %s", exc)
        return base
    except Exception as exc:
        logger.debug("[BRAIN] Failed to read neuromodulators: %s", exc)
        return _NEURO_DEFAULTS.copy()


def neuro_scale(base_value: float, neuro_level: float, sensitivity: float = 0.5) -> float:
    """Scale a base value by a neuromodulator level with safety clamps."""
    offset = (neuro_level - 0.5) * 2 * sensitivity
    multiplier = 1.0 + offset
    multiplier = max(NEURO_MIN_MULTIPLIER, min(NEURO_MAX_MULTIPLIER, multiplier))
    return base_value * multiplier


def ollama_health_check() -> tuple[bool, list[str]]:
    """Quick check for Ollama connectivity."""
    try:
        response = requests.get(f"{Config.OLLAMA_HOST}/api/tags", timeout=3)
        if response.status_code == 200:
            models = [model.get("name", "") for model in response.json().get("models", [])]
            return True, models
        return False, []
    except Exception:
        return False, []


def user_facing_llm_error(original_error: Exception | None = None, model: str | None = None) -> str:
    """Return a user-friendly LLM failure message."""
    model_tag = f" ({model})" if model else ""
    ollama_ok, available_models = ollama_health_check()

    if not ollama_ok:
        return (
            f"[LLM Error] Could not connect to Ollama{model_tag}.\n"
            "  - Is Ollama running? Try: ollama serve\n"
            "  - Is the model available? Try: ollama list\n"
            "  - For cloud models, check OLLAMA_API_KEY in .env"
        )

    if model and available_models:
        normalized = [item.split(":")[0] for item in available_models]
        model_base = model.split(":")[0]
        if model_base not in normalized and model not in available_models:
            return (
                f"[LLM Error] Model '{model}' not found on this Ollama instance.\n"
                f"  - Available models: {', '.join(available_models[:8])}\n"
                f"  - Pull it with: ollama pull {model}\n"
                "  - For cloud models, check OLLAMA_API_KEY in .env"
            )

    if original_error:
        err_type = type(original_error).__name__
        if isinstance(original_error, (ConnectionError, OSError)):
            return (
                f"[LLM Error] Connection failed for {model or 'model'}: {err_type}\n"
                "  - Ollama is reachable but the request failed.\n"
                f"  - The model may have crashed. Try: ollama run {model or '<model>'}"
            )
        if isinstance(original_error, TimeoutError) or "timeout" in str(original_error).lower():
            return (
                f"[LLM Error] Request timed out for {model or 'model'}.\n"
                "  - The model may be too large for your hardware.\n"
                "  - Try a smaller model or increase timeout."
            )
        return f"[LLM Error] {err_type} for {model or 'model'}: {original_error}"

    return f"[LLM Error] No response from {model or 'the language model'}.\n  - Try again or run: aura doctor"


def call_with_timeout(func: Callable, timeout: int = LLM_TIMEOUT, default: Any = None) -> Any:
    """Execute a function with timeout protection using the shared LLM pool."""
    try:
        from aura.pools import is_shutting_down
        if is_shutting_down():
            return default
    except Exception:
        logger.debug("Shutdown check failed in call_with_timeout", exc_info=True)
    try:
        future = _llm_pool_fn().submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning("LLM call timed out after %ss", timeout)
            future.cancel()
            return default
        except concurrent.futures.CancelledError:
            logger.warning("LLM call was cancelled")
            return default
        except (ConnectionError, OSError) as exc:
            logger.error("LLM connection error: %s", exc)
            return default
        except ValueError as exc:
            logger.error("LLM value error: %s", exc)
            return default
        except Exception as exc:
            logger.exception("Unexpected LLM error: %s: %s", type(exc).__name__, exc)
            return default
    except RuntimeError as exc:
        # Pool may be shut down either by our _shutdown_all() or by Python's own
        # ThreadPoolExecutor._python_exit() atexit handler, which fires in LIFO order
        # before our agent.shutdown() atexit. Check both flags.
        try:
            from aura.pools import is_shutting_down
            if is_shutting_down():
                return default
        except Exception:
            logger.debug("Shutdown re-check failed in call_with_timeout", exc_info=True)
        # Also detect interpreter-level shutdown (sys.is_finalizing() is Python 3.9+)
        if getattr(sys, 'is_finalizing', lambda: False)():
            return default
        # Non-shutdown RuntimeError from the shared pool is abnormal. Spawning an
        # ad-hoc ThreadPoolExecutor as a workaround masks the real problem and
        # violates the 3-pool discipline. Log loudly and fail closed.
        logger.error(
            "Shared LLM pool raised a non-shutdown RuntimeError (%s); returning default.",
            exc,
        )
        return default


def is_rate_limit_error(exc: Exception) -> bool:
    """Return True when an exception looks like a rate-limit error."""
    message = str(exc).lower()
    return "429" in message or "rate" in message or "too many" in message

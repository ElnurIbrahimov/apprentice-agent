"""Configuration management for the agent.

SECURITY: Thread-safe configuration with proper locking.
"""

import atexit
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# Load .env from the project root (not cwd, which may be different)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

logger = logging.getLogger(__name__)

# Thread lock for configuration changes
_config_lock = threading.RLock()

_validation_session = None
_validation_session_lock = threading.Lock()

# Background model validation future
_validation_future = None
_validation_future_lock = threading.Lock()

def _get_validation_session():
    global _validation_session
    with _validation_session_lock:
        if _validation_session is None:
            import requests
            _validation_session = requests.Session()
            atexit.register(_close_validation_session)
    return _validation_session


def _close_validation_session():
    global _validation_session
    with _validation_session_lock:
        if _validation_session is not None:
            try:
                _validation_session.close()
            except Exception as e:
                logger.debug(f"[Config] Failed to close validation HTTP session: {e}")
            _validation_session = None


# ============================================================================
#                    MODEL VALIDATION SYSTEM
# ============================================================================

# Verified cloud models via Ollama.com (Pro $20/month subscription)
# Updated Apr 2026 — all models accessed via Ollama cloud API
VERIFIED_CLOUD_MODELS = {
    # Fast / general
    "kimi-k2.5:cloud",                # 1T/32B MoE, multimodal, 256K, 92% MMLU
    "nemotron-3-super:cloud",         # 120B/12B Mamba-MoE, 1M ctx, 449 tok/s
    "glm-5:cloud",                    # 744B/40B MoE, 200K, 69 tok/s, lowest hallucination
    # Reasoning / planning
    "qwen3.5:397b-cloud",             # 397B/17B MoE, 262K, hybrid thinking, 201 langs
    "qwen3.5:cloud",                  # 9.65B dense, 262K, multimodal
    "deepseek-v3.2:cloud",            # 685B/37B MoE, 128K, cheapest ($0.14/M)
    "gemma4:31b-cloud",               # 31B dense, 256K, multimodal+audio, 85.2% MMLU-Pro
    # Code generation / debugging
    "glm-5.1:cloud",                  # 744B/40B MoE, 200K, +28% coding over GLM-5, 77.8% SWE
    "minimax-m2.7:cloud",             # 230B/10B MoE, 205K, SWE-Pro 56.2%, self-evolving
    "minimax-m2.5:cloud",             # 229B/10B MoE, 196K, 80.2% SWE-Bench
    "qwen3-coder:480b-cloud",         # 480B/35B MoE, 256K, 69.6% SWE-Bench
    "qwen3-coder-next:cloud",         # 80B/3B MoE, 256K, 71.3% SWE, 172 tok/s
    # General purpose
    "gpt-oss:120b-cloud",             # 117B/5.1B MoE, 128K, 97.9% AIME
}

# Local models (edge inference, embeddings, OCR)
VERIFIED_LOCAL_MODELS = {
    "nomic-embed-text:latest",        # RAG/embedding — 137M, MTEB 62.39
    "glm-ocr:latest",                 # OCR — 0.9B, OmniDocBench 94.6% #1
    "gemma4:e4b",                     # Edge — 9.6B dense, 128K, multimodal+audio
    "gemma4:e2b",                     # Edge — 5.1B dense, 128K, runs on RPi
}

_tags_cache: dict = {}
_tags_cache_ts: float = 0.0
_tags_cache_lock = threading.Lock()


def validate_model(model_name: str, ollama_host: str | None = None) -> bool:
    """
    Check if a model is available in Ollama.

    SECURITY: Validates model exists before use to prevent runtime errors.
    """
    # Cloud models are available if OLLAMA_API_KEY is set — no local HTTP check needed
    if model_name.endswith("-cloud") or model_name.endswith(":cloud"):
        return bool(os.getenv("OLLAMA_API_KEY"))

    import time as _time
    host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    global _tags_cache, _tags_cache_ts
    now = _time.time()
    with _tags_cache_lock:
        if now - _tags_cache_ts < 30.0 and _tags_cache:
            available_models = _tags_cache.get(host)
            if available_models is not None:
                # Don't refresh TTL on cache hit — let it expire after 30s
                if model_name in available_models:
                    return True
                base_name = model_name.split(":")[0]
                return any(m.startswith(base_name) for m in available_models)

    try:
        response = _get_validation_session().get(f"{host}/api/tags", timeout=5)
        if response.status_code == 200:
            available = [m["name"] for m in response.json().get("models", [])]
            with _tags_cache_lock:
                _tags_cache[host] = available
                _tags_cache_ts = now
            if model_name in available:
                return True
            base_name = model_name.split(":")[0]
            return any(m.startswith(base_name) for m in available)
    except Exception as e:
        logger.debug(f"[Config] Failed to validate model via Ollama tags: {e}")
    return False


def get_best_available_model(preferred: str, fallbacks: List[str], role: str = "unknown") -> str:
    """
    Get the best available model, trying preferred then fallbacks.

    SECURITY: Ensures we always have a working model.
    """
    # Try preferred first
    if validate_model(preferred):
        return preferred

    # Try fallbacks in order
    for fallback in fallbacks:
        if validate_model(fallback):
            logger.warning(f"[Config] Model '{preferred}' not available for {role}, using fallback: {fallback}")
            return fallback

    # Return preferred anyway and let it fail at runtime
    logger.error(f"[Config] No models available for {role}! Pull models with: ollama pull {fallbacks[0] if fallbacks else preferred}")
    return fallbacks[0] if fallbacks else preferred


class Config:
    _raw_ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # Validate scheme — reject non-http(s) to prevent SSRF via env misconfiguration
    # Case-insensitive check: HTTP://host and Http://host are valid
    OLLAMA_HOST: str = _raw_ollama_host if _raw_ollama_host.lower().startswith(("http://", "https://")) else "http://localhost:11434"

    # Feature toggles (configurable via env vars)
    KG_BRAIN_ENABLED: bool = os.getenv("KG_BRAIN_ENABLED", "true").lower() in ("true", "1", "yes")
    SKILL_LIBRARY_ENABLED: bool = os.getenv("SKILL_LIBRARY_ENABLED", "true").lower() in ("true", "1", "yes")

    # Auth (init-time default for middleware; runtime truth lives in api/auth.py::_auth_is_enabled)
    API_AUTH_ENABLED: bool = os.getenv("AURA_API_AUTH_ENABLED", "true").lower() in ("true", "1", "yes")

    # Budget-aware routing: when on, prefer the cheapest model in the selected
    # chain once session spend exceeds BUDGET_MAX_USD_PER_SESSION. SIMPLE tasks
    # always prefer the cheapest regardless of spend.
    BUDGET_MODE: bool = os.getenv("AURA_BUDGET_MODE", "false").lower() in ("true", "1", "yes")
    BUDGET_MAX_USD_PER_SESSION: float = float(os.getenv("AURA_BUDGET_MAX_USD", "0.50"))

    # SearXNG search instance (configurable)
    SEARXNG_URL: str = os.getenv("SEARXNG_URL", "http://localhost:8888")

    # ============================================================
    # MODEL CONFIGURATION — CLOUD-ONLY (Ollama Pro $20/month)
    # ============================================================
    # All models served via Ollama cloud API
    # Routing: fast → reason → code → vision → think → longctx

    # Model chains (first available is used as fallback)
    MODEL_FAST_CHAIN = [
        "nemotron-3-super:cloud",          # Primary: 449 tok/s, 1M ctx
        "kimi-k2.5:cloud",                # Fallback: strong general
        "glm-5.1:cloud",                   # Fallback: 744B MoE, strong all-around
        "glm-5:cloud",                     # Fallback: 69 tok/s, low hallucination
        "gemma4:31b-cloud",                # Fallback: 104 tok/s, multimodal
    ]
    MODEL_REASON_CHAIN = [
        "kimi-k2.5:cloud",                # Primary: 92% MMLU, 96.1% AIME, 256K
        "qwen3.5:397b-cloud",             # Fallback: 87.8 MMLU-Pro, hybrid thinking
        "glm-5.1:cloud",                   # Fallback: 744B MoE, +28% coding over GLM-5
        "glm-5:cloud",                     # Fallback: 96% MMLU, lowest hallucination
        "gemma4:31b-cloud",                # Fallback: 85.2% MMLU-Pro, 89.2% AIME
        "deepseek-v3.2:cloud",             # Fallback: 85.0 MMLU-Pro
    ]
    MODEL_CODE_CHAIN = [
        "minimax-m2.7:cloud",             # Primary: SWE-Pro 56.2%, self-evolving, 205K
        "minimax-m2.5:cloud",             # Fallback: 80.2% SWE-Bench
        "glm-5.1:cloud",                  # Fallback: 77.8% SWE, +28% coding over GLM-5
        "qwen3-coder:480b-cloud",          # Fallback: 69.6% SWE, 480B code specialist
        "qwen3-coder-next:cloud",          # Fallback: 71.3% SWE, 172 tok/s
        "deepseek-v3.2:cloud",             # Fallback: 67.8% SWE, cheapest
    ]
    MODEL_VISION_CHAIN = [
        "kimi-k2.5:cloud",                # Primary: native multimodal, 256K
        "gemma4:31b-cloud",                # Fallback: native vision+audio, 256K
        "qwen3.5:397b-cloud",             # Fallback: multimodal, 262K
    ]
    MODEL_THINK_CHAIN = [
        "qwen3.5:397b-cloud",             # Primary: hybrid think/non-think, 262K
        "kimi-k2.5:cloud",                # Fallback: 96.1% AIME
        "glm-5.1:cloud",                   # Fallback: 744B MoE, strong reasoning
        "glm-5:cloud",                     # Fallback: 92.7% AIME, low hallucination
        "gemma4:31b-cloud",                # Fallback: 89.2% AIME
    ]
    MODEL_LONGCTX_CHAIN = [
        "nemotron-3-super:cloud",          # Primary: 1M tokens, 449 tok/s
        "minimax-m2.7:cloud",             # Fallback: 205K, self-evolving
        "minimax-m2.5:cloud",              # Fallback: 196K
        "qwen3.5:397b-cloud",             # Fallback: 262K
        "kimi-k2.5:cloud",                 # Fallback: 256K
        "glm-5.1:cloud",                   # Fallback: 200K context
    ]

    # Primary defaults
    MODEL_FAST: str = os.getenv("MODEL_FAST", "nemotron-3-super:cloud")
    MODEL_REASON: str = os.getenv("MODEL_REASON", "kimi-k2.5:cloud")
    MODEL_CODE: str = os.getenv("MODEL_CODE", "minimax-m2.7:cloud")
    MODEL_VISION: str = os.getenv("MODEL_VISION", "kimi-k2.5:cloud")
    MODEL_THINK: str = os.getenv("MODEL_THINK", "qwen3.5:397b-cloud")
    MODEL_LONGCTX: str = os.getenv("MODEL_LONGCTX", "minimax-m2.7:cloud")

    MODEL_NAME: str = MODEL_REASON  # Default model (backward compat)

    @classmethod
    def _do_validate_models(cls) -> Dict[str, str]:
        """Internal: actually perform model validation (may be called in background)."""
        # Guard: reject known-weak API keys when auth is enabled
        _weak_keys = {"", "change-this-to-a-strong-random-key", "test", "admin", "password"}
        if cls.API_AUTH_ENABLED and cls.API_KEY in _weak_keys:
            logger.warning(
                "[Config] AURA_API_KEY is empty or a known placeholder. "
                "Set a strong random key: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )

        from concurrent.futures import ThreadPoolExecutor, as_completed
        roles = [
            ("fast", cls.MODEL_FAST, cls.MODEL_FAST_CHAIN),
            ("reason", cls.MODEL_REASON, cls.MODEL_REASON_CHAIN),
            ("code", cls.MODEL_CODE, cls.MODEL_CODE_CHAIN),
            ("vision", cls.MODEL_VISION, cls.MODEL_VISION_CHAIN),
            ("think", cls.MODEL_THINK, cls.MODEL_THINK_CHAIN),
            ("longctx", cls.MODEL_LONGCTX, cls.MODEL_LONGCTX_CHAIN),
        ]
        results = {}
        # Cap at 3 workers to respect Ollama Pro's 3-concurrent-model limit
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(get_best_available_model, preferred, fallbacks, role): role
                for role, preferred, fallbacks in roles
            }
            for future in as_completed(futures):
                role = futures[future]
                try:
                    results[role] = future.result()
                except Exception:
                    results[role] = None
        with _config_lock:
            if results.get("fast"):
                cls.MODEL_FAST = results["fast"]
            if results.get("reason"):
                cls.MODEL_REASON = results["reason"]
                cls.MODEL_NAME = results["reason"]
            if results.get("code"):
                cls.MODEL_CODE = results["code"]
            if results.get("vision"):
                cls.MODEL_VISION = results["vision"]
            if results.get("think"):
                cls.MODEL_THINK = results["think"]
            if results.get("longctx"):
                cls.MODEL_LONGCTX = results["longctx"]
        return results

    @classmethod
    def validate_models_on_startup(cls, background: bool = False) -> Dict[str, str]:
        """
        Validate all configured models and find best available.

        Args:
            background: If True, run validation in a background thread and
                       return immediately with empty dict. Results are applied
                       asynchronously. Use ``ensure_models_validated()`` to
                       block until done when a model is actually needed.

        Returns dict of role -> selected model (empty if background=True).

        SECURITY: Thread-safe with locking.
        """
        global _validation_future
        if background:
            from concurrent.futures import ThreadPoolExecutor
            with _validation_future_lock:
                if _validation_future is None:
                    # Single-thread pool that lives for the duration of validation
                    _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-validation")
                    _validation_future = _pool.submit(cls._do_validate_models)
                    # Allow the pool thread to be cleaned up after completion
                    _pool.shutdown(wait=False)
            return {}
        return cls._do_validate_models()

    @classmethod
    def ensure_models_validated(cls, timeout: float = 10.0) -> Dict[str, str]:
        """Wait for background model validation to complete (if running).

        If validation was never started or already completed synchronously,
        this returns immediately. Safe to call multiple times -- after the
        first completion the future is cleared so subsequent calls are free.
        """
        global _validation_future
        with _validation_future_lock:
            fut = _validation_future
        if fut is None:
            return {}
        if fut.done():
            # Already finished -- clear the future so we skip the lock next time
            with _validation_future_lock:
                _validation_future = None
            try:
                return fut.result()
            except Exception:
                return {}
        try:
            result = fut.result(timeout=timeout)
            with _validation_future_lock:
                _validation_future = None
            logger.info(f"[MODELS] Background validation complete: {result}")
            return result
        except Exception as e:
            logger.warning(f"[MODELS] Background validation failed/timed out: {e}")
            return {}

    @classmethod
    def get_model(cls, role: str) -> str:
        """
        Thread-safe getter for model by role.

        Lazily waits for background validation to complete on first call.

        Args:
            role: One of 'fast', 'reason', 'code', 'vision', 'think', 'longctx'

        Returns:
            Model name string
        """
        # Ensure background validation is done before returning a model
        cls.ensure_models_validated(timeout=10.0)
        with _config_lock:
            role_map = {
                'fast': cls.MODEL_FAST,
                'reason': cls.MODEL_REASON,
                'code': cls.MODEL_CODE,
                'vision': cls.MODEL_VISION,
                'think': cls.MODEL_THINK,
                'longctx': cls.MODEL_LONGCTX,
                'default': cls.MODEL_NAME,
            }
            return role_map.get(role, cls.MODEL_NAME)

    @classmethod
    def set_model(cls, role: str, model: str) -> bool:
        """
        Thread-safe setter for model by role.

        Args:
            role: One of 'fast', 'reason', 'code', 'vision', 'think', 'longctx'
            model: Model name string

        Returns:
            True if set successfully
        """
        with _config_lock:
            if role == 'fast':
                cls.MODEL_FAST = model
            elif role == 'reason':
                cls.MODEL_REASON = model
                cls.MODEL_NAME = model
            elif role == 'code':
                cls.MODEL_CODE = model
            elif role == 'vision':
                cls.MODEL_VISION = model
            elif role == 'think':
                cls.MODEL_THINK = model
            elif role == 'longctx':
                cls.MODEL_LONGCTX = model
            else:
                logger.warning(f"[CONFIG] Unknown role: {role}")
                return False
            logger.info(f"[CONFIG] Set {role} model to: {model}")
            return True

    @classmethod
    def get_all_models(cls) -> Dict[str, str]:
        """Thread-safe getter for all model configurations."""
        with _config_lock:
            return {
                'fast': cls.MODEL_FAST,
                'reason': cls.MODEL_REASON,
                'code': cls.MODEL_CODE,
                'vision': cls.MODEL_VISION,
                'think': cls.MODEL_THINK,
                'longctx': cls.MODEL_LONGCTX,
            }

    # System 1/System 2 confidence thresholds (Kahneman dual-process routing)
    S1_CONFIDENCE_THRESHOLD: float = 0.7   # Above this → System 1 (fast)
    S2_CONFIDENCE_THRESHOLD: float = 0.4   # Below this → System 2 (deliberative)

    # Perceptual hashing thresholds (dHash Hamming distance, 256-bit hash)
    PHASH_CHANGE_THRESHOLD: int = 12       # Visual change detection (cursor blinks < 12)
    PHASH_MAJOR_THRESHOLD: int = 20        # Major visual change (workflow transition)

    MEMORY_COLLECTION_NAME: str = "agent_memory"
    MAX_MEMORY_RESULTS: int = 5

    # PersonaPlex and Sesame voice removed — using external voice provider

    # MirrorMind Configuration (Tool #21) - Self-Critique System
    MIRRORMIND_ENABLED: bool = os.getenv("MIRRORMIND_ENABLED", "false").lower() == "true"
    MIRRORMIND_THRESHOLD: float = float(os.getenv("MIRRORMIND_THRESHOLD", "0.75"))
    MIRRORMIND_MAX_ITERATIONS: int = int(os.getenv("MIRRORMIND_MAX_ITERATIONS", "2"))

    # CognitiveTheater Configuration (Tool #22) - Multi-Perspective Reasoning
    COGNITIVE_THEATER_ENABLED: bool = os.getenv("COGNITIVE_THEATER_ENABLED", "true").lower() == "true"

    # Reflexion Configuration (Tool #25) - Learn From Mistakes
    REFLEXION_ENABLED: bool = os.getenv("REFLEXION_ENABLED", "true").lower() == "true"
    REFLEXION_MAX_ATTEMPTS: int = int(os.getenv("REFLEXION_MAX_ATTEMPTS", "3"))

    # SynapseForge Configuration (Tool #26) - Dynamic Tool Creation
    SYNAPSEFORGE_ENABLED: bool = os.getenv("SYNAPSEFORGE_ENABLED", "true").lower() == "true"

    # WorldSim Configuration (Tool #27) - Consequence Simulation
    WORLDSIM_ENABLED: bool = os.getenv("WORLDSIM_ENABLED", "true").lower() == "true"

    # Strategy Bandit Configuration — Adaptive reasoning strategy selection
    STRATEGY_BANDIT_ENABLED: bool = os.getenv("STRATEGY_BANDIT_ENABLED", "true").lower() == "true"
    STRATEGY_BANDIT_EPSILON: float = float(os.getenv("STRATEGY_BANDIT_EPSILON", "0.1"))
    STRATEGY_BANDIT_EVAL_ENABLED: bool = os.getenv("STRATEGY_BANDIT_EVAL_ENABLED", "true").lower() == "true"

    # Reasoning Template Library Configuration — Phase 3
    REASONING_TEMPLATES_ENABLED: bool = os.getenv("REASONING_TEMPLATES_ENABLED", "true").lower() == "true"

    # World Model Configuration — ADV-02: Persistent situational awareness
    WORLD_MODEL_ENABLED: bool = os.getenv("WORLD_MODEL_ENABLED", "true").lower() == "true"
    WORLD_MODEL_DB_PATH: str = os.getenv("WORLD_MODEL_DB_PATH", "")

    # World Model Extraction Configuration — ADV-02 Phase 2
    WORLD_MODEL_EXTRACTION_ENABLED: bool = os.getenv("WORLD_MODEL_EXTRACTION_ENABLED", "true").lower() == "true"
    WORLD_MODEL_EXTRACTION_MODEL: str = os.getenv("WORLD_MODEL_EXTRACTION_MODEL", "")
    WORLD_MODEL_EXTRACTION_MIN_INTERVAL: float = float(os.getenv("WORLD_MODEL_EXTRACTION_MIN_INTERVAL", "5.0"))

    # Proactive Awareness Configuration — ADV-02 Phase 3
    PROACTIVE_AWARENESS_ENABLED: bool = os.getenv("PROACTIVE_AWARENESS_ENABLED", "true").lower() == "true"
    PROACTIVE_AWARENESS_QUICK_AFTER_CHAT: bool = os.getenv("PROACTIVE_AWARENESS_QUICK_AFTER_CHAT", "true").lower() == "true"

    # Multi-User Consciousness Configuration (ADV-04)
    MULTI_USER_ENABLED: bool = os.getenv("AURA_MULTI_USER", "false").lower() == "true"
    MULTI_USER_DEFAULT_ID: str = os.getenv("AURA_DEFAULT_USER_ID", "default_user")
    MULTI_USER_SESSION_TIMEOUT: int = int(os.getenv("AURA_SESSION_TIMEOUT_MIN", "30"))

    # Trust Calibration (ADV-04) — controls how fast users gain trust
    TRUST_INCREMENT: float = float(os.getenv("TRUST_INCREMENT", "0.005"))
    TRUST_DECREMENT_ADVERSARIAL: float = float(os.getenv("TRUST_DECREMENT_ADVERSARIAL", "0.1"))
    TRUST_ACQUAINTANCE_MESSAGES: int = int(os.getenv("TRUST_ACQUAINTANCE_MESSAGES", "5"))
    TRUST_FAMILIAR_MESSAGES: int = int(os.getenv("TRUST_FAMILIAR_MESSAGES", "30"))
    TRUST_TRUSTED_MESSAGES: int = int(os.getenv("TRUST_TRUSTED_MESSAGES", "100"))
    TRUST_FAMILIAR_SCORE: float = float(os.getenv("TRUST_FAMILIAR_SCORE", "0.6"))
    TRUST_TRUSTED_SCORE: float = float(os.getenv("TRUST_TRUSTED_SCORE", "0.7"))

    # ============================================================
    # DIRECT API PROVIDER KEYS (loaded from env or set at runtime)
    # ============================================================
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROK_API_KEY: str = os.getenv("GROK_API_KEY", "")
    PERPLEXITY_API_KEY: str = os.getenv("PERPLEXITY_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    MINIMAX_API_KEY: str = os.getenv("MINIMAX_API_KEY", "")
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    KIMI_API_KEY: str = os.getenv("KIMI_API_KEY", "")
    GLM_API_KEY: str = os.getenv("GLM_API_KEY", "")

    # API Security Configuration (primary definition is above, line ~143)
    API_KEY: str = os.getenv("AURA_API_KEY", "")  # Set to enable API key auth
    API_RATE_LIMIT: int = int(os.getenv("AURA_API_RATE_LIMIT", "300"))  # requests per minute (300 for server deployments)
    API_CORS_ORIGINS: str = os.getenv("AURA_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:8000")

    # AURA v3.0 ALIVE System Configuration
    AURA_ENABLED: bool = os.getenv("AURA_ENABLED", "true").lower() == "true"
    AURA_SOUL: str = os.getenv("AURA_SOUL", "SOUL_PERSONAL")  # SOUL_PERSONAL or SOUL_ENTERPRISE
    AURA_PROACTIVE: bool = os.getenv("AURA_PROACTIVE", "false").lower() == "true"  # Disabled to prevent event loop blocking
    AURA_THINKING: bool = os.getenv("AURA_THINKING", "true").lower() == "true"
    AURA_HUMANIZE: bool = os.getenv("AURA_HUMANIZE", "true").lower() == "true"

    AURA_ENV: str = os.getenv("AURA_ENV", "development")  # "development" or "production"

    @classmethod
    def is_production(cls) -> bool:
        # Read env at call time (not class load time) so run_web.py --prod works
        return os.getenv("AURA_ENV", cls.AURA_ENV) == "production"

    # Voice Configuration
    VOICE_CONFIG = {
        "default_mode": "whisper",  # Using external voice provider
    }

    # Florence-2 Vision (local HuggingFace model — for image preprocessing only)
    FLORENCE2_MODEL: str = "microsoft/Florence-2-base"
    FLORENCE2_ENABLED: bool = os.getenv("FLORENCE2_ENABLED", "true").lower() == "true"

    # ============================================================
    # CENTRALIZED THRESHOLDS — referenced across subsystems
    # ============================================================

    # Salience Filter (aura.proactive.salience_filter)
    SALIENCE_FILTER_THRESHOLD: float = float(os.getenv("SALIENCE_FILTER_THRESHOLD", "0.3"))
    SALIENCE_LLM_TIMEOUT: float = float(os.getenv("SALIENCE_LLM_TIMEOUT", "5.0"))
    SALIENCE_SEEN_EVENT_TTL: float = float(os.getenv("SALIENCE_SEEN_EVENT_TTL", "3600.0"))
    SALIENCE_CLEANUP_INTERVAL: int = int(os.getenv("SALIENCE_CLEANUP_INTERVAL", "100"))
    SALIENCE_CLEANUP_PERIOD: float = float(os.getenv("SALIENCE_CLEANUP_PERIOD", "300.0"))

    # Theory of Mind (aura.proactive.theory_of_mind)
    TOM_EMA_ALPHA: float = float(os.getenv("TOM_EMA_ALPHA", "0.4"))

    # Proactive Awareness (aura.consciousness.proactive_awareness)
    PROACTIVE_MIN_CONFIDENCE: float = float(os.getenv("PROACTIVE_MIN_CONFIDENCE", "0.4"))

    # Conversation / Brain defaults (aura.brain)
    HISTORY_LIMIT: int = int(os.getenv("HISTORY_LIMIT", "20"))
    AUTO_RESET_INTERVAL: int = int(os.getenv("AUTO_RESET_INTERVAL", "15"))
    BUDGET_SMALL: int = int(os.getenv("BUDGET_SMALL", "300"))
    BUDGET_MEDIUM: int = int(os.getenv("BUDGET_MEDIUM", "1024"))
    BUDGET_LARGE: int = int(os.getenv("BUDGET_LARGE", "2048"))

    # Context compression (aura.memory.context_compressor)
    CONTEXT_COMPRESSION_THRESHOLD: int = int(os.getenv("CONTEXT_COMPRESSION_THRESHOLD", "80000"))
    CONTEXT_COMPRESSION_KEEP_LAST: int = int(os.getenv("CONTEXT_COMPRESSION_KEEP_LAST", "10"))

    # ============================================================
    # RELIABILITY UPGRADE — Phase 1-4 (2026-03)
    # ============================================================

    # Memory Write Gate
    ENABLE_MEMORY_WRITE_GATE: bool = os.getenv("ENABLE_MEMORY_WRITE_GATE", "true").lower() == "true"
    MEMORY_WRITE_THRESHOLD: float = float(os.getenv("MEMORY_WRITE_THRESHOLD", "0.15"))  # Lowered: personal AI OS, conversations should persist
    MEMORY_MERGE_THRESHOLD: float = float(os.getenv("MEMORY_MERGE_THRESHOLD", "0.88"))
    MEMORY_SUPERSEDE_THRESHOLD: float = float(os.getenv("MEMORY_SUPERSEDE_THRESHOLD", "0.80"))

    # Loop Guard
    ENABLE_LOOP_GUARD: bool = os.getenv("ENABLE_LOOP_GUARD", "true").lower() == "true"
    LOOP_GUARD_MAX_REPETITIONS: int = int(os.getenv("LOOP_GUARD_MAX_REPETITIONS", "3"))
    LOOP_GUARD_NOVELTY_THRESHOLD: float = float(os.getenv("LOOP_GUARD_NOVELTY_THRESHOLD", "0.25"))
    LOOP_GUARD_WINDOW_SIZE: int = int(os.getenv("LOOP_GUARD_WINDOW_SIZE", "20"))
    LOOP_GUARD_BUDGET: int = int(os.getenv("LOOP_GUARD_BUDGET", "40"))

    # Browser Agent
    ENABLE_BROWSER_POSTCONDITIONS: bool = os.getenv("ENABLE_BROWSER_POSTCONDITIONS", "true").lower() == "true"
    BROWSER_MAX_RETRIES: int = int(os.getenv("BROWSER_MAX_RETRIES", "3"))
    BROWSER_ABORT_ON_DOMAIN_DRIFT: bool = os.getenv("BROWSER_ABORT_ON_DOMAIN_DRIFT", "true").lower() == "true"

    # Outcome-aware routing
    ENABLE_OUTCOME_AWARE_ROUTING: bool = os.getenv("ENABLE_OUTCOME_AWARE_ROUTING", "true").lower() == "true"

    # Tool contracts
    ENABLE_TOOL_CONTRACTS: bool = os.getenv("ENABLE_TOOL_CONTRACTS", "true").lower() == "true"

    # Knowledge graph contradictions
    ENABLE_KG_CONTRADICTIONS: bool = os.getenv("ENABLE_KG_CONTRADICTIONS", "true").lower() == "true"

    # Dream consolidation
    DREAM_CLUSTER_BATCH_SIZE: int = int(os.getenv("DREAM_CLUSTER_BATCH_SIZE", "20"))
    DREAM_PRUNE_STALENESS_DAYS: int = int(os.getenv("DREAM_PRUNE_STALENESS_DAYS", "30"))
    DREAM_MIN_CLUSTER_SIZE: int = int(os.getenv("DREAM_MIN_CLUSTER_SIZE", "3"))
    DREAM_ENABLE_ROUTINE_EXTRACTION: bool = os.getenv("DREAM_ENABLE_ROUTINE_EXTRACTION", "true").lower() == "true"
    DREAM_ENABLE_GRAPH_DENSIFICATION: bool = os.getenv("DREAM_ENABLE_GRAPH_DENSIFICATION", "false").lower() == "true"

    # Consolidated Memory Store (Phase 2)
    AURA_MEMORY_DB_PATH: str = os.getenv("AURA_MEMORY_DB_PATH", "data/aura_memory.db")
    FADEM_HALF_LIFE_HOURS: float = float(os.getenv("FADEM_HALF_LIFE_HOURS", str(14 * 24)))  # 2 weeks default
    FADEM_PRUNE_THRESHOLD: float = float(os.getenv("FADEM_PRUNE_THRESHOLD", "0.05"))

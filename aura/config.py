"""Configuration management for the agent.

SECURITY: Thread-safe configuration with proper locking.
"""

import os
import logging
import threading
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Thread lock for configuration changes
_config_lock = threading.RLock()

_validation_session = None

def _get_validation_session():
    global _validation_session
    if _validation_session is None:
        import requests
        _validation_session = requests.Session()
    return _validation_session


# ============================================================================
#                    MODEL VALIDATION SYSTEM
# ============================================================================

# Verified cloud models via Ollama.com (Pro $20/month subscription)
# Updated Feb 2026 — all models accessed via Ollama cloud API
VERIFIED_CLOUD_MODELS = {
    # Fast / general conversation
    "gemini-3-flash-preview:cloud",   # Speed-optimized, low latency
    "nemotron-3-nano:30b-cloud",      # NVIDIA efficient 30B fallback
    "kimi-k2.5:cloud",                # General purpose, multimodal
    # Reasoning / planning
    "qwen3.5:397b-cloud",             # Deep planning, MMLU/GPQA leader
    "cogito-2.1:671b-cloud",          # Extended reasoning, MIT license
    "deepseek-v3.2:cloud",            # Strong all-rounder
    # Code generation / debugging
    "qwen3-coder:480b-cloud",         # Code-specialized at 480B scale
    "devstral-2:123b-cloud",          # Agentic coding, SWE-bench focus
    "qwen3-coder-next:cloud",         # Efficient code MoE variant
    # Vision / multimodal
    "qwen3-vl:235b-cloud",            # Only dedicated vision-language model
    # Deep thinking / chain-of-thought
    "kimi-k2-thinking:cloud",         # Explicit thinking mode, extended CoT
    # Long context / document analysis
    "minimax-m2.5:cloud",             # Million-token context windows
    # General purpose extras
    "mistral-large-3:675b-cloud",     # Large Mistral for general tasks
    "gpt-oss:120b-cloud",             # OSS GPT-style fallback
    "glm-5:cloud",                    # Zhipu AI general model
}

# Local models kept only for non-chat workloads (embeddings, OCR)
VERIFIED_LOCAL_MODELS = {
    "nomic-embed-text:latest",        # RAG/embedding — no cloud equivalent
    "glm-ocr:latest",                 # OCR — local processing
}

_tags_cache: dict = {}
_tags_cache_ts: float = 0.0


def validate_model(model_name: str, ollama_host: str = None) -> bool:
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
    if now - _tags_cache_ts < 30.0 and _tags_cache:
        available_models = _tags_cache.get(host)
        if available_models is not None:
            if model_name in available_models:
                return True
            base_name = model_name.split(":")[0]
            return any(m.startswith(base_name) for m in available_models)

    try:
        response = _get_validation_session().get(f"{host}/api/tags", timeout=5)
        if response.status_code == 200:
            available = [m["name"] for m in response.json().get("models", [])]
            _tags_cache[host] = available
            _tags_cache_ts = now
            if model_name in available:
                return True
            base_name = model_name.split(":")[0]
            return any(m.startswith(base_name) for m in available)
    except Exception:
        pass
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
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    CHROMADB_PATH: Path = Path(os.getenv("CHROMADB_PATH", "./data/chromadb"))

    # ============================================================
    # MODEL CONFIGURATION — CLOUD-ONLY (Ollama Pro $20/month)
    # ============================================================
    # All models served via Ollama cloud API
    # Routing: fast → reason → code → vision → think → longctx

    # Model chains (first available is used as fallback)
    MODEL_FAST_CHAIN = [
        "gemini-3-flash-preview:cloud",   # Primary: speed-optimized
        "nemotron-3-nano:30b-cloud",       # Fallback: efficient 30B
        "kimi-k2.5:cloud",                 # Fallback: general purpose
        "qwen3:8b",                        # Local fallback: offline-capable
        "qwen2:1.5b",                      # Local fallback: tiny/fast
    ]
    MODEL_REASON_CHAIN = [
        "qwen3.5:397b-cloud",              # Primary: deep planning
        "cogito-2.1:671b-cloud",           # Fallback: extended reasoning
        "deepseek-v3.2:cloud",             # Fallback: strong all-rounder
        "kimi-k2.5:cloud",                 # Fallback: general purpose
        "deepseek-r1:8b",                  # Local fallback: reasoning-capable
        "qwen3:8b",                        # Local fallback: offline-capable
    ]
    MODEL_CODE_CHAIN = [
        "qwen3-coder:480b-cloud",          # Primary: 480B code specialist
        "devstral-2:123b-cloud",           # Fallback: agentic code/SWE
        "qwen3-coder-next:cloud",          # Fallback: efficient code MoE
        "deepseek-v3.2:cloud",             # Fallback: strong at code
        "qwen2.5-coder:7b",               # Local fallback: code-specialized
        "deepseek-r1:8b",                  # Local fallback: offline code
    ]
    MODEL_VISION_CHAIN = [
        "qwen3-vl:235b-cloud",             # Primary: only dedicated VL model
        "kimi-k2.5:cloud",                 # Fallback: multimodal capable
        "gemini-3-flash-preview:cloud",    # Fallback: Gemini supports vision
        "llava:latest",                    # Local fallback: vision-capable
    ]
    MODEL_THINK_CHAIN = [
        "kimi-k2-thinking:cloud",          # Primary: dedicated thinking mode
        "cogito-2.1:671b-cloud",           # Fallback: extended reasoning
        "qwen3.5:397b-cloud",              # Fallback: deep planner
        "deepseek-r1:8b",                  # Local fallback: reasoning chain-of-thought
    ]
    MODEL_LONGCTX_CHAIN = [
        "minimax-m2.5:cloud",              # Primary: million-token context
        "kimi-k2.5:cloud",                 # Fallback: long context capable
        "qwen3.5:397b-cloud",              # Fallback: large context
        "qwen3:8b",                        # Local fallback: best local context window
    ]

    # Primary defaults
    MODEL_FAST: str = os.getenv("MODEL_FAST", "gemini-3-flash-preview:cloud")
    MODEL_REASON: str = os.getenv("MODEL_REASON", "qwen3.5:397b-cloud")
    MODEL_CODE: str = os.getenv("MODEL_CODE", "qwen3-coder:480b-cloud")
    MODEL_VISION: str = os.getenv("MODEL_VISION", "qwen3-vl:235b-cloud")
    MODEL_THINK: str = os.getenv("MODEL_THINK", "kimi-k2-thinking:cloud")
    MODEL_LONGCTX: str = os.getenv("MODEL_LONGCTX", "minimax-m2.5:cloud")

    MODEL_NAME: str = MODEL_REASON  # Default model (backward compat)

    @classmethod
    def validate_models_on_startup(cls) -> Dict[str, str]:
        """
        Validate all configured models and find best available.

        Call this once at startup to ensure models are available.
        Returns dict of role -> selected model.

        SECURITY: Thread-safe with locking.
        """
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
        with ThreadPoolExecutor(max_workers=6) as pool:
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
    def get_model(cls, role: str) -> str:
        """
        Thread-safe getter for model by role.

        Args:
            role: One of 'fast', 'reason', 'code', 'vision', 'think', 'longctx'

        Returns:
            Model name string
        """
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

    # PersonaPlex Configuration (Tool #17)
    # NOTE: Disabled by default due to huggingface-hub<1.0 conflict with transformers/sentence-transformers
    # moshi (required by PersonaPlex) requires huggingface-hub<1.0, but transformers/sentence-transformers
    # need huggingface-hub>=1.0 (breaking change in Dec 2024 release). They are incompatible.
    # To enable: export PERSONAPLEX_ENABLED=true (requires manual moshi installation & separate env)
    # Fallback: Use VoiceTool (pyttsx3/Whisper) or SesameTTS for voice functionality
    PERSONAPLEX_ENABLED: bool = os.getenv("PERSONAPLEX_ENABLED", "false").lower() == "true"

    # MirrorMind Configuration (Tool #21) - Self-Critique System
    MIRRORMIND_ENABLED: bool = os.getenv("MIRRORMIND_ENABLED", "true").lower() == "true"
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

    # Prompt Evolution Engine Configuration — Phase 4
    PROMPT_EVOLUTION_ENABLED: bool = os.getenv("PROMPT_EVOLUTION_ENABLED", "false").lower() == "true"
    PROMPT_EVOLUTION_INTERVAL: int = int(os.getenv("PROMPT_EVOLUTION_INTERVAL", "50"))

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

    # API Security Configuration
    API_AUTH_ENABLED: bool = os.getenv("AURA_API_AUTH_ENABLED", "false").lower() == "true"
    API_KEY: str = os.getenv("AURA_API_KEY", "")  # Set to enable API key auth
    API_RATE_LIMIT: int = int(os.getenv("AURA_API_RATE_LIMIT", "200"))  # requests per minute
    API_CORS_ORIGINS: str = os.getenv("AURA_CORS_ORIGINS", "*")  # wildcard required for WebSocket in Starlette 0.50+

    # AURA v3.0 ALIVE System Configuration
    AURA_ENABLED: bool = os.getenv("AURA_ENABLED", "true").lower() == "true"
    AURA_SOUL: str = os.getenv("AURA_SOUL", "SOUL_PERSONAL")  # SOUL_PERSONAL or SOUL_ENTERPRISE
    AURA_PROACTIVE: bool = os.getenv("AURA_PROACTIVE", "false").lower() == "true"  # Disabled to prevent event loop blocking
    AURA_THINKING: bool = os.getenv("AURA_THINKING", "true").lower() == "true"
    AURA_HUMANIZE: bool = os.getenv("AURA_HUMANIZE", "true").lower() == "true"

    AURA_ENV: str = os.getenv("AURA_ENV", "development")  # "development" or "production"

    @classmethod
    def is_production(cls) -> bool:
        return cls.AURA_ENV == "production"

    # Voice Configuration (Hybrid System)
    VOICE_CONFIG = {
        "default_mode": "pipeline",  # "pipeline" (Sesame) or "duplex" (PersonaPlex)
        "sesame": {
            "speaker": 0,           # Default speaker ID
            "sample_rate": 24000,
            "max_audio_length_ms": 30000
        },
        "personaplex": {
            "voice_prompt": "NATM1.pt",  # Natural Male 1
            "text_prompt": (
                "You are Aura, an intelligent AI assistant. "
                "You are wise, helpful, and occasionally witty with subtle sarcasm. "
                "You speak clearly and professionally."
            ),
            "cpu_offload": True  # Required for 8GB GPU
        }
    }

    # Vision model VRAM requirements (GB) — used by _can_fit_model() in vision.py
    VISION_MODEL_VRAM: Dict[str, float] = {
        "llava": 4.0,
        "llava:13b": 8.0,
        "llava:34b": 20.0,
        "minicpm-v": 6.0,
        "bakllava": 4.0,
    }

    # Florence-2 Vision (local HuggingFace model — for image preprocessing only)
    FLORENCE2_MODEL: str = "microsoft/Florence-2-base"
    FLORENCE2_ENABLED: bool = os.getenv("FLORENCE2_ENABLED", "true").lower() == "true"

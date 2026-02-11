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


# ============================================================================
#                    MODEL VALIDATION SYSTEM
# ============================================================================

# Verified working local models (via Ollama)
VERIFIED_LOCAL_MODELS = {
    "llama3:8b", "llama3:70b", "llama3.2", "llama3.2:3b",
    "qwen2:1.5b", "qwen2:7b", "qwen2.5:7b", "qwen2.5:14b",
    "qwen2.5-coder:7b", "qwen2.5-coder:14b",
    "qwen2.5-vl:7b",
    "deepseek-coder:6.7b", "deepseek-coder:33b",
    "mistral", "mistral:7b", "mixtral",
    "minicpm-v",
    "llava", "llava:7b", "llava:13b",
    "phi3", "phi3:mini",
    "codellama", "codellama:7b",
}

# Verified cloud models via Ollama.com (Pro $20/month subscription)
# Updated Feb 2026 with latest models
VERIFIED_CLOUD_MODELS = {
    # Reasoning - best for complex analysis
    "deepseek-v3.1:671b-cloud",   # Hybrid thinking/non-thinking, 671B params
    "cogito-2.1:671b-cloud",       # MIT license, general purpose
    "qwen3-next:80b-cloud",        # Efficient reasoning
    # Code - best for software engineering
    "devstral-2:123b-cloud",       # Codebase exploration, agents
    "glm-4.7-cloud",               # Advanced coding
    "devstral-small-2:24b-cloud",  # Lightweight code agent
    # Vision - best for multimodal
    "qwen3-vl:235b-cloud",         # Most powerful vision-language
    "kimi-k2.5-cloud",             # Native multimodal agentic
    # Legacy (still work)
    "gpt-oss:120b-cloud",
    "qwen3-coder:480b-cloud",
}

def validate_model(model_name: str, ollama_host: str = None) -> bool:
    """
    Check if a model is available in Ollama.

    SECURITY: Validates model exists before use to prevent runtime errors.
    """
    import requests

    host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        if response.status_code == 200:
            available = [m["name"] for m in response.json().get("models", [])]
            # Check exact match or base name match
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
    # MODEL CONFIGURATION - HYBRID LOCAL/CLOUD
    # ============================================================
    # Fast local models for quick responses, cloud for complex tasks
    # RTX 4060 (8GB VRAM) optimized

    # Model hierarchy (first available is used)
    # LOCAL FIRST for speed, CLOUD as quality fallback
    MODEL_FAST_CHAIN = ["mistral:7b", "llama3:8b", "qwen2:1.5b"]
    MODEL_REASON_CHAIN = ["llama3:8b", "deepseek-v3.1:671b-cloud", "cogito-2.1:671b-cloud", "kimi-k2.5-cloud", "mistral:7b"]
    MODEL_CODE_CHAIN = ["qwen2.5-coder:7b", "devstral-2:123b-cloud", "glm-4.7-cloud", "deepseek-coder:6.7b"]
    MODEL_VISION_CHAIN = ["minicpm-v", "qwen2.5-vl:7b", "llava", "kimi-k2.5-cloud", "qwen3-vl:235b-cloud", "llava:7b"]

    # Primary models - LOCAL for fast response
    MODEL_FAST: str = os.getenv("MODEL_FAST", "mistral:7b")
    MODEL_REASON: str = os.getenv("MODEL_REASON", "llama3:8b")
    MODEL_CODE: str = os.getenv("MODEL_CODE", "qwen2.5-coder:7b")
    MODEL_VISION: str = os.getenv("MODEL_VISION", "llava")

    MODEL_NAME: str = MODEL_REASON  # Default model (backward compat)

    # Cloud models for complex tasks (used by _select_model_for_complexity)
    # Updated Feb 2026 - Best models for $20/month Pro tier
    MODEL_REASON_CLOUD: str = "deepseek-v3.1:671b-cloud"   # 671B hybrid thinking model
    MODEL_CODE_CLOUD: str = "devstral-2:123b-cloud"         # Best for code agents
    MODEL_VISION_CLOUD: str = "qwen3-vl:235b-cloud"         # Most powerful vision-language

    # Local fallbacks
    MODEL_REASON_LOCAL: str = "llama3:8b"
    MODEL_CODE_LOCAL: str = "qwen2.5-coder:7b"
    MODEL_VISION_LOCAL: str = "llava"

    @classmethod
    def validate_models_on_startup(cls) -> Dict[str, str]:
        """
        Validate all configured models and find best available.

        Call this once at startup to ensure models are available.
        Returns dict of role -> selected model.

        SECURITY: Thread-safe with locking.
        """
        with _config_lock:
            results = {}

            # Validate each model type
            roles = [
                ("fast", cls.MODEL_FAST, cls.MODEL_FAST_CHAIN),
                ("reason", cls.MODEL_REASON, cls.MODEL_REASON_CHAIN),
                ("code", cls.MODEL_CODE, cls.MODEL_CODE_CHAIN),
                ("vision", cls.MODEL_VISION, cls.MODEL_VISION_CHAIN),
            ]

            for role, preferred, fallbacks in roles:
                selected = get_best_available_model(preferred, fallbacks, role)
                results[role] = selected

                # Update class attribute with validated model
                if role == "fast":
                    cls.MODEL_FAST = selected
                elif role == "reason":
                    cls.MODEL_REASON = selected
                    cls.MODEL_NAME = selected
                elif role == "code":
                    cls.MODEL_CODE = selected
                elif role == "vision":
                    cls.MODEL_VISION = selected

            return results

    @classmethod
    def get_model(cls, role: str) -> str:
        """
        Thread-safe getter for model by role.

        Args:
            role: One of 'fast', 'reason', 'code', 'vision'

        Returns:
            Model name string
        """
        with _config_lock:
            role_map = {
                'fast': cls.MODEL_FAST,
                'reason': cls.MODEL_REASON,
                'code': cls.MODEL_CODE,
                'vision': cls.MODEL_VISION,
                'default': cls.MODEL_NAME,
            }
            return role_map.get(role, cls.MODEL_NAME)

    @classmethod
    def set_model(cls, role: str, model: str) -> bool:
        """
        Thread-safe setter for model by role.

        Args:
            role: One of 'fast', 'reason', 'code', 'vision'
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
                'reason_cloud': cls.MODEL_REASON_CLOUD,
                'code_cloud': cls.MODEL_CODE_CLOUD,
                'vision_cloud': cls.MODEL_VISION_CLOUD,
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
    PERSONAPLEX_ENABLED: bool = os.getenv("PERSONAPLEX_ENABLED", "true").lower() == "true"

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
    STRATEGY_BANDIT_EVAL_ENABLED: bool = os.getenv("STRATEGY_BANDIT_EVAL_ENABLED", "false").lower() == "true"

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
    API_RATE_LIMIT: int = int(os.getenv("AURA_API_RATE_LIMIT", "60"))  # requests per minute
    API_CORS_ORIGINS: str = os.getenv("AURA_CORS_ORIGINS", "*")  # comma-separated origins, or * for dev

    # AURA v3.0 ALIVE System Configuration
    AURA_ENABLED: bool = os.getenv("AURA_ENABLED", "true").lower() == "true"
    AURA_SOUL: str = os.getenv("AURA_SOUL", "SOUL_PERSONAL")  # SOUL_PERSONAL or SOUL_ENTERPRISE
    AURA_PROACTIVE: bool = os.getenv("AURA_PROACTIVE", "false").lower() == "true"  # Disabled to prevent event loop blocking
    AURA_THINKING: bool = os.getenv("AURA_THINKING", "true").lower() == "true"
    AURA_HUMANIZE: bool = os.getenv("AURA_HUMANIZE", "true").lower() == "true"

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

    # VRAM Management
    GPU_VRAM_GB: int = 8  # RTX 4060
    SESAME_VRAM_GB: float = 4.5
    PERSONAPLEX_VRAM_GB: float = 8.0

    # Florence-2 Vision (HuggingFace transformers, not Ollama)
    FLORENCE2_MODEL: str = "microsoft/Florence-2-base"
    FLORENCE2_ENABLED: bool = os.getenv("FLORENCE2_ENABLED", "true").lower() == "true"
    VISION_VRAM_BUDGET_GB: float = float(os.getenv("VISION_VRAM_BUDGET_GB", "2.0"))

    # Estimated VRAM usage per vision model (GB)
    VISION_MODEL_VRAM = {
        "florence-2-base": 0.5,
        "qwen2.5-vl:7b": 5.0,
        "minicpm-v": 4.0,
        "llava": 4.5,
        "llava:7b": 4.5,
        "llava:13b": 8.0,
    }

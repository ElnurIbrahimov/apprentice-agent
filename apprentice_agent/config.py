"""Configuration management for the agent."""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
#                    MODEL VALIDATION SYSTEM
# ============================================================================

# Verified working local models (via Ollama)
VERIFIED_LOCAL_MODELS = {
    "llama3:8b", "llama3:70b", "llama3.2", "llama3.2:3b",
    "qwen2:1.5b", "qwen2:7b", "qwen2.5:7b", "qwen2.5:14b",
    "qwen2.5-coder:7b", "qwen2.5-coder:14b",
    "deepseek-coder:6.7b", "deepseek-coder:33b",
    "mistral", "mistral:7b", "mixtral",
    "llava", "llava:7b", "llava:13b",
    "phi3", "phi3:mini",
    "codellama", "codellama:7b",
}

# Verified cloud models via Ollama.com (may or may not be available)
VERIFIED_CLOUD_MODELS = {
    "gpt-oss:120b-cloud",
    "qwen3-coder:480b-cloud",
    "qwen3-vl:235b-cloud",
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
    # MODEL CONFIGURATION (Ollama - 100% FREE)
    # ============================================================
    # Uses local models by default with cloud fallback

    # Model hierarchy (first available is used)
    MODEL_FAST_CHAIN = ["qwen2:1.5b", "phi3:mini", "llama3.2:1b"]
    MODEL_REASON_CHAIN = ["gpt-oss:120b-cloud", "llama3:8b", "mistral:7b", "qwen2.5:7b"]
    MODEL_CODE_CHAIN = ["qwen3-coder:480b-cloud", "qwen2.5-coder:7b", "deepseek-coder:6.7b", "codellama:7b"]
    MODEL_VISION_CHAIN = ["qwen3-vl:235b-cloud", "llava", "llava:7b"]

    # Primary models (with auto-fallback on startup)
    MODEL_FAST: str = os.getenv("MODEL_FAST", "qwen2:1.5b")
    MODEL_REASON: str = os.getenv("MODEL_REASON", "llama3:8b")  # Safe local default
    MODEL_CODE: str = os.getenv("MODEL_CODE", "qwen2.5-coder:7b")  # Safe local default
    MODEL_VISION: str = os.getenv("MODEL_VISION", "llava")  # Safe local default

    MODEL_NAME: str = MODEL_REASON  # Default model (backward compat)

    # Explicit fallbacks
    MODEL_REASON_LOCAL: str = "llama3:8b"
    MODEL_CODE_LOCAL: str = "qwen2.5-coder:7b"
    MODEL_VISION_LOCAL: str = "llava"

    @classmethod
    def validate_models_on_startup(cls) -> Dict[str, str]:
        """
        Validate all configured models and find best available.

        Call this once at startup to ensure models are available.
        Returns dict of role -> selected model.
        """
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

    # AURA v3.0 ALIVE System Configuration
    AURA_ENABLED: bool = os.getenv("AURA_ENABLED", "true").lower() == "true"
    AURA_SOUL: str = os.getenv("AURA_SOUL", "SOUL_PERSONAL")  # SOUL_PERSONAL or SOUL_ENTERPRISE
    AURA_PROACTIVE: bool = os.getenv("AURA_PROACTIVE", "true").lower() == "true"
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

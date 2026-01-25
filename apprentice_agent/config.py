"""Configuration management for the agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    CHROMADB_PATH: Path = Path(os.getenv("CHROMADB_PATH", "./data/chromadb"))

    # Multi-model routing
    MODEL_FAST: str = "qwen2:1.5b"      # Simple tasks, greetings, short answers
    MODEL_REASON: str = "llama3:8b"     # Reasoning, planning, complex decisions
    MODEL_CODE: str = "deepseek-coder:6.7b"  # Code generation, debugging, scripts
    MODEL_VISION: str = "llava"         # Vision/image analysis
    MODEL_NAME: str = MODEL_REASON      # Default model (backward compat)

    MEMORY_COLLECTION_NAME: str = "agent_memory"
    MAX_MEMORY_RESULTS: int = 5

    # PersonaPlex Configuration (Tool #17)
    PERSONAPLEX_ENABLED: bool = os.getenv("PERSONAPLEX_ENABLED", "true").lower() == "true"

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

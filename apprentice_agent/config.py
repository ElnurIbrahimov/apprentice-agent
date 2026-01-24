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

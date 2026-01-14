"""Configuration management for the agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    CHROMADB_PATH: Path = Path(os.getenv("CHROMADB_PATH", "./data/chromadb"))
    MODEL_NAME: str = "llama3:8b"
    MEMORY_COLLECTION_NAME: str = "agent_memory"
    MAX_MEMORY_RESULTS: int = 5

"""
AURA v3.0 - ALIVE SYSTEM
========================
An emotionally present, proactive AI assistant.

Components:
- llm: Ollama LLM client with context injection
- memory: Markdown-based persistent memory + retriever
- emotion: Mood and emotional state management
- proactive: Heartbeat and background monitoring
- patterns: Cross-conversation pattern recognition
- thinking: Visible internal reasoning
- humanize: Response naturalization
- soul: Core personality configuration
- fast_path: Quick command handling (minimal)
"""

__version__ = "3.1.0"
__codename__ = "ALIVE"

# Main entry point
from .engine import AURAEngine, create_aura

__all__ = ["AURAEngine", "create_aura"]

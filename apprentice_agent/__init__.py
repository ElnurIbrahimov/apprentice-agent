"""Apprentice Agent - An AI agent with memory and reasoning capabilities."""

from .agent import ApprenticeAgent
from .memory import MemorySystem
from .brain import OllamaBrain

__version__ = "0.1.0"
__all__ = ["ApprenticeAgent", "MemorySystem", "OllamaBrain"]

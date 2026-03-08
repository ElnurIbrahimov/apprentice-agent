"""AURA - Autonomous Universal Reasoning Agent."""

__version__ = "4.3.0"
__all__ = ["ApprenticeAgent", "MemorySystem", "OllamaBrain"]


def __getattr__(name):
    if name == "ApprenticeAgent":
        from .agent import ApprenticeAgent
        return ApprenticeAgent
    if name == "MemorySystem":
        from .memory import MemorySystem
        return MemorySystem
    if name == "OllamaBrain":
        from .brain import OllamaBrain
        return OllamaBrain
    raise AttributeError(f"module 'aura' has no attribute {name!r}")

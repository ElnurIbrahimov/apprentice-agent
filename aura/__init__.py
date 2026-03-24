"""AURA - Autonomous Universal Reasoning Agent."""

__version__ = "4.6.0"
__all__ = ["ApprenticeAgent", "OllamaBrain"]


def __getattr__(name):
    if name == "ApprenticeAgent":
        from .agent import ApprenticeAgent
        return ApprenticeAgent
    if name == "OllamaBrain":
        from .brain import OllamaBrain
        return OllamaBrain
    raise AttributeError(f"module 'aura' has no attribute {name!r}")
